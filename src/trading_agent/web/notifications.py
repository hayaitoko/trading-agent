"""Aggregate the three alert sources into one notification feed.

Sources:
    * **Approvals** — pending proposals from :class:`ApprovalQueue` (actionable).
    * **Risk** — WARN/ERROR rows from the ``audit_log`` table written by
      :class:`AuditLogger` (kill switch, blocked orders, dispatch failures).
    * **Market** — recent moves from :class:`MarketMoveWatcher`.

Everything here is read-only; approve/reject happens in the web layer against
the queue directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..approval_queue import ApprovalQueue
    from ..db import DatabaseManager
    from .market_watch import MarketMoveWatcher


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


# Map raw audit messages to human-facing titles + severities. Anything not
# listed falls back to the raw message at "warning" severity.
_RISK_TITLES: dict[str, str] = {
    "kill_switch_active": "Kill switch active — trading halted",
    "position_size_blocked": "Order blocked: exceeds max position size",
    "hourly_trade_limit_blocked": "Order blocked: hourly trade limit reached",
    "daily_loss_limit_blocked": "Order blocked: daily loss limit reached",
    "dispatch_failed": "Order dispatch failed",
}


@dataclass
class Notification:
    """A single UI-facing alert, normalized across all sources."""

    id: str
    kind: str  # "approval" | "risk" | "market"
    severity: str  # "action" | "critical" | "warning" | "info"
    title: str
    body: str
    timestamp: str
    actionable: bool = False
    proposal_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class NotificationCenter:
    """Read-side aggregator over approvals, audit log, and market moves.

    Args:
        approval_queue: source of pending, actionable approvals.
        db: database manager whose ``audit_log`` table holds risk events.
        market_watch: source of recent market-move events.
        account_provider: optional callable returning an account snapshot dict
            (cash / positions / market value / pnl) for the UI header.
    """

    def __init__(
        self,
        approval_queue: ApprovalQueue,
        db: DatabaseManager,
        market_watch: MarketMoveWatcher,
        account_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.approval_queue = approval_queue
        self.db = db
        self.market_watch = market_watch
        self.account_provider = account_provider

    # --- Sections -----------------------------------------------------------

    def pending_approvals(self) -> list[Notification]:
        out: list[Notification] = []
        for record in self.approval_queue.pending():
            sig = record.signal
            side = str(sig.get("side", "?")).upper()
            asset = sig.get("asset", "?")
            amount = sig.get("amount", sig.get("quantity"))
            price = sig.get("price")
            body_bits = [f"{side} {amount} {asset}"]
            if price is not None:
                body_bits.append(f"@ {price:,.2f}")
            if sig.get("reason"):
                body_bits.append(f"— {sig['reason']}")
            out.append(
                Notification(
                    id=f"approval:{record.proposal_id}",
                    kind="approval",
                    severity="action",
                    title=f"Approve {side} {asset}?",
                    body=" ".join(body_bits),
                    timestamp=record.created_at.isoformat(),
                    actionable=True,
                    proposal_id=record.proposal_id,
                    data={"signal": sig, "expires_at": record.expires_at.isoformat()},
                )
            )
        return out

    def risk_alerts(self, limit: int = 20) -> list[Notification]:
        rows = self._fetch_risk_rows(limit)
        out: list[Notification] = []
        for row in rows:
            message = row["message"]
            level = (row["level"] or "WARN").upper()
            title = _RISK_TITLES.get(message, message.replace("_", " ").capitalize())
            out.append(
                Notification(
                    id=f"risk:{row['id']}",
                    kind="risk",
                    severity="critical" if level == "ERROR" else "warning",
                    title=title,
                    body=self._risk_body(row),
                    timestamp=row["timestamp"],
                    data={"module": row["module"], "level": level, "raw": message},
                )
            )
        return out

    def market_alerts(self, limit: int = 20) -> list[Notification]:
        out: list[Notification] = []
        for move in self.market_watch.recent(limit):
            arrow = "▲" if move.direction == "up" else "▼"
            pct = move.pct_change * 100.0
            out.append(
                Notification(
                    id=f"market:{move.symbol}:{move.timestamp}",
                    kind="market",
                    severity="info",
                    title=f"{move.symbol} {arrow} {pct:+.1f}%",
                    body=(
                        f"{move.current_price:,.2f} vs session open "
                        f"{move.reference_price:,.2f}"
                    ),
                    timestamp=move.timestamp,
                    data=move.as_dict(),
                )
            )
        return out

    # --- Snapshot -----------------------------------------------------------

    def snapshot(self, *, risk_limit: int = 20, market_limit: int = 20) -> dict[str, Any]:
        """Full payload for the UI: sections, counts, and optional account header."""
        approvals = self.pending_approvals()
        risk = self.risk_alerts(risk_limit)
        market = self.market_alerts(market_limit)
        payload: dict[str, Any] = {
            "generated_at": _utcnow_iso(),
            "counts": {
                "approvals": len(approvals),
                "risk": len(risk),
                "market": len(market),
            },
            "approvals": [n.as_dict() for n in approvals],
            "risk": [n.as_dict() for n in risk],
            "market": [n.as_dict() for n in market],
        }
        if self.account_provider is not None:
            try:
                payload["account"] = self.account_provider()
            except Exception as exc:  # never let a broker hiccup break the feed
                payload["account"] = {"error": str(exc)}
        return payload

    # --- Internals ----------------------------------------------------------

    def _fetch_risk_rows(self, limit: int) -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, timestamp, level, message, module, details "
                "FROM audit_log WHERE level IN ('WARN', 'ERROR') "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            cols = [c[0] for c in cursor.description]
            return [dict(zip(cols, row, strict=True)) for row in cursor.fetchall()]

    @staticmethod
    def _risk_body(row: dict[str, Any]) -> str:
        module = row.get("module")
        prefix = f"[{module}] " if module else ""
        details = row.get("details")
        if details and details not in ("{}", "null"):
            return f"{prefix}{details}"
        return f"{prefix}{row['message']}"
