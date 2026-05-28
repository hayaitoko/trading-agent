"""SQLite-backed approval queue for human-in-the-loop trade dispatch.

Two independent surfaces in this module:

1. :class:`ApprovalQueue` — original proposal-and-execute queue.  SignalRouter
   hands proposals here; a consumer calls :meth:`~ApprovalQueue.approve` or
   :meth:`~ApprovalQueue.reject`; the executor fires on approve.  All existing
   callers unchanged.

2. :class:`PendingTradeQueue` — WS-Agent A3 pre-approval lineage.  Stores the
   full ``propose → approve/deny → confirm/abandon/expire`` lifecycle for agent
   trades that require human sign-off.  Callbacks registered via
   :meth:`~PendingTradeQueue.register_callback` fire synchronously on every
   status change (A4's scheduler will wire them to event-driven turns).

Supporting dataclasses: :class:`TradeIntent`, :class:`FillResult`,
:class:`PendingTrade`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Union

PREAPPROVAL_TTL_MIN: int = int(os.environ.get("PREAPPROVAL_TTL_MIN", 5))


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


PathLike = Union[str, Path]


@dataclass
class ApprovalRecord:
    proposal_id: str
    signal: dict[str, Any]
    status: str          # 'pending' | 'approved' | 'rejected' | 'expired'
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None = None
    note: str | None = None
    execution_result: dict[str, Any] | None = None


class ApprovalQueue:
    """Persistent approval queue.

    ``executor`` is the callable invoked on approval — typically a wrapped
    ``broker.place_order``. If unset, approve() raises.
    """

    DEFAULT_TIMEOUT_SECONDS = 300

    def __init__(
        self,
        db_path: PathLike = "data/approvals.db",
        executor: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        default_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.executor = executor
        self.default_timeout_seconds = default_timeout_seconds
        # check_same_thread=False + an RLock lets a single queue serve consumers
        # on different threads (e.g. a web request handler vs. the strategy loop).
        # The RLock is re-entrant because approve() calls process_expirations()/get().
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS approvals (
                proposal_id      TEXT PRIMARY KEY,
                signal_json      TEXT NOT NULL,
                status           TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                expires_at       TEXT NOT NULL,
                decided_at       TEXT,
                note             TEXT,
                execution_result TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status)"
        )

    # --- Public API ---------------------------------------------------------

    def add(self, signal: dict[str, Any], timeout_seconds: int | None = None) -> str:
        """Enqueue a proposal. Returns the proposal_id."""
        proposal_id = str(uuid.uuid4())
        now = _utcnow()
        ttl = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        expires = now + timedelta(seconds=ttl)
        with self._lock:
            self._conn.execute(
                "INSERT INTO approvals(proposal_id, signal_json, status, created_at, expires_at) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (proposal_id, json.dumps(signal, default=str), now.isoformat(), expires.isoformat()),
            )
        return proposal_id

    def approve(self, proposal_id: str, note: str | None = None) -> dict[str, Any] | None:
        """Approve and execute. Returns the executor's result."""
        if self.executor is None:
            raise RuntimeError("Cannot approve: ApprovalQueue has no executor configured")

        with self._lock:
            self.process_expirations()
            record = self.get(proposal_id)
            if record is None:
                raise KeyError(f"Proposal {proposal_id} not found")
            if record.status != "pending":
                raise ValueError(f"Proposal {proposal_id} is {record.status!r}, not pending")

            result = self.executor(record.signal)
            now = _utcnow().isoformat()
            self._conn.execute(
                "UPDATE approvals SET status='approved', decided_at=?, note=?, execution_result=? "
                "WHERE proposal_id=?",
                (now, note, json.dumps(result, default=str), proposal_id),
            )
        return result

    def reject(self, proposal_id: str, note: str | None = None) -> None:
        with self._lock:
            record = self.get(proposal_id)
            if record is None:
                raise KeyError(f"Proposal {proposal_id} not found")
            if record.status != "pending":
                raise ValueError(f"Proposal {proposal_id} is {record.status!r}, not pending")
            now = _utcnow().isoformat()
            self._conn.execute(
                "UPDATE approvals SET status='rejected', decided_at=?, note=? WHERE proposal_id=?",
                (now, note, proposal_id),
            )

    def pending(self) -> list[ApprovalRecord]:
        """Return all currently pending (and not-yet-expired) proposals."""
        with self._lock:
            self.process_expirations()
            rows = self._conn.execute(
                "SELECT * FROM approvals WHERE status='pending' ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get(self, proposal_id: str) -> ApprovalRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM approvals WHERE proposal_id=?", (proposal_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def process_expirations(self) -> int:
        """Mark any pending proposals past their expires_at as ``expired``.
        Returns the count moved.
        """
        now_iso = _utcnow().isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE approvals SET status='expired', decided_at=? "
                "WHERE status='pending' AND expires_at <= ?",
                (now_iso, now_iso),
            )
            return cursor.rowcount or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- Helpers ------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ApprovalRecord:
        return ApprovalRecord(
            proposal_id=row["proposal_id"],
            signal=json.loads(row["signal_json"]),
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            decided_at=datetime.fromisoformat(row["decided_at"]) if row["decided_at"] else None,
            note=row["note"],
            execution_result=json.loads(row["execution_result"]) if row["execution_result"] else None,
        )


# ---------------------------------------------------------------------------
# WS-Agent A3: PendingTrade lineage dataclasses + PendingTradeQueue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeIntent:
    """Proposed trade parameters — the immutable intent record.

    Stored verbatim in ``pending_trades.intent_json``; never mutated after
    creation so the audit log reflects exactly what the agent asked for.
    """

    symbol: str
    side: str   # "BUY" | "SELL"
    qty: float
    stop: float | None = None
    take_profit: float | None = None
    trail: float | None = None


@dataclass(frozen=True)
class FillResult:
    """Broker fill outcome attached to a confirmed :class:`PendingTrade`."""

    order_id: str
    symbol: str
    side: str
    qty_filled: float
    fill_price: float | None = None
    status: str = "filled"


@dataclass(frozen=True)
class PendingTrade:
    """Full lifecycle record for a trade that went through the approval gate.

    Snapshots the state at a point in time; the DB row is the mutable truth.
    ``PendingTradeQueue`` returns a fresh instance on every read.
    """

    pending_trade_id: str
    trader_id: str
    proposed: TradeIntent
    proposed_at: datetime
    idempotency_key: str
    status: Literal[
        "awaiting_approval", "approved", "denied",
        "confirmed", "abandoned", "expired",
    ]
    approved_at: datetime | None
    approval_ttl_expires_at: datetime | None
    confirmed_at: datetime | None
    fill_result: FillResult | None
    note: str | None = None


class PendingTradeQueue:
    """Pre-approval lineage store for WS-Agent ACT tools.

    Lifecycle: ``propose`` → operator ``set_decision("approved" | "denied")``
    → trader ``confirm`` or ``abandon`` → automatic TTL ``expire_old``.

    Callbacks registered via :meth:`register_callback` fire synchronously on
    every status transition (approve, deny, expiry).  A4's scheduler wires
    them to event-driven ``decide()`` turns; A3 makes the mechanism available.

    The underlying SQLite table ``pending_trades`` lives in the same DB file as
    :class:`ApprovalQueue` (default ``data/approvals.db``) for operational
    simplicity, but the two tables are fully independent.
    """

    PREAPPROVAL_TTL_MIN: int = PREAPPROVAL_TTL_MIN

    def __init__(self, db_path: PathLike = "data/approvals.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        # Callbacks: pending_trade_id → list[fn(PendingTrade)]
        self._callbacks: dict[str, list[Callable[[PendingTrade], None]]] = defaultdict(list)

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_trades (
                pending_trade_id        TEXT PRIMARY KEY,
                trader_id               TEXT NOT NULL,
                idempotency_key         TEXT NOT NULL UNIQUE,
                intent_json             TEXT NOT NULL,
                proposed_at             TEXT NOT NULL,
                status                  TEXT NOT NULL,
                approved_at             TEXT,
                approval_ttl_expires_at TEXT,
                confirmed_at            TEXT,
                fill_result_json        TEXT,
                note                    TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_trades_trader "
            "ON pending_trades(trader_id, status)"
        )

    # --- Public API ----------------------------------------------------------

    def propose(
        self,
        trader_id: str,
        intent: TradeIntent,
        idempotency_key: str,
    ) -> PendingTrade:
        """Enqueue a proposed trade.  Returns a PendingTrade with status='awaiting_approval'.

        Raises :exc:`ValueError` if the idempotency_key is a duplicate (prevents
        crash-replay double-fires at the DB level).
        """
        now = _utcnow()
        ptid = str(uuid.uuid4())
        intent_json = json.dumps(
            {
                "symbol": intent.symbol,
                "side": intent.side,
                "qty": intent.qty,
                "stop": intent.stop,
                "take_profit": intent.take_profit,
                "trail": intent.trail,
            }
        )
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO pending_trades"
                    "(pending_trade_id, trader_id, idempotency_key, intent_json,"
                    " proposed_at, status) VALUES (?, ?, ?, ?, ?, 'awaiting_approval')",
                    (ptid, trader_id, idempotency_key, intent_json, now.isoformat()),
                )
            except sqlite3.IntegrityError:
                raise ValueError(
                    f"duplicate trade: idempotency_key {idempotency_key!r} already enqueued"
                )
        return PendingTrade(
            pending_trade_id=ptid,
            trader_id=trader_id,
            proposed=intent,
            proposed_at=now,
            idempotency_key=idempotency_key,
            status="awaiting_approval",
            approved_at=None,
            approval_ttl_expires_at=None,
            confirmed_at=None,
            fill_result=None,
        )

    def set_decision(
        self,
        pending_trade_id: str,
        decision: str,
        *,
        note: str | None = None,
    ) -> PendingTrade:
        """Approve or deny a pending trade.  Fires registered callbacks.

        ``decision`` must be ``"approved"`` or ``"denied"``.
        Raises :exc:`KeyError` if not found; :exc:`ValueError` if wrong status.
        """
        if decision not in ("approved", "denied"):
            raise ValueError(f"decision must be 'approved' or 'denied', got {decision!r}")
        now = _utcnow()
        ttl_expires: str | None = None
        if decision == "approved":
            ttl_expires = (now + timedelta(minutes=self.PREAPPROVAL_TTL_MIN)).isoformat()

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM pending_trades WHERE pending_trade_id=?",
                (pending_trade_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"PendingTrade {pending_trade_id!r} not found")
            if row["status"] != "awaiting_approval":
                raise ValueError(
                    f"PendingTrade {pending_trade_id!r} is {row['status']!r}, "
                    "not awaiting_approval"
                )
            self._conn.execute(
                "UPDATE pending_trades SET status=?, approved_at=?, "
                "approval_ttl_expires_at=?, note=? WHERE pending_trade_id=?",
                (decision, now.isoformat(), ttl_expires, note, pending_trade_id),
            )
            pt = self._row_to_pending_trade(
                self._conn.execute(
                    "SELECT * FROM pending_trades WHERE pending_trade_id=?",
                    (pending_trade_id,),
                ).fetchone()
            )
        # Fire callbacks outside the lock to prevent deadlock if a callback re-enters.
        self._fire_callbacks(pending_trade_id, pt)
        return pt

    def confirm(
        self,
        pending_trade_id: str,
        executor: Callable[[TradeIntent], FillResult],
    ) -> tuple[PendingTrade, FillResult]:
        """Execute a pre-approved trade.

        ``executor`` receives the :class:`TradeIntent` and must return a
        :class:`FillResult` (or raise on broker failure).

        Raises :exc:`KeyError` if not found; :exc:`ValueError` on wrong status or
        expired TTL; :exc:`RuntimeError` bubbles up from ``executor``.
        """
        now = _utcnow()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM pending_trades WHERE pending_trade_id=?",
                (pending_trade_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"PendingTrade {pending_trade_id!r} not found")
            status = row["status"]
            if status != "approved":
                raise ValueError(
                    f"PendingTrade {pending_trade_id!r} is {status!r}, not approved"
                )
            ttl_str = row["approval_ttl_expires_at"]
            if ttl_str and datetime.fromisoformat(ttl_str) < now:
                self._conn.execute(
                    "UPDATE pending_trades SET status='expired' WHERE pending_trade_id=?",
                    (pending_trade_id,),
                )
                raise ValueError(
                    f"PendingTrade {pending_trade_id!r} pre-approval TTL expired"
                )
            intent = self._row_to_intent(row)

        # Execute outside the lock so broker I/O doesn't block other threads.
        fill = executor(intent)

        fill_json = json.dumps(
            {
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side,
                "qty_filled": fill.qty_filled,
                "fill_price": fill.fill_price,
                "status": fill.status,
            }
        )
        with self._lock:
            self._conn.execute(
                "UPDATE pending_trades SET status='confirmed', confirmed_at=?, "
                "fill_result_json=? WHERE pending_trade_id=?",
                (now.isoformat(), fill_json, pending_trade_id),
            )
            pt = self._row_to_pending_trade(
                self._conn.execute(
                    "SELECT * FROM pending_trades WHERE pending_trade_id=?",
                    (pending_trade_id,),
                ).fetchone()
            )
        return pt, fill

    def abandon(self, pending_trade_id: str) -> PendingTrade:
        """Release a pre-approved trade without executing it.

        Raises :exc:`KeyError` if not found; :exc:`ValueError` if already
        confirmed, denied, or expired.
        """
        now = _utcnow()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM pending_trades WHERE pending_trade_id=?",
                (pending_trade_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"PendingTrade {pending_trade_id!r} not found")
            if row["status"] not in ("awaiting_approval", "approved"):
                raise ValueError(
                    f"PendingTrade {pending_trade_id!r} is {row['status']!r}, cannot abandon"
                )
            self._conn.execute(
                "UPDATE pending_trades SET status='abandoned', confirmed_at=? "
                "WHERE pending_trade_id=?",
                (now.isoformat(), pending_trade_id),
            )
            return self._row_to_pending_trade(
                self._conn.execute(
                    "SELECT * FROM pending_trades WHERE pending_trade_id=?",
                    (pending_trade_id,),
                ).fetchone()
            )

    def expire_old(self) -> int:
        """Mark all approved-but-TTL-elapsed trades as 'expired'.

        Fires registered callbacks for each newly expired record.
        Returns the count of rows moved to 'expired'.
        """
        now_iso = _utcnow().isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT pending_trade_id FROM pending_trades "
                "WHERE status='approved' AND approval_ttl_expires_at IS NOT NULL "
                "AND approval_ttl_expires_at <= ?",
                (now_iso,),
            ).fetchall()
            newly_expired = [r["pending_trade_id"] for r in rows]
            if newly_expired:
                placeholders = ",".join("?" * len(newly_expired))
                self._conn.execute(
                    f"UPDATE pending_trades SET status='expired' "
                    f"WHERE pending_trade_id IN ({placeholders})",
                    newly_expired,
                )
        for ptid in newly_expired:
            pt = self.get(ptid)
            if pt is not None:
                self._fire_callbacks(ptid, pt)
        return len(newly_expired)

    def get(self, pending_trade_id: str) -> PendingTrade | None:
        """Return the current snapshot of a pending trade, or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM pending_trades WHERE pending_trade_id=?",
                (pending_trade_id,),
            ).fetchone()
        return self._row_to_pending_trade(row) if row else None

    def pending_for_trader(self, trader_id: str) -> list[PendingTrade]:
        """Return all awaiting_approval and approved trades for a trader."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM pending_trades WHERE trader_id=? "
                "AND status IN ('awaiting_approval', 'approved') "
                "ORDER BY proposed_at ASC",
                (trader_id,),
            ).fetchall()
        return [self._row_to_pending_trade(r) for r in rows]

    def register_callback(
        self,
        pending_trade_id: str,
        fn: Callable[[PendingTrade], None],
    ) -> None:
        """Register ``fn`` to be called when this trade's status changes.

        ``fn`` receives the updated :class:`PendingTrade` snapshot.  Callbacks
        fire on approve, deny, and TTL expiry.  Multiple callbacks per trade are
        allowed and fire in registration order.  Exceptions in callbacks are
        silently swallowed to protect the decision flow.
        """
        self._callbacks[pending_trade_id].append(fn)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- Helpers -------------------------------------------------------------

    def _fire_callbacks(self, pending_trade_id: str, pt: PendingTrade) -> None:
        for fn in list(self._callbacks.get(pending_trade_id, [])):
            try:
                fn(pt)
            except Exception:
                pass

    @staticmethod
    def _row_to_intent(row: sqlite3.Row) -> TradeIntent:
        d = json.loads(row["intent_json"])
        return TradeIntent(
            symbol=d["symbol"],
            side=d["side"],
            qty=float(d["qty"]),
            stop=d.get("stop"),
            take_profit=d.get("take_profit"),
            trail=d.get("trail"),
        )

    @staticmethod
    def _row_to_fill(row: sqlite3.Row) -> FillResult | None:
        raw = row["fill_result_json"]
        if not raw:
            return None
        d = json.loads(raw)
        return FillResult(
            order_id=d["order_id"],
            symbol=d["symbol"],
            side=d["side"],
            qty_filled=float(d["qty_filled"]),
            fill_price=d.get("fill_price"),
            status=d.get("status", "filled"),
        )

    @classmethod
    def _row_to_pending_trade(cls, row: sqlite3.Row) -> PendingTrade:
        return PendingTrade(
            pending_trade_id=row["pending_trade_id"],
            trader_id=row["trader_id"],
            proposed=cls._row_to_intent(row),
            proposed_at=datetime.fromisoformat(row["proposed_at"]),
            idempotency_key=row["idempotency_key"],
            status=row["status"],
            approved_at=datetime.fromisoformat(row["approved_at"])
            if row["approved_at"]
            else None,
            approval_ttl_expires_at=datetime.fromisoformat(row["approval_ttl_expires_at"])
            if row["approval_ttl_expires_at"]
            else None,
            confirmed_at=datetime.fromisoformat(row["confirmed_at"])
            if row["confirmed_at"]
            else None,
            fill_result=cls._row_to_fill(row),
            note=row["note"],
        )
