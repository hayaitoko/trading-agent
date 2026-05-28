"""The bench: register competitors, fan price data to all of them, run a
cadence of autonomous decisions, and rank by paper P&L.

Thread-safety: the feed thread calls :meth:`observe_bar` / :meth:`observe_quote`,
a cadence thread calls :meth:`run_decisions`, and the web thread reads
:meth:`snapshot` / mutates the roster — all guarded by one ``RLock``.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..llm.trader import decision_to_signal
from ..paper_broker import PaperBroker
from ..risk_manager import RiskLimits, RiskManager
from ..signal_router import _signal_to_order

if TYPE_CHECKING:
    from ..audit import AuditLogger
    from ..llm.trader import Trader


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


@dataclass
class DecisionLogEntry:
    timestamp: str
    competitor: str
    symbol: str
    action: str
    quantity: float
    status: str  # filled | rejected | blocked | error | hold
    reason: str = ""
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "competitor": self.competitor,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class Competitor:
    name: str
    trader: Trader
    broker: PaperBroker
    risk: RiskManager
    initial_balance: float = 100_000.0
    style: str | None = None
    decisions: deque[DecisionLogEntry] = field(default_factory=lambda: deque(maxlen=50))
    last_comment: str = ""
    error: str | None = None
    decision_count: int = 0


class Bench:
    def __init__(
        self,
        symbols: list[str],
        *,
        initial_balance: float = 100_000.0,
        max_position_size: float = 1_000.0,
        audit: AuditLogger | None = None,
    ) -> None:
        self.symbols = list(symbols)
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.audit = audit
        self._competitors: dict[str, Competitor] = {}
        self._last_prices: dict[str, float] = {}
        self._lock = threading.RLock()
        self.started_at: str | None = None

    # --- Roster -------------------------------------------------------------

    def add_competitor(
        self,
        name: str,
        trader: Trader,
        *,
        initial_balance: float | None = None,
        max_position_size: float | None = None,
        style: str | None = None,
    ) -> Competitor:
        """Register a competitor with its own paper book.

        ``initial_balance`` / ``max_position_size`` default to the bench-wide
        values but can be set per trader (the add-trader wizard's starting cash).
        ``style`` is recorded for reporting; the trader itself carries the prompt.
        """
        with self._lock:
            if name in self._competitors:
                raise ValueError(f"Competitor {name!r} already registered")
            balance = self.initial_balance if initial_balance is None else float(initial_balance)
            max_pos = (
                self.max_position_size if max_position_size is None else float(max_position_size)
            )
            broker = PaperBroker(initial_balance=balance)
            broker.connect()
            # seed any known prices so valuation/fills work immediately
            if self._last_prices:
                broker.update_market_prices(dict(self._last_prices))
            risk = RiskManager(
                limits=RiskLimits(max_position_size=max_pos),
                kill_switch_file=None,
            )
            comp = Competitor(
                name=name,
                trader=trader,
                broker=broker,
                risk=risk,
                initial_balance=balance,
                style=style,
            )
            self._competitors[name] = comp
            return comp

    def remove_competitor(self, name: str) -> None:
        with self._lock:
            self._competitors.pop(name, None)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._competitors)

    # --- Market data fan-out ------------------------------------------------

    def observe_bar(self, bar: dict[str, Any]) -> None:
        symbol, close = bar.get("symbol"), bar.get("close")
        with self._lock:
            if symbol is not None and close is not None:
                self._last_prices[str(symbol)] = float(close)
                for comp in self._competitors.values():
                    comp.broker.update_market_prices({str(symbol): float(close)})
            for comp in self._competitors.values():
                comp.trader.observe(bar)
            self._check_hard_floors()

    def observe_quote(self, quote: dict[str, Any]) -> None:
        symbol = quote.get("symbol")
        if symbol is None:
            return
        bid, ask = quote.get("bid"), quote.get("ask")
        last = quote.get("price") or quote.get("last")
        with self._lock:
            if last is not None:
                self._last_prices[str(symbol)] = float(last)
            for comp in self._competitors.values():
                comp.broker.update_quote(
                    str(symbol),
                    bid=float(bid) if bid is not None else None,
                    ask=float(ask) if ask is not None else None,
                    last=float(last) if last is not None else None,
                )
            self._check_hard_floors()

    # --- Decision cadence ---------------------------------------------------

    def run_decisions(self) -> None:
        """One cadence tick: every competitor decides and (maybe) trades."""
        with self._lock:
            competitors = list(self._competitors.values())
        for comp in competitors:
            self._run_one(comp)

    def _run_one(self, comp: Competitor) -> None:
        account = {
            "cash": comp.broker.get_balance()["cash"],
            "positions": comp.broker.get_positions(),
        }
        result = comp.trader.decide(account)
        with self._lock:
            comp.decision_count += 1
            comp.last_comment = result.comment
            comp.error = result.error
            if result.error:
                comp.decisions.appendleft(
                    DecisionLogEntry(
                        _utcnow_iso(), comp.name, "-", "ERROR", 0.0, "error", detail=result.error
                    )
                )
                self._audit("error", comp.name, {"error": result.error})
                return
            for decision in result.decisions:
                self._apply_decision(comp, decision)

    def _apply_decision(self, comp: Competitor, decision: Any) -> None:
        signal = decision_to_signal(decision)
        if signal is None:
            return  # HOLD / zero qty
        log = lambda status, detail="": comp.decisions.appendleft(  # noqa: E731
            DecisionLogEntry(
                _utcnow_iso(), comp.name, decision.symbol, decision.action,
                decision.quantity, status, reason=decision.reason, detail=detail,
            )
        )
        if comp.risk.check_kill_switch():
            log("blocked", "kill switch")
            return
        if comp.risk.check_position_size(comp.name, signal["amount"]):
            log("blocked", "exceeds max position size")
            self._audit("position_size_blocked", comp.name, {"signal": signal})
            return
        try:
            result = comp.broker.place_order(_signal_to_order(signal))
        except Exception as exc:  # broker rejects, e.g. bad order
            log("error", str(exc))
            return
        status = (result or {}).get("status", "UNKNOWN")
        log("filled" if status == "FILLED" else "rejected", f"status={status}")
        self._audit("trade", comp.name, {"order": result})

    # --- Reporting ----------------------------------------------------------

    def leaderboard(self) -> list[dict[str, Any]]:
        with self._lock:
            prices = dict(self._last_prices)
            rows = []
            for comp in self._competitors.values():
                value = comp.broker.get_account_value(prices)
                base = comp.initial_balance
                pnl = value - base
                wins, losses = comp.broker.get_win_loss()
                rows.append(
                    {
                        "name": comp.name,
                        "model": getattr(comp.trader, "model", comp.name),
                        "account_value": value,
                        "initial_balance": base,
                        "cash": comp.broker.get_balance()["cash"],
                        "pnl": pnl,
                        "return_pct": (pnl / base * 100.0) if base else 0.0,
                        "realized_pnl": comp.broker.get_realized_pnl(),
                        "wins": wins,
                        "losses": losses,
                        "positions": comp.broker.get_positions(),
                        "trades": len(comp.broker.get_trade_history()),
                        "decisions": comp.decision_count,
                        "style": comp.style,
                        "last_comment": comp.last_comment,
                        "error": comp.error,
                    }
                )
        rows.sort(key=lambda r: r["account_value"], reverse=True)
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        return rows

    def recent_decisions(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            merged: list[DecisionLogEntry] = []
            for comp in self._competitors.values():
                merged.extend(comp.decisions)
        merged.sort(key=lambda e: e.timestamp, reverse=True)
        return [e.as_dict() for e in merged[:limit]]

    def snapshot(self) -> dict[str, Any]:
        return {
            "generated_at": _utcnow_iso(),
            "started_at": self.started_at,
            "symbols": self.symbols,
            "initial_balance": self.initial_balance,
            "last_prices": dict(self._last_prices),
            "leaderboard": self.leaderboard(),
            "recent_decisions": self.recent_decisions(),
        }

    def run_decisions_for_symbol(self, symbol: str) -> None:
        """Off-cadence tick for competitors that hold a position in ``symbol``.

        Called by the event-driven wake hook (P2) when a significant price move
        is detected. Only competitors with a non-zero position in the affected
        symbol are woken, keeping the blast radius narrow.
        """
        with self._lock:
            affected = [
                comp for comp in self._competitors.values()
                if comp.broker.get_position(symbol) is not None
            ]
        for comp in affected:
            self._run_one(comp)

    # --- internals ----------------------------------------------------------

    def _check_hard_floors(self) -> None:
        """Flatten any book whose equity has breached its catastrophic loss floor.

        Must be called while holding ``_lock`` (observe_bar / observe_quote do so).
        Default-off: fires only when ``comp.risk.limits.hard_floor_pct`` is set.
        """
        prices = dict(self._last_prices)
        for comp in self._competitors.values():
            if not comp.risk.check_hard_floor(
                comp.broker.get_account_value(prices),
                comp.initial_balance,
            ):
                continue
            comp.broker.flatten_all()
            comp.decisions.appendleft(
                DecisionLogEntry(
                    _utcnow_iso(), comp.name, "*", "FLATTEN", 0.0, "filled",
                    reason="hard floor breached",
                )
            )
            self._audit("hard_floor", comp.name, {})

    def _audit(self, event: str, competitor: str, details: dict[str, Any]) -> None:
        if self.audit is None:
            return
        level = "ERROR" if event in ("error",) else "WARN" if "blocked" in event else "INFO"
        self.audit.log(level, event, module=f"bench:{competitor}", details=details)
