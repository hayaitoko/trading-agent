"""Market-hours scheduler for the WS-Agent bench — A4.

Design role: sits above :class:`~trading_agent.bench.controller.BenchController`
and gates trader decision turns behind the ET-anchored live window.  The
controller's raw cadence clock keeps ticking; this scheduler decides whether
that tick should fire a turn and, if so, which turn type (SoD / regular / EoD /
event).

Integration with bench/controller.py (carry-over A4-c):
  A2 wired ``_do_scan_attention`` into :class:`BenchController`, which is the
  correct home for it — the controller has full access to bench state (last
  prices, pending approvals, competitors).  This scheduler does NOT duplicate
  that logic.  Instead, the controller calls :meth:`MarketScheduler.tick` once
  per cadence iteration; the scheduler delegates SoD/EoD/regular gating while
  the controller keeps owning the attention-queue scan.  This is the documented
  reconciliation of the A2 location discrepancy.

Callback wiring (carry-over A4-b):
  :class:`MarketScheduler` exposes :meth:`wire_pending_trade_callbacks`.  When
  the controller calls this for a newly registered PendingTradeQueue, the
  scheduler registers a closure that enqueues a callback turn for the affected
  trader on every status transition (approve, deny, TTL expire, confirm,
  abandon).  The callback turn fires via the bench's ``_run_one`` path with a
  synthetic ``wake_reason`` that describes the approval-state change.

Kill-switch soft halt (enforced at the ACT tool layer, NOT the scheduler):
  The kill switch is a *soft* halt: it blocks trading, not thinking.  When
  ``risk_manager.kill_switch_active`` is True, ACT tools return
  ``{ok:false, error:{kind:"unavailable", message:"bench halted by operator"}}``
  (A3's work).  This scheduler does NOT suppress turn firing and holds no
  risk-manager reference — SoD/EoD/regular/event/callback turns all continue to
  fire normally while the kill switch is active.  The trader still wakes, keeps
  full LOOK/NOTE access, and reaches ``hold()``/``pass()`` cleanly, so its state
  of mind is preserved for forensics.

After-hours protective-order fills:
  Tracked via :class:`~trading_agent.intel.lifecycle.LifecycleEngine`.
  A fill event that arrives during dormancy is queued and delivered as a
  dedicated event turn at the next live window, before SoD fires.

TZ-safety invariant:
  All time math is UTC-internal.  ET is used only for reading the market window
  from the Alpaca calendar.  No references to America/Los_Angeles or US/Pacific.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..intel.lifecycle import (
    AlpacaCalendar,
    LifecycleEngine,
    OrphanTurnStore,
)

if TYPE_CHECKING:
    from ..approval_queue import PendingTrade, PendingTradeQueue
    from ..bench.bench import Bench, Competitor

logger = logging.getLogger(__name__)


class MarketScheduler:
    """Market-hours scheduler: gates and classifies bench turns by ET window.

    Parameters
    ----------
    bench:
        The :class:`~trading_agent.bench.bench.Bench` instance.
    calendar:
        Shared :class:`AlpacaCalendar`.  Created internally when not provided.
    orphan_store:
        :class:`OrphanTurnStore` for crash recovery.  Created internally when
        not provided.
    api_key, api_secret:
        Alpaca credentials forwarded to :class:`AlpacaCalendar`.  Absent →
        static fallback market hours.
    """

    def __init__(
        self,
        bench: Bench,
        *,
        calendar: AlpacaCalendar | None = None,
        orphan_store: OrphanTurnStore | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        self.bench = bench
        self._calendar = calendar or AlpacaCalendar(api_key=api_key, api_secret=api_secret)
        self._orphan_store = orphan_store or OrphanTurnStore()
        self._engine = LifecycleEngine(self._calendar, self._orphan_store)
        self._lock = threading.Lock()

        # Pending-trade callback turn queue: list of (trader_id, wake_reason).
        # Drained on each tick() call; protected by _lock.
        self._callback_queue: list[tuple[str, str]] = []

        # Set of trader_ids for which crash-recovery has been attempted this session.
        self._recovered: set[str] = set()

    # --- Trader registration -------------------------------------------------

    def register_trader(
        self,
        trader_id: str,
        *,
        cadence_minutes: int = 30,
        extended_hours: bool = False,
    ) -> None:
        """Register a trader with lifecycle tracking.

        Should be called whenever a new trader is added to the bench.
        """
        self._engine.register_trader(
            trader_id,
            cadence_minutes=cadence_minutes,
            extended_hours=extended_hours,
        )

    def remove_trader(self, trader_id: str) -> None:
        """Remove a trader from lifecycle tracking."""
        self._engine.remove_trader(trader_id)

    # --- Tick entry point ----------------------------------------------------

    def tick(self, now: datetime | None = None) -> list[tuple[str, str, str]]:
        """Run one scheduler tick; return list of (trader_id, turn_type, wake_reason).

        Called by the bench controller on every cadence iteration (after
        ``bench.run_decisions()``).  Returns the turns that fired this tick so
        the controller can fire them via ``_run_one``.

        Note: the controller owns ``_scan_attention()`` — this tick does NOT
        call it (A4-c reconciliation).
        """
        if now is None:
            now = datetime.now(UTC)

        turns = self._engine.due_turns(now)

        # Drain the callback queue (approval-state changes from PendingTradeQueue).
        with self._lock:
            callbacks = list(self._callback_queue)
            self._callback_queue.clear()

        for trader_id, wake_reason in callbacks:
            turns.append((trader_id, "callback", wake_reason))

        return turns

    def fire_turns(self, turns: list[tuple[str, str, str]]) -> None:
        """Execute the turns returned by :meth:`tick` via the bench.

        Each turn wakes the affected trader with the appropriate turn type and
        wake reason.  Errors in individual turns never kill the loop.
        """
        for trader_id, turn_type, wake_reason in turns:
            try:
                comp = self.bench._competitors.get(trader_id)
                if comp is None:
                    continue
                self._fire_one(comp, turn_type, wake_reason)
            except Exception as exc:
                logger.warning("scheduler: turn %s/%s failed: %s", trader_id, turn_type, exc)

    def _fire_one(self, comp: Competitor, turn_type: str, wake_reason: str) -> None:
        """Wake one competitor with the given turn type and reason."""
        trader = comp.trader

        # Inject turn metadata into AgentTrader if supported.
        if hasattr(trader, "_current_turn_type"):
            trader._current_turn_type = turn_type  # type: ignore[attr-defined]
        if hasattr(trader, "_current_wake_reason"):
            trader._current_wake_reason = wake_reason  # type: ignore[attr-defined]

        # EoD: tell the trader no new positions (default strict).
        if turn_type == "EoD" and hasattr(trader, "_eod_no_new_positions"):
            trader._eod_no_new_positions = True  # type: ignore[attr-defined]

        self.bench._run_one(comp)

        # After EoD, reset the no-new-positions flag.
        if turn_type == "EoD" and hasattr(trader, "_eod_no_new_positions"):
            trader._eod_no_new_positions = False  # type: ignore[attr-defined]

        # After done_for_day terminal, update lifecycle state.
        if hasattr(trader, "_done_for_day_this_turn") and trader._done_for_day_this_turn:  # type: ignore[attr-defined]
            self._engine.mark_done_for_day(comp.name)
            trader._done_for_day_this_turn = False  # type: ignore[attr-defined]

    # --- Pending-trade callback wiring (carry-over A4-b) --------------------

    def wire_pending_trade_callbacks(
        self,
        pending_trade_queue: PendingTradeQueue,
        trader_id: str,
        pending_trade_id: str,
    ) -> None:
        """Register a callback so approval-state changes fire a new decide() turn.

        Called by the ACT toolkit after proposing a trade.  On approve, deny, or
        TTL expiry, a callback turn is enqueued for the trader.  The callback
        turn fires with the appropriate wake_reason so the trader knows what
        happened and can respond (confirm_trade / abandon_trade / reassess).

        This is carry-over A4-b: A3 built the register_callback mechanism; A4
        connects it to the lifecycle engine's turn queue.
        """
        def _on_status_change(pt: PendingTrade) -> None:
            status = pt.status
            if status == "approved":
                reason = (
                    f"trade_id={pending_trade_id} was approved at "
                    f"{_format_utc(pt.approved_at)}, "
                    f"pre-approval TTL {_ttl_remaining(pt)} remaining"
                )
            elif status == "denied":
                note = pt.note or "no reason given"
                reason = f"trade_id={pending_trade_id} was denied ({note})"
            elif status == "expired":
                reason = f"trade_id={pending_trade_id} approval expired (TTL elapsed)"
            else:
                return  # confirmed/abandoned don't need a callback turn

            with self._lock:
                self._callback_queue.append((trader_id, reason))

        pending_trade_queue.register_callback(pending_trade_id, _on_status_change)

    # --- Crash recovery (carry-over A4-a) -----------------------------------

    def recover_orphans(self) -> None:
        """Detect and re-queue orphaned turns from a previous crash.

        Called once at startup by the controller.  For each orphaned turn,
        queues a recovery turn with the ORIGINAL turn_id (not a fresh UUID)
        so idempotency keys match and double-fires are caught by the
        PendingTradeQueue UNIQUE constraint.

        The trader receives a first-look block with previous_attempt_tools
        populated from the orphaned turn's tool_names_called list.

        Design note (in commit message as required): direct-execution trades
        (no PendingTradeQueue) rely on turn_id reuse alone for dedup, because
        the risk manager's in-memory idempotency set resets on restart.  A crash
        between "broker filled" and "turn completed" could theoretically double-
        fire a direct trade.  This is an accepted limitation for the current
        scope.  DB-UNIQUE durable dedup for direct trades is deferred as a
        future hardening step.
        """
        for orphan in self._orphan_store.all_orphans():
            if orphan.trader_id in self._recovered:
                continue
            self._recovered.add(orphan.trader_id)

            comp = self.bench._competitors.get(orphan.trader_id)
            if comp is None:
                # Trader no longer exists; clean up.
                self._orphan_store.complete(orphan.turn_id)
                continue

            trader = comp.trader
            if not hasattr(trader, "_current_turn_id"):
                # Not an AgentTrader; skip.
                self._orphan_store.complete(orphan.turn_id)
                continue

            logger.info(
                "scheduler: recovering orphan turn %s for trader %s "
                "(tools called: %s)",
                orphan.turn_id,
                orphan.trader_id,
                orphan.tool_names_called,
            )

            # Inject the orphan turn_id and previous-attempt annotation.
            trader._current_turn_id = orphan.turn_id  # type: ignore[attr-defined]
            trader._recovery_previous_attempt = list(orphan.tool_names_called)  # type: ignore[attr-defined]

            # Fire a recovery turn immediately (event type, special wake reason).
            wake_reason = (
                f"crash-recovery: previous turn {orphan.turn_id} did not complete "
                f"(tools called: {', '.join(orphan.tool_names_called) or 'none'})"
            )
            try:
                self._fire_one(comp, "event", wake_reason)
            except Exception as exc:
                logger.warning("scheduler: recovery turn failed for %s: %s", orphan.trader_id, exc)

            # Clear orphan record after recovery attempt.
            self._orphan_store.complete(orphan.turn_id)

    # --- After-hours fill event queueing -----------------------------------

    def queue_ah_fill(
        self,
        trader_id: str,
        symbol: str,
        fill_type: str,
        fill_time_et: str,
    ) -> None:
        """Queue an after-hours protective-order fill for the next live window.

        Called when a stop/TP fires during dormancy.  The event is NOT merged
        into SoD — it fires as a dedicated event turn before SoD.
        """
        self._engine.queue_ah_fill_event(trader_id, symbol, fill_type, fill_time_et)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_utc(dt: datetime | None) -> str:
    if dt is None:
        return "unknown time"
    return dt.strftime("%H:%M UTC")


def _ttl_remaining(pt: PendingTrade) -> str:
    expires = pt.approval_ttl_expires_at
    if expires is None:
        return "unknown"
    # approval_ttl_expires_at from PendingTradeQueue is stored as naive UTC.
    now_utc = datetime.now(UTC).replace(tzinfo=None)
    expires_naive = expires.replace(tzinfo=None) if expires.tzinfo is not None else expires
    remaining = (expires_naive - now_utc).total_seconds()
    if remaining <= 0:
        return "expired"
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    return f"{minutes}m{seconds:02d}s"
