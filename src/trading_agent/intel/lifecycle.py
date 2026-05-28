"""Market-hours lifecycle engine for the WS-Agent trader — A4.

State diagram (prose):
  DORMANT → LIVE (on SoD trigger at T-60min before NYSE open)
  LIVE → DORMANT (on EoD trigger at T+30min after NYSE close)
  LIVE → LIVE (regular cadence every N minutes during RTH)
  LIVE → LIVE (event-driven wakes: watchpoint, reminder, approval-callback)
  DORMANT → DORMANT (research agent still runs; after-hours protective fills queued)

All time math is UTC-internal.  Market window boundaries are anchored to
America/New_York (ET) via the Alpaca calendar, which knows about half-days,
holidays, and DST.  The server's local timezone is never used for market logic.

Live window definition:
  T-60min before market open → T+30min after market close (ET, per Alpaca calendar)

Extended-hours window (opt-in per trader, default off):
  04:00–09:30 ET pre-market  +  16:00–20:00 ET after-hours

Dormant window:
  Anything outside live or extended-hours windows.  Research agent continues
  running independently — only the trader decision loop pauses.

Special turn types fired by the lifecycle engine:
  - SoD (Start-of-Day): at T-60min before open.  Trader absorbs overnight
    intel (reads whatever briefs exist via the research_brief() tool — the SoD
    guidance directs it to) and seeds watchpoints.
  - EoD (End-of-Day): at T+30min after close.  Trader reflects, locks
    overnight protections.  No new positions by default (configurable).
  - regular: per-trader cadence during RTH.
  - event: watchpoint trip, reminder fire, protective-order fill, etc.
  - callback: approval-state change (approve/deny/expire).

Pre-SoD research hydration (DEFERRED — plan §A4 contract item):
  The plan calls for the research agent's overnight batch to be hydrated into
  the research_brief() cache *before* SoD fires.  An explicit scheduler-triggered
  hydration handshake is NOT implemented here: the lifecycle engine holds no
  research-service reference, and triggering a (cost-gated, WS-C CostGate'd)
  research pass from the scheduler is out of A4's scope.  Functionally the trader
  still gets overnight research on the SoD turn — the research agent runs on its
  own background schedule and the trader pulls available briefs via the
  research_brief() LOOK tool (A1), which the SoD special-prompt guidance tells it
  to do.  The guaranteed-fresh pre-SoD trigger is deferred to the Situation+
  Forecast Track C integration, which wires providers/research as scheduled passes.

After-hours protective-order fills:
  When a stop/TP fires during dormancy, the event is queued and delivered
  as a dedicated event turn at the next live window (NOT merged into SoD).
  Message format: "Your stop on {symbol} hit at {HH:MM} ET while you were
  dormant."

Crash recovery (carry-over A4-a):
  On restart, the engine detects orphaned turns (started but no terminal action).
  It fires a new turn with the orphaned turn_id (NOT a fresh UUID) to preserve
  idempotency-key parity.  The previous_attempt field in TurnContext is populated
  with tool names called in the orphaned turn.  See :class:`OrphanTurnStore`.

TZ-safety invariant:
  This module imports ONLY ``zoneinfo.ZoneInfo("America/New_York")`` for ET
  anchoring.  Never imports or references America/Los_Angeles, US/Pacific, PT,
  PST, or PDT.  All calendar comparisons happen in UTC; ET conversion is done
  only at the boundary (open/close time lookup from Alpaca calendar).

Failure mode:
  Alpaca calendar fetch failure → falls back to static US equity market hours
  (09:30–16:00 ET Mon–Fri, holidays not modelled).  Logged at WARNING level.
  The bench cadence loop still runs; the trader may fire outside a true window
  on a holiday — acceptable degradation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — ET-anchored, never hardcoded as a local offset
# ---------------------------------------------------------------------------

_ET = ZoneInfo("America/New_York")

# Static fallback market hours (used when Alpaca calendar is unavailable).
# America/New_York is the authoritative zone; DST is handled by ZoneInfo.
_OPEN_STATIC = time(9, 30)
_CLOSE_STATIC = time(16, 0)

# Live window extension: SoD fires 60 min before open; EoD fires 30 min after close.
SOD_LEAD_MINUTES: int = int(os.environ.get("SOD_LEAD_MINUTES", 60))
EOD_TRAIL_MINUTES: int = int(os.environ.get("EOD_TRAIL_MINUTES", 30))

# Extended-hours sessions (ET clock times).
_PRE_MARKET_OPEN = time(4, 0)
_PRE_MARKET_CLOSE = time(9, 30)  # same as regular open
_AFTER_HOURS_OPEN = time(16, 0)  # same as regular close
_AFTER_HOURS_CLOSE = time(20, 0)

# Path for persisting orphan turn records across restarts.
_DEFAULT_ORPHAN_DB = Path("data/orphan_turns.json")


# ---------------------------------------------------------------------------
# Alpaca calendar wrapper
# ---------------------------------------------------------------------------

@dataclass
class MarketDay:
    """One market day: date + open/close in UTC."""

    date: datetime  # midnight UTC (calendar date)
    open_utc: datetime
    close_utc: datetime

    @property
    def open_et(self) -> datetime:
        return self.open_utc.astimezone(_ET)

    @property
    def close_et(self) -> datetime:
        return self.close_utc.astimezone(_ET)


class AlpacaCalendar:
    """Wrapper around the Alpaca trading calendar API.

    Fetches market days for a rolling window and caches them.  Falls back to
    static US equity hours when credentials are absent or the fetch fails.

    The calendar is holiday-aware and half-day-aware — every open/close
    timestamp comes from Alpaca, so DST transitions and early closures are
    automatically handled.

    Parameters
    ----------
    api_key, api_secret:
        Alpaca credentials.  When absent (None / empty), the static fallback
        is used for every call.
    paper:
        Whether to use the Alpaca paper-trading base URL.  Doesn't matter
        for the calendar endpoint (same for both), but passed through for
        consistency.
    cache_ttl_seconds:
        How long to cache the fetched calendar before refreshing.
        Default: 3600 (1 hour).
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        paper: bool = True,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self._key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret = api_secret or os.environ.get("ALPACA_SECRET_KEY", "")
        self._paper = paper
        self._cache_ttl = cache_ttl_seconds
        self._cache: list[MarketDay] = []
        self._cache_until: datetime = datetime.min.replace(tzinfo=UTC)
        self._lock = threading.Lock()

    def get_day(self, dt: datetime) -> MarketDay | None:
        """Return the market day for ``dt`` (UTC), or None if it's a non-trading day.

        Comparison is against the market open's ET date (not the raw ``date``
        field, which is midnight UTC and would be the day before in ET).
        """
        dt_et_date = dt.astimezone(_ET).date()
        days = self._get_calendar(dt.date(), (dt + timedelta(days=1)).date())
        for day in days:
            open_et_date = day.open_utc.astimezone(_ET).date()
            if open_et_date == dt_et_date:
                return day
        return None

    def get_next_open(self, now: datetime) -> datetime | None:
        """Return the UTC datetime of the next market open on or after ``now``."""
        for day in self._get_calendar(now.date(), (now + timedelta(days=14)).date()):
            if day.open_utc >= now:
                return day.open_utc
        return None

    def _get_calendar(self, start: Any, end: Any) -> list[MarketDay]:
        """Fetch (or return cached) calendar days for the given date range."""
        with self._lock:
            if self._cache and datetime.now(UTC) < self._cache_until:
                return self._cache
            self._cache = self._fetch(start, end)
            self._cache_until = datetime.now(UTC) + timedelta(seconds=self._cache_ttl)
            return self._cache

    def _fetch(self, start: Any, end: Any) -> list[MarketDay]:
        """Fetch calendar from Alpaca.  Returns static fallback days on any failure."""
        if not self._key or not self._secret:
            return self._static_days(start, end)
        try:
            from datetime import date as _date

            from alpaca.trading.client import TradingClient
            from alpaca.trading.models import Calendar as AlpacaCalendarModel
            from alpaca.trading.requests import GetCalendarRequest

            client = TradingClient(
                api_key=self._key,
                secret_key=self._secret,
                paper=self._paper,
            )
            # GetCalendarRequest accepts date objects.
            start_date = _date.fromisoformat(str(start)) if isinstance(start, str) else start
            end_date = _date.fromisoformat(str(end)) if isinstance(end, str) else end
            req = GetCalendarRequest(start=start_date, end=end_date)
            raw_calendar = client.get_calendar(filters=req)
            days: list[MarketDay] = []
            for cal in raw_calendar:
                if not isinstance(cal, AlpacaCalendarModel):
                    continue
                # cal.open and cal.close are datetime objects from alpaca-py v0.20+
                open_dt = _ensure_utc(cal.open)
                close_dt = _ensure_utc(cal.close)
                date_dt = datetime(
                    cal.date.year, cal.date.month, cal.date.day, tzinfo=UTC
                )
                days.append(MarketDay(date=date_dt, open_utc=open_dt, close_utc=close_dt))
            return days
        except Exception as exc:
            logger.warning("Alpaca calendar fetch failed (%s); using static fallback", exc)
            return self._static_days(start, end)

    @staticmethod
    def _static_days(start: Any, end: Any) -> list[MarketDay]:
        """Generate static Mon-Fri 09:30-16:00 ET days for the range (holiday-unaware)."""
        from datetime import date as _date

        if isinstance(start, str):
            start = _date.fromisoformat(start)
        if isinstance(end, str):
            end = _date.fromisoformat(end)

        days: list[MarketDay] = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Mon-Fri
                open_et = datetime(current.year, current.month, current.day,
                                   _OPEN_STATIC.hour, _OPEN_STATIC.minute, tzinfo=_ET)
                close_et = datetime(current.year, current.month, current.day,
                                    _CLOSE_STATIC.hour, _CLOSE_STATIC.minute, tzinfo=_ET)
                days.append(MarketDay(
                    date=datetime(current.year, current.month, current.day, tzinfo=UTC),
                    open_utc=open_et.astimezone(UTC),
                    close_utc=close_et.astimezone(UTC),
                ))
            current = current + timedelta(days=1)
        return days


def _ensure_utc(dt: datetime) -> datetime:
    """Return ``dt`` as timezone-aware UTC.  Alpaca-py returns naive datetimes
    that are actually UTC, or tz-aware datetimes."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ---------------------------------------------------------------------------
# Window computation
# ---------------------------------------------------------------------------

@dataclass
class LiveWindow:
    """The live decision window for one trading day.

    ``sod_utc``: when the SoD turn fires (T-60min before open).
    ``open_utc``: regular session open.
    ``close_utc``: regular session close.
    ``eod_utc``: when the EoD turn fires (T+30min after close).
    """

    date_et: str  # YYYY-MM-DD in ET, for human display
    sod_utc: datetime
    open_utc: datetime
    close_utc: datetime
    eod_utc: datetime
    is_half_day: bool = False

    @property
    def open_et(self) -> datetime:
        """Market open in ET (for human display in wake_reason strings)."""
        return self.open_utc.astimezone(_ET)

    @property
    def close_et(self) -> datetime:
        """Market close in ET (for human display in wake_reason strings)."""
        return self.close_utc.astimezone(_ET)

    @property
    def is_live(self) -> bool:
        """True while we are inside the live window (sod → eod inclusive)."""
        now = datetime.now(UTC)
        return self.sod_utc <= now <= self.eod_utc

    @property
    def is_rth(self) -> bool:
        """True during regular trading hours (open → close)."""
        now = datetime.now(UTC)
        return self.open_utc <= now < self.close_utc

    def is_extended_hours_active(self) -> bool:
        """True during extended-hours sessions (pre-market or after-hours ET)."""
        now_et = datetime.now(UTC).astimezone(_ET)
        t = now_et.time()
        pre = _PRE_MARKET_OPEN <= t < _PRE_MARKET_CLOSE
        post = _AFTER_HOURS_OPEN <= t < _AFTER_HOURS_CLOSE
        return pre or post


def compute_live_window(
    calendar: AlpacaCalendar,
    now: datetime | None = None,
) -> LiveWindow | None:
    """Return today's :class:`LiveWindow`, or None if today is a non-trading day.

    ``now`` is the reference UTC time (default: wall clock).  Uses the Alpaca
    calendar to determine the actual open/close for the day (handles half-days,
    holidays).
    """
    if now is None:
        now = datetime.now(UTC)

    day = calendar.get_day(now)
    if day is None:
        return None

    date_et = day.open_et.strftime("%Y-%m-%d")
    open_utc = day.open_utc
    close_utc = day.close_utc

    # Half-day detection: close before 15:00 ET
    is_half = close_utc.astimezone(_ET).time() < time(15, 0)

    return LiveWindow(
        date_et=date_et,
        sod_utc=open_utc - timedelta(minutes=SOD_LEAD_MINUTES),
        open_utc=open_utc,
        close_utc=close_utc,
        eod_utc=close_utc + timedelta(minutes=EOD_TRAIL_MINUTES),
        is_half_day=is_half,
    )


# ---------------------------------------------------------------------------
# Trader lifecycle state
# ---------------------------------------------------------------------------

TraderPhase = Literal["dormant", "sod_pending", "live", "eod_pending", "done"]


@dataclass
class TraderLifecycle:
    """Per-trader lifecycle state tracked by the scheduler.

    Mutable — the scheduler updates phase, turn timestamps, and cadence state.
    """

    trader_id: str
    phase: TraderPhase = "dormant"

    # When the next cadence tick is due (UTC).
    next_cadence_utc: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Whether the SoD and EoD turns have fired for today.
    sod_fired_date: str | None = None   # YYYY-MM-DD ET
    eod_fired_date: str | None = None

    # done_for_day flag — set when trader calls done_for_day() terminal.
    # Reset at the next SoD.
    done_for_day: bool = False

    # Extended-hours flag (per-trader config).
    extended_hours: bool = False

    # Cadence minutes (per-trader config, default 30).
    cadence_minutes: int = 30

    # Queue of after-hours protective-order fill events to deliver at next live window.
    pending_ah_fills: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orphaned-turn persistence (carry-over A4-a)
# ---------------------------------------------------------------------------

@dataclass
class OrphanedTurn:
    """A turn that started but did not reach a terminal action.

    Persisted to disk so crash recovery survives a restart.  On the next
    live window, a new turn fires with this turn_id (NOT a fresh UUID) to
    preserve idempotency key parity.
    """

    turn_id: str
    trader_id: str
    started_at: str  # ISO format UTC
    tool_names_called: list[str]  # names only, no stale results


class OrphanTurnStore:
    """Disk-backed store for orphaned turns.

    Uses a simple JSON file at ``db_path``.  All operations are protected by
    an RLock so concurrent writes from the bench thread are safe.

    Design choice (committed in A4): the orphan store persists turn_ids so
    that crash recovery fires with the *same* turn_id.  This means:
      - idempotency_key = sha256(trader_id, turn_id, symbol, side, qty)
        is identical in both the original and the recovery turn.
      - The PendingTradeQueue UNIQUE constraint on idempotency_key catches
        double-fires at the DB level.
      - Direct-execution trades (no PendingTradeQueue) rely on turn_id
        reuse alone, because the risk manager's in-memory idempotency set is
        reset on restart.  This is an accepted limitation: a crash between
        "broker filled" and "turn completed" would double-fire a direct trade.
        The alternative (DB-backed idempotency for direct trades) is deferred
        as a future hardening step — see commit message for rationale.
    """

    def __init__(self, db_path: Path = _DEFAULT_ORPHAN_DB) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._turns: dict[str, OrphanedTurn] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text())
                for item in raw:
                    t = OrphanedTurn(**item)
                    self._turns[t.turn_id] = t
        except Exception as exc:
            logger.warning("OrphanTurnStore load failed (%s); starting fresh", exc)
            self._turns = {}

    def _save(self) -> None:
        try:
            payload = [
                {
                    "turn_id": t.turn_id,
                    "trader_id": t.trader_id,
                    "started_at": t.started_at,
                    "tool_names_called": t.tool_names_called,
                }
                for t in self._turns.values()
            ]
            self._path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.warning("OrphanTurnStore save failed (%s)", exc)

    def register(self, turn_id: str, trader_id: str, tool_names: list[str] | None = None) -> None:
        """Register a turn as in-progress (may become orphaned on crash)."""
        with self._lock:
            self._turns[turn_id] = OrphanedTurn(
                turn_id=turn_id,
                trader_id=trader_id,
                started_at=datetime.now(UTC).isoformat(),
                tool_names_called=list(tool_names or []),
            )
            self._save()

    def update_tools(self, turn_id: str, tool_names: list[str]) -> None:
        """Update the tool list for an in-progress turn."""
        with self._lock:
            if turn_id in self._turns:
                self._turns[turn_id].tool_names_called = list(tool_names)
                self._save()

    def complete(self, turn_id: str) -> None:
        """Mark a turn as completed (remove from orphan tracking)."""
        with self._lock:
            self._turns.pop(turn_id, None)
            self._save()

    def get_orphans_for_trader(self, trader_id: str) -> list[OrphanedTurn]:
        """Return all orphaned turns for a trader (for crash recovery)."""
        with self._lock:
            return [t for t in self._turns.values() if t.trader_id == trader_id]

    def all_orphans(self) -> list[OrphanedTurn]:
        """Return all orphaned turns (called on startup to detect crashes)."""
        with self._lock:
            return list(self._turns.values())

    def clear_for_trader(self, trader_id: str) -> None:
        """Remove all orphan records for a trader (after recovery)."""
        with self._lock:
            to_remove = [tid for tid, t in self._turns.items() if t.trader_id == trader_id]
            for tid in to_remove:
                self._turns.pop(tid, None)
            if to_remove:
                self._save()


# ---------------------------------------------------------------------------
# Lifecycle engine
# ---------------------------------------------------------------------------

class LifecycleEngine:
    """Coordinates market-hours gating, SoD/EoD turns, and per-trader cadence.

    Design role: sits between the bench's cadence clock and the individual
    trader's ``decide()`` method.  The engine decides *when* to fire a turn and
    *what type* of turn it is; the bench executes it.

    Parameters
    ----------
    calendar:
        :class:`AlpacaCalendar` instance (shared across all traders on the bench).
    orphan_store:
        :class:`OrphanTurnStore` for crash-recovery.  None → crash recovery skipped.
    """

    def __init__(
        self,
        calendar: AlpacaCalendar,
        orphan_store: OrphanTurnStore | None = None,
    ) -> None:
        self.calendar = calendar
        self.orphan_store = orphan_store
        self._lifecycles: dict[str, TraderLifecycle] = {}
        self._lock = threading.Lock()

    def register_trader(
        self,
        trader_id: str,
        *,
        cadence_minutes: int = 30,
        extended_hours: bool = False,
    ) -> None:
        """Register a trader with the lifecycle engine."""
        with self._lock:
            self._lifecycles[trader_id] = TraderLifecycle(
                trader_id=trader_id,
                cadence_minutes=cadence_minutes,
                extended_hours=extended_hours,
            )

    def remove_trader(self, trader_id: str) -> None:
        """Unregister a trader from lifecycle tracking."""
        with self._lock:
            self._lifecycles.pop(trader_id, None)

    def get_lifecycle(self, trader_id: str) -> TraderLifecycle | None:
        with self._lock:
            return self._lifecycles.get(trader_id)

    def check_window(self, trader_id: str, now: datetime | None = None) -> LiveWindow | None:
        """Return today's live window, or None if dormant."""
        return compute_live_window(self.calendar, now)

    def is_live(self, trader_id: str, now: datetime | None = None) -> bool:
        """True if the trader should be active right now."""
        if now is None:
            now = datetime.now(UTC)
        lc = self.get_lifecycle(trader_id)
        if lc is None:
            return False
        window = compute_live_window(self.calendar, now)
        if window is None:
            return False
        if window.sod_utc <= now <= window.eod_utc:
            return True
        if lc.extended_hours and window.is_extended_hours_active():
            return True
        return False

    def due_turns(self, now: datetime | None = None) -> list[tuple[str, str, str]]:
        """Return list of (trader_id, turn_type, wake_reason) tuples for turns due now.

        Called by the scheduler on every cadence tick.  Returns only the turns
        that should fire at ``now``.
        """
        if now is None:
            now = datetime.now(UTC)

        results: list[tuple[str, str, str]] = []

        with self._lock:
            lifecycles = dict(self._lifecycles)

        for trader_id, lc in lifecycles.items():
            window = compute_live_window(self.calendar, now)
            if window is None:
                continue  # non-trading day

            date_et = window.date_et

            # --- SoD: fires at T-60min if not yet fired today ---
            if (
                lc.sod_fired_date != date_et
                and now >= window.sod_utc
            ):
                with self._lock:
                    lc.sod_fired_date = date_et
                    lc.done_for_day = False  # reset done_for_day at SoD
                    lc.phase = "live"
                    # Reset cadence clock so first regular turn fires on next tick
                    # (at now + cadence_minutes), not at some stale construction-time value.
                    lc.next_cadence_utc = now + timedelta(minutes=lc.cadence_minutes)
                # Deliver any queued after-hours fills BEFORE the SoD turn (not after).
                for fill_event in lc.pending_ah_fills:
                    results.append((trader_id, "event", fill_event["wake_reason"]))
                lc.pending_ah_fills.clear()
                results.append((trader_id, "SoD",
                                 f"start-of-day: market opens at {window.open_et.strftime('%H:%M ET')}"))
                continue

            # --- EoD: fires at T+30min if not yet fired today ---
            if (
                lc.eod_fired_date != date_et
                and now >= window.eod_utc
            ):
                with self._lock:
                    lc.eod_fired_date = date_et
                    lc.phase = "done"
                results.append((trader_id, "EoD",
                                 f"end-of-day: market closed at {window.close_et.strftime('%H:%M ET')}"))
                continue

            # Outside live window and no extended-hours → dormant
            if not (window.sod_utc <= now <= window.eod_utc):
                if not (lc.extended_hours and window.is_extended_hours_active()):
                    continue

            # Skip regular turns if done_for_day
            if lc.done_for_day:
                continue

            # --- Regular cadence tick ---
            if now >= lc.next_cadence_utc:
                with self._lock:
                    lc.next_cadence_utc = now + timedelta(minutes=lc.cadence_minutes)
                results.append((trader_id, "regular", "scheduled"))

        return results

    def mark_done_for_day(self, trader_id: str) -> None:
        """Called when a trader emits done_for_day() terminal."""
        with self._lock:
            lc = self._lifecycles.get(trader_id)
            if lc is not None:
                lc.done_for_day = True

    def queue_ah_fill_event(
        self,
        trader_id: str,
        symbol: str,
        fill_type: str,
        fill_time_et: str,
    ) -> None:
        """Queue an after-hours protective-order fill for delivery at next SoD.

        Called when a stop/TP fires during dormancy.  The event is NOT merged
        into SoD — it fires as a separate event turn at the start of the next
        live window, before SoD.

        Parameters
        ----------
        trader_id:
            The trader to notify.
        symbol:
            The symbol whose protective order filled.
        fill_type:
            "stop" | "take_profit" | "trail".
        fill_time_et:
            Human-readable ET time string, e.g. "03:42 ET".
        """
        with self._lock:
            lc = self._lifecycles.get(trader_id)
            if lc is not None:
                wake_reason = (
                    f"Your {fill_type} on {symbol} hit at {fill_time_et} while you were dormant."
                )
                lc.pending_ah_fills.append({"wake_reason": wake_reason, "symbol": symbol})
