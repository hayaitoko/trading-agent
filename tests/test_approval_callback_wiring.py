"""Integration tests for the A3→A4 approval callback wiring.

Ensures that:
  1. ``TradeTool.run()`` registers a scheduler callback after proposing a trade.
  2. Approving a pending trade via ``PendingTradeQueue.set_decision("approved")``
     enqueues a callback turn with ``wake_reason`` containing "approved".
  3. Denying a pending trade enqueues a callback turn with "denied" in the reason,
     and no trade is executed.
  4. TTL expiry fires a callback turn with "expired" in the reason, and no trade
     is executed.
  5. ``BenchController._expire_pending_trades()`` calls ``expire_old()`` on the
     attached PendingTradeQueue so TTL callbacks are swept per cadence tick.

Design: these are integration-style tests that wire real objects (no mocks for the
queue/scheduler path) to exercise the full propose→callback→turn flow.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from trading_agent.approval_queue import PendingTradeQueue, TradeIntent
from trading_agent.bench.scheduler import MarketScheduler
from trading_agent.intel.lifecycle import AlpacaCalendar, MarketDay
from trading_agent.intel.tools.act._base import ActToolBase
from trading_agent.intel.tools.act.trade import TradeTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = ZoneInfo("America/New_York")


def _make_market_day(
    year: int = 2026, month: int = 6, day: int = 2
) -> MarketDay:
    """Build a MarketDay for a typical US equity trading day."""
    open_et = datetime(year, month, day, 9, 30, tzinfo=_ET)
    close_et = datetime(year, month, day, 16, 0, tzinfo=_ET)
    return MarketDay(
        date=datetime(year, month, day).date(),
        open_utc=open_et.astimezone(UTC),
        close_utc=close_et.astimezone(UTC),
    )


class _FixedCalendar(AlpacaCalendar):
    """Calendar that always returns a fixed MarketDay."""

    def __init__(self, day: MarketDay) -> None:
        self._day = day

    def get_day(self, dt: datetime) -> MarketDay | None:
        return self._day


def _make_scheduler(bench: Any, day: MarketDay | None = None) -> MarketScheduler:
    if day is None:
        day = _make_market_day()
    cal = _FixedCalendar(day)
    return MarketScheduler(bench, calendar=cal)


def _make_stub_bench() -> Any:
    """Return a minimal bench stub compatible with MarketScheduler."""
    bench = MagicMock()
    bench._competitors = {}
    return bench


def _make_ptq(tmp_path: Path) -> PendingTradeQueue:
    return PendingTradeQueue(db_path=tmp_path / "approvals.db")


def _make_trade_tool(
    ptq: PendingTradeQueue,
    scheduler: MarketScheduler,
    trader_id: str = "trader-alpha",
    turn_id: str = "turn-001",
) -> TradeTool:
    return TradeTool(
        broker=None,
        risk_manager=None,
        pending_trade_queue=ptq,
        trader_id=trader_id,
        turn_id=turn_id,
        requires_approval=True,
        scheduler=scheduler,
    )


# ---------------------------------------------------------------------------
# 1. TradeTool.run() wires the callback when requires_approval=True
# ---------------------------------------------------------------------------

def test_trade_tool_wires_callback_on_propose(tmp_path: Path) -> None:
    """After propose(), a callback is registered so approve fires a scheduler turn."""
    ptq = _make_ptq(tmp_path)
    bench = _make_stub_bench()
    day = _make_market_day()
    scheduler = _make_scheduler(bench, day)

    tool = _make_trade_tool(ptq, scheduler)
    result = tool.run("AAPL", "BUY", 10.0)

    assert result.ok is True
    pending_trade_id = result.data["pending_trade_id"]  # type: ignore[index]
    assert result.data["status"] == "awaiting_approval"  # type: ignore[index]

    # At this point a callback should be registered.  Approve and tick to confirm.
    ptq.set_decision(pending_trade_id, "approved")

    now = day.open_utc + timedelta(minutes=5)
    turns = scheduler.tick(now=now)

    cb = [(tid, tt, wr) for tid, tt, wr in turns if tt == "callback"]
    assert len(cb) >= 1, f"Expected a callback turn; got: {turns}"
    tid, _, wr = cb[0]
    assert tid == "trader-alpha"
    assert "approved" in wr.lower()
    assert pending_trade_id in wr


# ---------------------------------------------------------------------------
# 2. No scheduler → callback is silently skipped, propose still succeeds
# ---------------------------------------------------------------------------

def test_trade_tool_no_scheduler_still_proposes(tmp_path: Path) -> None:
    """When no scheduler is attached, trade() proposes without wiring a callback."""
    ptq = _make_ptq(tmp_path)
    tool = TradeTool(
        broker=None,
        risk_manager=None,
        pending_trade_queue=ptq,
        trader_id="trader-beta",
        turn_id="turn-002",
        requires_approval=True,
        scheduler=None,  # no scheduler
    )
    result = tool.run("TSLA", "SELL", 5.0)
    assert result.ok is True
    assert result.data["status"] == "awaiting_approval"  # type: ignore[index]

    # Pending trade exists in DB even without a scheduler.
    pending_trade_id = result.data["pending_trade_id"]  # type: ignore[index]
    pt = ptq.get(pending_trade_id)
    assert pt is not None
    assert pt.status == "awaiting_approval"


# ---------------------------------------------------------------------------
# 3. Approve → callback turn carries correct wake_reason; deny → "denied"
# ---------------------------------------------------------------------------

def test_approve_callback_wake_reason(tmp_path: Path) -> None:
    """Approving a pending trade schedules a callback turn with 'approved' reason."""
    ptq = _make_ptq(tmp_path)
    bench = _make_stub_bench()
    day = _make_market_day()
    scheduler = _make_scheduler(bench, day)

    tool = _make_trade_tool(ptq, scheduler, trader_id="t-approve", turn_id="turn-app")
    result = tool.run("NVDA", "BUY", 2.0)
    pending_trade_id = result.data["pending_trade_id"]  # type: ignore[index]

    ptq.set_decision(pending_trade_id, "approved")

    now = day.open_utc + timedelta(minutes=10)
    turns = scheduler.tick(now=now)

    cb = [(tid, tt, wr) for tid, tt, wr in turns if tt == "callback" and tid == "t-approve"]
    assert len(cb) == 1
    wr = cb[0][2]
    assert "approved" in wr.lower()
    assert pending_trade_id in wr


def test_deny_callback_wake_reason(tmp_path: Path) -> None:
    """Denying a pending trade schedules a callback turn with 'denied' reason."""
    ptq = _make_ptq(tmp_path)
    bench = _make_stub_bench()
    day = _make_market_day()
    scheduler = _make_scheduler(bench, day)

    tool = _make_trade_tool(ptq, scheduler, trader_id="t-deny", turn_id="turn-deny")
    result = tool.run("MSFT", "SELL", 3.0)
    pending_trade_id = result.data["pending_trade_id"]  # type: ignore[index]

    ptq.set_decision(pending_trade_id, "denied", note="risk limit")

    now = day.open_utc + timedelta(minutes=5)
    turns = scheduler.tick(now=now)

    cb = [(tid, tt, wr) for tid, tt, wr in turns if tt == "callback" and tid == "t-deny"]
    assert len(cb) == 1
    wr = cb[0][2]
    assert "denied" in wr.lower()
    # Note text should appear in wake_reason.
    assert "risk limit" in wr


def test_deny_does_not_execute_trade(tmp_path: Path) -> None:
    """Deny keeps the trade in denied status — no fill is recorded."""
    ptq = _make_ptq(tmp_path)
    bench = _make_stub_bench()
    scheduler = _make_scheduler(bench)

    tool = _make_trade_tool(ptq, scheduler, trader_id="t-deny2", turn_id="turn-d2")
    result = tool.run("AMD", "BUY", 1.0)
    pending_trade_id = result.data["pending_trade_id"]  # type: ignore[index]

    ptq.set_decision(pending_trade_id, "denied")

    pt = ptq.get(pending_trade_id)
    assert pt is not None
    assert pt.status == "denied"
    assert pt.fill_result is None


# ---------------------------------------------------------------------------
# 4. TTL expiry fires callback with "expired" in wake_reason
# ---------------------------------------------------------------------------

def test_expire_callback_wake_reason(tmp_path: Path) -> None:
    """TTL expiry fires a callback turn with 'expired' in the wake_reason."""
    ptq = _make_ptq(tmp_path)
    # Force immediate expiry by setting PREAPPROVAL_TTL_MIN = 0.
    ptq.PREAPPROVAL_TTL_MIN = 0  # type: ignore[assignment]

    bench = _make_stub_bench()
    day = _make_market_day()
    scheduler = _make_scheduler(bench, day)

    tool = _make_trade_tool(ptq, scheduler, trader_id="t-expire", turn_id="turn-exp")
    result = tool.run("GOOG", "BUY", 1.0)
    pending_trade_id = result.data["pending_trade_id"]  # type: ignore[index]

    # Approve first (required for TTL expiry path; "awaiting_approval" rows are
    # not swept by expire_old — only "approved" ones are).
    ptq.set_decision(pending_trade_id, "approved")

    # Re-wire the callback for the scheduler (set_decision fires the first callback
    # for the "approved" status change; expire_old fires a second one for "expired").
    # Wire a second callback so we catch the expiry event.
    bench2 = _make_stub_bench()
    scheduler2 = _make_scheduler(bench2, day)
    scheduler2.wire_pending_trade_callbacks(ptq, "t-expire", pending_trade_id)

    # Force expiry.
    ptq.expire_old()

    now = day.open_utc + timedelta(minutes=5)
    turns = scheduler2.tick(now=now)

    cb = [(tid, tt, wr) for tid, tt, wr in turns if tt == "callback" and tid == "t-expire"]
    assert len(cb) >= 1, f"Expected expiry callback; got: {turns}"
    assert "expired" in cb[0][2].lower()


def test_expire_does_not_execute_trade(tmp_path: Path) -> None:
    """An expired approved trade has no fill_result — execution is suppressed."""
    ptq = _make_ptq(tmp_path)
    ptq.PREAPPROVAL_TTL_MIN = 0  # type: ignore[assignment]

    bench = _make_stub_bench()
    scheduler = _make_scheduler(bench)

    tool = _make_trade_tool(ptq, scheduler, trader_id="t-exp2", turn_id="turn-e2")
    result = tool.run("META", "BUY", 1.0)
    pending_trade_id = result.data["pending_trade_id"]  # type: ignore[index]

    ptq.set_decision(pending_trade_id, "approved")
    ptq.expire_old()

    pt = ptq.get(pending_trade_id)
    assert pt is not None
    assert pt.status == "expired"
    assert pt.fill_result is None


# ---------------------------------------------------------------------------
# 5. BenchController._expire_pending_trades() sweeps the PendingTradeQueue
# ---------------------------------------------------------------------------

def test_bench_controller_expire_pending_trades_sweeps(tmp_path: Path) -> None:
    """BenchController._expire_pending_trades() calls ptq.expire_old() each tick."""
    from trading_agent.bench.controller import BenchController

    ptq = _make_ptq(tmp_path)
    ptq.PREAPPROVAL_TTL_MIN = 0  # type: ignore[assignment]

    bench = _make_stub_bench()
    # Controller needs a minimal client mock.
    client = MagicMock()
    client.list_models.return_value = []
    controller = BenchController(
        bench,
        client,
        symbols=["AAPL"],
        pending_trade_queue=ptq,
    )

    # Manually propose + approve a trade so it can expire.
    intent = TradeIntent(symbol="AAPL", side="BUY", qty=5.0)
    pt = ptq.propose("trader-ctrl", intent, "idem-ctrl")
    ptq.set_decision(pt.pending_trade_id, "approved")

    # expire_old has not been called yet.
    snapshot_before = ptq.get(pt.pending_trade_id)
    assert snapshot_before is not None
    assert snapshot_before.status == "approved"

    # Now call the controller method — it should sweep the expired record.
    controller._expire_pending_trades()

    snapshot_after = ptq.get(pt.pending_trade_id)
    assert snapshot_after is not None
    assert snapshot_after.status == "expired"


# ---------------------------------------------------------------------------
# 6. ActToolBase accepts scheduler kwarg; default is None
# ---------------------------------------------------------------------------

def test_act_tool_base_accepts_scheduler() -> None:
    """ActToolBase stores the scheduler reference when provided."""
    scheduler_mock = MagicMock()
    tool = ActToolBase(
        trader_id="x",
        turn_id="y",
        scheduler=scheduler_mock,
    )
    assert tool.scheduler is scheduler_mock


def test_act_tool_base_scheduler_defaults_none() -> None:
    """ActToolBase.scheduler is None when not provided."""
    tool = ActToolBase(trader_id="x", turn_id="y")
    assert tool.scheduler is None


# ---------------------------------------------------------------------------
# 7. Full end-to-end: propose→callback enqueued→scheduler fires turn
# ---------------------------------------------------------------------------

def test_end_to_end_approve_then_scheduler_fires_callback_turn(tmp_path: Path) -> None:
    """Full integration: propose trade, approve it, scheduler turn fires with trader id."""
    ptq = _make_ptq(tmp_path)
    bench = _make_stub_bench()
    day = _make_market_day()
    scheduler = _make_scheduler(bench, day)

    # Propose via TradeTool (wires the callback automatically).
    tool = TradeTool(
        broker=None,
        risk_manager=None,
        pending_trade_queue=ptq,
        trader_id="e2e-trader",
        turn_id="e2e-turn",
        requires_approval=True,
        scheduler=scheduler,
    )
    res = tool.run("AAPL", "BUY", 7.0)
    assert res.ok is True
    ptid = res.data["pending_trade_id"]  # type: ignore[index]

    # Operator approves.
    ptq.set_decision(ptid, "approved")

    # Scheduler tick drains the callback queue.
    now = day.open_utc + timedelta(minutes=15)
    turns = scheduler.tick(now=now)

    cb = [t for t in turns if t[1] == "callback" and t[0] == "e2e-trader"]
    assert len(cb) == 1, f"Expected exactly one callback turn; got: {turns}"
    wake_reason = cb[0][2]
    assert "approved" in wake_reason.lower()
    assert ptid in wake_reason
