"""P2: Event-driven wake hooks — threshold cross wakes scoped model decisions."""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

from trading_agent.bench import Bench
from trading_agent.bench.controller import _WAKE_DEDUP_SECONDS, BenchController
from trading_agent.llm.trader import DecisionResult
from trading_agent.web.market_watch import MarketMove, MarketMoveWatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _move(symbol: str = "AAPL", pct: float = -0.05) -> MarketMove:
    ref = 100.0
    cur = ref * (1 + pct)
    return MarketMove(
        symbol=symbol,
        reference_price=ref,
        current_price=cur,
        pct_change=pct,
        direction="down" if pct < 0 else "up",
    )


class ScriptedTrader:
    def __init__(self, name: str, script: list[DecisionResult] | None = None) -> None:
        self.name = name
        self.model = name
        self._script = iter(script or [])
        self.decided: list[dict[str, Any]] = []

    def observe(self, bar: Any) -> None:
        pass

    def decide(self, account: Any) -> DecisionResult:
        self.decided.append(dict(account))
        return next(self._script, DecisionResult())


def _make_controller(
    symbols: list[str] | None = None,
    *,
    market_watcher: MarketMoveWatcher | None = None,
) -> tuple[BenchController, Bench]:
    syms = symbols or ["AAPL"]
    bench = Bench(syms, initial_balance=10_000.0)
    client = MagicMock()
    ctrl = BenchController(
        bench,
        client,
        symbols=syms,
        market_watcher=market_watcher,
    )
    return ctrl, bench


# ---------------------------------------------------------------------------
# Threshold cross triggers a scoped re-decision
# ---------------------------------------------------------------------------


def test_on_market_move_wakes_book_with_position() -> None:
    """A threshold cross should cause the model to decide once for the symbol."""
    ctrl, bench = _make_controller()
    trader = ScriptedTrader("t")
    comp = bench.add_competitor("t", trader)
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})

    # Open a position so run_decisions_for_symbol picks up this competitor.
    comp.broker.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 5})

    ctrl._running = True  # fake the running flag (no real loop)
    ctrl.on_market_move(_move("AAPL", -0.05))

    # The trader's decide() should have been called once (for the wake).
    assert len(trader.decided) == 1


def test_on_market_move_skips_book_without_position() -> None:
    """Only competitors WITH a position in the symbol should be woken."""
    ctrl, bench = _make_controller()
    trader_aapl = ScriptedTrader("aapl")
    trader_goog = ScriptedTrader("goog")
    bench.add_competitor("aapl", trader_aapl)
    bench.add_competitor("goog", trader_goog)
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench.observe_bar({"symbol": "GOOG", "close": 200.0})

    # Only aapl holds AAPL.
    bench._competitors["aapl"].broker.place_order(
        {"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 5}
    )

    ctrl._running = True
    ctrl.on_market_move(_move("AAPL", -0.05))

    assert len(trader_aapl.decided) == 1  # woken
    assert len(trader_goog.decided) == 0  # not woken — no AAPL position


# ---------------------------------------------------------------------------
# De-dup window prevents storms
# ---------------------------------------------------------------------------


def test_dedup_window_prevents_second_wake() -> None:
    """A second move within _WAKE_DEDUP_SECONDS must not fire again."""
    ctrl, bench = _make_controller()
    trader = ScriptedTrader("t")
    comp = bench.add_competitor("t", trader)
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    comp.broker.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 5})

    ctrl._running = True
    ctrl.on_market_move(_move("AAPL", -0.02))  # first cross — fires
    ctrl.on_market_move(_move("AAPL", -0.04))  # second cross within window — suppressed

    assert len(trader.decided) == 1  # only one wake


def test_dedup_window_allows_wake_after_expiry() -> None:
    """After the de-dup window elapses, the next move should fire again."""
    ctrl, bench = _make_controller()
    trader = ScriptedTrader("t")
    comp = bench.add_competitor("t", trader)
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    comp.broker.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 5})

    ctrl._running = True
    ctrl.on_market_move(_move("AAPL", -0.02))  # first wake

    # Backdate the last_wake timestamp so the window has expired.
    ctrl._last_wake["AAPL"] = datetime.now(UTC) - timedelta(seconds=_WAKE_DEDUP_SECONDS + 1)
    ctrl.on_market_move(_move("AAPL", -0.04))  # should fire again

    assert len(trader.decided) == 2


# ---------------------------------------------------------------------------
# Running=False guard: hook does nothing when loop is stopped
# ---------------------------------------------------------------------------


def test_on_market_move_no_op_when_not_running() -> None:
    ctrl, bench = _make_controller()
    trader = ScriptedTrader("t")
    comp = bench.add_competitor("t", trader)
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    comp.broker.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 5})

    # ctrl._running is False (default) — hook must be silent.
    ctrl.on_market_move(_move("AAPL", -0.05))
    assert len(trader.decided) == 0


# ---------------------------------------------------------------------------
# Hard floor still fires if model is too slow
# ---------------------------------------------------------------------------


def test_hard_floor_fires_regardless_of_wake_dedup() -> None:
    """The hard floor (P0 broker-level) is deterministic and fires independently
    of the P2 soft-stop wake mechanism, even while the de-dup window is active."""
    ctrl, bench = _make_controller()
    trader = ScriptedTrader("t")
    comp = bench.add_competitor("t", trader)

    # Enable the hard floor: 15% loss → auto-flatten.
    from trading_agent.risk_manager import RiskLimits, RiskManager
    comp.risk = RiskManager(limits=RiskLimits(hard_floor_pct=15.0), kill_switch_file=None)

    # Build up a long position.
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    comp.broker.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 80})

    # Fake the dedup window so the soft-stop would be suppressed.
    ctrl._running = True
    ctrl._last_wake["AAPL"] = datetime.now(UTC)

    # Price crashes: equity = $2000 + 80*73 = $7840 < $8500 (15% of $10000).
    # This goes through observe_bar → _check_hard_floors, not the wake path.
    bench.observe_bar({"symbol": "AAPL", "close": 73.0})

    # Hard floor should have auto-flattened regardless.
    assert comp.broker.get_position("AAPL") is None


# ---------------------------------------------------------------------------
# MarketMoveWatcher integration — threshold cross emits exactly one wake
# ---------------------------------------------------------------------------


def test_market_move_watcher_emits_one_wake_per_band() -> None:
    """A move that crosses one threshold band triggers on_market_move exactly once."""
    watcher = MarketMoveWatcher(threshold_pct=2.0)
    ctrl, bench = _make_controller(market_watcher=watcher)
    trader = ScriptedTrader("t")
    comp = bench.add_competitor("t", trader)
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    comp.broker.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 5})

    ctrl._running = True
    wake_calls: list[MarketMove] = []
    original = ctrl.on_market_move
    ctrl.on_market_move = lambda m: (wake_calls.append(m), original(m))  # type: ignore[method-assign]

    # Feed ticks that cross one 2% band (reference=100 → price 97 = -3%).
    for px in [99.5, 98.0, 97.0]:
        move = watcher.observe("AAPL", px)
        if move is not None:
            ctrl.on_market_move(move)

    # Only one band was crossed, so exactly one wake.
    assert len(wake_calls) == 1
    assert wake_calls[0].symbol == "AAPL"
