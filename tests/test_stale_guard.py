"""P1: Stale-decision guard — TTL, price-drift, and position-mismatch checks."""

from datetime import UTC, datetime, timedelta
from typing import Any

from trading_agent.bench import Bench
from trading_agent.bench.bench import _STALE_TTL_SECONDS, _DecisionSnapshot
from trading_agent.llm.trader import DecisionResult, TradeDecision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision(action: str, symbol: str = "AAPL", qty: float = 5.0) -> TradeDecision:
    return TradeDecision(symbol=symbol, action=action, quantity=qty)


def _now_snap(bench: Bench, comp_name: str | None = None, *, prices: dict[str, float] | None = None) -> _DecisionSnapshot:
    """Build a fresh (non-stale) snapshot from current bench prices."""
    return _DecisionSnapshot(
        prices=prices if prices is not None else dict(bench._last_prices),
        positions={},
        ts=datetime.now(UTC),
    )


def _old_snap(bench: Bench, *, seconds_ago: float = _STALE_TTL_SECONDS + 1) -> _DecisionSnapshot:
    """Build a snapshot that is TTL-expired."""
    return _DecisionSnapshot(
        prices=dict(bench._last_prices),
        positions={},
        ts=datetime.now(UTC) - timedelta(seconds=seconds_ago),
    )


class ScriptedTrader:
    def __init__(self, name: str, script: list[DecisionResult] | None = None) -> None:
        self.name = name
        self.model = name
        self._script = iter(script or [])

    def observe(self, bar: Any) -> None:
        pass

    def decide(self, account: Any) -> DecisionResult:
        return next(self._script, DecisionResult())


def _bench_with_aapl(price: float = 100.0, balance: float = 10_000.0) -> tuple[Bench, Any]:
    bench = Bench(["AAPL"], initial_balance=balance)
    comp = bench.add_competitor("t", ScriptedTrader("t"))
    bench.observe_bar({"symbol": "AAPL", "close": price})
    return bench, comp


# ---------------------------------------------------------------------------
# TTL — expired decision is discarded
# ---------------------------------------------------------------------------


def test_ttl_expired_decision_is_discarded() -> None:
    bench, comp = _bench_with_aapl()
    # Buy 5 shares — should be blocked because snapshot is stale.
    snap = _old_snap(bench)
    d = _decision("BUY")
    bench._apply_decision(comp, d, snap)
    assert comp.broker.get_position("AAPL") is None  # no trade placed
    last = comp.decisions[0]
    assert last.status == "blocked"
    assert "stale:ttl" in last.detail


def test_ttl_not_expired_decision_proceeds() -> None:
    bench, comp = _bench_with_aapl()
    snap = _now_snap(bench)
    d = _decision("BUY")
    bench._apply_decision(comp, d, snap)
    # Should fill — market price is set, not stale.
    pos = comp.broker.get_position("AAPL")
    assert pos is not None and pos.quantity == 5.0


# ---------------------------------------------------------------------------
# Price drift — decision based on significantly different price is discarded
# ---------------------------------------------------------------------------


def test_drift_past_tolerance_drops_decision() -> None:
    bench, comp = _bench_with_aapl(price=100.0)
    # Snapshot price = 100, but now price has moved 2% (> 1% default threshold).
    snap = _DecisionSnapshot(
        prices={"AAPL": 100.0},
        positions={},
        ts=datetime.now(UTC),
    )
    bench.observe_bar({"symbol": "AAPL", "close": 102.1})  # >2% drift
    d = _decision("BUY")
    bench._apply_decision(comp, d, snap)
    # Not filled because drift exceeds tolerance.
    assert comp.broker.get_position("AAPL") is None
    last = comp.decisions[0]
    assert "stale:drift" in last.detail


def test_drift_within_tolerance_allows_decision() -> None:
    bench, comp = _bench_with_aapl(price=100.0)
    snap = _DecisionSnapshot(
        prices={"AAPL": 100.0},
        positions={},
        ts=datetime.now(UTC),
    )
    # Move price by only 0.5% — within the 1% tolerance.
    bench.observe_bar({"symbol": "AAPL", "close": 100.5})
    d = _decision("BUY")
    bench._apply_decision(comp, d, snap)
    pos = comp.broker.get_position("AAPL")
    assert pos is not None and pos.quantity == 5.0


# ---------------------------------------------------------------------------
# Position mismatch — assumed long but now flat (stop fired while model thought)
# ---------------------------------------------------------------------------


def test_position_mismatch_assumed_long_now_flat_drops_sell() -> None:
    bench, comp = _bench_with_aapl()
    # Manually give the broker a long position so we can then clear it.
    comp.broker.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 10})

    # Snapshot reflects the long.
    snap = _DecisionSnapshot(
        prices={"AAPL": 100.0},
        positions={"AAPL": 10.0},   # snapshot assumed long
        ts=datetime.now(UTC),
    )
    # A stop order fires and flattens the position externally.
    comp.broker.flatten_all()
    assert comp.broker.get_position("AAPL") is None

    # SELL decision: snapshot said long, but now flat → stale:position
    d = _decision("SELL")
    bench._apply_decision(comp, d, snap)
    # Should be blocked (not fill a spurious short-open).
    last = comp.decisions[0]
    assert "stale:position" in last.detail


def test_position_mismatch_assumed_flat_long_still_present_no_drop() -> None:
    """SELL when snapshot also saw a long AND position is still there → passes."""
    bench, comp = _bench_with_aapl()
    comp.broker.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 5})
    snap = _DecisionSnapshot(
        prices={"AAPL": 100.0},
        positions={"AAPL": 5.0},
        ts=datetime.now(UTC),
    )
    d = _decision("SELL", qty=5.0)
    bench._apply_decision(comp, d, snap)
    # Position should be closed (SELL filled).
    assert comp.broker.get_position("AAPL") is None


# ---------------------------------------------------------------------------
# No snapshot → guard is skipped (backward-compatible path)
# ---------------------------------------------------------------------------


def test_no_snapshot_guard_is_skipped() -> None:
    bench, comp = _bench_with_aapl()
    d = _decision("BUY")
    bench._apply_decision(comp, d, None)  # no snapshot passed
    pos = comp.broker.get_position("AAPL")
    assert pos is not None and pos.quantity == 5.0


# ---------------------------------------------------------------------------
# End-to-end: run_decisions attaches a snapshot so concurrent moves can be caught
# ---------------------------------------------------------------------------


def test_run_decisions_passes_snapshot_to_apply() -> None:
    """run_decisions() should log 'filled' (fresh snapshot, no staleness)."""
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    result = DecisionResult(decisions=[TradeDecision("AAPL", "BUY", 5)])
    bench.add_competitor("t", ScriptedTrader("t", [result]))
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench.run_decisions()
    decisions = bench.recent_decisions()
    assert decisions and decisions[0]["status"] == "filled"
