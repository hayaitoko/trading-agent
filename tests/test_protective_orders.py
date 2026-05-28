"""P0: Protective order mechanics — stop_loss, take_profit, trailing_stop, hard floor."""

import pytest

from trading_agent.bench import Bench
from trading_agent.enums import OrderType
from trading_agent.llm.trader import DecisionResult, TradeDecision
from trading_agent.paper_broker import PaperBroker, OrderStatus
from trading_agent.risk_manager import RiskLimits, RiskManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broker(balance: float = 10_000.0) -> PaperBroker:
    b = PaperBroker(initial_balance=balance)
    b.connect()
    return b


def _long(broker: PaperBroker, symbol: str, qty: float, price: float) -> None:
    """Open a long position at a known price."""
    broker.update_market_prices({symbol: price})
    broker.place_order({"symbol": symbol, "side": "BUY", "order_type": "market", "amount": qty})


class ScriptedTrader:
    def __init__(self, name: str, script: list[DecisionResult] | None = None) -> None:
        self.name = name
        self.model = name
        self._script = iter(script or [])

    def observe(self, bar: object) -> None:
        pass

    def decide(self, account: object) -> DecisionResult:
        return next(self._script, DecisionResult())


# ---------------------------------------------------------------------------
# Stop-loss
# ---------------------------------------------------------------------------


def test_stop_fires_when_price_reaches_trigger() -> None:
    b = _broker()
    _long(b, "AAPL", 10, 100.0)

    # Place a stop-loss: sell 10 AAPL if price falls to 95
    b.place_order({
        "symbol": "AAPL",
        "side": "SELL",
        "order_type": "stop",
        "amount": 10,
        "stop_price": 95.0,
    })

    assert b.get_position("AAPL") is not None  # still long before trigger
    b.update_market_prices({"AAPL": 95.0})     # exactly at trigger
    assert b.get_position("AAPL") is None       # flat after stop fires


def test_stop_does_not_fire_above_trigger() -> None:
    b = _broker()
    _long(b, "AAPL", 5, 100.0)
    b.place_order({
        "symbol": "AAPL",
        "side": "SELL",
        "order_type": "stop",
        "amount": 5,
        "stop_price": 90.0,
    })
    b.update_market_prices({"AAPL": 91.0})  # above trigger — should not fire
    assert b.get_position("AAPL") is not None


def test_stop_fires_below_trigger() -> None:
    b = _broker()
    _long(b, "AAPL", 5, 100.0)
    b.place_order({
        "symbol": "AAPL",
        "side": "SELL",
        "order_type": "stop",
        "amount": 5,
        "stop_price": 90.0,
    })
    b.update_market_prices({"AAPL": 85.0})  # gap-down through trigger
    assert b.get_position("AAPL") is None


# ---------------------------------------------------------------------------
# Take-profit
# ---------------------------------------------------------------------------


def test_take_profit_fires_at_target() -> None:
    b = _broker()
    _long(b, "AAPL", 10, 100.0)
    b.place_order({
        "symbol": "AAPL",
        "side": "SELL",
        "order_type": "take_profit",
        "amount": 10,
        "stop_price": 110.0,
    })
    b.update_market_prices({"AAPL": 109.0})  # not yet
    assert b.get_position("AAPL") is not None
    b.update_market_prices({"AAPL": 110.0})  # exactly at target
    assert b.get_position("AAPL") is None


def test_take_profit_uses_price_field_if_no_stop_price() -> None:
    b = _broker()
    _long(b, "AAPL", 10, 100.0)
    b.place_order({
        "symbol": "AAPL",
        "side": "SELL",
        "order_type": "take_profit",
        "amount": 10,
        "price": 115.0,  # alternative field
    })
    b.update_market_prices({"AAPL": 115.0})
    assert b.get_position("AAPL") is None


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------


def test_trailing_stop_fires_when_peak_then_drops() -> None:
    b = _broker()
    _long(b, "AAPL", 10, 100.0)
    # Trailing stop: trail by $5.  Initial trigger = 95.
    b.place_order({
        "symbol": "AAPL",
        "side": "SELL",
        "order_type": "trailing_stop",
        "amount": 10,
        "stop_price": 95.0,
        "trail_amount": 5.0,
    })
    # Price rises to 105 — trigger should ratchet to 100.
    b.update_market_prices({"AAPL": 105.0})
    assert b.get_position("AAPL") is not None  # 105 > 100, still open
    # Price rises to 110 — trigger ratchets to 105.
    b.update_market_prices({"AAPL": 110.0})
    assert b.get_position("AAPL") is not None  # 110 > 105, still open
    # Price drops to 105 — exactly at the ratcheted trigger → fires.
    b.update_market_prices({"AAPL": 105.0})
    assert b.get_position("AAPL") is None


def test_trailing_stop_does_not_ratchet_down() -> None:
    """Trigger must never move against us when price drops before rising."""
    b = _broker()
    _long(b, "AAPL", 10, 100.0)
    b.place_order({
        "symbol": "AAPL",
        "side": "SELL",
        "order_type": "trailing_stop",
        "amount": 10,
        "stop_price": 95.0,
        "trail_amount": 5.0,
    })
    # Price dips to 97 — trigger stays at 95 (not lowered).
    b.update_market_prices({"AAPL": 97.0})
    assert b.get_position("AAPL") is not None  # above 95

    # Price drops to 95 — now fires.
    b.update_market_prices({"AAPL": 95.0})
    assert b.get_position("AAPL") is None


def test_trailing_stop_inferred_from_trail_amount_only() -> None:
    """If only trail_amount is given (no stop_price), compute stop from market."""
    b = _broker()
    _long(b, "AAPL", 5, 100.0)
    # trail_amount=10 → initial stop = 100 - 10 = 90
    b.place_order({
        "symbol": "AAPL",
        "side": "SELL",
        "order_type": "trailing_stop",
        "amount": 5,
        "trail_amount": 10.0,
    })
    b.update_market_prices({"AAPL": 91.0})  # above computed 90 — open
    assert b.get_position("AAPL") is not None
    b.update_market_prices({"AAPL": 89.0})  # below 90 — fires
    assert b.get_position("AAPL") is None


def test_trailing_stop_locks_profit() -> None:
    """Classic profit-lock: enter at 100, trail by 5.  Ride to 120, then pull back 5."""
    b = _broker()
    _long(b, "AAPL", 10, 100.0)
    initial_cash = b.balance

    b.place_order({
        "symbol": "AAPL",
        "side": "SELL",
        "order_type": "trailing_stop",
        "amount": 10,
        "stop_price": 95.0,
        "trail_amount": 5.0,
    })
    # Ride up to 120 — trigger ratchets to 115.
    for px in [105.0, 110.0, 115.0, 120.0]:
        b.update_market_prices({"AAPL": px})
    assert b.get_position("AAPL") is not None  # still riding

    # Pull back to 115 — fires at 115 (or current market fill).
    b.update_market_prices({"AAPL": 115.0})
    assert b.get_position("AAPL") is None  # stopped out

    # Should have realised a profit (locked in at ratcheted stop, not original entry).
    assert b.balance > initial_cash


# ---------------------------------------------------------------------------
# Defaults-off: existing order types unaffected
# ---------------------------------------------------------------------------


def test_protective_defaults_off_market_order_unchanged() -> None:
    b = _broker()
    b.update_market_prices({"AAPL": 100.0})
    result = b.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 1})
    assert result is not None
    assert result["status"] == "FILLED"


def test_protective_defaults_off_limit_order_unchanged() -> None:
    b = _broker()
    b.update_market_prices({"AAPL": 100.0})
    result = b.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "limit", "amount": 1, "price": 99.0})
    assert result is not None
    assert result["status"] == "PENDING"  # 100 > 99, not marketable yet


# ---------------------------------------------------------------------------
# Hard floor — RiskManager.check_hard_floor
# ---------------------------------------------------------------------------


def test_hard_floor_disabled_by_default() -> None:
    r = RiskManager()
    assert not r.check_hard_floor(80_000.0, 100_000.0)  # 20% loss, but no floor set


def test_hard_floor_returns_false_below_threshold() -> None:
    r = RiskManager(limits=RiskLimits(hard_floor_pct=20.0))
    assert not r.check_hard_floor(85_000.0, 100_000.0)  # 15% loss < 20%


def test_hard_floor_returns_true_at_threshold() -> None:
    r = RiskManager(limits=RiskLimits(hard_floor_pct=20.0))
    assert r.check_hard_floor(80_000.0, 100_000.0)  # exactly 20%


def test_hard_floor_returns_true_past_threshold() -> None:
    r = RiskManager(limits=RiskLimits(hard_floor_pct=10.0))
    assert r.check_hard_floor(88_000.0, 100_000.0)  # 12% loss > 10%


# ---------------------------------------------------------------------------
# Hard floor — broker-level auto-flatten via hard_floor_pct
# ---------------------------------------------------------------------------


def test_broker_hard_floor_flattens_at_breach() -> None:
    b = _broker(10_000.0)
    b.hard_floor_pct = 20.0  # auto-flatten if equity drops 20%
    _long(b, "AAPL", 50, 100.0)  # spend $5000; remaining cash $5000

    # Drop to $58 → equity = $5000 + 50*58 = $7900 < $8000 (20% floor of $10000)
    b.update_market_prices({"AAPL": 58.0})
    assert b.get_position("AAPL") is None  # flattened


def test_broker_hard_floor_does_not_fire_above_threshold() -> None:
    b = _broker(10_000.0)
    b.hard_floor_pct = 20.0
    _long(b, "AAPL", 50, 100.0)

    # Drop to $62 → equity = $5000 + 50*62 = $8100 > $8000 — should not flatten
    b.update_market_prices({"AAPL": 62.0})
    assert b.get_position("AAPL") is not None


def test_broker_hard_floor_off_by_default() -> None:
    b = _broker(10_000.0)
    _long(b, "AAPL", 99, 100.0)  # near-all-in
    # Crash to 0.01 — no floor set, position stays
    b.update_market_prices({"AAPL": 0.01})
    assert b.get_position("AAPL") is not None  # floor is None, no auto-flatten


# ---------------------------------------------------------------------------
# Hard floor — bench-level via RiskLimits.hard_floor_pct
# ---------------------------------------------------------------------------


def test_bench_hard_floor_flattens_via_observe_bar() -> None:
    """The bench's observe_bar path triggers hard-floor auto-flatten."""
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    # Add a competitor with a 15% hard-floor.
    comp = bench.add_competitor(
        "test",
        ScriptedTrader("test"),
    )
    # Wire the risk manager with a hard floor.
    from trading_agent.risk_manager import RiskLimits, RiskManager
    comp.risk = RiskManager(limits=RiskLimits(hard_floor_pct=15.0), kill_switch_file=None)

    # Buy 80 shares @ 100 → position $8000, cash $2000, equity $10000
    comp.broker.update_market_prices({"AAPL": 100.0})
    comp.broker.place_order({"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 80})

    # Price falls to 73 → equity = $2000 + 80*73 = $7840 < $8500 (15% floor of $10000)
    bench.observe_bar({"symbol": "AAPL", "close": 73.0})
    assert comp.broker.get_position("AAPL") is None  # auto-flattened
