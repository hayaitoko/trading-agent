"""Tests for Signal, Order, Position dataclasses and SignalType/PositionType/OrderType enums."""

from __future__ import annotations

from datetime import datetime

import pytest

from trading_agent.enums import OrderType
from trading_agent.models import (
    Order,
    Position,
    PositionType,
    Signal,
    SignalType,
)

# --- enums --------------------------------------------------------------------


def test_signal_type_values():
    assert SignalType.BUY.value == "buy"
    assert SignalType.SELL.value == "sell"
    assert SignalType.HOLD.value == "hold"


def test_position_type_values():
    assert PositionType.LONG.value == "long"
    assert PositionType.SHORT.value == "short"


def test_order_type_values_lowercase():
    # OrderType is re-exported by models from .enums; values are lowercase.
    assert OrderType.MARKET.value == "market"
    assert OrderType.LIMIT.value == "limit"
    assert OrderType.STOP.value == "stop"
    assert OrderType.STOP_LIMIT.value == "stop_limit"


# --- Signal -------------------------------------------------------------------


def test_signal_minimal_construction():
    ts = datetime(2026, 1, 1, 12, 0, 0)
    s = Signal(symbol="AAPL", type=SignalType.BUY, strength=0.75, timestamp=ts)
    assert s.symbol == "AAPL"
    assert s.type is SignalType.BUY
    assert s.strength == 0.75
    assert s.timestamp == ts
    assert s.price is None
    assert s.metadata is None


def test_signal_with_optional_fields():
    ts = datetime(2026, 1, 1)
    meta = {"source": "mean_reversion"}
    s = Signal(
        symbol="BTC-USD",
        type=SignalType.SELL,
        strength=0.5,
        timestamp=ts,
        price=42_000.0,
        metadata=meta,
    )
    assert s.price == 42_000.0
    assert s.metadata == meta


# --- Order --------------------------------------------------------------------


def test_order_minimal_construction_defaults_timestamp():
    o = Order(symbol="AAPL", type=OrderType.MARKET, quantity=10)
    assert o.symbol == "AAPL"
    assert o.type is OrderType.MARKET
    assert o.quantity == 10
    assert o.price is None
    # timestamp auto-populates via __post_init__
    assert isinstance(o.timestamp, datetime)


def test_order_timestamp_preserved_when_supplied():
    ts = datetime(2026, 5, 22, 9, 30)
    o = Order(symbol="TSLA", type=OrderType.LIMIT, quantity=2.5, price=180.0, timestamp=ts)
    assert o.timestamp == ts
    assert o.price == 180.0


def test_order_post_init_uses_close_to_now():
    before = datetime.now()
    o = Order(symbol="AAPL", type=OrderType.MARKET, quantity=1)
    after = datetime.now()
    assert before <= o.timestamp <= after


# --- Position -----------------------------------------------------------------


def test_position_construction_with_defaults():
    ts = datetime(2026, 3, 1)
    p = Position(symbol="AAPL", type=PositionType.LONG, quantity=5, entry_price=150.0, timestamp=ts)
    assert p.symbol == "AAPL"
    assert p.type is PositionType.LONG
    assert p.quantity == 5
    assert p.entry_price == 150.0
    assert p.timestamp == ts
    assert p.profit_loss == 0.0


def test_position_with_profit_loss():
    p = Position(
        symbol="TSLA",
        type=PositionType.SHORT,
        quantity=2,
        entry_price=200.0,
        timestamp=datetime(2026, 4, 1),
        profit_loss=12.5,
    )
    assert p.type is PositionType.SHORT
    assert p.profit_loss == 12.5


# --- Construction errors ------------------------------------------------------


def test_signal_requires_required_fields():
    with pytest.raises(TypeError):
        Signal(symbol="AAPL", type=SignalType.BUY)  # type: ignore[call-arg]


def test_order_requires_required_fields():
    with pytest.raises(TypeError):
        Order(symbol="AAPL")  # type: ignore[call-arg]
