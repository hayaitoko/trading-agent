"""SignalRouter dispatch tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_agent.enums import Mode, OrderSide, OrderType
from trading_agent.signal_router import SignalRouter, _signal_to_order


class FakeBroker:
    """Records every place_order call and returns a synthetic order receipt."""

    def __init__(self) -> None:
        self.placed: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.next_result: dict[str, Any] | None = None
        self.return_none = False

    def place_order(self, order_details: dict[str, Any]) -> dict[str, Any] | None:
        self.placed.append(order_details)
        if self.return_none:
            return None
        return self.next_result or {"order_id": f"ord-{len(self.placed)}", "status": "FILLED"}

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return True

    def get_positions(self) -> list[dict[str, Any]]:
        return [{"symbol": "AAPL", "quantity": 10}]


class FakeQueue:
    def __init__(self) -> None:
        self.added: list[dict[str, Any]] = []

    def add(self, signal: dict[str, Any]) -> str:
        self.added.append(signal)
        return f"prop-{len(self.added)}"


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker()


@pytest.fixture
def queue() -> FakeQueue:
    return FakeQueue()


@pytest.fixture
def router(broker: FakeBroker, queue: FakeQueue) -> SignalRouter:
    return SignalRouter(broker, approval_queue=queue, global_mode=Mode.AUTONOMOUS)


# --- Autonomous dispatch ------------------------------------------------------

def test_autonomous_places_order(router: SignalRouter, broker: FakeBroker) -> None:
    signal = {"asset": "AAPL", "side": "LONG", "amount": 5.0}
    result = router.dispatch(signal)
    assert result is not None
    assert len(broker.placed) == 1
    placed = broker.placed[0]
    assert placed["symbol"] == "AAPL"
    assert placed["side"] is OrderSide.BUY
    assert placed["order_type"] is OrderType.MARKET
    assert placed["amount"] == 5.0


def test_autonomous_short_maps_to_sell(router: SignalRouter, broker: FakeBroker) -> None:
    router.dispatch({"asset": "AAPL", "side": "SHORT", "amount": 5.0})
    assert broker.placed[0]["side"] is OrderSide.SELL


def test_neutral_signal_is_noop(router: SignalRouter, broker: FakeBroker, queue: FakeQueue) -> None:
    result = router.dispatch({"asset": "AAPL", "side": "NEUTRAL", "amount": 1.0})
    assert result is None
    assert broker.placed == []
    assert queue.added == []


def test_limit_order_passes_price(router: SignalRouter, broker: FakeBroker) -> None:
    router.dispatch(
        {"asset": "AAPL", "side": "LONG", "amount": 1.0, "type": "limit", "price": 150.0}
    )
    placed = broker.placed[0]
    assert placed["order_type"] is OrderType.LIMIT
    assert placed["price"] == 150.0


def test_quantity_key_is_accepted(router: SignalRouter, broker: FakeBroker) -> None:
    router.dispatch({"asset": "AAPL", "side": "LONG", "quantity": 7.5})
    assert broker.placed[0]["amount"] == 7.5


def test_amount_zero_is_passed_through(router: SignalRouter, broker: FakeBroker) -> None:
    router.dispatch({"asset": "AAPL", "side": "LONG", "amount": 0})
    assert broker.placed[0]["amount"] == 0.0


def test_buy_sell_strings_also_work(router: SignalRouter, broker: FakeBroker) -> None:
    router.dispatch({"asset": "AAPL", "side": "BUY", "amount": 1.0})
    router.dispatch({"asset": "AAPL", "side": "SELL", "amount": 1.0})
    assert broker.placed[0]["side"] is OrderSide.BUY
    assert broker.placed[1]["side"] is OrderSide.SELL


def test_enum_side_passed_through(router: SignalRouter, broker: FakeBroker) -> None:
    router.dispatch({"asset": "AAPL", "side": OrderSide.SELL, "amount": 1.0})
    assert broker.placed[0]["side"] is OrderSide.SELL


def test_broker_returning_none_raises(router: SignalRouter, broker: FakeBroker) -> None:
    broker.return_none = True
    with pytest.raises(RuntimeError, match="Broker failed"):
        router.dispatch({"asset": "AAPL", "side": "LONG", "amount": 1.0})


# --- Approval dispatch --------------------------------------------------------

def test_approval_mode_routes_to_queue(
    router: SignalRouter, broker: FakeBroker, queue: FakeQueue
) -> None:
    router.set_global_mode(Mode.APPROVAL)
    signal = {"asset": "AAPL", "side": "LONG", "amount": 5.0}
    result = router.dispatch(signal)
    assert result == "prop-1"
    assert broker.placed == []
    assert queue.added == [signal]


def test_per_asset_mode_overrides_global(
    router: SignalRouter, broker: FakeBroker, queue: FakeQueue
) -> None:
    router.set_asset_mode("AAPL", Mode.APPROVAL)
    router.dispatch({"asset": "AAPL", "side": "LONG", "amount": 1.0})
    router.dispatch({"asset": "TSLA", "side": "LONG", "amount": 1.0})
    assert len(queue.added) == 1 and queue.added[0]["asset"] == "AAPL"
    assert len(broker.placed) == 1 and broker.placed[0]["symbol"] == "TSLA"


def test_approval_mode_without_queue_raises(broker: FakeBroker) -> None:
    router = SignalRouter(broker, approval_queue=None, global_mode=Mode.APPROVAL)
    with pytest.raises(RuntimeError, match="no approval_queue"):
        router.dispatch({"asset": "AAPL", "side": "LONG", "amount": 1.0})


# --- Error cases --------------------------------------------------------------

def test_missing_asset_raises(router: SignalRouter) -> None:
    with pytest.raises(ValueError, match="asset"):
        router.dispatch({"side": "LONG", "amount": 1.0})


def test_missing_side_raises(router: SignalRouter) -> None:
    with pytest.raises(ValueError, match="side"):
        router.dispatch({"asset": "AAPL", "amount": 1.0})


def test_missing_amount_raises(router: SignalRouter) -> None:
    with pytest.raises(ValueError, match="amount"):
        router.dispatch({"asset": "AAPL", "side": "LONG"})


def test_invalid_side_raises(router: SignalRouter) -> None:
    with pytest.raises(ValueError, match="Invalid side"):
        router.dispatch({"asset": "AAPL", "side": "DIAGONAL", "amount": 1.0})


def test_invalid_order_type_raises(router: SignalRouter) -> None:
    with pytest.raises(ValueError, match="Invalid order type"):
        router.dispatch({"asset": "AAPL", "side": "LONG", "amount": 1.0, "type": "telepathic"})


# --- Passthroughs -------------------------------------------------------------

def test_cancel_order_delegates(router: SignalRouter, broker: FakeBroker) -> None:
    assert router.cancel_order("abc") is True
    assert broker.cancelled == ["abc"]


def test_get_positions_delegates(router: SignalRouter) -> None:
    positions = router.get_positions()
    assert positions == [{"symbol": "AAPL", "quantity": 10}]


# --- _signal_to_order direct tests --------------------------------------------

def test_signal_to_order_canonical_shape() -> None:
    order = _signal_to_order(
        {"asset": "AAPL", "side": "LONG", "amount": 5.0, "type": "limit", "price": 100.0}
    )
    assert order == {
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "amount": 5.0,
        "price": 100.0,
    }


def test_signal_to_order_accepts_enum_inputs() -> None:
    order = _signal_to_order(
        {
            "asset": "AAPL",
            "side": OrderSide.SELL,
            "amount": 1.0,
            "type": OrderType.MARKET,
        }
    )
    assert order["side"] is OrderSide.SELL
    assert order["order_type"] is OrderType.MARKET


# --- Misc ---------------------------------------------------------------------

def test_unknown_mode_raises(broker: FakeBroker) -> None:
    router = SignalRouter(broker, approval_queue=FakeQueue(), global_mode=Mode.AUTONOMOUS)
    sentinel = MagicMock()
    router.global_mode = sentinel  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown mode"):
        router.dispatch({"asset": "AAPL", "side": "LONG", "amount": 1.0})
