"""Tests for PaperBroker specifics not covered by the contract suite.

Focus: not-connected guard, return-dict shape, buy-then-sell flat,
insufficient-cash REJECTED, missing-market-price REJECTED, balance accounting
across multiple trades, unknown-id get_order_status, get_account_value math.
"""

from __future__ import annotations

import pytest

from trading_agent.enums import OrderSide, OrderType
from trading_agent.paper_broker import PaperBroker


@pytest.fixture
def broker() -> PaperBroker:
    b = PaperBroker(initial_balance=10_000.0)
    b.connect()
    b.update_market_prices({"AAPL": 100.0, "TSLA": 200.0})
    return b


# --- connection guard --------------------------------------------------------


def test_place_order_raises_when_not_connected():
    b = PaperBroker(initial_balance=1_000.0)
    # Did not call connect()
    with pytest.raises(RuntimeError):
        b.place_order(
            {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
        )


def test_get_order_status_raises_when_not_connected():
    b = PaperBroker()
    with pytest.raises(RuntimeError):
        b.get_order_status("any")


def test_cancel_order_raises_when_not_connected():
    b = PaperBroker()
    with pytest.raises(RuntimeError):
        b.cancel_order("any")


# --- return dict shape -------------------------------------------------------


def test_place_order_returns_expected_keys(broker: PaperBroker):
    result = broker.place_order(
        {
            "symbol": "AAPL",
            "side": OrderSide.BUY,
            "order_type": OrderType.MARKET,
            "amount": 1.0,
        }
    )
    assert result is not None
    expected = {
        "order_id",
        "symbol",
        "side",
        "order_type",
        "quantity",
        "price",
        "status",
        "filled_quantity",
        "filled_price",
    }
    assert expected.issubset(result.keys())
    assert result["status"] == "FILLED"
    assert result["filled_price"] == 100.0
    assert result["side"] == "BUY"
    assert result["order_type"] == "market"


# --- buy then sell flat ------------------------------------------------------


def test_buy_then_sell_flattens_position(broker: PaperBroker):
    buy = broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 5.0}
    )
    assert buy["status"] == "FILLED"
    assert broker.get_position("AAPL") is not None

    sell = broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 5.0}
    )
    assert sell["status"] == "FILLED"
    assert broker.get_position("AAPL") is None  # flat
    # Cash should be back to initial (no commission modelled)
    assert broker.get_balance()["cash"] == pytest.approx(10_000.0)


# --- rejection paths ---------------------------------------------------------


def test_insufficient_cash_buy_rejected(broker: PaperBroker):
    # 200 shares at $100 = $20,000 > $10,000 cash
    result = broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 200.0}
    )
    assert result["status"] == "REJECTED"
    assert broker.get_balance()["cash"] == pytest.approx(10_000.0)
    assert broker.get_position("AAPL") is None


def test_missing_market_price_rejected(broker: PaperBroker):
    # No price set for SYMX in market_prices
    result = broker.place_order(
        {"symbol": "SYMX", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    assert result["status"] == "REJECTED"


def test_sell_without_position_rejected(broker: PaperBroker):
    result = broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    assert result["status"] == "REJECTED"


# --- balance accounting across trades ---------------------------------------


def test_balance_accounting_multiple_trades(broker: PaperBroker):
    # Buy 10 AAPL @ 100  -> cash 9000
    broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    assert broker.get_balance()["cash"] == pytest.approx(9_000.0)

    # Buy 5 TSLA @ 200 -> cash 8000
    broker.place_order(
        {"symbol": "TSLA", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 5.0}
    )
    assert broker.get_balance()["cash"] == pytest.approx(8_000.0)

    # Price moves; sell 5 AAPL @ 110 -> +550 -> cash 8550
    broker.update_market_prices({"AAPL": 110.0})
    broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 5.0}
    )
    assert broker.get_balance()["cash"] == pytest.approx(8_550.0)


# --- unknown order id --------------------------------------------------------


def test_get_order_status_unknown_id_returns_empty(broker: PaperBroker):
    assert broker.get_order_status("does-not-exist") == {}


# --- get_account_value ------------------------------------------------------


def test_get_account_value_combines_cash_and_positions(broker: PaperBroker):
    broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    # cash = 9000, holds 10 AAPL. If AAPL mark = 120, total = 9000 + 10*120 = 10200.
    val = broker.get_account_value({"AAPL": 120.0})
    assert val == pytest.approx(10_200.0)


def test_get_account_value_ignores_unpriced_symbols(broker: PaperBroker):
    broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    # No mark for AAPL in arg -> only cash contributes.
    val = broker.get_account_value({})
    assert val == pytest.approx(9_000.0)
