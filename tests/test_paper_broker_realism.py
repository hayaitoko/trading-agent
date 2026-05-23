"""Tests for the PaperBroker realism features: bid/ask fills, slippage,
commission, limit-order matching, and the market-hours gate."""

from __future__ import annotations

import pytest

from trading_agent.enums import OrderSide, OrderType
from trading_agent.paper_broker import PaperBroker


def _broker(**kwargs) -> PaperBroker:
    b = PaperBroker(initial_balance=100_000.0, **kwargs)
    b.connect()
    return b


# --- bid/ask aware fills -----------------------------------------------------


def test_market_buy_fills_at_ask():
    b = _broker()
    b.update_quote("AAPL", bid=149.0, ask=151.0)
    result = b.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    assert result["status"] == "FILLED"
    assert result["filled_price"] == 151.0  # paid the ask


def test_market_sell_fills_at_bid():
    b = _broker()
    b.update_quote("AAPL", bid=149.0, ask=151.0)
    b.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    sell = b.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    assert sell["filled_price"] == 149.0  # received the bid


def test_quote_updates_market_prices_mid_for_valuation():
    b = _broker()
    b.update_quote("AAPL", bid=100.0, ask=102.0)
    # No explicit last -> mid (101) used for get_quote / valuation
    assert b.get_quote("AAPL")["price"] == 101.0


# --- slippage ----------------------------------------------------------------


def test_slippage_widens_buy_price():
    b = _broker(slippage_bps=10.0)  # 10 bps = 0.10%
    b.update_market_prices({"AAPL": 100.0})
    result = b.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    assert result["filled_price"] == pytest.approx(100.10)


def test_slippage_lowers_sell_price():
    b = _broker(slippage_bps=10.0)
    b.update_market_prices({"AAPL": 100.0})
    b.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    sell = b.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    assert sell["filled_price"] == pytest.approx(99.90)


# --- commission --------------------------------------------------------------


def test_commission_deducted_from_cash():
    b = _broker(commission_bps=20.0)  # 0.20%
    b.update_market_prices({"AAPL": 100.0})
    b.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    # notional 1000, commission 0.20% = 2.0 -> cash = 100000 - 1000 - 2
    assert b.get_balance()["cash"] == pytest.approx(98_998.0)


# --- limit-order matching ----------------------------------------------------


def test_buy_limit_fills_when_price_drops_to_limit():
    b = _broker()
    b.update_market_prices({"AAPL": 150.0})
    order = b.place_order(
        {
            "symbol": "AAPL",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "amount": 1.0,
            "price": 145.0,
        }
    )
    assert order["status"] == "PENDING"  # 150 > 145, not marketable yet

    b.update_market_prices({"AAPL": 144.0})  # crosses the limit
    status = b.get_order_status(order["order_id"])
    assert status["status"] == "FILLED"
    assert status["filled_price"] == 144.0  # filled at the better price
    assert b.get_position("AAPL").quantity == 1.0


def test_sell_limit_fills_when_price_rises_to_limit():
    b = _broker()
    b.update_market_prices({"AAPL": 100.0})
    b.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    sell = b.place_order(
        {
            "symbol": "AAPL",
            "side": OrderSide.SELL,
            "order_type": OrderType.LIMIT,
            "amount": 1.0,
            "price": 110.0,
        }
    )
    assert sell["status"] == "PENDING"  # 100 < 110

    b.update_market_prices({"AAPL": 112.0})  # crosses
    status = b.get_order_status(sell["order_id"])
    assert status["status"] == "FILLED"
    assert status["filled_price"] == 112.0
    assert b.get_position("AAPL") is None  # flat again


def test_marketable_limit_fills_immediately():
    b = _broker()
    b.update_market_prices({"AAPL": 150.0})
    # Buy limit above market is immediately marketable; fills at the better price.
    order = b.place_order(
        {
            "symbol": "AAPL",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "amount": 1.0,
            "price": 155.0,
        }
    )
    assert order["status"] == "FILLED"
    assert order["filled_price"] == 150.0


def test_cancelled_limit_does_not_fill_on_later_cross():
    b = _broker()
    b.update_market_prices({"AAPL": 150.0})
    order = b.place_order(
        {
            "symbol": "AAPL",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "amount": 1.0,
            "price": 145.0,
        }
    )
    assert b.cancel_order(order["order_id"]) is True
    b.update_market_prices({"AAPL": 140.0})  # would have crossed
    assert b.get_order_status(order["order_id"])["status"] == "CANCELLED"


# --- market-hours gate -------------------------------------------------------


def test_market_order_rejected_when_market_closed():
    b = _broker(is_market_open=lambda _s: False)
    b.update_market_prices({"AAPL": 100.0})
    result = b.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    assert result["status"] == "REJECTED"


def test_limit_order_queues_while_closed_then_fills_when_open():
    state = {"open": False}
    b = _broker(is_market_open=lambda _s: state["open"])
    b.update_market_prices({"AAPL": 150.0})
    order = b.place_order(
        {
            "symbol": "AAPL",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "amount": 1.0,
            "price": 160.0,  # marketable, but market is closed
        }
    )
    assert order["status"] == "PENDING"

    state["open"] = True
    b.update_market_prices({"AAPL": 150.0})  # re-trigger matching now that we're open
    assert b.get_order_status(order["order_id"])["status"] == "FILLED"


def test_us_equity_clock_returns_callable():
    from trading_agent.market_hours import is_us_equity_market_open, us_equity_clock

    clock = us_equity_clock()
    assert clock("AAPL") == is_us_equity_market_open()
