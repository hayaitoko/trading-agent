from decimal import Decimal

import pytest

from trading_agent.brokers import InsufficientFundsError, MockBroker
from trading_agent.models import Order


def _quotes(prices: dict[str, Decimal]):
    return lambda ticker: prices[ticker]


async def test_market_buy_reduces_cash_and_adds_position():
    broker = MockBroker(cash=Decimal("10000"), quote_fn=_quotes({"AAPL": Decimal("100")}))
    await broker.place_order(Order(ticker="AAPL", side="buy", qty=10))

    assert await broker.get_cash() == Decimal("9000")
    positions = await broker.get_positions()
    assert len(positions) == 1
    assert positions[0].ticker == "AAPL"
    assert positions[0].qty == 10
    assert positions[0].avg_cost == Decimal("100")


async def test_market_sell_returns_cash_and_removes_position():
    broker = MockBroker(cash=Decimal("10000"), quote_fn=_quotes({"AAPL": Decimal("100")}))
    await broker.place_order(Order(ticker="AAPL", side="buy", qty=10))
    await broker.place_order(Order(ticker="AAPL", side="sell", qty=10))

    assert await broker.get_cash() == Decimal("10000")
    assert await broker.get_positions() == []


async def test_insufficient_cash_on_buy():
    broker = MockBroker(cash=Decimal("500"), quote_fn=_quotes({"AAPL": Decimal("100")}))
    with pytest.raises(InsufficientFundsError):
        await broker.place_order(Order(ticker="AAPL", side="buy", qty=10))


async def test_oversell_rejected():
    broker = MockBroker(cash=Decimal("10000"), quote_fn=_quotes({"AAPL": Decimal("100")}))
    await broker.place_order(Order(ticker="AAPL", side="buy", qty=5))
    with pytest.raises(InsufficientFundsError):
        await broker.place_order(Order(ticker="AAPL", side="sell", qty=10))


async def test_average_cost_blends_on_subsequent_buys():
    prices = {"AAPL": Decimal("100")}
    broker = MockBroker(cash=Decimal("10000"), quote_fn=lambda t: prices[t])
    await broker.place_order(Order(ticker="AAPL", side="buy", qty=10))
    prices["AAPL"] = Decimal("200")
    await broker.place_order(Order(ticker="AAPL", side="buy", qty=10))

    positions = await broker.get_positions()
    assert positions[0].qty == 20
    assert positions[0].avg_cost == Decimal("150")


async def test_account_value_includes_unrealized():
    prices = {"AAPL": Decimal("100")}
    broker = MockBroker(cash=Decimal("10000"), quote_fn=lambda t: prices[t])
    await broker.place_order(Order(ticker="AAPL", side="buy", qty=10))
    prices["AAPL"] = Decimal("150")

    assert await broker.get_account_value() == Decimal("10500")
