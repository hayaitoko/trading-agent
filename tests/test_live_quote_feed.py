"""Tests for LiveQuoteFeed: real quotes -> PaperBroker fills + bus events."""

from __future__ import annotations

import asyncio
from typing import Any

from trading_agent.data_feed import MessageBus
from trading_agent.enums import OrderSide, OrderType
from trading_agent.feeds import LiveQuoteFeed
from trading_agent.paper_broker import PaperBroker


class FakeQuoteSource:
    """Stand-in for AlpacaBroker / CCXTBroker get_quote()."""

    def __init__(self, quotes: dict[str, dict[str, Any]]):
        self._quotes = quotes
        self.calls: list[str] = []

    def get_quote(self, symbol: str) -> dict[str, Any]:
        self.calls.append(symbol)
        if symbol not in self._quotes:
            raise ValueError(f"no quote for {symbol}")
        return self._quotes[symbol]


def test_poll_feeds_real_quote_into_paper_broker():
    bus = MessageBus()
    src = FakeQuoteSource({"AAPL": {"bid": 149.0, "ask": 151.0, "price": 150.0}})
    broker = PaperBroker(initial_balance=100_000.0)
    broker.connect()
    feed = LiveQuoteFeed(bus, src, ["AAPL"], paper_broker=broker)

    feed.poll_once()

    # Broker now holds the real quote: a market buy should fill at the ask.
    result = broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    assert result["filled_price"] == 151.0


def test_poll_publishes_quote_and_bar_topics():
    bus = MessageBus()
    src = FakeQuoteSource({"BTC/USDT": {"bid": 60000.0, "ask": 60001.0, "last": 60000.5}})
    feed = LiveQuoteFeed(bus, src, ["BTC/USDT"])

    quotes: list[dict] = []
    bars: list[dict] = []
    bus.subscribe("quote.BTC/USDT", quotes.append)
    bus.subscribe("bar.BTC/USDT", bars.append)

    feed.poll_once()

    assert len(quotes) == 1
    assert quotes[0]["bid"] == 60000.0
    assert len(bars) == 1
    assert bars[0]["close"] == 60000.5  # synthesized from last


def test_poll_skips_symbol_on_quote_error():
    bus = MessageBus()
    src = FakeQuoteSource({"AAPL": {"price": 150.0}})  # MSFT will raise
    feed = LiveQuoteFeed(bus, src, ["AAPL", "MSFT"])

    results = feed.poll_once()
    assert "AAPL" in results
    assert "MSFT" not in results  # error swallowed, AAPL still processed


def test_price_falls_back_to_mid_when_no_last_or_price():
    bus = MessageBus()
    src = FakeQuoteSource({"AAPL": {"bid": 100.0, "ask": 102.0}})
    broker = PaperBroker(initial_balance=100_000.0)
    broker.connect()
    feed = LiveQuoteFeed(bus, src, ["AAPL"], paper_broker=broker)

    feed.poll_once()
    assert broker.get_quote("AAPL")["price"] == 101.0  # mid


def test_run_loop_bounded_by_max_polls():
    bus = MessageBus()
    src = FakeQuoteSource({"AAPL": {"price": 150.0}})
    feed = LiveQuoteFeed(bus, src, ["AAPL"], poll_interval=0.0)

    polls = asyncio.run(feed.run(max_polls=3))
    assert polls == 3
    assert src.calls.count("AAPL") == 3


def test_subscribe_unsubscribe_symbols():
    bus = MessageBus()
    src = FakeQuoteSource({"AAPL": {"price": 150.0}, "MSFT": {"price": 400.0}})
    feed = LiveQuoteFeed(bus, src, ["AAPL"])

    asyncio.run(feed.subscribe_symbols(["MSFT"]))
    results = feed.poll_once()
    assert set(results) == {"AAPL", "MSFT"}

    asyncio.run(feed.unsubscribe_symbols(["AAPL"]))
    results = feed.poll_once()
    assert set(results) == {"MSFT"}
