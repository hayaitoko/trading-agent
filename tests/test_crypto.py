"""Crypto tradeability: 24/7 market clock, fractional fills, and a crypto ticker
feed driving a scripted trader on the bench end-to-end."""

from __future__ import annotations

from typing import Any

import pytest

from trading_agent.bench.bench import Bench
from trading_agent.enums import AssetClass, OrderSide, OrderType
from trading_agent.feeds.crypto_ticker import CryptoTickerFeed
from trading_agent.llm.trader import DecisionResult, TradeDecision
from trading_agent.market_hours import is_crypto_symbol, market_clock
from trading_agent.paper_broker import PaperBroker

# --- asset-class classification ----------------------------------------------


def test_asset_class_enum_values() -> None:
    assert AssetClass.CRYPTO.value == "crypto"
    assert AssetClass.EQUITY.value == "equity"


def test_is_crypto_symbol_by_pair_notation() -> None:
    assert is_crypto_symbol("BTC/USDT") is True
    assert is_crypto_symbol("AAPL") is False


def test_is_crypto_symbol_by_explicit_set() -> None:
    crypto = {"XBTUSD"}  # an exchange that doesn't use slash notation
    assert is_crypto_symbol("XBTUSD", crypto) is True
    assert is_crypto_symbol("BTC/USDT", crypto) is False  # set overrides heuristic


# --- 24/7 market clock -------------------------------------------------------


def test_market_clock_crypto_always_open_equity_closed() -> None:
    clock = market_clock({"BTC/USDT"}, equity_open=lambda: False)
    assert clock("BTC/USDT") is True  # crypto bypasses the closed equity session
    assert clock("AAPL") is False


def test_market_clock_defers_to_equity_session_when_open() -> None:
    clock = market_clock({"BTC/USDT"}, equity_open=lambda: True)
    assert clock("AAPL") is True
    assert clock("BTC/USDT") is True


def test_crypto_market_order_fills_while_equity_market_closed() -> None:
    b = PaperBroker(initial_balance=100_000.0, is_market_open=market_clock(equity_open=lambda: False))
    b.connect()
    b.update_market_prices({"BTC/USDT": 60_000.0, "AAPL": 150.0})
    crypto = b.place_order(
        {"symbol": "BTC/USDT", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 0.1}
    )
    equity = b.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    assert crypto["status"] == "FILLED"
    assert equity["status"] == "REJECTED"  # equity gated, crypto not


# --- fractional quantities ---------------------------------------------------


def test_fractional_crypto_fill() -> None:
    b = PaperBroker(initial_balance=100_000.0)
    b.connect()
    b.update_market_prices({"BTC/USDT": 60_000.0})
    result = b.place_order(
        {"symbol": "BTC/USDT", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 0.25}
    )
    assert result["status"] == "FILLED"
    assert result["filled_quantity"] == 0.25
    pos = b.get_position("BTC/USDT")
    assert pos.quantity == 0.25
    # 0.25 BTC @ 60k = 15k spent.
    assert b.get_balance()["cash"] == pytest.approx(85_000.0)


def test_fractional_partial_close_pnl() -> None:
    b = PaperBroker(initial_balance=100_000.0)
    b.connect()
    b.update_market_prices({"ETH/USDT": 3_000.0})
    b.place_order(
        {"symbol": "ETH/USDT", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.5}
    )
    b.update_market_prices({"ETH/USDT": 3_200.0})
    b.place_order(
        {"symbol": "ETH/USDT", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 0.5}
    )
    # 0.5 * (3200 - 3000) = 100
    assert b.get_realized_pnl() == pytest.approx(100.0)
    assert b.get_position("ETH/USDT").quantity == pytest.approx(1.0)


# --- crypto ticker feed ------------------------------------------------------


class _FakeExchangeSource:
    """A ccxt-style quote source returning canned tickers."""

    def __init__(self, tickers: dict[str, dict[str, Any]]) -> None:
        self._tickers = tickers

    def get_quote(self, symbol: str) -> dict[str, Any]:
        if symbol not in self._tickers:
            raise ValueError(f"unknown symbol {symbol}")
        return self._tickers[symbol]


def test_crypto_ticker_feed_pushes_quote_and_bar() -> None:
    src = _FakeExchangeSource(
        {"BTC/USDT": {"bid": 59_999.0, "ask": 60_001.0, "last": 60_000.0, "volume": 12.0}}
    )
    quotes: list[dict[str, Any]] = []
    bars: list[dict[str, Any]] = []
    feed = CryptoTickerFeed(
        src, ["BTC/USDT"], on_quote=quotes.append, on_bar=bars.append
    )
    raw = feed.poll_once()
    assert "BTC/USDT" in raw
    assert quotes[0]["price"] == 60_000.0
    assert quotes[0]["bid"] == 59_999.0
    assert bars[0]["close"] == 60_000.0
    assert bars[0]["volume"] == 12.0


def test_crypto_ticker_feed_skips_bad_symbol() -> None:
    src = _FakeExchangeSource({"BTC/USDT": {"last": 60_000.0}})
    quotes: list[dict[str, Any]] = []
    feed = CryptoTickerFeed(src, ["BTC/USDT", "DOGE/USDT"], on_quote=quotes.append)
    raw = feed.poll_once()  # DOGE raises, BTC ok
    assert set(raw) == {"BTC/USDT"}
    assert len(quotes) == 1


# --- end-to-end: a scripted trader trading crypto on the bench ---------------


class _ScriptedCryptoTrader:
    """Buys a fixed fractional amount of a crypto symbol on its first decision."""

    def __init__(self, symbol: str, amount: float) -> None:
        self.name = "scripted-crypto"
        self.model = "scripted"
        self.symbol = symbol
        self.amount = amount
        self._fired = False

    def observe(self, bar: dict[str, Any]) -> None:  # noqa: D401 - protocol method
        pass

    def decide(self, account: dict[str, Any]) -> DecisionResult:
        if self._fired:
            return DecisionResult(comment="done")
        self._fired = True
        return DecisionResult(
            decisions=[TradeDecision(self.symbol, "BUY", self.amount, "scripted entry")],
            comment="entry",
        )


def test_crypto_trades_end_to_end_on_bench() -> None:
    bench = Bench(["BTC/USDT"], initial_balance=100_000.0)
    trader = _ScriptedCryptoTrader("BTC/USDT", 0.5)
    bench.add_competitor(trader.name, trader)

    # Feed a crypto tick straight into the bench's observe callbacks.
    src = _FakeExchangeSource(
        {"BTC/USDT": {"bid": 59_999.0, "ask": 60_001.0, "last": 60_000.0, "volume": 5.0}}
    )
    feed = CryptoTickerFeed(
        src, ["BTC/USDT"], on_quote=bench.observe_quote, on_bar=bench.observe_bar
    )
    feed.poll_once()

    bench.run_decisions()

    board = bench.leaderboard()
    row = board[0]
    positions = {p["symbol"]: p["quantity"] for p in row["positions"]}
    assert positions["BTC/USDT"] == pytest.approx(0.5)
    # 0.5 BTC bought at the ask (60_001) → cash drawn down.
    assert row["cash"] == pytest.approx(100_000.0 - 0.5 * 60_001.0)
    assert row["trades"] == 1
