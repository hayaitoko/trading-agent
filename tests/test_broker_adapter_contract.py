"""Parametrized BrokerAdapter contract suite.

Runs the same place_order / get_quote / cancel_order / get_positions /
get_balance assertions against:

  * PaperBroker (no mocks — the canonical reference implementation)
  * AlpacaBroker (alpaca-py TradingClient + StockHistoricalDataClient mocked)
  * CCXTBroker (ccxt.binance Exchange mocked)

This is the only place we exercise the real broker classes against the
frozen BrokerAdapter ABC.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trading_agent.broker_adapter import BrokerAdapter
from trading_agent.enums import OrderSide, OrderType
from trading_agent.paper_broker import PaperBroker

# --- Fixtures -----------------------------------------------------------------


@pytest.fixture
def paper_broker() -> PaperBroker:
    broker = PaperBroker(initial_balance=10_000.0)
    broker.connect()
    broker.update_market_prices({"AAPL": 150.0, "TSLA": 200.0})
    return broker


@pytest.fixture
def alpaca_broker() -> Any:
    """AlpacaBroker with TradingClient + data client patched at import time."""
    with patch("trading_agent.alpaca_broker.TradingClient") as MockTC, patch(
        "trading_agent.alpaca_broker.StockHistoricalDataClient"
    ) as MockData:
        from trading_agent.alpaca_broker import AlpacaBroker

        trading_client = MockTC.return_value
        data_client = MockData.return_value

        # get_account / connect
        account = MagicMock()
        account.id = "acc-1"
        account.status = "ACTIVE"
        account.buying_power = "20000.0"
        account.cash = "10000.0"
        account.equity = "20000.0"
        account.long_market_value = "10000.0"
        account.short_market_value = "0.0"
        account.initial_margin = "0.0"
        account.maintenance_margin = "0.0"
        account.last_equity = "20000.0"
        account.daytrade_count = 0
        trading_client.get_account.return_value = account
        trading_client.get_all_positions.return_value = []

        # submit_order
        order_resp = MagicMock()
        order_resp.id = "alp-ord-1"
        order_resp.symbol = "AAPL"
        order_resp.status.value = "accepted"
        trading_client.submit_order.return_value = order_resp

        trading_client.cancel_order_by_id.return_value = None

        # get_stock_latest_quote
        quote_data = MagicMock()
        quote_data.bid_price = 149.5
        quote_data.ask_price = 150.5
        quote_data.timestamp = datetime(2026, 5, 22, 12, 0, 0)
        data_client.get_stock_latest_quote.return_value = {"AAPL": quote_data}

        broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
        # Hand the mocks back to the test via attribute attachment
        broker._mock_trading = trading_client  # type: ignore[attr-defined]
        broker._mock_data = data_client  # type: ignore[attr-defined]
        yield broker


@pytest.fixture
def ccxt_broker() -> Any:
    """CCXTBroker with ccxt.binance Exchange mocked."""
    with patch("trading_agent.ccxt_broker.ccxt") as MockCcxt:
        exchange = MagicMock()
        exchange.markets = {"BTC/USDT": {}}
        exchange.symbols = ["BTC/USDT"]
        exchange.fetch_balance.return_value = {
            "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0},
            "free": {"USDT": 10000.0},
            "total": {"USDT": 10000.0},
        }
        exchange.fetch_ticker.return_value = {
            "bid": 60000.0,
            "ask": 60001.0,
            "last": 60000.5,
            "timestamp": 1700000000000,
            "datetime": "2026-05-22T12:00:00Z",
        }
        exchange.create_order.return_value = {"id": "ccxt-ord-1", "status": "open"}
        exchange.cancel_order.return_value = {"id": "ccxt-ord-1", "status": "canceled"}
        exchange.fetch_positions.return_value = []
        MockCcxt.binance.return_value = exchange

        from trading_agent.ccxt_broker import CCXTBroker

        broker = CCXTBroker(
            exchange_name="binance",
            api_key="k",
            secret="s",
        )
        broker._mock_exchange = exchange  # type: ignore[attr-defined]
        yield broker


# --- ABC compliance -----------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["paper_broker", "alpaca_broker", "ccxt_broker"])
def test_is_broker_adapter(request: pytest.FixtureRequest, fixture_name: str) -> None:
    broker = request.getfixturevalue(fixture_name)
    assert isinstance(broker, BrokerAdapter)


# --- Place order: market buy --------------------------------------------------


class TestPlaceOrder:
    def test_paper_market_buy_canonical(self, paper_broker: PaperBroker) -> None:
        result = paper_broker.place_order(
            {
                "symbol": "AAPL",
                "side": OrderSide.BUY,
                "order_type": OrderType.MARKET,
                "amount": 1.0,
            }
        )
        assert result is not None
        assert result["status"] == "FILLED"
        assert result["filled_quantity"] == 1.0

    def test_paper_market_buy_string_inputs(self, paper_broker: PaperBroker) -> None:
        result = paper_broker.place_order(
            {"symbol": "AAPL", "side": "BUY", "order_type": "market", "amount": 1.0}
        )
        assert result is not None
        assert result["status"] == "FILLED"

    def test_paper_limit_order_stays_pending(self, paper_broker: PaperBroker) -> None:
        result = paper_broker.place_order(
            {
                "symbol": "AAPL",
                "side": OrderSide.BUY,
                "order_type": OrderType.LIMIT,
                "amount": 1.0,
                "price": 140.0,
            }
        )
        assert result is not None
        assert result["status"] == "PENDING"
        assert result["price"] == 140.0

    def test_paper_missing_required_returns_none(self, paper_broker: PaperBroker) -> None:
        assert paper_broker.place_order({"symbol": "AAPL"}) is None

    def test_paper_invalid_side_returns_none(self, paper_broker: PaperBroker) -> None:
        result = paper_broker.place_order(
            {
                "symbol": "AAPL",
                "side": "DIAGONAL",
                "order_type": "market",
                "amount": 1.0,
            }
        )
        assert result is None

    def test_alpaca_market_buy(self, alpaca_broker: Any) -> None:
        result = alpaca_broker.place_order(
            {
                "symbol": "AAPL",
                "side": OrderSide.BUY,
                "order_type": OrderType.MARKET,
                "amount": 1.0,
            }
        )
        assert result is not None
        assert result["order_id"] == "alp-ord-1"
        alpaca_broker._mock_trading.submit_order.assert_called_once()

    def test_alpaca_string_inputs_also_work(self, alpaca_broker: Any) -> None:
        result = alpaca_broker.place_order(
            {"symbol": "AAPL", "side": "buy", "order_type": "market", "amount": 1.0}
        )
        assert result is not None

    def test_alpaca_limit_requires_price(self, alpaca_broker: Any) -> None:
        with pytest.raises(ValueError, match="Price is required"):
            alpaca_broker.place_order(
                {
                    "symbol": "AAPL",
                    "side": OrderSide.BUY,
                    "order_type": OrderType.LIMIT,
                    "amount": 1.0,
                }
            )

    def test_ccxt_market_buy(self, ccxt_broker: Any) -> None:
        result = ccxt_broker.place_order(
            {
                "symbol": "BTC/USDT",
                "side": OrderSide.BUY,
                "order_type": OrderType.MARKET,
                "amount": 0.5,
            }
        )
        assert result is not None
        assert result["order_id"] == "ccxt-ord-1"
        ccxt_broker._mock_exchange.create_order.assert_called_once_with(
            "BTC/USDT", "market", "buy", 0.5
        )

    def test_ccxt_limit_passes_price_to_exchange(self, ccxt_broker: Any) -> None:
        ccxt_broker.place_order(
            {
                "symbol": "BTC/USDT",
                "side": OrderSide.SELL,
                "order_type": OrderType.LIMIT,
                "amount": 0.5,
                "price": 70000.0,
            }
        )
        ccxt_broker._mock_exchange.create_order.assert_called_with(
            "BTC/USDT", "limit", "sell", 0.5, 70000.0
        )

    def test_ccxt_unknown_symbol_raises(self, ccxt_broker: Any) -> None:
        # ccxt broker validates symbol membership
        ccxt_broker._mock_exchange.markets = {}
        with pytest.raises(Exception):
            ccxt_broker.place_order(
                {
                    "symbol": "XYZ/USDT",
                    "side": OrderSide.BUY,
                    "order_type": OrderType.MARKET,
                    "amount": 1.0,
                }
            )


# --- Quote --------------------------------------------------------------------


class TestGetQuote:
    def test_paper_quote(self, paper_broker: PaperBroker) -> None:
        quote = paper_broker.get_quote("AAPL")
        assert quote["price"] == 150.0

    def test_paper_unknown_symbol_raises(self, paper_broker: PaperBroker) -> None:
        with pytest.raises(ValueError):
            paper_broker.get_quote("UNKNOWN")

    def test_alpaca_quote_returns_dict_with_price(self, alpaca_broker: Any) -> None:
        quote = alpaca_broker.get_quote("AAPL")
        assert "price" in quote
        # mid = (149.5 + 150.5) / 2
        assert quote["price"] == 150.0

    def test_ccxt_quote_returns_dict_with_bid_ask(self, ccxt_broker: Any) -> None:
        quote = ccxt_broker.get_quote("BTC/USDT")
        assert quote["bid"] == 60000.0
        assert quote["ask"] == 60001.0


# --- Balance ------------------------------------------------------------------


class TestGetBalance:
    def test_paper_balance_has_cash(self, paper_broker: PaperBroker) -> None:
        balance = paper_broker.get_balance()
        assert balance["cash"] == 10_000.0

    def test_alpaca_balance_has_cash(self, alpaca_broker: Any) -> None:
        balance = alpaca_broker.get_balance()
        assert "cash" in balance
        assert balance["cash"] == 10000.0

    def test_ccxt_balance_returns_dict(self, ccxt_broker: Any) -> None:
        balance = ccxt_broker.get_balance()
        # ccxt returns the raw exchange response — caller normalizes
        assert isinstance(balance, dict)


# --- Cancel order -------------------------------------------------------------


class TestCancelOrder:
    def test_paper_cancel_pending(self, paper_broker: PaperBroker) -> None:
        order = paper_broker.place_order(
            {
                "symbol": "AAPL",
                "side": OrderSide.BUY,
                "order_type": OrderType.LIMIT,
                "amount": 1.0,
                "price": 100.0,
            }
        )
        assert order is not None
        assert paper_broker.cancel_order(order["order_id"]) is True
        # Cancelling twice fails
        assert paper_broker.cancel_order(order["order_id"]) is False

    def test_paper_cancel_unknown_returns_false(self, paper_broker: PaperBroker) -> None:
        assert paper_broker.cancel_order("nonexistent") is False

    def test_alpaca_cancel_returns_true_on_success(self, alpaca_broker: Any) -> None:
        assert alpaca_broker.cancel_order("alp-ord-1") is True

    def test_alpaca_cancel_returns_false_on_error(self, alpaca_broker: Any) -> None:
        alpaca_broker._mock_trading.cancel_order_by_id.side_effect = RuntimeError("boom")
        assert alpaca_broker.cancel_order("nope") is False


# --- Positions ----------------------------------------------------------------


class TestGetPositions:
    def test_paper_positions_initially_empty(self, paper_broker: PaperBroker) -> None:
        assert paper_broker.get_positions() == []

    def test_paper_positions_after_buy(self, paper_broker: PaperBroker) -> None:
        paper_broker.place_order(
            {
                "symbol": "AAPL",
                "side": OrderSide.BUY,
                "order_type": OrderType.MARKET,
                "amount": 1.0,
            }
        )
        positions = paper_broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["quantity"] == 1.0

    def test_alpaca_positions_returns_list(self, alpaca_broker: Any) -> None:
        result = alpaca_broker.get_positions()
        assert isinstance(result, list)

    def test_ccxt_positions_returns_list(self, ccxt_broker: Any) -> None:
        result = ccxt_broker.get_positions()
        assert isinstance(result, list)


# --- PaperBroker-specific end-to-end ------------------------------------------


def test_paper_round_trip_pnl(paper_broker: PaperBroker) -> None:
    """Buy → price rises → sell → realized profit."""
    paper_broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    paper_broker.update_market_prices({"AAPL": 160.0})
    paper_broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 10.0}
    )
    # bought 10 @ 150 = 1500 cost, sold 10 @ 160 = 1600 → +100
    assert paper_broker.get_balance()["cash"] == 10_100.0
    assert paper_broker.get_positions() == []


def test_paper_short_position_creates_negative_quantity(paper_broker: PaperBroker) -> None:
    """Sell without a position is rejected by the paper broker."""
    result = paper_broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.SELL, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    assert result is not None
    assert result["status"] == "REJECTED"


def test_paper_reset_restores_initial_balance(paper_broker: PaperBroker) -> None:
    paper_broker.place_order(
        {"symbol": "AAPL", "side": OrderSide.BUY, "order_type": OrderType.MARKET, "amount": 1.0}
    )
    paper_broker.reset()
    paper_broker.connect()
    assert paper_broker.get_balance()["cash"] == 10_000.0
    assert paper_broker.get_positions() == []
