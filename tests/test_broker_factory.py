"""Tests for broker_factory: env/config -> real broker adapters (mocked SDKs)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trading_agent.broker_factory import build_alpaca_broker, build_ccxt_broker
from trading_agent.config import ConfigError

# --- Alpaca ------------------------------------------------------------------


def test_build_alpaca_defaults_to_paper(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret123")
    monkeypatch.delenv("ALPACA_PAPER", raising=False)

    with patch("trading_agent.alpaca_broker.TradingClient") as MockTC, patch(
        "trading_agent.alpaca_broker.StockHistoricalDataClient"
    ):
        broker = build_alpaca_broker()
        assert broker.paper is True
        # paper flag propagated to the SDK client
        _, kwargs = MockTC.call_args
        assert kwargs.get("paper") is True


def test_build_alpaca_explicit_live_overrides_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_PAPER", "true")

    with patch("trading_agent.alpaca_broker.TradingClient"), patch(
        "trading_agent.alpaca_broker.StockHistoricalDataClient"
    ):
        broker = build_alpaca_broker(paper=False)
        assert broker.paper is False


def test_build_alpaca_paper_env_false(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_PAPER", "false")

    with patch("trading_agent.alpaca_broker.TradingClient"), patch(
        "trading_agent.alpaca_broker.StockHistoricalDataClient"
    ):
        broker = build_alpaca_broker()
        assert broker.paper is False


def test_build_alpaca_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ConfigError, match="Alpaca credentials missing"):
        build_alpaca_broker()


def test_build_alpaca_explicit_args_bypass_env(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with patch("trading_agent.alpaca_broker.TradingClient"), patch(
        "trading_agent.alpaca_broker.StockHistoricalDataClient"
    ):
        broker = build_alpaca_broker(api_key="a", secret_key="b")
        assert broker.api_key == "a"


# --- CCXT --------------------------------------------------------------------


def test_build_ccxt_public_no_keys_needed(monkeypatch):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_SECRET_KEY", raising=False)
    with patch("trading_agent.ccxt_broker.ccxt") as MockCcxt:
        build_ccxt_broker("binance")
        # constructed binance with empty creds (public data is fine)
        MockCcxt.binance.assert_called_once()
        cfg = MockCcxt.binance.call_args[0][0]
        assert cfg["apiKey"] == ""


def test_build_ccxt_reads_env_keys(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "envkey")
    monkeypatch.setenv("BINANCE_SECRET_KEY", "envsecret")
    with patch("trading_agent.ccxt_broker.ccxt") as MockCcxt:
        build_ccxt_broker("binance")
        cfg = MockCcxt.binance.call_args[0][0]
        assert cfg["apiKey"] == "envkey"
        assert cfg["secret"] == "envsecret"
