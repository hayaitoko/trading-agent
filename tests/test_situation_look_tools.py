"""WS-Situation A2+wiring — LOOK tool wiring tests for world_events,
prediction_market_odds, and options_iv.

Tests
-----
WorldEventsTool
  - flag off (settings_store=None)           → disabled error
  - flag off (settings_store returns False)  → disabled error
  - flag on, provider=None                   → disabled error
  - flag on, provider raises                 → network_error
  - flag on, provider returns bins+articles  → ok=True, correct shape
  - flag on, explicit theme passed           → ok=True with that theme in data

PredictionMarketOddsTool
  - flag off → disabled error
  - flag on, provider=None → disabled error
  - flag on, provider raises → network_error
  - flag on, provider returns events → ok=True, correct shape

OptionsIVTool
  - flag off → disabled error
  - flag on, provider=None → disabled error
  - flag on, provider raises → network_error
  - flag on, provider returns quotes with IV → ok=True, iv in result
  - flag on, NTM filter applied (spot prices injected)
  - flag on, no-IV quotes fallback included

settings_store DEFAULTS include the three new flags (all False)
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_agent.config.settings_store import DEFAULTS
from trading_agent.instruments.options import OptionContract, OptionQuote, OptionRight
from trading_agent.intel.tools.look.options_iv import OptionsIVTool
from trading_agent.intel.tools.look.prediction_market_odds import PredictionMarketOddsTool
from trading_agent.intel.tools.look.world_events import WorldEventsTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(flag: str, value: bool) -> Any:
    """Minimal duck-typed settings_store."""
    store = MagicMock()
    store.get = lambda user_id, key, default=None: (
        value if key == flag else DEFAULTS.get(key, default)
    )
    return store


def _gdelt_provider(
    *,
    bins=None,
    articles=None,
    raise_on_call: Exception | None = None,
) -> Any:
    """Fake GDELTProvider."""
    if bins is None:
        bins = [
            SimpleNamespace(
                bucket_start=datetime(2026, 5, 28, 12, 0, tzinfo=UTC),
                value=50.0,
                unit="mentions",
            )
        ]
    if articles is None:
        articles = [
            SimpleNamespace(
                title="Conflict Spreads",
                url="https://reuters.com/conflict",
                published=datetime(2026, 5, 28, 10, 0, tzinfo=UTC),
                source_domain="reuters.com",
                tone=-4.5,
            )
        ]

    provider = MagicMock()
    if raise_on_call:
        provider.timeline_volume.side_effect = raise_on_call
        provider.top_articles.side_effect = raise_on_call
    else:
        provider.timeline_volume.return_value = bins
        provider.top_articles.return_value = articles
    return provider


def _pm_provider(
    *,
    events=None,
    raise_on_call: Exception | None = None,
) -> Any:
    """Fake PredictionMarketsProvider."""
    if events is None:
        events = [
            SimpleNamespace(
                venue="polymarket",
                event_id="fed-cut",
                title="Fed Rate Cut June",
                outcomes=["Yes", "No"],
                prices=[0.72, 0.28],
                liquidity=10_000.0,
                volume_24h=3_000.0,
                end_date=datetime(2026, 6, 30, tzinfo=UTC),
                restricted=False,
            )
        ]
    provider = MagicMock()
    if raise_on_call:
        provider.event_odds.side_effect = raise_on_call
    else:
        provider.event_odds.return_value = events
    return provider


def _chain_provider(
    *,
    quotes=None,
    raise_on_call: Exception | None = None,
) -> Any:
    """Fake AlpacaOptionChainProvider."""
    if quotes is None:
        c = OptionContract("AAPL", date(2026, 6, 19), 150.0, OptionRight.CALL)
        quotes = [
            OptionQuote(
                contract=c,
                bid=2.0,
                ask=2.4,
                last=2.2,
                implied_vol=0.28,
                greeks={"delta": 0.5, "gamma": 0.02, "theta": -0.05, "vega": 0.12, "rho": 0.01},
            )
        ]
    provider = MagicMock()
    if raise_on_call:
        provider.get_chain.side_effect = raise_on_call
    else:
        provider.get_chain.return_value = quotes
    return provider


# ---------------------------------------------------------------------------
# DEFAULTS include new flags
# ---------------------------------------------------------------------------


def test_defaults_include_situation_gdelt_false() -> None:
    assert DEFAULTS.get("SITUATION_GDELT") is False


def test_defaults_include_situation_prediction_markets_false() -> None:
    assert DEFAULTS.get("SITUATION_PREDICTION_MARKETS") is False


def test_defaults_include_situation_options_iv_false() -> None:
    assert DEFAULTS.get("SITUATION_OPTIONS_IV") is False


# ---------------------------------------------------------------------------
# WorldEventsTool
# ---------------------------------------------------------------------------


def test_world_events_flag_off_settings_none() -> None:
    tool = WorldEventsTool(trader_id="Alpha")
    result = tool()
    assert not result.ok
    assert result.error.kind == "disabled"


def test_world_events_flag_off_returns_disabled() -> None:
    settings = _settings("SITUATION_GDELT", False)
    tool = WorldEventsTool(trader_id="Alpha", settings_store=settings)
    result = tool()
    assert not result.ok
    assert result.error.kind == "disabled"


def test_world_events_flag_on_provider_none() -> None:
    settings = _settings("SITUATION_GDELT", True)
    tool = WorldEventsTool(trader_id="Alpha", settings_store=settings, gdelt_provider=None)
    result = tool()
    assert not result.ok
    assert result.error.kind == "disabled"


def test_world_events_flag_on_provider_raises_network_error() -> None:
    settings = _settings("SITUATION_GDELT", True)
    provider = _gdelt_provider(raise_on_call=RuntimeError("connection refused"))
    tool = WorldEventsTool(trader_id="Alpha", settings_store=settings, gdelt_provider=provider)
    result = tool()
    assert not result.ok
    assert result.error.kind == "network_error"


def test_world_events_flag_on_returns_bins_and_articles() -> None:
    settings = _settings("SITUATION_GDELT", True)
    provider = _gdelt_provider()
    tool = WorldEventsTool(trader_id="Alpha", settings_store=settings, gdelt_provider=provider)
    result = tool(theme="WAR", timespan="24h")
    assert result.ok
    assert result.data["theme"] == "WAR"
    assert result.data["timespan"] == "24h"
    assert isinstance(result.data["bins"], list)
    assert isinstance(result.data["articles"], list)
    assert result.data["bins"][0]["unit"] == "mentions"


def test_world_events_default_theme_queries_multiple_themes() -> None:
    """Calling with theme=None queries the default theme list."""
    settings = _settings("SITUATION_GDELT", True)
    provider = _gdelt_provider()
    tool = WorldEventsTool(trader_id="Alpha", settings_store=settings, gdelt_provider=provider)
    result = tool()  # no theme
    assert result.ok
    # Default theme string is comma-joined default list
    assert "WAR" in result.data["theme"]


def test_world_events_article_dict_shape() -> None:
    settings = _settings("SITUATION_GDELT", True)
    provider = _gdelt_provider()
    tool = WorldEventsTool(trader_id="Alpha", settings_store=settings, gdelt_provider=provider)
    result = tool(theme="WAR")
    assert result.ok
    articles = result.data["articles"]
    assert len(articles) >= 1
    a = articles[0]
    assert "title" in a
    assert "url" in a
    assert "published" in a
    assert "source_domain" in a
    assert "tone" in a


# ---------------------------------------------------------------------------
# PredictionMarketOddsTool
# ---------------------------------------------------------------------------


def test_prediction_market_odds_flag_off_returns_disabled() -> None:
    tool = PredictionMarketOddsTool(trader_id="Alpha")
    result = tool("economics")
    assert not result.ok
    assert result.error.kind == "disabled"


def test_prediction_market_odds_flag_on_provider_none() -> None:
    settings = _settings("SITUATION_PREDICTION_MARKETS", True)
    tool = PredictionMarketOddsTool(trader_id="Alpha", settings_store=settings, pm_provider=None)
    result = tool("economics")
    assert not result.ok
    assert result.error.kind == "disabled"


def test_prediction_market_odds_flag_on_provider_raises() -> None:
    settings = _settings("SITUATION_PREDICTION_MARKETS", True)
    provider = _pm_provider(raise_on_call=RuntimeError("timeout"))
    tool = PredictionMarketOddsTool(
        trader_id="Alpha", settings_store=settings, pm_provider=provider
    )
    result = tool("economics")
    assert not result.ok
    assert result.error.kind == "network_error"


def test_prediction_market_odds_flag_on_returns_events() -> None:
    settings = _settings("SITUATION_PREDICTION_MARKETS", True)
    provider = _pm_provider()
    tool = PredictionMarketOddsTool(
        trader_id="Alpha", settings_store=settings, pm_provider=provider
    )
    result = tool("economics")
    assert result.ok
    assert result.data["category"] == "economics"
    events = result.data["events"]
    assert len(events) == 1
    e = events[0]
    assert e["venue"] == "polymarket"
    assert e["title"] == "Fed Rate Cut June"
    assert e["outcomes"] == ["Yes", "No"]
    assert e["prices"] == pytest.approx([0.72, 0.28])
    assert e["end_date"] is not None


def test_prediction_market_odds_event_dict_shape() -> None:
    settings = _settings("SITUATION_PREDICTION_MARKETS", True)
    provider = _pm_provider()
    tool = PredictionMarketOddsTool(
        trader_id="Alpha", settings_store=settings, pm_provider=provider
    )
    result = tool("any")
    assert result.ok
    e = result.data["events"][0]
    for key in ("venue", "event_id", "title", "outcomes", "prices", "liquidity",
                "volume_24h", "end_date", "restricted"):
        assert key in e, f"missing key: {key}"


# ---------------------------------------------------------------------------
# OptionsIVTool
# ---------------------------------------------------------------------------


def test_options_iv_flag_off_returns_disabled() -> None:
    tool = OptionsIVTool(trader_id="Alpha")
    result = tool("AAPL")
    assert not result.ok
    assert result.error.kind == "disabled"


def test_options_iv_flag_on_provider_none() -> None:
    settings = _settings("SITUATION_OPTIONS_IV", True)
    tool = OptionsIVTool(trader_id="Alpha", settings_store=settings, chain_provider=None)
    result = tool("AAPL")
    assert not result.ok
    assert result.error.kind == "disabled"


def test_options_iv_flag_on_provider_raises() -> None:
    settings = _settings("SITUATION_OPTIONS_IV", True)
    provider = _chain_provider(raise_on_call=RuntimeError("no connection"))
    tool = OptionsIVTool(trader_id="Alpha", settings_store=settings, chain_provider=provider)
    result = tool("AAPL")
    assert not result.ok
    assert result.error.kind == "network_error"


def test_options_iv_flag_on_returns_contracts_with_iv() -> None:
    settings = _settings("SITUATION_OPTIONS_IV", True)
    provider = _chain_provider()
    tool = OptionsIVTool(trader_id="Alpha", settings_store=settings, chain_provider=provider)
    result = tool("AAPL")
    assert result.ok
    assert result.data["symbol"] == "AAPL"
    contracts = result.data["contracts"]
    assert len(contracts) == 1
    c = contracts[0]
    assert c["implied_vol"] == pytest.approx(0.28)
    assert c["greeks"] is not None
    assert c["greeks"]["delta"] == pytest.approx(0.5)


def test_options_iv_contract_dict_shape() -> None:
    settings = _settings("SITUATION_OPTIONS_IV", True)
    provider = _chain_provider()
    tool = OptionsIVTool(trader_id="Alpha", settings_store=settings, chain_provider=provider)
    result = tool("AAPL")
    assert result.ok
    c = result.data["contracts"][0]
    for key in ("occ", "underlying", "strike", "right", "expiry",
                "bid", "ask", "mark", "implied_vol", "greeks"):
        assert key in c, f"missing key: {key}"


def test_options_iv_ntm_filter_with_spot() -> None:
    """With spot injected, contracts outside ±20% band are filtered."""
    settings = _settings("SITUATION_OPTIONS_IV", True)
    # Spot = 150; band = 120–180.  One contract at 200 (out of band).
    c_in = OptionContract("AAPL", date(2026, 6, 19), 150.0, OptionRight.CALL)
    c_out = OptionContract("AAPL", date(2026, 6, 19), 200.0, OptionRight.CALL)
    quotes = [
        OptionQuote(contract=c_in, bid=2.0, ask=2.4, implied_vol=0.28),
        OptionQuote(contract=c_out, bid=0.1, ask=0.2, implied_vol=0.45),
    ]
    provider = _chain_provider(quotes=quotes)
    tool = OptionsIVTool(
        trader_id="Alpha",
        settings_store=settings,
        chain_provider=provider,
        spot_prices={"AAPL": 150.0},
    )
    result = tool("AAPL")
    assert result.ok
    strikes = [c["strike"] for c in result.data["contracts"]]
    assert 150.0 in strikes
    assert 200.0 not in strikes


def test_options_iv_fallback_to_all_when_no_iv() -> None:
    """When no contracts have IV, all contracts are included (no silent drop)."""
    settings = _settings("SITUATION_OPTIONS_IV", True)
    c = OptionContract("AAPL", date(2026, 6, 19), 150.0, OptionRight.CALL)
    quotes = [OptionQuote(contract=c, bid=2.0, ask=2.4)]  # no IV
    provider = _chain_provider(quotes=quotes)
    tool = OptionsIVTool(trader_id="Alpha", settings_store=settings, chain_provider=provider)
    result = tool("AAPL")
    assert result.ok
    assert len(result.data["contracts"]) == 1
    assert result.data["contracts"][0]["implied_vol"] is None
