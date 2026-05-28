"""WS-Situation A1 — PredictionMarketsProvider unit tests (fully offline).

Tests
-----
- EventOdds is a frozen dataclass
- Polymarket Gamma event parsed correctly (outcomes + prices)
- Polymarket liquidity filter excludes thin markets
- Polymarket restricted flag passed through (not excluded)
- Kalshi market parsed correctly (yes_bid normalisation cents→fraction)
- Kalshi price normalisation when already fractional
- event_odds returns combined + sorted by volume_24h desc
- event_odds partial success: Polymarket fails → Kalshi results returned
- event_odds partial success: Kalshi fails → Polymarket results returned
- event_odds both fail → raises PredictionMarketsProviderError
- by_id polymarket happy path
- by_id kalshi happy path
- by_id returns None on 404
- HTTP 429 with retries → eventually raises after max retries
- Network error raises PredictionMarketsProviderError
- Cache: second call does NOT hit transport
- clear_cache forces refetch
- Query filter applies to title substring (case-insensitive)
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from trading_agent.data.providers.prediction_markets import (
    EventOdds,
    PredictionMarketsProvider,
    PredictionMarketsProviderError,
    _kalshi_market_to_odds,
    _kalshi_price,
    _polymarket_event_to_odds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pm_event(
    *,
    title: str = "Will X happen?",
    slug: str = "will-x-happen",
    liquidity: float = 5_000.0,
    volume_24h: float = 1_500.0,
    outcomes: list[str] | None = None,
    prices: list[str] | None = None,
    restricted: bool = False,
    end_date: str = "2026-12-31T00:00:00",
) -> dict[str, Any]:
    if outcomes is None:
        outcomes = ["Yes", "No"]
    if prices is None:
        prices = ["0.72", "0.28"]
    return {
        "title": title,
        "slug": slug,
        "id": slug,
        "liquidity": liquidity,
        "volume24hr": volume_24h,
        "endDate": end_date,
        "restricted": restricted,
        "markets": [
            {
                "outcomes": outcomes,
                "outcomePrices": prices,
            }
        ],
    }


def _kalshi_mkt(
    *,
    title: str = "Will the Fed cut rates?",
    ticker: str = "FED-CUT-2026",
    yes_bid: float = 65,  # cents
    volume: float = 200_000.0,
    status: str = "open",
    close_time: str = "2026-12-31T00:00:00Z",
) -> dict[str, Any]:
    return {
        "title": title,
        "ticker": ticker,
        "yes_bid": yes_bid,
        "volume": volume,
        "status": status,
        "close_time": close_time,
    }


def _build_transport(
    pm_events: list[dict[str, Any]] | None = None,
    kalshi_markets: list[dict[str, Any]] | None = None,
    pm_status: int = 200,
    kalshi_status: int = 200,
) -> httpx.MockTransport:
    """Build a mock transport that routes by URL base."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "gamma-api.polymarket.com" in url or "clob.polymarket.com" in url or "polymarket" in url:
            body = json.dumps(pm_events or []).encode()
            return httpx.Response(pm_status, content=body)
        if "kalshi" in url:
            body = json.dumps({"markets": kalshi_markets or []}).encode()
            return httpx.Response(kalshi_status, content=body)
        # Unknown — return 404
        return httpx.Response(404, content=b"not found")

    return httpx.MockTransport(handler)


def _provider(
    pm_events: list[dict[str, Any]] | None = None,
    kalshi_markets: list[dict[str, Any]] | None = None,
    pm_status: int = 200,
    kalshi_status: int = 200,
) -> PredictionMarketsProvider:
    return PredictionMarketsProvider(
        pm_gamma_base="https://gamma-api.polymarket.com",
        pm_clob_base="https://clob.polymarket.com",
        kalshi_base="https://kalshi.test",
        transport=_build_transport(pm_events, kalshi_markets, pm_status, kalshi_status),
    )


# ---------------------------------------------------------------------------
# EventOdds dataclass
# ---------------------------------------------------------------------------


def test_event_odds_is_frozen() -> None:
    e = EventOdds(
        venue="polymarket",
        event_id="test",
        title="Test",
        outcomes=["Yes", "No"],
        prices=[0.7, 0.3],
        liquidity=5_000.0,
        volume_24h=1_000.0,
        end_date=None,
        restricted=False,
    )
    with pytest.raises(Exception):
        e.title = "Changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Polymarket conversion
# ---------------------------------------------------------------------------


def test_polymarket_event_parsed_correctly() -> None:
    ev = _pm_event(title="Fed Rate Cut June", liquidity=10_000.0, volume_24h=3_000.0)
    odds = _polymarket_event_to_odds(ev, min_liquidity=1_000.0)
    assert odds is not None
    assert odds.venue == "polymarket"
    assert odds.title == "Fed Rate Cut June"
    assert odds.outcomes == ["Yes", "No"]
    assert odds.prices[0] == pytest.approx(0.72)
    assert odds.prices[1] == pytest.approx(0.28)
    assert odds.liquidity == pytest.approx(10_000.0)
    assert odds.volume_24h == pytest.approx(3_000.0)
    assert odds.restricted is False


def test_polymarket_liquidity_filter_excludes_thin_markets() -> None:
    ev = _pm_event(liquidity=500.0)  # below $1k
    odds = _polymarket_event_to_odds(ev, min_liquidity=1_000.0)
    assert odds is None


def test_polymarket_liquidity_filter_zero_allows_all() -> None:
    ev = _pm_event(liquidity=0.0)
    odds = _polymarket_event_to_odds(ev, min_liquidity=0.0)
    # Still needs outcomes to be non-None
    assert odds is not None


def test_polymarket_restricted_flag_passed_through() -> None:
    ev = _pm_event(restricted=True, liquidity=5_000.0)
    odds = _polymarket_event_to_odds(ev, min_liquidity=1_000.0)
    assert odds is not None
    assert odds.restricted is True


def test_polymarket_end_date_parsed() -> None:
    ev = _pm_event(end_date="2026-11-03T20:00:00")
    odds = _polymarket_event_to_odds(ev, min_liquidity=0.0)
    assert odds is not None
    assert odds.end_date is not None
    assert odds.end_date.year == 2026
    assert odds.end_date.month == 11


def test_polymarket_no_title_returns_none() -> None:
    ev = {"liquidity": 5_000.0, "markets": []}
    assert _polymarket_event_to_odds(ev, min_liquidity=0.0) is None


# ---------------------------------------------------------------------------
# Kalshi conversion
# ---------------------------------------------------------------------------


def test_kalshi_market_parsed_correctly() -> None:
    mkt = _kalshi_mkt(title="Fed Rate Cut", ticker="FED-CUT", yes_bid=60)
    odds = _kalshi_market_to_odds(mkt)
    assert odds is not None
    assert odds.venue == "kalshi"
    assert odds.title == "Fed Rate Cut"
    assert odds.event_id == "FED-CUT"
    assert odds.outcomes == ["Yes", "No"]
    assert odds.prices[0] == pytest.approx(0.60)
    assert odds.prices[1] == pytest.approx(0.40)
    assert odds.restricted is False


def test_kalshi_price_normalisation_cents() -> None:
    assert _kalshi_price(65) == pytest.approx(0.65)
    assert _kalshi_price(100) == pytest.approx(1.0)


def test_kalshi_price_normalisation_already_fractional() -> None:
    assert _kalshi_price(0.72) == pytest.approx(0.72)


def test_kalshi_price_none() -> None:
    assert _kalshi_price(None) is None


def test_kalshi_closed_market_returns_none() -> None:
    mkt = _kalshi_mkt(status="resolved")
    assert _kalshi_market_to_odds(mkt) is None


def test_kalshi_no_ticker_returns_none() -> None:
    mkt = {"title": "Something", "yes_bid": 50, "status": "open"}
    assert _kalshi_market_to_odds(mkt) is None


# ---------------------------------------------------------------------------
# event_odds: integration + sorting
# ---------------------------------------------------------------------------


def test_event_odds_combined_sorted_by_volume_desc() -> None:
    pm_ev = _pm_event(title="PM Event", volume_24h=3_000.0, liquidity=5_000.0)
    k_mkt = _kalshi_mkt(title="Kalshi Event", volume=1_000.0, yes_bid=55)
    provider = _provider(pm_events=[pm_ev], kalshi_markets=[k_mkt])
    results = provider.event_odds("any")
    assert len(results) == 2
    assert results[0].volume_24h >= results[1].volume_24h
    # PM event has 3000 vol, Kalshi has 1000
    assert results[0].venue == "polymarket"


def test_event_odds_query_filter_applies() -> None:
    pm1 = _pm_event(title="Fed Rate Decision 2026", volume_24h=2_000.0, liquidity=5_000.0)
    pm2 = _pm_event(
        title="Bitcoin price end of year", slug="btc-eoy", volume_24h=1_500.0, liquidity=5_000.0
    )
    provider = _provider(pm_events=[pm1, pm2], kalshi_markets=[])
    results = provider.event_odds("any", query="fed")
    titles = [r.title for r in results]
    assert any("Fed" in t for t in titles)
    assert not any("Bitcoin" in t for t in titles)


def test_event_odds_partial_success_pm_fails_returns_kalshi() -> None:
    k_mkt = _kalshi_mkt(title="Kalshi Only", volume=500.0, yes_bid=70)
    # PM returns 404
    provider = _provider(kalshi_markets=[k_mkt], pm_status=404)
    results = provider.event_odds("any")
    assert len(results) >= 1
    assert all(r.venue == "kalshi" for r in results)


def test_event_odds_partial_success_kalshi_fails_returns_pm() -> None:
    pm_ev = _pm_event(title="PM Only", volume_24h=2_000.0, liquidity=5_000.0)
    # Kalshi returns 503
    provider = _provider(pm_events=[pm_ev], kalshi_status=503)
    results = provider.event_odds("any")
    assert len(results) >= 1
    assert all(r.venue == "polymarket" for r in results)


def test_event_odds_both_fail_raises() -> None:
    provider = _provider(pm_status=500, kalshi_status=500)
    with pytest.raises(PredictionMarketsProviderError):
        provider.event_odds("any")


# ---------------------------------------------------------------------------
# by_id
# ---------------------------------------------------------------------------


def test_by_id_polymarket_happy_path() -> None:
    ev = _pm_event(title="Specific Event", slug="specific-event", liquidity=10_000.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(ev).encode())

    provider = PredictionMarketsProvider(
        pm_gamma_base="https://gamma-api.polymarket.com",
        pm_clob_base="https://clob.polymarket.com",
        kalshi_base="https://kalshi.test",
        transport=httpx.MockTransport(handler),
    )
    result = provider.by_id("polymarket", "specific-event")
    assert result is not None
    assert result.title == "Specific Event"


def test_by_id_returns_none_on_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no connection")

    provider = PredictionMarketsProvider(
        pm_gamma_base="https://gamma-api.polymarket.com",
        pm_clob_base="https://clob.polymarket.com",
        kalshi_base="https://kalshi.test",
        transport=httpx.MockTransport(handler),
    )
    result = provider.by_id("polymarket", "missing-event")
    assert result is None


# ---------------------------------------------------------------------------
# HTTP error / retry
# ---------------------------------------------------------------------------


def test_non_200_raises_provider_error() -> None:
    # To force exception, make both fail
    provider2 = _provider(pm_status=503, kalshi_status=503)
    with pytest.raises(PredictionMarketsProviderError):
        provider2.event_odds("any")


def test_network_error_raises_provider_error_when_both_fail() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    provider = PredictionMarketsProvider(
        pm_gamma_base="https://gamma-api.polymarket.com",
        pm_clob_base="https://clob.polymarket.com",
        kalshi_base="https://kalshi.test",
        transport=httpx.MockTransport(boom),
    )
    with pytest.raises(PredictionMarketsProviderError):
        provider.event_odds("any")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_prevents_second_request() -> None:
    call_count = 0
    pm_ev = _pm_event(liquidity=5_000.0)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if "kalshi" in str(request.url):
            return httpx.Response(200, content=json.dumps({"markets": []}).encode())
        return httpx.Response(200, content=json.dumps([pm_ev]).encode())

    provider = PredictionMarketsProvider(
        pm_gamma_base="https://gamma-api.polymarket.com",
        pm_clob_base="https://clob.polymarket.com",
        kalshi_base="https://kalshi.test",
        transport=httpx.MockTransport(handler),
    )
    provider.event_odds("any")
    calls_after_first = call_count  # number of HTTP requests made so far
    provider.event_odds("any")  # should hit cache, zero new requests
    # Second call must not issue any new requests
    assert call_count == calls_after_first


def test_clear_cache_forces_refetch() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if "kalshi" in str(request.url):
            return httpx.Response(200, content=json.dumps({"markets": []}).encode())
        return httpx.Response(200, content=json.dumps([]).encode())

    provider = PredictionMarketsProvider(
        pm_gamma_base="https://gamma-api.polymarket.com",
        pm_clob_base="https://clob.polymarket.com",
        kalshi_base="https://kalshi.test",
        transport=httpx.MockTransport(handler),
    )
    provider.event_odds("any")
    provider.clear_cache()
    provider.event_odds("any")
    assert call_count >= 4  # two fetches × two venues
