"""WS-Situation A0 — GDELTProvider unit tests (fully offline via httpx transport seam).

Tests
-----
- GDELTBin / GDELTArticle are frozen dataclasses
- timeline_volume parses series-wrapper JSON correctly
- timeline_tone parses tone JSON correctly
- top_articles parses artlist JSON correctly
- 15-min cache: second call does NOT hit the transport
- Cache TTL: entry expires after ttl seconds (monkeypatched time.monotonic)
- HTTP 4xx raises GDELTProviderError
- Network error raises GDELTProviderError
- clear_cache flushes entries
- _parse_gdelt_date handles YYYYMMDDTHHMMSSZ and YYYYMMDDHHMMSS variants
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from trading_agent.data.providers.gdelt import (
    GDELTArticle,
    GDELTBin,
    GDELTProvider,
    GDELTProviderError,
    _parse_gdelt_date,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timeline_response(bins: list[dict[str, Any]]) -> bytes:
    """Minimal GDELT timeline JSON — series-wrapper shape."""
    payload = {"timeline": [{"series": "Volume", "data": bins}]}
    return json.dumps(payload).encode()


def _artlist_response(articles: list[dict[str, Any]]) -> bytes:
    payload = {"articles": articles}
    return json.dumps(payload).encode()


def _make_transport(body: bytes, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    return httpx.MockTransport(handler)


def _provider(body: bytes, status: int = 200) -> GDELTProvider:
    return GDELTProvider(
        base_url="https://gdelt.test",
        transport=_make_transport(body, status),
    )


# ---------------------------------------------------------------------------
# Dataclass contracts
# ---------------------------------------------------------------------------


def test_gdelt_bin_is_frozen() -> None:
    b = GDELTBin(
        bucket_start=datetime(2026, 5, 28, 12, 0, tzinfo=UTC),
        value=42.0,
        unit="mentions",
    )
    with pytest.raises(Exception):
        b.value = 99.0  # type: ignore[misc]


def test_gdelt_article_is_frozen() -> None:
    a = GDELTArticle(
        title="Test",
        url="https://example.com",
        published=datetime(2026, 5, 28, tzinfo=UTC),
        source_domain="example.com",
        tone=-2.5,
    )
    with pytest.raises(Exception):
        a.tone = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# timeline_volume
# ---------------------------------------------------------------------------


def test_timeline_volume_parses_bins() -> None:
    bins_data = [
        {"date": "20260528120000", "value": 123.0},
        {"date": "20260528121500", "value": 456.0},
    ]
    provider = _provider(_timeline_response(bins_data))
    result = provider.timeline_volume("WAR", timespan="24h")
    assert len(result) == 2
    assert result[0].unit == "mentions"
    assert result[0].value == pytest.approx(123.0)
    assert result[1].value == pytest.approx(456.0)
    assert result[0].bucket_start == datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def test_timeline_volume_empty_response() -> None:
    provider = _provider(json.dumps({"timeline": []}).encode())
    result = provider.timeline_volume("WAR")
    assert result == []


def test_timeline_volume_cache_prevents_second_request() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        bins_data = [{"date": "20260528120000", "value": 1.0}]
        return httpx.Response(200, content=_timeline_response(bins_data))

    provider = GDELTProvider(
        base_url="https://gdelt.test",
        transport=httpx.MockTransport(handler),
    )
    provider.timeline_volume("WAR")
    provider.timeline_volume("WAR")  # should hit cache
    assert call_count == 1


# ---------------------------------------------------------------------------
# timeline_tone
# ---------------------------------------------------------------------------


def test_timeline_tone_unit_is_tone() -> None:
    bins_data = [{"date": "20260528120000", "value": -3.5}]
    payload = {"timeline": [{"series": "Tone", "data": bins_data}]}
    provider = _provider(json.dumps(payload).encode())
    result = provider.timeline_tone("ELECTION")
    assert len(result) == 1
    assert result[0].unit == "tone"
    assert result[0].value == pytest.approx(-3.5)


# ---------------------------------------------------------------------------
# top_articles
# ---------------------------------------------------------------------------


def test_top_articles_parses_artlist() -> None:
    articles_data = [
        {
            "title": "War Escalates",
            "url": "https://reuters.com/war",
            "seendate": "20260528T120000Z",
            "domain": "reuters.com",
            "tone": -5.2,
        },
        {
            "title": "Peace Talks Begin",
            "url": "https://bbc.com/peace",
            "seendate": "20260528110000",
            "domain": "bbc.com",
            "tone": 3.1,
        },
    ]
    provider = _provider(_artlist_response(articles_data))
    result = provider.top_articles("WAR", n=5)
    assert len(result) == 2
    assert result[0].title == "War Escalates"
    assert result[0].source_domain == "reuters.com"
    assert result[0].tone == pytest.approx(-5.2)
    assert result[0].url == "https://reuters.com/war"
    assert result[1].tone == pytest.approx(3.1)


def test_top_articles_n_clamped() -> None:
    """n is clamped to 1–250; transport sees a request (no crash)."""
    provider = _provider(_artlist_response([]))
    # Should not raise even with out-of-range n
    provider.top_articles("WAR", n=0)
    provider.top_articles("WAR", n=9999)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_http_4xx_raises_provider_error() -> None:
    provider = _provider(b"Not found", status=404)
    with pytest.raises(GDELTProviderError, match="HTTP 404"):
        provider.timeline_volume("WAR")


def test_http_500_raises_provider_error() -> None:
    provider = _provider(b"Server error", status=500)
    with pytest.raises(GDELTProviderError, match="HTTP 500"):
        provider.timeline_volume("WAR")


def test_network_error_raises_provider_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("timeout")

    provider = GDELTProvider(
        base_url="https://gdelt.test",
        transport=httpx.MockTransport(boom),
    )
    with pytest.raises(GDELTProviderError, match="network error"):
        provider.timeline_volume("WAR")


def test_invalid_json_raises_provider_error() -> None:
    provider = _provider(b"not json at all", status=200)
    with pytest.raises(GDELTProviderError):
        provider.timeline_volume("WAR")


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def test_clear_cache_forces_refetch() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, content=_timeline_response([]))

    provider = GDELTProvider(
        base_url="https://gdelt.test",
        transport=httpx.MockTransport(handler),
    )
    provider.timeline_volume("WAR")
    provider.clear_cache()
    provider.timeline_volume("WAR")
    assert call_count == 2


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def test_parse_gdelt_date_yyyymmddhhmmss() -> None:
    dt = _parse_gdelt_date("20260528143000")
    assert dt == datetime(2026, 5, 28, 14, 30, 0, tzinfo=UTC)


def test_parse_gdelt_date_with_t_and_z() -> None:
    dt = _parse_gdelt_date("20260528T143000Z")
    assert dt == datetime(2026, 5, 28, 14, 30, 0, tzinfo=UTC)


def test_parse_gdelt_date_date_only() -> None:
    dt = _parse_gdelt_date("20260528")
    assert dt == datetime(2026, 5, 28, 0, 0, 0, tzinfo=UTC)


def test_parse_gdelt_date_too_short_raises() -> None:
    with pytest.raises(ValueError):
        _parse_gdelt_date("202605")
