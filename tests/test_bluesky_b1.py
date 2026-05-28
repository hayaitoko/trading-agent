"""Tests for bluesky.py B1 additions: BlueskyListSource, BlueskyAuthorSource,
and the resolve_starter_pack helper.

All tests use httpx.MockTransport — no live network calls.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from trading_agent.ingest.fetchers.base import SourceError
from trading_agent.ingest.fetchers.bluesky import (
    BlueskyAuthorSource,
    BlueskyListSource,
    resolve_starter_pack,
)

# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------

_FEED_PAYLOAD = {
    "feed": [
        {
            "post": {
                "uri": "at://did:plc:abc/app.bsky.feed.post/rkey1",
                "indexedAt": "2026-05-28T12:00:00Z",
                "author": {"handle": "user1.bsky.social"},
                "record": {
                    "text": "$SPY looks bullish today, big vol move incoming",
                    "createdAt": "2026-05-28T11:59:00Z",
                },
            }
        },
        {
            "post": {
                "uri": "at://did:plc:abc/app.bsky.feed.post/rkey2",
                "indexedAt": "2026-05-28T12:01:00Z",
                "author": {"handle": "user2.bsky.social"},
                "record": {
                    "text": "Rates are going to drive everything this week",
                    "createdAt": "2026-05-28T12:01:00Z",
                },
            }
        },
        {
            # Empty text — should be skipped
            "post": {
                "uri": "at://did:plc:abc/app.bsky.feed.post/rkey3",
                "indexedAt": "2026-05-28T12:02:00Z",
                "author": {"handle": "user3.bsky.social"},
                "record": {"text": ""},
            }
        },
    ]
}

_STARTER_PACK_PAYLOAD = {
    "starterPack": {
        "uri": "at://did:plc:owner/app.bsky.graph.starterpack/rkey",
        "cid": "bafy...",
        "record": {
            "$type": "app.bsky.graph.starterpack",
            "name": "Test Finance Pack",
            "list": "at://did:plc:owner/app.bsky.graph.list/listkey",
        },
    }
}

_RESOLVE_HANDLE_PAYLOAD = {"did": "did:plc:owner"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# BlueskyListSource
# ---------------------------------------------------------------------------


class TestBlueskyListSource:
    def test_fetches_list_feed_and_parses_posts(self) -> None:
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            captured["list"] = req.url.params.get("list")
            return httpx.Response(200, json=_FEED_PAYLOAD)

        async def go():
            async with _client(handler) as c:
                src = BlueskyListSource("list1", c)
                return await src.fetch({
                    "list_uri": "at://did:plc:owner/app.bsky.graph.list/listkey"
                })

        items = _run(go())
        assert "getListFeed" in captured["url"]
        assert captured["list"] == "at://did:plc:owner/app.bsky.graph.list/listkey"
        # 3 feed items, 1 has empty text → 2 items returned
        assert len(items) == 2
        assert items[0].source_id == "list1"
        assert "$SPY" in items[0].text

    def test_extracts_cashtag_as_ticker(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_FEED_PAYLOAD)

        async def go():
            async with _client(handler) as c:
                src = BlueskyListSource("list1", c)
                return await src.fetch({
                    "list_uri": "at://did:plc:owner/app.bsky.graph.list/listkey"
                })

        items = _run(go())
        # First post has $SPY → ticker extracted
        assert items[0].ticker == "SPY"
        # Second post has no cashtag → ticker is None
        assert items[1].ticker is None

    def test_respects_explicit_ticker_override(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_FEED_PAYLOAD)

        async def go():
            async with _client(handler) as c:
                src = BlueskyListSource("list1", c)
                return await src.fetch({
                    "list_uri": "at://did:plc:owner/app.bsky.graph.list/listkey",
                    "ticker": "QQQ",
                })

        items = _run(go())
        # Explicit ticker overrides cashtag extraction
        assert all(it.ticker == "QQQ" for it in items)

    def test_requires_list_uri(self) -> None:
        async def go():
            async with _client(lambda r: httpx.Response(200, json={})) as c:
                return await BlueskyListSource("l", c).fetch({})

        with pytest.raises(SourceError, match="list_uri"):
            _run(go())

    def test_rejects_non_at_uri(self) -> None:
        async def go():
            async with _client(lambda r: httpx.Response(200, json={})) as c:
                return await BlueskyListSource("l", c).fetch({
                    "list_uri": "https://bsky.app/profile/foo/lists/bar"
                })

        with pytest.raises(SourceError, match="AT-URI"):
            _run(go())

    def test_http_error_raises_source_error(self) -> None:
        async def go():
            async with _client(lambda r: httpx.Response(503, text="down")) as c:
                return await BlueskyListSource("l", c).fetch({
                    "list_uri": "at://did:plc:x/app.bsky.graph.list/y"
                })

        with pytest.raises(SourceError):
            _run(go())

    def test_limit_caps_items(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_FEED_PAYLOAD)

        async def go():
            async with _client(handler) as c:
                src = BlueskyListSource("l", c)
                return await src.fetch({
                    "list_uri": "at://did:plc:x/app.bsky.graph.list/y",
                    "limit": 1,
                })

        items = _run(go())
        assert len(items) == 1

    def test_link_constructed_from_at_uri(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_FEED_PAYLOAD)

        async def go():
            async with _client(handler) as c:
                src = BlueskyListSource("l", c)
                return await src.fetch({
                    "list_uri": "at://did:plc:owner/app.bsky.graph.list/listkey"
                })

        items = _run(go())
        assert items[0].url.startswith("https://bsky.app/profile/")
        assert "rkey1" in items[0].url


# ---------------------------------------------------------------------------
# BlueskyAuthorSource
# ---------------------------------------------------------------------------


class TestBlueskyAuthorSource:
    def test_fetches_author_feed(self) -> None:
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            captured["actor"] = req.url.params.get("actor")
            return httpx.Response(200, json=_FEED_PAYLOAD)

        async def go():
            async with _client(handler) as c:
                src = BlueskyAuthorSource("auth1", c)
                return await src.fetch({"handle": "joeweisenthal.bsky.social"})

        items = _run(go())
        assert "getAuthorFeed" in captured["url"]
        assert captured["actor"] == "joeweisenthal.bsky.social"
        assert len(items) == 2

    def test_requires_handle(self) -> None:
        async def go():
            async with _client(lambda r: httpx.Response(200, json={})) as c:
                return await BlueskyAuthorSource("a", c).fetch({})

        with pytest.raises(SourceError, match="handle"):
            _run(go())

    def test_non_json_raises(self) -> None:
        async def go():
            async with _client(lambda r: httpx.Response(200, text="<html>")) as c:
                return await BlueskyAuthorSource("a", c).fetch(
                    {"handle": "user.bsky.social"}
                )

        with pytest.raises(SourceError):
            _run(go())

    def test_empty_feed_returns_empty_list(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"feed": []})

        async def go():
            async with _client(handler) as c:
                src = BlueskyAuthorSource("a", c)
                return await src.fetch({"handle": "user.bsky.social"})

        items = _run(go())
        assert items == []

    def test_author_cashtag_extraction(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_FEED_PAYLOAD)

        async def go():
            async with _client(handler) as c:
                src = BlueskyAuthorSource("a", c)
                return await src.fetch({"handle": "user.bsky.social"})

        items = _run(go())
        assert items[0].ticker == "SPY"  # extracted from $SPY in text


# ---------------------------------------------------------------------------
# resolve_starter_pack
# ---------------------------------------------------------------------------


class TestResolveStarterPack:
    def test_resolves_at_uri_directly(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert "getStarterPack" in str(req.url)
            return httpx.Response(200, json=_STARTER_PACK_PAYLOAD)

        async def go():
            async with _client(handler) as c:
                dummy = BlueskyListSource("dummy", c)
                return await resolve_starter_pack(
                    dummy,
                    "at://did:plc:owner/app.bsky.graph.starterpack/rkey",
                )

        result = _run(go())
        assert result == "at://did:plc:owner/app.bsky.graph.list/listkey"

    def test_resolves_bsky_app_url(self) -> None:
        calls: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            url = str(req.url)
            calls.append(url)
            if "resolveHandle" in url:
                return httpx.Response(200, json=_RESOLVE_HANDLE_PAYLOAD)
            if "getStarterPack" in url:
                return httpx.Response(200, json=_STARTER_PACK_PAYLOAD)
            return httpx.Response(404)

        async def go():
            async with _client(handler) as c:
                dummy = BlueskyListSource("dummy", c)
                return await resolve_starter_pack(
                    dummy,
                    "https://bsky.app/starter-pack/someuser.bsky.social/3lbgeejdteh2u",
                )

        result = _run(go())
        # resolveHandle was called first, then getStarterPack
        assert any("resolveHandle" in u for u in calls)
        assert any("getStarterPack" in u for u in calls)
        assert result == "at://did:plc:owner/app.bsky.graph.list/listkey"

    def test_raises_on_non_at_non_bskyapp_url(self) -> None:
        async def go():
            async with _client(lambda r: httpx.Response(200, json={})) as c:
                dummy = BlueskyListSource("dummy", c)
                return await resolve_starter_pack(dummy, "https://example.com/pack")

        with pytest.raises(SourceError, match="AT-URI or bsky.app URL"):
            _run(go())

    def test_raises_on_bad_bskyapp_url_format(self) -> None:
        async def go():
            async with _client(lambda r: httpx.Response(200, json={})) as c:
                dummy = BlueskyListSource("dummy", c)
                # URL path too short — no rkey
                return await resolve_starter_pack(
                    dummy, "https://bsky.app/starter-pack/onlyone"
                )

        with pytest.raises(SourceError, match="cannot parse"):
            _run(go())

    def test_raises_on_missing_list_field(self) -> None:
        bad_payload = {
            "starterPack": {
                "uri": "at://...",
                "record": {"name": "No list field"},
            }
        }

        async def go():
            async with _client(lambda r: httpx.Response(200, json=bad_payload)) as c:
                dummy = BlueskyListSource("dummy", c)
                return await resolve_starter_pack(
                    dummy, "at://did:plc:x/app.bsky.graph.starterpack/y"
                )

        with pytest.raises(SourceError):
            _run(go())

    def test_raises_on_http_error(self) -> None:
        async def go():
            async with _client(lambda r: httpx.Response(404, text="not found")) as c:
                dummy = BlueskyListSource("dummy", c)
                return await resolve_starter_pack(
                    dummy, "at://did:plc:x/app.bsky.graph.starterpack/y"
                )

        with pytest.raises(SourceError):
            _run(go())


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


def test_adapters_registry_includes_b1_kinds() -> None:
    """BlueskyListSource and BlueskyAuthorSource are registered in ADAPTERS."""
    from trading_agent.ingest.fetchers import ADAPTERS
    assert "bluesky_list" in ADAPTERS
    assert "bluesky_author" in ADAPTERS
    assert "bluesky" in ADAPTERS  # cashtag kind still present


def test_b1_kinds_config_schema_documented() -> None:
    """Both B1 adapters expose a CONFIG_SCHEMA."""
    assert "list_uri" in BlueskyListSource.CONFIG_SCHEMA
    assert "handle" in BlueskyAuthorSource.CONFIG_SCHEMA


# ---------------------------------------------------------------------------
# Seed sources — Bluesky seeds
# ---------------------------------------------------------------------------


def test_seed_includes_bluesky_list_and_author_rows(tmp_path) -> None:
    """seed_finance_sources inserts bluesky_list and bluesky_author rows."""
    from trading_agent.config.db import Database
    from trading_agent.ingest.seed_sources import (
        _BSKY_AUTHOR_SEEDS,
        _BSKY_LIST_SEEDS,
        seed_finance_sources,
    )

    db = Database(tmp_path / "config.db")
    seed_finance_sources(db, "u1")

    list_rows = db.query(
        "SELECT * FROM sources WHERE user_id = ? AND kind = 'bluesky_list'", ("u1",)
    )
    author_rows = db.query(
        "SELECT * FROM sources WHERE user_id = ? AND kind = 'bluesky_author'", ("u1",)
    )

    assert len(list_rows) == len(_BSKY_LIST_SEEDS)
    assert len(author_rows) == len(_BSKY_AUTHOR_SEEDS)


def test_bluesky_list_seeds_have_at_uris(tmp_path) -> None:
    """All bluesky_list seeds carry an AT-URI list_uri."""
    from trading_agent.config.db import Database
    from trading_agent.ingest.seed_sources import seed_finance_sources

    db = Database(tmp_path / "config.db")
    seed_finance_sources(db, "u1")
    rows = db.query(
        "SELECT config_json FROM sources WHERE user_id = ? AND kind = 'bluesky_list'",
        ("u1",),
    )
    for row in rows:
        cfg = json.loads(row["config_json"])
        assert cfg["list_uri"].startswith("at://"), f"expected AT-URI, got {cfg['list_uri']!r}"


def test_bluesky_author_seeds_have_handles(tmp_path) -> None:
    """All bluesky_author seeds carry a non-empty handle."""
    from trading_agent.config.db import Database
    from trading_agent.ingest.seed_sources import seed_finance_sources

    db = Database(tmp_path / "config.db")
    seed_finance_sources(db, "u1")
    rows = db.query(
        "SELECT config_json FROM sources WHERE user_id = ? AND kind = 'bluesky_author'",
        ("u1",),
    )
    for row in rows:
        cfg = json.loads(row["config_json"])
        assert cfg["handle"], "handle must be non-empty"
        assert "bsky.social" in cfg["handle"], f"expected bsky.social handle, got {cfg['handle']!r}"
