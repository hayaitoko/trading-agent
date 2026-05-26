"""Source adapters parse canned payloads via httpx.MockTransport — no live network.

Same offline DI pattern as tests/test_openrouter.py: an AsyncClient wired to a
MockTransport so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from trading_agent.ingest.fetchers.base import SourceError
from trading_agent.ingest.fetchers.reddit import RedditSource
from trading_agent.ingest.fetchers.rss import RssSource
from trading_agent.ingest.fetchers.stocktwits import StockTwitsSource

RSS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Market Feed</title>
  <item><title>AAPL pops</title><description>up 5%</description>
        <link>http://feed/1</link><pubDate>Wed, 15 Nov 2023 12:00:00 GMT</pubDate></item>
  <item><title>MSFT slips</title><description>down a bit</description>
        <link>http://feed/2</link></item>
</channel></rss>"""

ATOM_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry><title>NVDA news</title><summary>chips</summary>
         <link href="http://atom/1"/><updated>2023-11-15T12:00:00Z</updated></entry>
</feed>"""

REDDIT_JSON = {
    "data": {
        "children": [
            {"data": {"title": "YOLO calls", "selftext": "to the moon",
                      "permalink": "/r/wsb/c/1", "created_utc": 1700000000}},
            {"data": {"title": "", "selftext": "", "permalink": "/r/wsb/c/2",
                      "created_utc": 1700000001}},  # empty -> skipped
        ]
    }
}

STOCKTWITS_JSON = {
    "messages": [
        {"id": 11, "body": "bullish AF", "created_at": "2023-11-15T12:00:00Z",
         "symbols": [{"symbol": "aapl"}]},
        {"id": 12, "body": "", "created_at": "2023-11-15T12:01:00Z"},  # empty -> skipped
    ]
}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(coro):
    return asyncio.run(coro)


# --- RSS ---------------------------------------------------------------------


def test_rss_parses_rss20_with_ticker_stamp() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=RSS_XML)

    async def go():
        async with _client(handler) as c:
            src = RssSource("rss1", c)
            return await src.fetch({"url": "http://feed.xml", "ticker": "AAPL"})

    items = _run(go())
    assert [it.url for it in items] == ["http://feed/1", "http://feed/2"]
    assert all(it.source_id == "rss1" and it.ticker == "AAPL" for it in items)
    assert "AAPL pops" in items[0].text
    assert items[0].ts == "2023-11-15T12:00:00+00:00"  # pubDate normalized


def test_rss_parses_atom() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=ATOM_XML)

    async def go():
        async with _client(handler) as c:
            return await RssSource("a1", c).fetch({"url": "http://atom.xml"})

    items = _run(go())
    assert len(items) == 1
    assert items[0].url == "http://atom/1"
    assert "NVDA news" in items[0].text


def test_rss_limit_caps_items() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=RSS_XML)

    async def go():
        async with _client(handler) as c:
            return await RssSource("r", c).fetch({"url": "http://feed.xml", "limit": 1})

    assert len(_run(go())) == 1


def test_rss_requires_url() -> None:
    async def go():
        async with _client(lambda r: httpx.Response(200, text=RSS_XML)) as c:
            return await RssSource("r", c).fetch({})

    with pytest.raises(SourceError):
        _run(go())


def test_http_error_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    async def go():
        async with _client(handler) as c:
            return await RssSource("r", c).fetch({"url": "http://feed.xml"})

    with pytest.raises(SourceError):
        _run(go())


# --- Reddit ------------------------------------------------------------------


def test_reddit_parses_listing_and_skips_empty() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json=REDDIT_JSON)

    async def go():
        async with _client(handler) as c:
            return await RedditSource("wsb", c).fetch({"subreddit": "wallstreetbets", "listing": "new"})

    items = _run(go())
    assert len(items) == 1  # empty post dropped
    assert items[0].url == "https://www.reddit.com/r/wsb/c/1"
    assert "YOLO calls" in items[0].text
    assert "/r/wallstreetbets/new.json" in seen["url"]
    assert "trading-agent" in (seen["ua"] or "")  # descriptive UA sent (Reddit blocks default)


def test_reddit_requires_subreddit() -> None:
    async def go():
        async with _client(lambda r: httpx.Response(200, json=REDDIT_JSON)) as c:
            return await RedditSource("r", c).fetch({})

    with pytest.raises(SourceError):
        _run(go())


def test_reddit_bad_listing_falls_back_to_new() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json=REDDIT_JSON)

    async def go():
        async with _client(handler) as c:
            return await RedditSource("r", c).fetch({"subreddit": "stocks", "listing": "garbage"})

    _run(go())
    assert "/new.json" in captured["url"]


# --- StockTwits --------------------------------------------------------------


def test_stocktwits_symbol_stream_stamps_ticker() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json=STOCKTWITS_JSON)

    async def go():
        async with _client(handler) as c:
            return await StockTwitsSource("st", c).fetch({"symbol": "AAPL"})

    items = _run(go())
    assert len(items) == 1  # empty body dropped
    assert items[0].ticker == "AAPL"
    assert items[0].url == "https://stocktwits.com/message/11"
    assert "/streams/symbol/AAPL.json" in captured["url"]


def test_stocktwits_trending_uses_message_symbol() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/streams/trending.json" in str(request.url)
        return httpx.Response(200, json=STOCKTWITS_JSON)

    async def go():
        async with _client(handler) as c:
            return await StockTwitsSource("st", c).fetch({"stream": "trending"})

    items = _run(go())
    assert items[0].ticker == "AAPL"  # promoted from message.symbols, upper-cased


def test_stocktwits_non_json_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    async def go():
        async with _client(handler) as c:
            return await StockTwitsSource("st", c).fetch({"symbol": "AAPL"})

    with pytest.raises(SourceError):
        _run(go())


def test_json_payloads_are_well_formed() -> None:
    # guards the fixtures themselves (no network, just shape)
    assert json.loads(json.dumps(REDDIT_JSON))["data"]["children"]
    assert json.loads(json.dumps(STOCKTWITS_JSON))["messages"]
