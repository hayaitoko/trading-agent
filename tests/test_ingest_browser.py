"""Browser adapter is isolated behind the Source interface.

Playwright is optional and not installed in CI — the adapter must degrade to a
clean :class:`BrowserUnavailable` (a SourceError) that the worker catches, never
breaking the other sources. With an injected fake manager it produces RawItems
like any other source.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from trading_agent.config.db import Database
from trading_agent.ingest.fetchers.base import SourceError
from trading_agent.ingest.fetchers.browser import (
    BrowserManager,
    BrowserSource,
    BrowserUnavailable,
)
from trading_agent.ingest.registry import BoundSource
from trading_agent.ingest.store import IngestStore
from trading_agent.ingest.worker import IngestWorker


class FakeManager:
    async def fetch_text(self, url, *, selector=None, wait_until="domcontentloaded", timeout_ms=15000):
        return f"rendered {url} [{selector}]", url + "#final"


def _no_network_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: (_ for _ in ()).throw(AssertionError("no network expected"))
    ))


def test_browser_unavailable_when_playwright_missing() -> None:
    # playwright is not installed in this env -> launching must raise cleanly.
    async def go():
        return await BrowserManager().fetch_text("http://example.com")

    with pytest.raises(BrowserUnavailable):
        asyncio.run(go())


def test_unavailable_is_a_source_error() -> None:
    # so the worker's `except ... SourceError`-style isolation catches it
    assert issubclass(BrowserUnavailable, SourceError)


def test_browser_source_with_injected_manager_yields_items() -> None:
    async def go():
        src = BrowserSource("br1", manager=FakeManager())  # type: ignore[arg-type]
        return await src.fetch({"url": "https://x.com/foo", "selector": "article", "ticker": "TSLA"})

    items = asyncio.run(go())
    assert len(items) == 1
    assert items[0].source_id == "br1"
    assert items[0].ticker == "TSLA"
    assert items[0].url == "https://x.com/foo#final"
    assert "rendered https://x.com/foo [article]" in items[0].text


def test_browser_source_requires_url() -> None:
    async def go():
        return await BrowserSource("br1", manager=FakeManager()).fetch({})  # type: ignore[arg-type]

    with pytest.raises(SourceError):
        asyncio.run(go())


def test_worker_isolates_missing_browser(tmp_path) -> None:
    """A browser source with no playwright must not break other sources."""
    store = IngestStore(Database(tmp_path / "config.db"))

    class GoodSource:
        kind = "good"
        source_id = "good"

        async def fetch(self, config):
            from trading_agent.ingest.fetchers.base import RawItem
            return [RawItem("good", "hello", "http://good/1", "2023-11-15T12:00:00+00:00")]

    class StubRegistry:
        def build(self, user_id, client, *, browser_manager=None):
            return [
                BoundSource("br", "browser", BrowserSource("br"), {"url": "http://x"}),
                BoundSource("good", "good", GoodSource(), {}),
            ]

    worker = IngestWorker(store, StubRegistry(), client_factory=_no_network_client)
    written = asyncio.run(worker.run_once("u1"))
    assert written == 1  # good source landed; browser failure isolated
    assert [it.url for it in store.drain("u1")] == ["http://good/1"]
