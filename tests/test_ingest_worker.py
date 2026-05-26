"""Worker concurrency + registry. No live network.

The headline acceptance test: ~10 sources, each a fake that sleeps one
"round-trip", complete a cycle in ~one round-trip (proving asyncio.gather
fan-out, not serial), land deduped in the store, and one failing source never
blocks the rest.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from trading_agent.config.db import Database
from trading_agent.ingest.fetchers.base import RawItem, SourceError
from trading_agent.ingest.fetchers.reddit import RedditSource
from trading_agent.ingest.fetchers.rss import RssSource
from trading_agent.ingest.registry import BoundSource, SourceRegistry
from trading_agent.ingest.store import IngestStore
from trading_agent.ingest.worker import IngestWorker

ROUND_TRIP = 0.1


class SleepySource:
    """Fake Source: waits one round-trip, then returns deterministic items."""

    kind = "sleepy"

    def __init__(self, source_id: str, n_items: int = 1, fail: bool = False) -> None:
        self.source_id = source_id
        self.n_items = n_items
        self.fail = fail

    async def fetch(self, config) -> list[RawItem]:
        await asyncio.sleep(ROUND_TRIP)
        if self.fail:
            raise SourceError(f"{self.source_id} boom")
        return [
            RawItem(self.source_id, f"item {i}", f"http://{self.source_id}/{i}",
                    "2023-11-15T12:00:00+00:00")
            for i in range(self.n_items)
        ]


class FakeRegistry:
    """Returns a fixed set of bound sources regardless of user/client."""

    def __init__(self, sources: list[SleepySource]) -> None:
        self._sources = sources

    def build(self, user_id, client, *, browser_manager=None) -> list[BoundSource]:
        return [BoundSource(s.source_id, s.kind, s, {}) for s in self._sources]


def _no_network_client() -> httpx.AsyncClient:
    # any accidental request fails loudly instead of hitting the network
    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected network call to {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(boom))


def _worker(store: IngestStore, registry, **kw) -> IngestWorker:
    return IngestWorker(store, registry, client_factory=_no_network_client, **kw)


@pytest.fixture
def store(tmp_path) -> IngestStore:
    return IngestStore(Database(tmp_path / "config.db"))


def test_ten_sources_fetch_concurrently(store: IngestStore) -> None:
    sources = [SleepySource(f"s{i}") for i in range(10)]
    worker = _worker(store, FakeRegistry(sources))

    start = time.perf_counter()
    written = asyncio.run(worker.run_once("u1"))
    elapsed = time.perf_counter() - start

    assert written == 10
    assert store.count("u1") == 10
    # serial would be 10 * ROUND_TRIP = 1.0s; concurrent is ~1 round-trip.
    assert elapsed < ROUND_TRIP * 4, f"took {elapsed:.3f}s — not concurrent"


def test_second_cycle_dedups(store: IngestStore) -> None:
    worker = _worker(store, FakeRegistry([SleepySource(f"s{i}", n_items=2) for i in range(5)]))
    assert asyncio.run(worker.run_once("u1")) == 10
    # re-running yields the same urls -> all deduped, nothing new written
    assert asyncio.run(worker.run_once("u1")) == 0
    assert store.count("u1") == 10


def test_one_failing_source_does_not_block_others(store: IngestStore) -> None:
    sources = [SleepySource("good1"), SleepySource("bad", fail=True), SleepySource("good2")]
    worker = _worker(store, FakeRegistry(sources))

    written = asyncio.run(worker.run_once("u1"))
    assert written == 2  # both good sources landed; failure isolated
    urls = {it.url for it in store.drain("u1")}
    assert urls == {"http://good1/0", "http://good2/0"}


def test_no_sources_is_noop(store: IngestStore) -> None:
    worker = _worker(store, FakeRegistry([]))
    assert asyncio.run(worker.run_once("u1")) == 0


def test_run_forever_honors_max_cycles_and_cadence(store: IngestStore) -> None:
    worker = _worker(store, FakeRegistry([SleepySource("s0")]), cadence=0.01)
    cycles = asyncio.run(worker.run_forever("u1", max_cycles=3))
    assert cycles == 3
    assert store.count("u1") == 1  # same item each cycle, deduped


def test_run_forever_stops_on_event(store: IngestStore) -> None:
    worker = _worker(store, FakeRegistry([SleepySource("s0")]), cadence=10.0)

    async def go():
        stop = asyncio.Event()

        async def stopper():
            await asyncio.sleep(ROUND_TRIP * 1.5)
            stop.set()

        cycles, _ = await asyncio.gather(
            worker.run_forever("u1", stop=stop), stopper()
        )
        return cycles

    # cadence is 10s but stop fires mid-sleep -> loop exits promptly after cycle 1
    cycles = asyncio.run(asyncio.wait_for(go(), timeout=2.0))
    assert cycles >= 1


# --- registry over the real sources table -----------------------------------


def test_registry_builds_enabled_adapters(tmp_path) -> None:
    db = Database(tmp_path / "config.db")
    db.execute(
        "INSERT INTO sources (id, user_id, kind, name, config_json, enabled) VALUES "
        "('a','u1','rss','Feed','{\"url\":\"http://x\"}',1),"
        "('b','u1','reddit','WSB','{\"subreddit\":\"wsb\"}',1),"
        "('c','u1','rss','Disabled','{}',0),"
        "('d','u1','bogus','Unknown','{}',1),"
        "('e','u2','rss','OtherUser','{}',1)"
    )
    reg = SourceRegistry(db)

    async def go():
        async with _no_network_client() as client:
            return reg.build("u1", client)

    built = asyncio.run(go())

    kinds = sorted(b.kind for b in built)
    assert kinds == ["reddit", "rss"]  # disabled, unknown, and other-user excluded
    by_kind = {b.kind: b for b in built}
    assert isinstance(by_kind["rss"].source, RssSource)
    assert isinstance(by_kind["reddit"].source, RedditSource)
    assert by_kind["rss"].config == {"url": "http://x"}


def test_registry_kinds_exposes_config_schema(tmp_path) -> None:
    reg = SourceRegistry(Database(tmp_path / "config.db"))
    schema = reg.kinds()
    assert {"rss", "reddit", "stocktwits", "browser"} <= set(schema)
    assert "url" in schema["rss"]
