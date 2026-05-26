"""``IngestWorker`` — runs every enabled source concurrently on a cadence.

The whole point of this stream: fan out ~10 sources with ``asyncio.gather`` over
one shared HTTP client, so a cycle costs ~one round-trip, not the sum. Each
source is awaited with ``return_exceptions=True`` — **no source can block or kill
the others**; a failure is logged and the rest still land.

**Location-agnostic:** the worker depends only on :class:`IngestStore` and
:class:`SourceRegistry` (both DB-backed). Run it in-process on the Pi today, or
as a standalone process / on another host tomorrow — same code, same DB:

    python -m trading_agent.ingest.worker            # all users, INGEST_CADENCE_SECONDS

No model is called anywhere here, so there is no spend to cost-gate.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx

from ..config.db import Database
from .fetchers.base import DEFAULT_TIMEOUT, RawItem
from .fetchers.browser import BrowserManager
from .registry import SourceRegistry
from .store import IngestStore

logger = logging.getLogger(__name__)

DEFAULT_CADENCE = float(os.environ.get("INGEST_CADENCE_SECONDS", "60"))

ClientFactory = Callable[[], httpx.AsyncClient]


class IngestWorker:
    def __init__(
        self,
        store: IngestStore,
        registry: SourceRegistry,
        *,
        cadence: float = DEFAULT_CADENCE,
        timeout: float = DEFAULT_TIMEOUT,
        client_factory: ClientFactory | None = None,
        browser_manager: BrowserManager | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.cadence = cadence
        self.timeout = timeout
        self._client_factory = client_factory
        self._browser_manager = browser_manager

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._client_factory is not None:
            client = self._client_factory()
        else:
            client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)
        try:
            yield client
        finally:
            await client.aclose()

    async def fetch_cycle(self, user_id: str, client: httpx.AsyncClient) -> int:
        """One concurrent pass over a user's enabled sources → rows written.

        Sources run in parallel; exceptions are isolated per-source.
        """
        built = self.registry.build(user_id, client, browser_manager=self._browser_manager)
        if not built:
            return 0
        results = await asyncio.gather(*(b.fetch() for b in built), return_exceptions=True)
        items: list[RawItem] = []
        for bound, result in zip(built, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "ingest: source %s (%s) failed: %s", bound.source_id, bound.kind, result
                )
                continue
            items.extend(result)
        written = self.store.append(user_id, items)
        logger.info(
            "ingest: user=%s sources=%d fetched=%d new=%d",
            user_id, len(built), len(items), written,
        )
        return written

    async def run_once(self, user_id: str) -> int:
        """A single fetch cycle (manages its own client). Handy for tests/cron."""
        async with self._client() as client:
            return await self.fetch_cycle(user_id, client)

    async def run_forever(
        self,
        user_id: str,
        *,
        stop: asyncio.Event | None = None,
        max_cycles: int | None = None,
    ) -> int:
        """Loop fetch cycles every ``cadence`` seconds until ``stop`` is set.

        Returns the number of cycles run. ``max_cycles`` bounds it for tests; the
        sleep is interruptible (waits on ``stop``) for prompt shutdown.
        """
        stop = stop or asyncio.Event()
        cycles = 0
        async with self._client() as client:
            while not stop.is_set():
                try:
                    await self.fetch_cycle(user_id, client)
                except Exception:  # a cycle-level fault must not kill the loop
                    logger.exception("ingest: cycle failed for user=%s", user_id)
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.cadence)
                except TimeoutError:
                    pass
        return cycles


def _users_with_sources(db: Database) -> list[str]:
    rows = db.query("SELECT DISTINCT user_id FROM sources WHERE enabled = 1")
    return [r["user_id"] for r in rows]


async def _run(cadence: float, db_path: str | None) -> None:
    db = Database(db_path) if db_path else Database()
    store = IngestStore(db)
    registry = SourceRegistry(db)
    worker = IngestWorker(store, registry, cadence=cadence)
    users = _users_with_sources(db)
    if not users:
        logger.warning("ingest: no users with enabled sources; nothing to do")
        return
    logger.info("ingest: starting worker for %d user(s), cadence=%ss", len(users), cadence)
    stop = asyncio.Event()
    try:
        await asyncio.gather(*(worker.run_forever(uid, stop=stop) for uid in users))
    except asyncio.CancelledError:  # pragma: no cover
        stop.set()
        raise


def main(argv: list[str] | None = None) -> None:
    """Standalone entrypoint: ``python -m trading_agent.ingest.worker``."""
    parser = argparse.ArgumentParser(description="Trading-agent ingestion worker")
    parser.add_argument(
        "--cadence", type=float, default=DEFAULT_CADENCE,
        help="seconds between fetch cycles (default: $INGEST_CADENCE_SECONDS or 60)",
    )
    parser.add_argument(
        "--db", default=None, help="path to config.db (default: $TRADING_AGENT_DB)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(_run(args.cadence, args.db))
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("ingest: shutting down")


if __name__ == "__main__":  # pragma: no cover
    main()
