"""``SourceRegistry`` — enabled ``sources`` rows → instantiated adapters.

Reads the WS-0 ``sources`` table (``kind``, ``config_json``, ``enabled``) and maps
each ``kind`` to its adapter class. Every adapter shares the one injected
``httpx.AsyncClient`` (connection pooling = concurrency). Unknown kinds are
skipped with a warning rather than raising, so one stray config row can't take
the worker down.

The ``kind`` → config-schema map (:meth:`kinds`) is the documentation the WS-0
``/api/sources`` CRUD surface can advertise; this stream owns the adapters, WS-0
owns the route.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from ..config.db import Database
from .fetchers import ADAPTERS
from .fetchers.base import RawItem, Source
from .fetchers.browser import BrowserManager, BrowserSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BoundSource:
    """An instantiated adapter paired with its decoded config, ready to fetch."""

    source_id: str
    kind: str
    source: Source
    config: dict[str, Any]

    async def fetch(self) -> list[RawItem]:
        return await self.source.fetch(self.config)


class SourceRegistry:
    def __init__(
        self, db: Database, adapters: Mapping[str, Callable[..., Source]] | None = None
    ) -> None:
        self._db = db
        self._adapters: dict[str, Callable[..., Source]] = dict(adapters or ADAPTERS)

    def kinds(self) -> dict[str, dict[str, str]]:
        """``kind`` → its ``CONFIG_SCHEMA`` (for docs / the sources CRUD UI)."""
        return {
            kind: dict(getattr(cls, "CONFIG_SCHEMA", {}))
            for kind, cls in self._adapters.items()
        }

    def build(
        self,
        user_id: str,
        client: httpx.AsyncClient,
        *,
        browser_manager: BrowserManager | None = None,
    ) -> list[BoundSource]:
        """Instantiate every *enabled* source for ``user_id``.

        ``browser_manager`` (optional) is shared across all browser sources so
        they reuse one headless browser; absent, each falls back to the process
        default.
        """
        rows = self._db.query(
            "SELECT id, kind, config_json FROM sources WHERE user_id = ? AND enabled = 1",
            (user_id,),
        )
        built: list[BoundSource] = []
        for row in rows:
            kind = row["kind"]
            cls = self._adapters.get(kind)
            if cls is None:
                logger.warning("ingest: unknown source kind %r (id=%s), skipping", kind, row["id"])
                continue
            try:
                config = json.loads(row["config_json"]) or {}
            except json.JSONDecodeError:
                logger.warning("ingest: bad config_json for source %s, skipping", row["id"])
                continue
            if kind == BrowserSource.kind and browser_manager is not None:
                adapter: Source = cls(row["id"], client, manager=browser_manager)
            else:
                adapter = cls(row["id"], client)
            built.append(BoundSource(row["id"], kind, adapter, config))
        return built
