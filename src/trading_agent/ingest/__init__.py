"""WS-B · Ingestion layer.

Pulls many social/news sources concurrently and cheaply, landing *raw* items in
a SQLite-backed store for WS-C (research) to digest. This is an I/O problem, not
a model problem — async HTTP, one shared client, ``asyncio.gather`` over enabled
sources. No model is ever called here, so there is nothing to cost-gate.

The worker is **location-agnostic**: it talks only to :class:`IngestStore` (the
DB), so the same code runs in-process on the Pi today or as a separate
process/host over the LAN later, with no rewrite.

Public surface (see ``design/handoff/CONTRACTS.md §WS-B``):
- :class:`RawItem`         — one fetched item.
- :class:`Source`          — adapter protocol (``kind`` + ``async fetch``).
- :class:`IngestStore`     — ``append`` / ``drain`` over the ``raw_items`` table.
- :class:`SourceRegistry`  — enabled ``sources`` rows → instantiated adapters.

The runner ``IngestWorker`` lives in :mod:`trading_agent.ingest.worker` and is
*not* re-exported here on purpose: it is the ``python -m
trading_agent.ingest.worker`` entrypoint, and importing it eagerly from the
package would make running that module under ``-m`` emit a runpy double-import
warning. Import it directly: ``from trading_agent.ingest.worker import IngestWorker``.
"""

from __future__ import annotations

from .fetchers.base import RawItem, Source
from .registry import SourceRegistry
from .store import IngestStore

__all__ = [
    "IngestStore",
    "RawItem",
    "Source",
    "SourceRegistry",
]
