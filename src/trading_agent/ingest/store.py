"""``IngestStore`` — the raw-item landing zone (the WS-B↔WS-C seam).

Backed by a ``raw_items`` table in the same per-user ``config.db`` (WS-0's
:class:`~trading_agent.config.db.Database`). The worker only ever talks to this
store, which is what makes ingestion **location-agnostic**: move the worker to
another host and point it at the same DB and nothing else changes.

Dedup is by ``(user_id, source_id, url)`` — re-fetching a feed that still lists
yesterday's posts is a no-op. ``fetched_at`` (our ingest time) is separate from
``RawItem.ts`` (the source's reported time) and is the cursor ``drain`` reads.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ..config.db import Database
from .fetchers.base import RawItem

# DDL lives here (per the brief: "add its DDL here, keyed by user_id"). Idempotent
# CREATE IF NOT EXISTS so it composes with WS-0's bootstrap without owning it.
SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    source_id  TEXT NOT NULL,
    ticker     TEXT,
    text       TEXT NOT NULL,
    url        TEXT NOT NULL,
    ts         TEXT NOT NULL,                 -- source-reported time (ISO-8601)
    fetched_at REAL NOT NULL,                 -- our ingest time; the drain cursor
    UNIQUE(user_id, source_id, url)
);
CREATE INDEX IF NOT EXISTS idx_raw_items_drain ON raw_items(user_id, fetched_at);
"""


class IngestStore:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._db.connect().executescript(SCHEMA)

    def append(self, user_id: str, items: Sequence[RawItem]) -> int:
        """Insert ``items`` for ``user_id``, deduped by ``(source_id, url)``.

        Returns the number of *new* rows actually written. Existing rows are left
        untouched (``INSERT OR IGNORE``), so a source's repeated listings collapse.
        """
        if not items:
            return 0
        fetched_at = time.time()
        rows = [
            (user_id, it.source_id, it.ticker, it.text, it.url, it.ts, fetched_at)
            for it in items
        ]
        conn = self._db.connect()
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO raw_items "
            "(user_id, source_id, ticker, text, url, ts, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes - before

    def drain(self, user_id: str, since: float = 0.0) -> list[RawItem]:
        """Items ingested after the ``since`` watermark, oldest first.

        ``since`` is a ``fetched_at`` value (epoch seconds); pair with
        :meth:`latest_fetched_at` to checkpoint. WS-C calls this to pull the
        backlog it has not yet digested.
        """
        rows = self._db.query(
            "SELECT source_id, ticker, text, url, ts FROM raw_items "
            "WHERE user_id = ? AND fetched_at > ? ORDER BY fetched_at ASC, id ASC",
            (user_id, since),
        )
        return [
            RawItem(
                source_id=r["source_id"],
                text=r["text"],
                url=r["url"],
                ts=r["ts"],
                ticker=r["ticker"],
            )
            for r in rows
        ]

    def latest_fetched_at(self, user_id: str) -> float:
        """Newest ingest watermark for ``user_id`` (``0.0`` if empty). Consumers
        advance their cursor to this after a successful drain."""
        row = self._db.query_one(
            "SELECT MAX(fetched_at) AS hi FROM raw_items WHERE user_id = ?", (user_id,)
        )
        if row is None or row["hi"] is None:
            return 0.0
        return float(row["hi"])

    def count(self, user_id: str) -> int:
        """Total stored items for ``user_id`` (diagnostics / tests)."""
        row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM raw_items WHERE user_id = ?", (user_id,)
        )
        return int(row["n"]) if row is not None else 0
