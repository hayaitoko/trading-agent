"""``ResearchStore`` — where per-ticker briefs live (the WS-C read surface).

A brief is stored two ways, on purpose:

- a **structured SQLite row** in ``research_briefs`` (same per-user ``config.db``
  as the rest of WS-0), so the Research tab can list/filter briefs fast without
  touching any vectors — this is the source of truth for :meth:`get` /
  :meth:`recent`;
- a **point in the shared vector collection** (``r:{user_id}:briefs``, riding on
  WS-D's :class:`~trading_agent.memory.vector.base.VectorStore`), so traders and
  the manager can recall briefs *semantically* via :meth:`search`.

Briefs are **shared per user, not per trader** (``CONTRACTS.md §WS-C``): the
collection keys on ``user_id`` only and every trader reads the same briefs
read-only. Contrast WS-D's private per-(user, trader) lesson memory.

The vector write is **best-effort**: embeddings require a local endpoint
(``D-memory.md``: embeddings never leave the box) which may not be configured or
reachable on a fresh Pi. If embedding fails the structured row is still written
— the Research tab keeps working; only semantic recall is unavailable until a
local embed endpoint is up. The SQLite row is the durable record.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from ..config.db import Database
from ..memory.embed import Embedder, EmbedError
from ..memory.vector.base import Hit, VectorStore

logger = logging.getLogger(__name__)

KIND = "briefs"

# Idempotent DDL kept beside the store (mirrors IngestStore's pattern): a
# CREATE ... IF NOT EXISTS so it composes with WS-0's bootstrap without owning it.
SCHEMA = """
CREATE TABLE IF NOT EXISTS research_briefs (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    summary       TEXT NOT NULL,
    sentiment     REAL NOT NULL DEFAULT 0.0,   -- [-1, 1], negative=bearish
    catalysts_json TEXT NOT NULL DEFAULT '[]',
    sources_json  TEXT NOT NULL DEFAULT '[]',
    ts            TEXT NOT NULL,               -- ISO-8601 the brief refers to
    created_at    REAL NOT NULL                -- our write time; the list cursor
);
CREATE INDEX IF NOT EXISTS idx_research_recent ON research_briefs(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_research_ticker ON research_briefs(user_id, ticker, created_at);
"""


@dataclass
class Brief:
    """One per-ticker research brief (``CONTRACTS.md §WS-C``).

    The first six fields match the contract exactly and are positional; ``id``
    and ``created_at`` are store bookkeeping appended with defaults so a brief
    can still be built as ``Brief(ticker, summary, sentiment, catalysts,
    sources, ts)``.
    """

    ticker: str
    summary: str
    sentiment: float
    catalysts: list[str]
    sources: list[str]
    ts: str
    id: str = ""
    created_at: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        """Vector-point payload: enough to rebuild the brief from a search hit."""
        return {
            "id": self.id,
            "user_id": "",  # filled by the store at upsert time
            "ticker": self.ticker,
            "summary": self.summary,
            "sentiment": self.sentiment,
            "catalysts": list(self.catalysts),
            "sources": list(self.sources),
            "ts": self.ts,
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Brief:
        return cls(
            ticker=payload.get("ticker", ""),
            summary=payload.get("summary", ""),
            sentiment=float(payload.get("sentiment", 0.0)),
            catalysts=list(payload.get("catalysts", [])),
            sources=list(payload.get("sources", [])),
            ts=payload.get("ts", ""),
            id=payload.get("id", ""),
            created_at=float(payload.get("created_at", 0.0)),
        )


def research_collection_for(user_id: str) -> str:
    """The shared per-user briefs collection. No trader in the key: research is
    shared across a user's traders (cf. memory's per-trader ``u:…:lessons``)."""
    return f"r:{user_id}:{KIND}"


class ResearchStore:
    """Structured + vector storage for shared per-user research briefs."""

    def __init__(
        self,
        db: Database,
        vector: VectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._db = db
        self._db.connect().executescript(SCHEMA)
        self._vector = vector
        self._embedder = embedder

    # --- write ---------------------------------------------------------------

    def put(self, user_id: str, brief: Brief) -> Brief:
        """Persist ``brief`` for ``user_id`` (SQL row + best-effort vector point).

        Returns the stored brief with its ``id`` / ``created_at`` filled in.
        """
        if not brief.id:
            brief.id = uuid.uuid4().hex
        if not brief.created_at:
            brief.created_at = time.time()
        self._db.execute(
            "INSERT OR REPLACE INTO research_briefs "
            "(id, user_id, ticker, summary, sentiment, catalysts_json, sources_json, ts, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                brief.id,
                user_id,
                brief.ticker,
                brief.summary,
                float(brief.sentiment),
                json.dumps(list(brief.catalysts)),
                json.dumps(list(brief.sources)),
                brief.ts,
                brief.created_at,
            ),
        )
        self._index(user_id, brief)
        return brief

    def _index(self, user_id: str, brief: Brief) -> None:
        """Best-effort vector upsert. Embedding may be unavailable (no local
        endpoint); the structured row is the durable record either way."""
        if self._vector is None or self._embedder is None:
            return
        try:
            vector = self._embedder.embed(f"{brief.ticker}: {brief.summary}")
        except EmbedError as exc:
            logger.warning("research brief %s not vectorized: %s", brief.id, exc)
            return
        payload = brief.to_payload()
        payload["user_id"] = user_id
        self._vector.upsert(research_collection_for(user_id), brief.id, vector, payload)

    # --- read (structured, fast) --------------------------------------------

    def get(self, user_id: str, ticker: str) -> list[Brief]:
        """All briefs for ``ticker``, newest first."""
        rows = self._db.query(
            "SELECT * FROM research_briefs WHERE user_id = ? AND ticker = ? "
            "ORDER BY created_at DESC, id DESC",
            (user_id, ticker.upper()),
        )
        return [self._row_to_brief(r) for r in rows]

    def recent(self, user_id: str, n: int = 20) -> list[Brief]:
        """The ``n`` most recent briefs for the user, across tickers."""
        if n <= 0:
            return []
        rows = self._db.query(
            "SELECT * FROM research_briefs WHERE user_id = ? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (user_id, n),
        )
        return [self._row_to_brief(r) for r in rows]

    def count(self, user_id: str) -> int:
        row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM research_briefs WHERE user_id = ?", (user_id,)
        )
        return int(row["n"]) if row is not None else 0

    # --- read (semantic; needs a vector store + embedder) --------------------

    def search(self, user_id: str, query: str, k: int = 5) -> list[Brief]:
        """Most-similar briefs to ``query`` (for trader context / manager chat).

        Returns ``[]`` when no vector store/embedder is wired or embedding fails
        — callers fall back to :meth:`recent`.
        """
        if self._vector is None or self._embedder is None or k <= 0:
            return []
        try:
            vector = self._embedder.embed(query)
        except EmbedError:
            return []
        hits: list[Hit] = self._vector.search(
            research_collection_for(user_id), vector, k, flt={"user_id": user_id}
        )
        out: list[Brief] = []
        for h in hits:
            brief = Brief.from_payload(h.payload)
            brief.id = h.id
            out.append(brief)
        return out

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_brief(row: Any) -> Brief:
        return Brief(
            ticker=row["ticker"],
            summary=row["summary"],
            sentiment=float(row["sentiment"]),
            catalysts=json.loads(row["catalysts_json"]),
            sources=json.loads(row["sources_json"]),
            ts=row["ts"],
            id=row["id"],
            created_at=float(row["created_at"]),
        )
