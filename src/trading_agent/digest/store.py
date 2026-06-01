"""DigestStore — persists compiled analyst digests.

Each digest record is:
  - Keyed by ``(user_id, universe_key)`` where ``universe_key`` is a canonical
    sorted join of the trader's symbols (e.g. ``"AAPL,NVDA,TSLA"``).
  - Written as a single SQLite row in ``analyst_digests``.
  - Best-effort indexed as a vector point in the shared memory vault so
    ``search_context(query)`` can retrieve its chunks semantically.

Schema is idempotent (CREATE ... IF NOT EXISTS) so it composes with WS-0's
bootstrap without owning it.

Token budget
------------
``max_chars`` (default 2000) caps the rendered digest string.  The compiler
truncates to this budget before storing; the trader pulls the stored string
verbatim (one cheap read, no model call).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from ..config.db import Database

logger = logging.getLogger(__name__)

# Maximum stored digest size in characters (~500 tokens @ 4 chars/token).
DEFAULT_MAX_CHARS: int = 2000

# Staleness ceiling: digests older than this many seconds are considered stale
# (the DigestDaemon refreshes before this is reached in normal operation).
DIGEST_STALE_SECONDS: int = 3600

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyst_digests (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    universe_key    TEXT NOT NULL,
    as_of           REAL NOT NULL,
    digest_text     TEXT NOT NULL,
    headlines_json  TEXT NOT NULL DEFAULT '[]',
    regime_label    TEXT,
    material_flag   INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_digest_user_universe
    ON analyst_digests(user_id, universe_key);
CREATE INDEX IF NOT EXISTS idx_digest_user_created
    ON analyst_digests(user_id, created_at);
"""


def universe_key(symbols: list[str]) -> str:
    """Canonical, sorted symbol join used as the per-universe digest key."""
    return ",".join(sorted(s.upper() for s in symbols if s))


@dataclass
class Digest:
    """One compiled analyst digest for a symbol universe."""

    user_id: str
    universe_key: str
    as_of: float  # Unix timestamp when the source data was current
    digest_text: str  # token-budgeted compact text
    headlines: list[str]  # top headline one-liners
    regime_label: str | None  # e.g. "elevated", "risk-off"
    material_flag: bool  # True when a high-impact event was detected
    id: str = ""
    created_at: float = 0.0

    def age_seconds(self) -> float:
        return time.time() - self.as_of

    def is_stale(self, threshold: int = DIGEST_STALE_SECONDS) -> bool:
        return self.age_seconds() > threshold


class DigestStore:
    """Structured storage for analyst digests with best-effort vector index."""

    def __init__(
        self,
        db: Database,
        vector: Any = None,
        embedder: Any = None,
    ) -> None:
        self._db = db
        self._db.connect().executescript(SCHEMA)
        self._vector = vector
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Write

    def put(self, digest: Digest) -> Digest:
        """Persist (upsert) a compiled digest.  Returns the stored record."""
        if not digest.id:
            digest.id = uuid.uuid4().hex
        if not digest.created_at:
            digest.created_at = time.time()
        self._db.execute(
            "INSERT OR REPLACE INTO analyst_digests "
            "(id, user_id, universe_key, as_of, digest_text, headlines_json, "
            "regime_label, material_flag, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                digest.id,
                digest.user_id,
                digest.universe_key,
                digest.as_of,
                digest.digest_text,
                json.dumps(list(digest.headlines)),
                digest.regime_label,
                int(digest.material_flag),
                digest.created_at,
            ),
        )
        self._index(digest)
        return digest

    def _index(self, digest: Digest) -> None:
        """Best-effort vector upsert of the digest text."""
        if self._vector is None or self._embedder is None:
            return
        try:
            from ..memory.embed import EmbedError

            # Reuse the per-user briefs collection so search_context() can
            # search across both briefs and digests with a single query.
            collection = f"d:{digest.user_id}:digests"
            try:
                vec = self._embedder.embed(digest.digest_text[:1000])
            except EmbedError as exc:
                logger.debug("digest not vectorized (embed error): %s", exc)
                return
            payload = {
                "kind": "digest",
                "user_id": digest.user_id,
                "universe_key": digest.universe_key,
                "digest_text": digest.digest_text,
                "regime_label": digest.regime_label,
                "material_flag": digest.material_flag,
                "as_of": digest.as_of,
            }
            self._vector.upsert(collection, digest.id, vec, payload)
        except Exception as exc:
            logger.debug("digest vector index failed: %s", exc)

    # ------------------------------------------------------------------
    # Read

    def get_latest(self, user_id: str, symbols: list[str]) -> Digest | None:
        """Return the most recent digest for the given universe, or None."""
        uk = universe_key(symbols)
        row = self._db.query_one(
            "SELECT * FROM analyst_digests "
            "WHERE user_id = ? AND universe_key = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id, uk),
        )
        return self._row_to_digest(row) if row is not None else None

    def get_by_key(self, user_id: str, uk: str) -> Digest | None:
        """Return the most recent digest for a pre-computed universe_key."""
        row = self._db.query_one(
            "SELECT * FROM analyst_digests "
            "WHERE user_id = ? AND universe_key = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id, uk),
        )
        return self._row_to_digest(row) if row is not None else None

    def recent(self, user_id: str, n: int = 10) -> list[Digest]:
        """Most recent ``n`` digests for a user (across universes)."""
        rows = self._db.query(
            "SELECT * FROM analyst_digests WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, max(1, n)),
        )
        return [self._row_to_digest(r) for r in rows]

    def search_vector(
        self, user_id: str, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """Semantic search over the digest vault for a user.

        Returns a list of payload dicts (not Digest objects) so the tool can
        render them without a full ORM round-trip.  Falls back to an empty
        list when the vector store / embedder is absent.
        """
        if self._vector is None or self._embedder is None or k <= 0:
            return []
        try:
            from ..memory.embed import EmbedError

            try:
                vec = self._embedder.embed(query)
            except EmbedError:
                return []

            collection = f"d:{user_id}:digests"
            hits = self._vector.search(collection, vec, k, flt={"user_id": user_id})
            return [dict(h.payload) for h in hits]
        except Exception as exc:
            logger.debug("digest vector search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _row_to_digest(row: Any) -> Digest:
        return Digest(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            universe_key=str(row["universe_key"]),
            as_of=float(row["as_of"]),
            digest_text=str(row["digest_text"]),
            headlines=json.loads(row["headlines_json"]),
            regime_label=row["regime_label"],
            material_flag=bool(row["material_flag"]),
            created_at=float(row["created_at"]),
        )
