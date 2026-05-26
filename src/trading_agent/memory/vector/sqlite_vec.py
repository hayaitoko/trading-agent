"""Default :class:`~trading_agent.memory.vector.base.VectorStore`: vectors live
*inside SQLite*, no external service — the right call for a Pi 4B.

Vectors are stored as compact float32 BLOBs in a single ``vec_points`` table
keyed by ``(collection, id)``, with the payload as JSON alongside. Equality
filters (the namespacing keys: ``user_id`` / ``trader_id`` / ``status``) are
pushed into SQL via ``json_extract`` so we never load a foreign trader's rows.

Ranking has two paths, transparent to callers:

- **sqlite-vec extension present** → distance is computed in-SQL with
  ``vec_distance_cosine`` and ordered/limited by the engine (the fast path; the
  ``sqlite-vec`` wheel ships aarch64 builds, so it works on the Pi too).
- **extension unavailable** → the filtered candidate rows are scored with a
  numpy cosine in Python. Tiny personal corpus, so brute force is fine, and CI
  never has to load a native extension.

Both paths return identical ordering and cosine-similarity scores.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import (
    Filter,
    Hit,
    StoredPoint,
    cosine_similarity,
    pack_vector,
    unpack_vector,
)

# Separate from config.db: vectors can grow and shouldn't bloat the per-user
# config database. Override with TRADING_AGENT_MEMORY_DB.
DEFAULT_MEMORY_DB = os.environ.get("TRADING_AGENT_MEMORY_DB", "data/memory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vec_points (
    collection TEXT NOT NULL,
    id         TEXT NOT NULL,
    vector     BLOB NOT NULL,        -- little-endian float32
    payload    TEXT NOT NULL,        -- JSON
    PRIMARY KEY (collection, id)
);
CREATE INDEX IF NOT EXISTS idx_vec_points_collection ON vec_points(collection);
"""


def _try_load_vec(conn: sqlite3.Connection) -> bool:
    """Best-effort load of the sqlite-vec extension. False if unavailable."""
    try:
        import sqlite_vec  # local import: optional acceleration only
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except (AttributeError, sqlite3.OperationalError):
        # Python built without extension support, or load blocked.
        return False


class SqliteVecStore:
    """SQLite-backed vector store with thread-local connections (WAL)."""

    def __init__(self, path: str | Path = DEFAULT_MEMORY_DB) -> None:
        self.path = str(path)
        self._local = threading.local()
        # Realize the schema (and learn whether the extension is available) now.
        self._conn()

    # --- connection management ----------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            p = Path(self.path)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.has_vec = _try_load_vec(conn)
            conn.executescript(_SCHEMA)
            self._local.conn = conn
        return conn

    @property
    def has_vec_extension(self) -> bool:
        """Whether this thread's connection loaded the sqlite-vec extension."""
        self._conn()
        return bool(getattr(self._local, "has_vec", False))

    # --- filter -> SQL -------------------------------------------------------

    @staticmethod
    def _where(collection: str, flt: Filter | None) -> tuple[str, list[Any]]:
        clauses = ["collection = ?"]
        params: list[Any] = [collection]
        for key, value in (flt or {}).items():
            clauses.append(f"json_extract(payload, '$.{key}') = ?")
            params.append(value)
        return " AND ".join(clauses), params

    # --- CONTRACTS triple ----------------------------------------------------

    def upsert(
        self, collection: str, id: str, vector: Sequence[float], payload: dict[str, Any]
    ) -> None:
        self._conn().execute(
            """
            INSERT INTO vec_points (collection, id, vector, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(collection, id) DO UPDATE SET
                vector = excluded.vector, payload = excluded.payload
            """,
            (collection, id, pack_vector(vector), json.dumps(payload)),
        )

    def search(
        self, collection: str, vector: Sequence[float], k: int, flt: Filter | None = None
    ) -> list[Hit]:
        if k <= 0:
            return []
        conn = self._conn()
        where, params = self._where(collection, flt)
        if getattr(self._local, "has_vec", False):
            rows = conn.execute(
                f"""
                SELECT id, payload, vec_distance_cosine(vector, ?) AS dist
                FROM vec_points WHERE {where}
                ORDER BY dist ASC LIMIT ?
                """,
                (pack_vector(vector), *params, k),
            ).fetchall()
            return [
                Hit(id=r["id"], score=1.0 - float(r["dist"]), payload=json.loads(r["payload"]))
                for r in rows
            ]
        # Fallback: filter in SQL, score in Python.
        rows = conn.execute(
            f"SELECT id, vector, payload FROM vec_points WHERE {where}", params
        ).fetchall()
        scored = [
            Hit(
                id=r["id"],
                score=cosine_similarity(vector, unpack_vector(r["vector"])),
                payload=json.loads(r["payload"]),
            )
            for r in rows
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def delete(self, collection: str, id: str) -> None:
        self._conn().execute(
            "DELETE FROM vec_points WHERE collection = ? AND id = ?", (collection, id)
        )

    # --- management helpers --------------------------------------------------

    def get(self, collection: str, id: str) -> StoredPoint | None:
        row = self._conn().execute(
            "SELECT id, vector, payload FROM vec_points WHERE collection = ? AND id = ?",
            (collection, id),
        ).fetchone()
        if row is None:
            return None
        return StoredPoint(
            id=row["id"], vector=unpack_vector(row["vector"]), payload=json.loads(row["payload"])
        )

    def iter_points(self, collection: str, flt: Filter | None = None) -> list[StoredPoint]:
        where, params = self._where(collection, flt)
        rows = self._conn().execute(
            f"SELECT id, vector, payload FROM vec_points WHERE {where}", params
        ).fetchall()
        return [
            StoredPoint(
                id=r["id"], vector=unpack_vector(r["vector"]), payload=json.loads(r["payload"])
            )
            for r in rows
        ]

    def set_payload(self, collection: str, id: str, payload: dict[str, Any]) -> None:
        self._conn().execute(
            "UPDATE vec_points SET payload = ? WHERE collection = ? AND id = ?",
            (json.dumps(payload), collection, id),
        )

    def count(self, collection: str, flt: Filter | None = None) -> int:
        where, params = self._where(collection, flt)
        row = self._conn().execute(
            f"SELECT COUNT(*) AS n FROM vec_points WHERE {where}", params
        ).fetchone()
        return int(row["n"])
