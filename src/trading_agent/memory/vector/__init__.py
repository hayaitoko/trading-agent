"""Vector store backends + the setting-driven factory.

``vstore`` (a per-user setting, default ``"sqlite-vec"``) decides which backend a
:class:`~trading_agent.memory.store.MemoryStore` rides on. Both satisfy the same
:class:`VectorStore` protocol, so the rest of WS-D is backend-agnostic.
"""

from __future__ import annotations

from typing import Any

from .base import (
    Filter,
    Hit,
    StoredPoint,
    VectorStore,
    cosine_similarity,
    matches_filter,
    pack_vector,
    unpack_vector,
)
from .sqlite_vec import DEFAULT_MEMORY_DB, SqliteVecStore

__all__ = [
    "DEFAULT_MEMORY_DB",
    "Filter",
    "Hit",
    "SqliteVecStore",
    "StoredPoint",
    "VectorStore",
    "cosine_similarity",
    "make_vector_store",
    "matches_filter",
    "pack_vector",
    "unpack_vector",
]


def make_vector_store(vstore: str = "sqlite-vec", **kwargs: Any) -> VectorStore:
    """Build the backend named by the ``vstore`` setting.

    - ``"sqlite-vec"`` (default): ``SqliteVecStore(path=...)`` — vectors in SQLite.
    - ``"qdrant"``: ``QdrantVectorStore(url=/path=/location=...)`` — external or
      embedded Qdrant.

    Unknown values fall back to sqlite-vec rather than failing a trader's memory.
    """
    name = (vstore or "sqlite-vec").strip().lower()
    if name == "qdrant":
        from .qdrant import QdrantVectorStore

        qkeys = {"url", "path", "location", "api_key"}
        return QdrantVectorStore(**{k: v for k, v in kwargs.items() if k in qkeys})
    path = kwargs.get("path", DEFAULT_MEMORY_DB)
    return SqliteVecStore(path=path)
