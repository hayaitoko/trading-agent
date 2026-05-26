"""VectorStore protocol + shared value types and math helpers.

The cross-stream contract (``CONTRACTS.md §Stores & agent interfaces``) is the
``upsert`` / ``search`` / ``delete`` triple — that is all WS-C (research) and
WS-E (manager) lean on. WS-D's own memory hygiene needs a little more (read a
point back, iterate a namespace, flip a payload field for a soft-delete), so the
protocol additionally declares ``get`` / ``iter_points`` / ``set_payload`` /
``count``. These are *additive*: anything satisfying the contract triple plus
these helpers is a valid store, and both shipped impls (sqlite-vec, qdrant)
provide all of them.

Scores are **cosine similarity in [-1, 1], higher = closer** — identical
semantics across both backends so callers never special-case the store.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Hit:
    """A search result: a stored point plus its similarity to the query."""

    id: str
    score: float  # cosine similarity, higher = more similar
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoredPoint:
    """A point read back out of the store (vector + payload)."""

    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


# A search filter: payload field -> required exact value (e.g.
# {"trader_id": "alpha", "status": "active"}). Equality only — that is all the
# memory namespacing needs, and it maps cleanly onto both backends.
Filter = dict[str, Any]


@runtime_checkable
class VectorStore(Protocol):
    """Backend-agnostic vector store. Impls: sqlite-vec (default), qdrant.

    Collections are created lazily on first ``upsert``. ``id`` is a caller-chosen
    stable string (re-upserting the same id overwrites).
    """

    # --- CONTRACTS.md triple -------------------------------------------------
    def upsert(
        self, collection: str, id: str, vector: Sequence[float], payload: dict[str, Any]
    ) -> None: ...

    def search(
        self, collection: str, vector: Sequence[float], k: int, flt: Filter | None = None
    ) -> list[Hit]: ...

    def delete(self, collection: str, id: str) -> None: ...

    # --- WS-D management helpers (hygiene / reflection) ----------------------
    def get(self, collection: str, id: str) -> StoredPoint | None: ...

    def iter_points(self, collection: str, flt: Filter | None = None) -> list[StoredPoint]: ...

    def set_payload(self, collection: str, id: str, payload: dict[str, Any]) -> None: ...

    def count(self, collection: str, flt: Filter | None = None) -> int: ...


# --- math / serialization shared by impls ------------------------------------


def pack_vector(vector: Sequence[float]) -> bytes:
    """Pack a float vector into a compact little-endian float32 blob."""
    arr = np.asarray(vector, dtype=np.float32)
    return arr.tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    """Inverse of :func:`pack_vector`."""
    return np.frombuffer(blob, dtype=np.float32).tolist()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 when either side is a zero vector."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def matches_filter(payload: dict[str, Any], flt: Filter | None) -> bool:
    """True when ``payload`` satisfies every ``key == value`` in ``flt``."""
    if not flt:
        return True
    return all(payload.get(k) == v for k, v in flt.items())
