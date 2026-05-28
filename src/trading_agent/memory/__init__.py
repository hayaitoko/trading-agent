"""WS-D — Memory.

Private, namespaced long-term memory for each trader, plus the shared vector
home WS-C's research store rides on. SQLite-vec by default (vectors inside
SQLite, Pi-friendly), Qdrant optional, both behind one :class:`VectorStore`
protocol. A local embedder keeps embeddings on-box. Reflection is gated
(cap + dedup) and hygiene keeps the store from flooding the way Artoo's does.

Typical wiring (per user/session)::

    store = make_vector_store(settings.get(user_id, "vstore"))
    embedder = make_embedder(registry, settings, user_id)   # or FakeEmbedder() in tests
    memory = MemoryStore(store, embedder)
    memory.remember(user_id, trader_id, "keep single-name risk under ~0.9%")
    memory.recall(user_id, trader_id, "how much to risk per trade")
"""

from __future__ import annotations

from .embed import (
    DEFAULT_DIM,
    DEFAULT_EMBED_MODEL,
    Embedder,
    EmbedError,
    FakeEmbedder,
    LocalEmbedder,
    make_embedder,
)
from .hygiene import BM25, Hygiene, HygieneReport
from .reflect import (
    CostGate,
    CostGateError,
    DualReflectionOutput,
    LearningLoop,
    OutcomeRecord,
    ReflectionResult,
    Reflector,
    Skipped,
)
from .store import KIND, STATUS_ACTIVE, STATUS_ARCHIVED, Lesson, MemoryStore, collection_for
from .vector import (
    Hit,
    SqliteVecStore,
    StoredPoint,
    VectorStore,
    make_vector_store,
)

__all__ = [
    "BM25",
    "DEFAULT_DIM",
    "DEFAULT_EMBED_MODEL",
    "KIND",
    "STATUS_ACTIVE",
    "STATUS_ARCHIVED",
    "CostGate",
    "CostGateError",
    "DualReflectionOutput",
    "EmbedError",
    "Embedder",
    "FakeEmbedder",
    "Hit",
    "Hygiene",
    "HygieneReport",
    "LearningLoop",
    "Lesson",
    "LocalEmbedder",
    "MemoryStore",
    "OutcomeRecord",
    "ReflectionResult",
    "Reflector",
    "Skipped",
    "SqliteVecStore",
    "StoredPoint",
    "VectorStore",
    "collection_for",
    "make_embedder",
    "make_vector_store",
]
