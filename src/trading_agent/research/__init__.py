"""WS-C — Research.

One *shared* research agent turns the raw items WS-B ingests into per-ticker
**briefs** every trader can read. One batched pass, shared per user (not per
trader) → a big quality lift for tiny per-trader cost.

Two pieces (``CONTRACTS.md §Stores & agent interfaces``):

- :class:`ResearchStore` — briefs live in the shared vector collection (per user)
  *and* a structured SQLite row, so the Research tab lists them fast while
  traders/manager can recall them semantically.
- :class:`ResearchAgent` — drains :class:`~trading_agent.ingest.store.IngestStore`,
  groups by ticker, makes **one batched cheap-model call** through the endpoint
  registry, parses :class:`Brief` objects, and stores them. It is **cost-gated**:
  it never loops uncapped and refuses to spend past the user's daily $ ceiling.

Typical wiring (per user/session)::

    store = ResearchStore(db, vector_store, embedder)
    agent = ResearchAgent(ingest_store, store, registry, settings)
    agent.run(user_id, tickers=None, ref=ModelRef(endpoint_id, "google/gemini-3.5-flash"))
    store.recent(user_id, 20)            # what the Research tab shows
"""

from __future__ import annotations

from .agent import ResearchAgent
from .store import Brief, ResearchStore, research_collection_for

__all__ = [
    "Brief",
    "ResearchAgent",
    "ResearchStore",
    "research_collection_for",
]
