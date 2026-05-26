# WS-D · Memory (Wave 1, parallel)

**Goal:** give each trader **private, namespaced** long-term memory (lessons it learns), plus the
shared research store's vector home — mirroring Artoo's SQLite+vector approach but **with the
namespacing Artoo lacks**, sized for a Pi.

**Depends on:** WS-0 (db, settings for embed model/vstore). **Blocks:** WS-C (shares VectorStore),
WS-E (manager recalls memory). Both can stub against your interface until you land.

**Owns (create):**
- `memory/vector/` — `VectorStore` impls: `sqlite_vec.py` (**default**, vectors inside SQLite) and
  `qdrant.py` (optional; honor the user's `vstore` setting). Both behind the `CONTRACTS.md VectorStore`
  protocol. `embed(text)` calls a **local** embedder (Ollama; model from `user_settings.embed_model`,
  default `bge-small-en-v1.5`, 384-dim). No WAN calls for embeddings.
- `memory/store.py` — `MemoryStore.remember(user_id, trader_id, lesson, tags)` /
  `recall(user_id, trader_id, query, k)`. **Namespacing is the whole point:** every point's payload
  carries `user_id` + `trader_id`; recall filters to that pair. Private memory is never cross-trader.
- `memory/reflect.py` — after a decision/round, distill durable, decision-changing lessons (not a
  journal). Gated: cap writes, dedup near-duplicates before insert.
- `memory/hygiene.py` — port Artoo's discipline: `status: active|archived` soft-delete, a dedup pass
  (BM25 + semantic), a staleness sweep (archive cold >N days). Expose as callables a scheduler can run.

**Steps:** VectorStore protocol + sqlite-vec impl → local embed() → MemoryStore w/ (user,trader)
filter → reflection (gated/deduped) → hygiene callables → qdrant impl behind same protocol → tests.

**Acceptance:**
- Two traders' memories are isolated: trader A's `recall` never returns trader B's lessons (test it).
- sqlite-vec path works with a stub/local embedder in CI (don't require a live Ollama in tests — inject
  a fake `embed`).
- Reflection writes are capped + deduped; hygiene archives via `status`, never hard-deletes silently.
- vstore switch (sqlite-vec ↔ qdrant) is a setting, both pass the same store tests.
- ruff + mypy green.

**Out of scope:** the Research *agent* logic (WS-C) — you provide the store/embeddings it uses.
