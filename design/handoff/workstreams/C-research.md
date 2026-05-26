# WS-C · Research agent + store (Wave 2, parallel)

**Goal:** one shared research agent that turns ingested raw items into per-ticker **briefs** every
trader can read. One pass, shared → big quality lift for tiny per-trader cost.

**Depends on:** WS-B (`IngestStore`), WS-D (`VectorStore`/embeddings), WS-0 (endpoint registry).
Stub B/D against their interfaces to start early. **Blocks:** WS-G2 (Research tab), helps WS-E.

**Owns (create):**
- `research/store.py` — `ResearchStore.put/get/recent` per `CONTRACTS.md`. Briefs live in the shared
  vector collection (per user, NOT per trader) + a SQLite row for the structured fields so the
  Research tab can list them fast.
- `research/agent.py` — `ResearchAgent.run(user_id, tickers, ref)`: `IngestStore.drain` → group by
  ticker → **one batched cheap-model call** (via `EndpointRegistry.chat`, model from
  `user_settings.research_model`) → parse into `Brief{summary,sentiment,catalysts,sources,ts}` → `put`.
  **Cost-gated:** runs on the configured cadence or an explicit `/api/research/run`; respects the
  per-user daily $ ceiling. Optionally delegate cheap sentiment extraction to a small/local model.
- Fill in `web/routers/research.py`: `GET /api/research` (list recent briefs for the user),
  `POST /api/research/run` (trigger a pass; gated). Response shape matches the cockpit `RESEARCH` mock.

**Steps:** ResearchStore (vector + sql rows) → agent.run batched+parsed+gated → wire research router →
optional: feed recent briefs into trader context (coordinate with WS-A's context_block) → tests.

**Acceptance:**
- `run` with a fake ingest feed + mocked model produces briefs in the store; `GET /api/research`
  returns them in the cockpit's expected shape.
- Cost gate respected (no run without trigger/cadence; honors daily ceiling) — tested.
- Briefs are shared (per user), not per-trader; traders read read-only.
- ruff + mypy green.

**Out of scope:** ingestion fetchers (WS-B). The private per-trader memory (WS-D).
