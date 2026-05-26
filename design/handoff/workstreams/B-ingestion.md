# WS-B · Ingestion layer (Wave 1, parallel)

**Goal:** pull ~10 sources near-constantly and concurrently, cheaply. **This is an I/O problem, not a
model problem** — async HTTP, not a browser per site. Produce raw items into a store for WS-C to digest.

**Depends on:** WS-0 (db, `sources` table, settings). **Blocks:** WS-C (research consumes the store).

**Owns (create):**
- `ingest/store.py` — `IngestStore.append(user_id, items)` / `drain(user_id, since)` over SQLite
  (a `raw_items` table; add its DDL here, keyed by user_id). Dedup by (source_id,url).
- `ingest/fetchers/` — `Source` adapters per `CONTRACTS.md`:
  - `rss.py` (any feed URL), `reddit.py` (WSB via Reddit JSON — no browser), `stocktwits.py` (API).
    These are async `aiohttp` fetchers; cheap and concurrent.
  - `browser.py` — **only** for JS-walled sites (e.g. X). One shared headless browser, multiple
    contexts; keep this set tiny. Make it a clean adapter that can later run **out-of-process / on
    another host** (see location-agnostic note).
- `ingest/registry.py` — reads enabled rows from `sources` (WS-0 table) → instantiates adapters.
- `ingest/worker.py` — runs all enabled sources concurrently (`asyncio.gather`) on a cadence; writes
  to `IngestStore`. **Location-agnostic:** the worker talks ONLY to `IngestStore` (DB), so the same
  code runs in-process on the Pi now, or as a separate process/host later — no rewrite. Make the
  worker entrypoint runnable standalone (`python -m trading_agent.ingest.worker`).
- Routes already stubbed in `web/routers/config.py` for `/api/sources` (WS-0 implemented CRUD) — you
  just provide the adapter `kind`s and their config schema; document them.

**Steps:** store + raw_items DDL → rss/reddit/stocktwits async adapters → registry from `sources` →
concurrent worker w/ cadence + standalone entrypoint → browser adapter last (isolated, optional) →
tests with fake HTTP (no live network in CI).

**Acceptance:**
- 10 fake sources fetched concurrently in a test complete in ~1 round-trip time (proves concurrency),
  land deduped in `IngestStore`.
- Worker runs standalone and in-process; cadence configurable; no source can block the others.
- Browser adapter isolated behind the `Source` interface (can be disabled/missing without breaking).
- ruff + mypy green.

**Out of scope:** turning raw items into briefs/sentiment (WS-C). Deciding final source list (config).
