# Trading-agent — Agent Architecture (design)

**Purpose:** get everyone back in the loop on how the agents actually work, and design the
fuller organization the cockpit already hints at (research agent, real data/history,
per-trader memory, a manager). Nothing here is built yet unless tagged ✅.

**Legend:** ✅ exists today · 🟡 partial · 🔵 net-new work

---

## 0. Where we are today (the honest baseline)

Each "trader" is **one model, trading blind off a short price strip, alone.**

- A **Competitor** = one `LLMTrader` (a single OpenRouter model) + its own `PaperBroker`
  ($100k isolated book) + its own `RiskManager`. ✅ (`bench/bench.py:56`)
- A **round** (`run_decisions`) loops each competitor independently: build prompt → ask model →
  get `{action, symbol, qty, reason}` → risk-check → fill in *its own* book → log. ✅
- What a model actually sees each decision (`llm/trader.py:143`): **cash, current positions,
  the tradable-symbol list, and the last 30 _closing prices_ per symbol.** That's the whole
  worldview. No candles, no volume, no fundamentals, no news, no past events. 🟡
- Between rounds it remembers **nothing** except that rolling 30-price buffer; the bench keeps
  the last 50 decision-log rows + a one-line comment per competitor. ✅ (`bench/bench.py:61`)
- **No research agent. No shared knowledge. No memory. No manager.** Confirmed by code search.

So the real system is **N blindfolded solo contestants reacting to 30 recent prices.** The
cockpit shows the org we *want*; this doc designs how to get there.

---

## 1. Target component map

| Component | Role | Status |
|---|---|---|
| **Market data feed** | One Alpaca data key → bars/quotes pushed to every book | ✅ `bench.observe_bar/observe_quote` |
| **History & fundamentals service** | Deep historical bars + fundamentals + corporate events, queryable by any agent | ✅ `data/history.HistoryService` — `history()` tool |
| **Research agent** | One shared agent: ingests news/filings → writes per-user **briefs** | ✅ WS-C `research/` — `research_brief()` + `request_research()` tools |
| **Research store** | Shared, read-only-to-traders store of briefs | ✅ `research/store.ResearchStore` |
| **Trader (per account)** | ReAct tool-using agent; calls LOOK/NOTE/ACT tools; exits with hold/pass/trade | ✅ WS-Agent A0–A6 `llm/trader.AgentTrader` |
| **Per-trader memory** | Private, namespaced lessons each trader learns and recalls | ✅ WS-D `memory/store.MemoryStore` — `memory_search()` + `reflect()` tools |
| **Reflection step** | `reflect()` tool writes durable lessons (with tool-call provenance) | ✅ WS-D + WS-Agent A2 |
| **Manager / overseer** | Watches all books; answers operator chat; responds to `ask_manager()` calls | ✅ WS-E `manager/` — `ask_manager()` tool (cost-gated) |
| **Risk manager (per book)** | Kill switch, limits, idempotency checks before any fill | ✅ `risk_manager.py` |
| **Approval queue** | Operator gate; trades await approval; callback turns fire on state changes | ✅ WS-Agent A3 `approval_queue.py` + `web/routers/approvals.py` |
| **Attention queue** | Trader-set reminders + watchpoints; scheduler fires event turns on trip | ✅ WS-Agent A2 `intel/attention_queue.py` |
| **Lifecycle scheduler** | ET-anchored SoD/EoD/regular/event turns; crash recovery; kill-switch soft halt | ✅ WS-Agent A4 `bench/scheduler.py` + `intel/lifecycle.py` |
| **Turn trace + observability** | Full tool-call trace per turn; cockpit replay tiles; cost rollup | ✅ WS-Agent A5 `intel/turn_store.py` + `web/routers/traces.py` |
| **Tutorial mode** | First N turns for new traders: tools → memory → watchpoints orientation | ✅ WS-Agent A6 `prompts/tutorial.py` |
| **Notification center** | Stock-requests + alerts + fills to the operator | ✅ WS-H `web/routers/requests.py`, `web/notifications.py` |
| **Bench controller** | Add/remove traders, cadence, start/stop, tick | ✅ `bench/controller.py` |

---

## 2. The decision loop (as-built — WS-Agent tool-using model)

**Status:** ✅ fully shipped as of WS-Agent A6.  See `design/TRADER-AGENT.md` for
the complete specification and per-component docs.

Each trader is a ReAct-style tool-using agent, not a structured-output pipeline.
The trader decides what data to look at, can skip turns, trade multiple symbols
at once, or do nothing — unconstrained by what the operator stuffed into the prompt.

```
                         ┌─────────────────────────────────────────────┐
   news / filings  ───▶  │  RESEARCH AGENT (shared, per-user cadence)   │ ✅ WS-C
   (RSS + Bluesky)       │  → per-ticker briefs in ResearchStore        │
                         └───────────────┬─────────────────────────────┘
                                         ▼
   Alpaca bars/quotes ──▶  HISTORY SERVICE ✅ ──▶  RESEARCH STORE ✅ (shared read-only)
                                         │                       │
                                         ▼                       ▼
                 ┌──────────────────────────────────────────────────────────┐
   each turn:    │  AGENT TRADER (one model per book)        ✅ WS-Agent A0–A6  │
                 │                                                          │
                 │  always-on first-look context:                           │
                 │    identity · account · wake_reason · turn_type · time   │
                 │    cadence · attention counts · cost-so-far              │
                 │    [directed notes · recent reflections · context hint]  │
                 │                                                          │
                 │  tool loop (until terminal):                             │
                 │    LOOK tools: history, news, research_brief, situation, │
                 │                account_state, memory_search, ask_manager │
                 │    NOTE tools: reflect, remind_me, watchpoint,           │
                 │                watch_symbol / unwatch_symbol             │
                 │    ACT tools:  trade, trade_batch, confirm_trade,        │
                 │                update_protective_order, abandon_trade    │
                 │    END:        hold(reason) · pass() · done_for_day()    │
                 │                trade* / confirm_trade*                   │
                 └──────────┬──────────────────────────┬───────────────────┘
                            ▼                          ▼
                     RISK CHECK ✅              reflect() → MEMORY STORE ✅ WS-D
                            │                  (per-trader, namespaced)
                            ▼
              APPROVAL QUEUE ✅ ──(approve)──▶ PaperBroker fill ✅ ──▶ Turn trace ✅ A5

   MANAGER ✅ WS-E: reads bench snapshot → answers operator chat, flags traders,
                    responds to trader ask_manager() calls (cost-gated ≤1/turn).
                    Manager never discloses paper/live status or peer-trader state.
```

Key point: **the trader chooses what to look at**.  Data sources are tools the
trader calls when it decides it needs them, not context stuffed unconditionally
into the prompt.  Tool-call provenance (which tools fired before a trade) is
recorded in the turn trace and measured in the P6 calibration experiment.

**Lifecycle (ET-anchored, ✅ A4):**
- T−60 min → SoD turn (absorb overnight intel, seed watchpoints).
- RTH: regular cadence + event/reminder/callback turns.
- T+30 min → EoD turn (reflect, lock protective orders).
- Dormant outside that window; research agent continues independently.
- Kill switch: ACT tools return `{ok:false, error:"unavailable"}`; LOOK/NOTE
  tools and hold()/pass() continue to work.

**Tutorial mode (✅ A6):** new traders (`tutorial_remaining=3` default) get
three structured turns (tools → memory → watchpoints) before operating freely.
See §8 of `design/TRADER-AGENT.md`.

---

## 3. Memory design — directly addressing the overlap worry

Two separate stores, on purpose:

- **Shared research store** (read-only to traders): "What's happening with NVDA and why."
  Facts about the world. One copy, everyone reads it.
- **Private per-trader memory** (read+write, namespaced by trader): "*I* got burned oversizing
  momentum on 5/19, so I cap risk at 0.9%." Lessons about *itself*. Never visible to others.

**No overlap by construction** — memory rows are keyed by `trader_id`; a trader only ever
retrieves its own. The shared layer is the *only* shared knowledge, and it's facts, not opinions.

**Hygiene (the flooding concern):**
- Reflection writes are **gated** — only durable, decision-changing lessons, not a journal of
  every tick. Cap rows per trader; dedup near-identical lessons; status-flag stale ones.
- Retrieval is **relevance-scoped**: at decision time a trader pulls only memories about the
  symbols/strategies in play, not its whole history.
- **Reuse Artoo's stack (decided):** **SQLite** for structured/short-term (decision log, chat
  history, recent briefs) + a **vector store** for long-term semantic memory, with embeddings
  generated **locally** — *zero per-call cost, no WAN round-trip*. The embedder + vector store are
  **sized for the deploy target (Pi 4 8GB — see §5b)**: a small 384-dim embedder, not Artoo's
  heavy mxbai-large. Also port Artoo's hygiene crons (`duplicate_digest` dedup, `staleness_sweep`
  cold-archive, `memory_hygiene` reconcile) and its `status: active|archived` soft-delete flag —
  that's the anti-flooding discipline already built.
- **The one change vs Artoo:** Artoo uses a *single global* Qdrant collection with **no
  namespacing**. For trading we **must** separate per trader — a `trader_id` payload filter, or
  per-trader collections (`book_<slug>`), mirroring artoo-web's planned `proj_<slug>`. Plus one
  **shared `research` collection** everyone reads. That's what makes "no overlap" real.

```
SQLite:  decisions / chat turns / sessions / recent briefs        (short-term, structured)
Qdrant:  book_<trader>  → private lessons {text,tags,status,...}   (long-term, namespaced)
         research        → shared briefs {ticker,summary,...}      (long-term, read-only)
```

---

## 4. Research agent

- **One agent, one cheap model**, on a configurable cadence (before each round, or hourly).
- **Sources are a configurable registry (decided), not hardcoded** — the customer brings their own.
  Each source is a pluggable adapter behind one interface; enable/configure in **Settings**
  (canonical) and/or via **manager-chat skills** ("add WallStreetBets", "research NVDA now").
  Adapter kinds:
  - **API** — Alpaca News (already keyed, cheapest), Finnhub/Polygon/NewsAPI (customer's pick).
  - **RSS** — any feed URL.
  - **Browser-scraped social** — **WallStreetBets, Twitter/X, etc.** via Artoo's `browser_task`
    pattern (headless Chromium), which sidesteps Twitter's paywalled API. Brittle + ToS-gray, but
    the cost-effective homelab path. A cheap model (Artoo's `quick` worker, Inception Mercury-2)
    turns a pile of posts into sentiment/themes for pennies.
- **Output:** a per-ticker **brief** → `{summary, sentiment, catalysts[], source_links[], ts}`,
  written to the shared `research` store. **Shared on purpose:** one pass, every trader reads it →
  big quality lift for tiny marginal cost.
- **Surfaces in the cockpit:** the **Research tab** (currently mock) becomes this store's view;
  source on/off lives in **Settings**.

### Ingestion layer (decided): API-first, concurrent, location-agnostic
Lukas needs ~10 sources updating near-constantly with low latency. Key reframe:
- **Concurrency is an I/O problem, not a model one.** ~10 sources running constantly = async HTTP
  fetchers on one event loop; the Pi handles hundreds of in-flight requests trivially. A local model
  does **not** speed scraping up — it only ever helps the *digestion* step.
- **Most sources are HTTP/JSON/RSS/streaming, not browser** — Reddit/WSB (JSON), StockTwits (API),
  news (RSS/API). Run them all concurrently + constantly, cheap. **Only JS-walled sites (e.g. X)
  need a browser** → keep that set tiny (one browser w/ multiple contexts, or an alt-frontend/API).
- **Decoupled:** fetchers (constant, concurrent) → raw queue/store → cheap model digests on a
  cadence. Freshness comes from the fetchers, not the model; the model isn't in the hot path.
- **Location-agnostic by design (decided: build for 1, support 2 + 4 later):** each fetcher/scraper
  is a worker behind a defined interface (writes to the shared queue/store over LAN). The same code
  runs **(now)** API-first on the Pi, **(later)** with a dedicated x86 scraper box for the browser
  fleet, or **(option)** all-in-one on a single bigger box — no rewrite, just *where* workers deploy.
- **Latency reality:** the **price feed (Alpaca stream) is the real-time path**; social/news is an
  inherently laggier signal — scrape concurrently for freshness, don't chase tick-speed on sentiment.

---

## 5. Data & history layer

Today traders see 30 closes. Target: every agent can ask for —
- **Deep historical bars** (e.g., 1–2y daily + a recent intraday window), full OHLCV not just close.
- **Fundamentals** (earnings dates, basic ratios) where the provider supports it.
- **Corporate events** ("what happened in the past" — earnings beats/misses, splits, big gaps).

Alpaca already supplies historical bars via the same key feeding the live books; fundamentals/
events likely need a second provider. This is the **lowest-complexity, highest-leverage** change
— right now the models are nearly blind.

- **History depth = configurable (decided).** Pattern recognition wants *more* history, but you
  can't dump years of bars into a prompt (tokens/cost/context blow up). So: **store** deep history,
  but **feed** the model a downsampled long view + a dense recent window (e.g. daily bars for 1–2y +
  intraday for the last few days), or precomputed pattern features. Make the depth a setting;
  default generous, let the customer tune the history-vs-cost trade.

## 5b. Deploy target: Raspberry Pi 4B (8 GB, USB3 SATA SSD, heatsinks)

Embeddings must stay **local (no WAN)**. The LLM calls go to OpenRouter in the cloud, so they don't
tax the Pi at all — only embeddings + storage + (optional) scraping run on-box.

**Reality check on the actual rig:** a Pi 4B 8GB with an SSD over USB3 and proper cooling is more
capable than my first pass implied. Most of the "go as light as possible" framing was overcautious.

- **Embedding model.** `bge-small-en-v1.5` (~33M, **384-dim**) is still a great *default* — fast,
  small vectors, plenty good. But this rig is **not** forced to the smallest: `nomic-embed-text`
  (137M, 768-dim) runs comfortably, and even `mxbai-large` (335M) is fine for *occasional* embedding
  (it's per-write, not per-token). Pick on quality vs speed, not survival. (Now configurable in
  **Settings → Memory engine**.)
- **Vector store — Qdrant is fine here; `sqlite-vec` is a *simplicity* choice, not a hardware one.**
  I walked this back: on an 8GB Pi with SSD, Qdrant runs comfortably for a memory store this size
  (384-dim vectors are tiny; even 100k+ of them is a few hundred MB). The real trade is operational:
  - **Qdrant** — you *already run it for Artoo*, so it's the parity/known-ops choice, richer filtering.
  - **`sqlite-vec`** — fewer moving parts (lives in the SQLite file, no daemon), nice for a single box.
  Either works on this hardware. Lean Qdrant for Artoo-parity, sqlite-vec for minimalism.
- **Browser scraping — I overstated this.** Headless Chromium is the heaviest *local* piece, but on
  this rig **occasional, scheduled, single-session** scraping (a few pages every 15–60 min) is fine.
  The only thing that actually bites is **many concurrent, continuous** browser sessions. So: one
  session at a time, scheduled, prefer HTTP/JSON (Reddit) where it exists. No hardware change needed
  for the realistic workload.

**If you ever outgrow it** (heavy continuous multi-source scraping, or running a *local LLM* not just
an embedder): a **Pi 5** (much faster CPU/IO) or a small **x86 mini-PC (Intel N100/N150, 16GB)** is
the sweet spot — similar idle power, far more headroom, and x86 smooths some ML tooling. Not needed
for the current plan; only if the workload grows.

---

## 6. Manager / overseer + chat

- **One model**, the thing you talk to in the left-rail chat. **Model is configurable (decided),
  cheap default** — it runs often (summaries/chat/flags), so default to something cheap like
  `google/gemini-3.5-flash`, switchable up to GLM-5.1/DeepSeek in Settings. (Same for the research
  agent's model.)
- **Reads:** the bench snapshot (all books), recent decisions across traders, risk state, and the
  research store. **Does not trade** — it advises, summarizes, and **raises flags** as notifications.
- **Chat layer (🔵):** a conversation store (SQLite, like Artoo's `turns`/`sessions` tables) + an
  endpoint; context = fleet snapshot + research + (optionally) relevant trader memories. Backs
  **saved chats**.
- Optional later: the manager may *propose* an action (e.g., "pause Gemini, it's chasing") that
  lands in the **approval queue / notifications** — never auto-executed.

### Manager skills — port a subset of Artoo's (decided)
Artoo already implements these; bring the relevant ones over so the manager can *act*, not just talk:
- **Scheduling / crons** (`scheduler.py` croniter loop + one-shot `remind`) → schedule research
  passes, bench rounds, reminders ("run research every morning", "tick the bench hourly").
- **Memory ops** (`search_memory` / `save_memory` / `delete_memory`) → read/write Qdrant memory.
- **Browser** (`browser_task` / `browser_run`) → the social-media ingestion path above.
- **Worker delegation** (`spawn_worker`: `quick` / `deep` / `general`) → offload cheap sentiment to
  `quick`, deeper analysis to `deep`, instead of burning the manager model on everything.
- **Notifications** (`to_telegram`) → wire to the cockpit notification center (and/or Telegram).
- **Skip** the irrelevant ones: `github`, `local`, `build_status`, `generate_image`.

## 6b. Provider / endpoint registry (decided)

Models aren't tied to a single provider. A **registry of endpoints** (configured in **Settings →
AI endpoints**) holds **many at once**, each `{type, name, base_url, api_key, enabled}`:
- Types: **OpenRouter / OpenAI / Anthropic / Local** (OpenAI-compatible — Ollama/llama.cpp/LM Studio,
  e.g. on the Pi). Adding a local model is just another entry → "optional local models" are first-class.
- **Multiple active simultaneously.** Each agent (trader / research / manager) picks *which* endpoint
  + model it runs on — so cloud models for the traders and a cheap local model for, say, research
  digestion can run side by side.
- Keys live **server-side** in the real build (the cockpit mock keeps them in `localStorage`).
- This is the seam that makes the OpenRouter-vs-OpenAI-vs-Anthropic-vs-local choice a setting, not a
  code change.

---

## 7. Assignment, management & stock-requests

- **Assignment today:** add a model → it becomes a Competitor with a fresh $100k book. That's it.
- **Target adds:** each trader has a **universe** (which symbols it may trade) and a **style**.
  The Add-a-trader wizard already collects model/cash/style — it would also set the universe.
- **Stock-requests (wires the notification center):** a trader may emit a request to trade a
  symbol *outside* its universe → lands in the **approval queue / notification center** →
  you Allow → the symbol is added to that trader's universe. Makes the bell's "stock requests" real.
- **Advisor notes (🔵):** the per-account notes box needs a store (`notes(trader_id, ts, text)`).

---

## 8. How each piece maps to the cockpit (so the UI becomes honest)

| Cockpit surface | Backed by |
|---|---|
| Accounts grid + fleet strip | `Bench.leaderboard()` / `snapshot()` ✅ |
| Account window (positions, history) | `PaperBroker` ✅ ; advisor notes 🔵 |
| Positions tab | aggregate of all books' positions ✅ |
| Research tab | research store 🔵 |
| Leaderboard + compare chart | `leaderboard()` ✅ ; equity-curve history 🔵 |
| Approvals | `ApprovalQueue` ✅ |
| Risk (editable limits) | `RiskManager` ✅ (limit *editing* persists 🔵) |
| Activity | bench decision log + audit ✅ |
| Bench Control | `BenchController` ✅ |
| Manager chat + saved chats | manager agent + chat store 🔵 |
| Notifications (stock-requests, alerts) | `NotificationCenter`/`MarketMoveWatcher` 🟡 + request flow 🔵 |

---

## 9. Suggested build order (each phase wires a cockpit surface to live data)

1. **Data & history layer** — give traders real OHLCV history + fundamentals. Biggest quality
   lift, least new machinery. Wires: Accounts/Positions/Leaderboard to live `Bench`.
2. **Research agent + store** — shared briefs feed trader context. Wires: Research tab.
3. **Per-trader memory + reflection** — namespaced, hygiene-gated. Wires: the Memory panel.
4. **Manager + chat** — the left-rail overseer + saved chats. Wires: chat, notification flags.
5. **Stock-requests + advisor notes** — wires: notification center requests + account notes.

Wire the read-only views (1) before the spend-y agents (2–4); keep every model call cost-gated.

---

## 10. Open decisions (need Lukas)

**Decided 2026-05-26 (Lukas):**
- Sources = **configurable registry** (Settings + chat skills), not hardcoded. **Social media is
  IN** (WallStreetBets / Twitter) via browser-scraping + a cheap model.
- Memory = **Artoo's stack: SQLite + Qdrant + Ollama (mxbai-embed) embeddings**, with **per-trader
  namespacing** added, reusing Artoo's hygiene crons.
- Manager + research models = **configurable, cheap default** (e.g. `google/gemini-3.5-flash`), and
  the cockpit shows a **live per-day $ estimate** for the model × cadence combo (built into Bench
  Control's Research-agent card).
- **Reuse a subset of Artoo's skills** (scheduling, memory, browser, worker delegation, notifications).
- **Deploy target = Pi 4 8GB, embeddings local (no WAN).** → small 384-dim embedder
  (`bge-small-en-v1.5` default), and lean toward **`sqlite-vec`** over Qdrant on the Pi; run the
  heavy browser-scraping **off-box** (see §5b).
- **History depth = configurable** (store deep, feed a downsampled long + dense recent window).
- **Ingestion = API/RSS-first + async-concurrent on the Pi, location-agnostic** (build for the Pi
  now; support a dedicated x86 scraper box and/or single-box consolidation later, no rewrite).
- **Provider/endpoint registry = multiple endpoints at once** (OpenRouter/OpenAI/Anthropic/local),
  configured in Settings; each agent binds to one. Local model endpoints are first-class.

**Still open:**
1. **Research $ ceiling** — a hard daily spend cap on top of the cadence estimate?
2. **Universe policy** — per-trader default universe + whether there's a master allowlist that
   stock-requests must fall within.
3. **Exact embedder + store** — `bge-small` vs `all-MiniLM`; `sqlite-vec` vs Qdrant (lock once we
   benchmark on the actual Pi).
4. **Where scraping runs** — on Artoo's host vs the Pi vs a worker.
5. **Notifications** — cockpit-only, Telegram (Artoo's `to_telegram`), or both?

---

## Appendix — existing code inventory (✅ to build on)

- `llm/trader.py` — `LLMTrader`/`StrategyTrader`, `TradeDecision`, `DecisionResult`, `_build_context` (30-close context, `lookback`)
- `bench/bench.py` — `Bench`, `Competitor` (decisions deque 50, `last_comment`), `run_decisions`, `observe_bar/observe_quote`, `leaderboard`, `recent_decisions`, `snapshot`
- `bench/controller.py` — `BenchController`: `add_model`, `set_cadence`, `start/stop`, `tick_now`, `available_models`, `status`
- `paper_broker.py` — `PaperBroker`: `get_balance`, `get_positions`, `get_trade_history`, `get_account_value`
- `risk_manager.py` — `RiskManager`: kill switch, `check_*`, limits, exposure
- `approval_queue.py` — `ApprovalQueue`: add/approve/reject/pending/get (SQLite, thread-safe)
- `web/notifications.py` — `NotificationCenter.snapshot()`; `web/market_watch.py` — `MarketMoveWatcher`
- `llm/openrouter.py` — `OpenRouterClient.chat` / `list_models`
