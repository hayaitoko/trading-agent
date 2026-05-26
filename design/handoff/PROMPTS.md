# Subagent prompts — copy-paste

Each agent is told to **orient in the repo and find its own instructions** (`design/handoff/`) before
coding. Fire Wave 0 alone; then Wave 1 (4 agents in one message); then Wave 2 (4 agents in one message).

---

## PREAMBLE (prepended to every prompt below)

> You are working in the repo `/home/hayai/projects/trading-agent-build-a-python-trading`. Use its
> `.venv` for everything (`.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/mypy`).
>
> **Step 1 — get oriented and find your instructions.** Explore the repo: list the tree, skim
> `src/trading_agent/`, then read `design/handoff/README.md` and `design/handoff/CONTRACTS.md` in
> full. CONTRACTS.md is the shared law — schemas, interfaces, and the FastAPI router table everything
> builds against. Then open your workstream brief (named below) in `design/handoff/workstreams/` —
> that brief is your spec.
>
> **Rules:** stay strictly within your workstream's "Owns" file list. Put any new HTTP routes in your
> own router file (never a shared one). Never hardcode a model provider/base-url/key — resolve through
> the endpoint registry. Cost-gate anything that calls a paid model. Do NOT edit `design/cockpit.html`
> or `design/` docs (the cockpit-wiring stream works on the `web/static/` copy only).
>
> **Definition of done:** implement your brief behind the CONTRACTS interfaces; add tests; then run
> `.venv/bin/pytest`, `.venv/bin/ruff check .`, `.venv/bin/mypy src` and report the results. If you had
> to change a shared interface in CONTRACTS.md, say so explicitly in your final summary.

---

## WAVE 0 — fire this ONE agent first; it blocks everything

> [PREAMBLE] Your workstream is **Workstream 0 — Foundation**: read
> `design/handoff/workstreams/00-foundation.md` and implement it. Build the shared spine the other
> seven streams depend on: SQLite bootstrap, real per-user local accounts + auth, per-user settings
> store, the provider/endpoint registry (OpenRouter/OpenAI/Anthropic/local, multiple active at once),
> and a FastAPI app that mounts one router per stream — fully implementing `config.py` and stubbing
> every other route as HTTP 501 with the correct method/path and `current_user` dependency. Make the
> CONTRACTS.md interfaces real and importable. Do not implement other streams' business logic.

---

## WAVE 1 — fire these FOUR together (one message), after Wave 0 is green

> [PREAMBLE] Your workstream is **A — Data & history**: read
> `design/handoff/workstreams/A-data-history.md` and implement it. Build the historical-bars +
> fundamentals service and a richer trader context block (replacing the 30-close prompt), wired as an
> optional injection into `llm/trader.py` so existing bench tests still pass.

> [PREAMBLE] Your workstream is **B — Ingestion**: read
> `design/handoff/workstreams/B-ingestion.md` and implement it. Build the async, concurrent
> source-fetcher layer (RSS/Reddit-JSON/StockTwits adapters + an isolated browser adapter) and a
> location-agnostic worker that writes raw items to a store. This is an I/O concurrency problem, not a
> model one — prove ~10 sources fetch concurrently. No live network in tests.

> [PREAMBLE] Your workstream is **D — Memory**: read
> `design/handoff/workstreams/D-memory.md` and implement it. Build the VectorStore (sqlite-vec default,
> Qdrant optional) with a local embedder, and a MemoryStore namespaced by `(user_id, trader_id)` so
> traders' private lessons never overlap, plus gated reflection and Artoo-style hygiene. Inject a fake
> embedder in tests.

> [PREAMBLE] Your workstream is **G — Cockpit wiring, PHASE 1 ONLY**: read
> `design/handoff/workstreams/G-cockpit-wiring.md` and implement **Phase 1** (read-only surfaces).
> Copy `design/cockpit.html` to `web/static/cockpit.html`, serve it, and replace its mock data for
> login, settings, Accounts, Positions, Leaderboard, Activity, Risk, and Approvals with live `fetch()`
> calls to the existing routes. Keep the look and interactions identical — data plumbing only. Leave
> Phase-2 surfaces on mock data for now.

---

## WAVE 2 — fire these FOUR together (one message), once Wave 1's deps are in

> [PREAMBLE] Your workstream is **C — Research agent + store**: read
> `design/handoff/workstreams/C-research.md` and implement it. Build the ResearchStore and the shared
> research agent that batches ingested items into per-ticker briefs via the endpoint registry
> (cost-gated, cheap model), and fill the research router. Stub the ingestion/memory interfaces if
> those streams aren't merged yet.

> [PREAMBLE] Your workstream is **E — Manager + chat**: read
> `design/handoff/workstreams/E-manager-chat.md` and implement it. Build the overseer agent that reads
> the live bench snapshot + research + memory and chats with the operator (configurable cheap model,
> cost-gated), with conversation persistence, and fill the manager router. The manager advises/flags
> only — it must never trade.

> [PREAMBLE] Your workstream is **H — Stock-requests + advisor notes**: read
> `design/handoff/workstreams/H-requests-notes.md` and implement it. Build the per-(user,scope,ref)
> notes store, the stock-request flow (trader asks → notification → operator allows → that trader's
> universe updates), and merge requests/alerts/fills into the notifications router.

> [PREAMBLE] Your workstream is **G — Cockpit wiring, PHASE 2**: read
> `design/handoff/workstreams/G-cockpit-wiring.md` and implement **Phase 2**. Wire the Research tab,
> Manager chat + saved chats, Notification center (incl. stock-request allow/decline), advisor notes,
> and the add-a-trader wizard to their live routers. Keep the design identical — only swap mock data
> for `fetch()`.
