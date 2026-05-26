# Session handoff — trading-agent build (2026-05-26)

Pick-up notes for the next session/agent. Read this, then `AGENT-ARCHITECTURE.md` and `CONTRACTS.md`.

## What this is
A multi-model paper-trading harness with its own web UI ("the cockpit"). N AI "traders" each run an
isolated $100k paper book; a manager/overseer chats with the operator; a shared research agent feeds
news/social; per-trader memory; all configurable. Deploy target: a Raspberry Pi 4B 8GB.

## Where things live
- **Repo:** `/home/hayai/projects/trading-agent-build-a-python-trading` · use `.venv`.
- **Remote:** `github.com/hayaitoko/trading-agent` — **NOT pushed this session** (all local).
- **Branch:** `ws0-foundation` (NOT merged to `main` yet).
- **Visual spec (do not edit):** `design/cockpit.html` (the finished mock, served on `0.0.0.0:8090`).
- **Live cockpit (the copy that gets wired):** `src/trading_agent/web/static/cockpit.html`, served at `/`.
- **Design/plan:** `design/AGENT-ARCHITECTURE.md` (the org + decisions), `design/handoff/` (CONTRACTS,
  README, PROMPTS, per-workstream briefs).

## Build status — Waves 0, 1, 2 DONE and verified green
`.venv/bin/pytest` → **495 passed**; `ruff` clean; `mypy src` clean (85 files). Committed in 4 commits
on `ws0-foundation` (foundation → docs → Wave1 → Wave2). No deletions, no CONTRACTS interface breaks.

- **WS-0 Foundation:** per-user SQLite (users/auth/sessions), per-user settings, endpoint registry
  (OpenRouter/OpenAI/Anthropic/local, multi-active), FastAPI app + one router per stream.
- **WS-A** data/history service + optional richer trader context (backward-compatible).
- **WS-B** ingestion: async concurrent fetchers (httpx) + store + location-agnostic worker.
- **WS-D** memory: sqlite-vec(default)/qdrant, namespaced by `(user_id, trader_id)`, reflection, hygiene.
- **WS-C** research agent + store; `/api/research*` **live** (cost-gated).
- **WS-E** manager + chat; `/api/chat*` **live**; manager has **no trading path** (asserted).
- **WS-H** notes + stock-requests + notifications; those routes **live**; trader universe storage.
- **WS-G1/G2** cockpit fully wired with an `api()` helper that **falls back to mock on any non-2xx** —
  so every surface renders today and lights up as its backend lands. Login/settings/endpoints are live.

## What's actually LIVE vs still MOCK
- **Live:** auth/login, settings (theme/limits/embed/vstore), endpoints CRUD, sources CRUD, research,
  manager chat + saved chats, notifications (stock-requests), notes.
- **Still MOCK (the gap):** Accounts / Leaderboard / Positions / Activity / Risk / Approvals — their
  routers (`bench.py`/`risk.py`/`approvals.py`) are **still 501**; no Wave touched them. The cockpit
  shows mock there via fallback. **This is WS-I** (`workstreams/I-engine-wiring.md`).

## OPEN ITEMS / NEXT (priority order)
1. **WS-I — engine wiring (the big one):** implement bench/risk/approvals routers from the live
   `Bench`/`RiskManager`/`ApprovalQueue`, add the **`POST /api/accounts` create route** (wizard), and
   attach `app.state.bench` + `app.state.market_watch` in the serve entrypoint. Brief written.
2. **Manager model picker is cosmetic** — left-rail selector doesn't drive the chat; reconcile per
   WS-I ("persist selection to the settings key `resolve_manager_ref` reads" is the low-coupling fix).
3. **app.state wiring** — until the serve process attaches live `bench`/`market_watch`, notifications =
   only stock-requests and the manager answers without book context (both degrade gracefully).
4. (optional) centralize WS-H's `trader_universe`/`notification_reads` tables into `config/db.py`.
5. **Not pushed / not merged** — decide when to merge `ws0-foundation` → `main` and `git push`
   (Lukas wanted GitHub untouched until he says so).
6. Real provider keys + a live Pi run (Alpaca data/paper, OpenRouter, local Ollama embedder) — the
   tests all mock external calls; nothing has hit a real API yet this session.

## Verify / run
```
.venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy src
.venv/bin/uvicorn trading_agent.web.app:app --host 0.0.0.0 --port 8000   # real app at http://10.0.0.26:8000/
.venv/bin/python -m http.server 8090 --bind 0.0.0.0 --directory design   # the mock spec
```

## Gotchas (carried from this session)
- **Bind `0.0.0.0`, never `127.0.0.1`** — host is artoo @ 10.0.0.26, browser is a different LAN machine.
- `design/cockpit.html` = untouchable spec; wire the `web/static/cockpit.html` copy only.
- Subagents are run by Lukas in **separate terminals** (not the in-app Agent tool); he fires a wave,
  they don't commit, the integrator (you) checkpoints between waves. Keep the contracts-first + distinct
  file-ownership + per-stream-router discipline.
- Endpoint API masks keys (`key_preview`/`has_key`, never raw); cockpit maps `base_url`↔`url`, `enabled`↔`on`.
- Agent memory writes are allowlisted (no prompts). Key memory: `trading-agent-agent-architecture`,
  `trading-agent-cockpit-redesign`.

---

## ▶ NEXT AGENT: hand this WS-I prompt to Lukas (do NOT run it in your own context)

WS-I is run the same way as the other workstreams — as a **separate Claude Code instance Lukas fires
in its own terminal**, not inside your context. So **don't implement WS-I yourself.** Instead, give
Lukas the copy-paste prompt below and let him launch it. (After it finishes, do the integration pass +
checkpoint commit, like the earlier waves.)

> You are working in the repo `/home/hayai/projects/trading-agent-build-a-python-trading`. Use its
> `.venv` for everything. **Step 1 — get oriented:** explore the repo, then read
> `design/handoff/SESSION-HANDOFF.md`, `design/handoff/CONTRACTS.md` (the shared law, incl. the
> "Runtime wiring via app.state" and the `POST /api/accounts` create route), and your brief
> `design/handoff/workstreams/I-engine-wiring.md` — that brief is your spec.
>
> Implement **Workstream I — Engine wiring:** make `web/routers/bench.py`, `risk.py`, and
> `approvals.py` return live data from `app.state.bench` (`Bench`/`BenchController`), `RiskManager`,
> and `ApprovalQueue`; add `POST /api/accounts` (the add-trader wizard create route); attach
> `app.state.bench` + `app.state.market_watch` in the serve entrypoint; and reconcile the cosmetic
> manager model picker (persist the cockpit selector to the settings key `resolve_manager_ref` reads).
> Remove the routes you implement from `tests/test_foundation.py`'s `STUB_ROUTES` (shed-your-own-rows).
>
> **Rules:** stay within the files named in the brief; match the cockpit's expected response shapes
> (see the mock arrays in `design/cockpit.html`); keep response keys aligned with CONTRACTS. Add tests.
> Before finishing run `.venv/bin/pytest`, `.venv/bin/ruff check .`, `.venv/bin/mypy src` and report
> results. Flag any shared-interface change. Do not push to GitHub or merge branches.
