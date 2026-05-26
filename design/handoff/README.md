# Trading-agent — implementation handoff

This folder is the build plan for turning the cockpit mock into a real system. It exists because the
**UI is finished and well ahead of the backend** — the mock defines the target, this handoff says how
to build it, parallelized across subagents.

**Read first, in order:**
1. `design/AGENT-ARCHITECTURE.md` — the *what* and *why* (the org, decisions taken).
2. `design/handoff/CONTRACTS.md` — the shared seams (schemas, interfaces, routers). **Everything
   builds against this.** Do not diverge from it without updating it.
3. Your workstream brief in `design/handoff/workstreams/`.

The finished cockpit mock is `design/cockpit.html` (served on `0.0.0.0:8090`). Keep it as the visual
spec; the cockpit-wiring stream (G) replaces its mock data with live `fetch()` calls.

---

## The per-user settings answer (important)

The mock persists settings in **`localStorage` = per-browser, NOT per-user**. The login is a stub.
**The real build must store all per-user state server-side, keyed by `user_id`:** theme, risk limits,
endpoints, sources, embed model, saved chats, advisor notes, per-trader universe. Login → resolve
`user_id` → load that user's config. This is foundational (Workstream 0), and it's the same seam that
later enables multi-user / Alpaca-OAuth distribution. Where state is *also* per-trader (memory, notes),
it's keyed by **`(user_id, trader_id)`**.

---

## Principles (every stream follows these)

- **Contracts-first.** `CONTRACTS.md` is the law. Implement *behind* the interfaces there so streams
  compose without merge wars.
- **Own distinct files.** Each stream has an explicit "Owns" file list. Don't edit outside it. New
  HTTP routes go in **your own FastAPI router** (`web/routers/<stream>.py`), never a shared file.
- **All model calls go through the endpoint registry** (`CONTRACTS.md §Endpoint resolution`). No
  hardcoded provider/base-url/key anywhere. Local models are just another endpoint.
- **Cost-gate spend.** Anything that calls a paid model is triggered explicitly or on a cadence with a
  budget — never an uncapped loop. Mirror the cockpit's cost estimator assumptions.
- **Keep it green.** Before finishing: `.venv/bin/pytest`, `.venv/bin/ruff check .`, `.venv/bin/mypy`
  must pass. Add tests for what you build.
- **Deploy target = Pi 4B 8GB** (USB3 SSD, cooling). Local embedder + sqlite-vec/Qdrant; ingestion
  workers are **location-agnostic** (run on the Pi now, movable to an x86 box later — see Workstream B).

## Run & test

```
cd /home/hayai/projects/trading-agent-build-a-python-trading
.venv/bin/pytest -q            # 269 baseline tests must stay green
.venv/bin/ruff check .
.venv/bin/mypy src
# serve cockpit (mock) for visual reference — bind 0.0.0.0, host is artoo @ 10.0.0.26:
.venv/bin/python -m http.server 8090 --bind 0.0.0.0 --directory design
# the real app (after Workstream 0 builds it):
.venv/bin/uvicorn trading_agent.web.app:app --host 0.0.0.0 --port 8000
```

---

## Parallelization plan

```
WAVE 0  (1 agent, BLOCKS everything — must land first)
  └─ WS-0  Foundation: package scaffold, SQLite bootstrap, users + real local auth,
           per-user settings store, provider/endpoint registry, FastAPI app + router includes,
           and the typed stubs/route-stubs from CONTRACTS.md (return 501 until filled).

WAVE 1  (fire these 4 in PARALLEL — independent file sets)
  ├─ WS-A  Data & history        (provider adapters + history service + richer trader context)
  ├─ WS-B  Ingestion             (source registry + async fetchers + worker iface + raw store)
  ├─ WS-D  Memory                (sqlite-vec/Qdrant store, (user,trader) namespacing, reflection, hygiene)
  └─ WS-G1 Cockpit wiring R/O    (Accounts/Positions/Leaderboard/Activity/Risk → live bench/risk/approvals)

WAVE 2  (fire in PARALLEL once their deps below are in; each may stub an unfinished upstream)
  ├─ WS-C  Research agent+store  (needs B; reads memory schema from D)
  ├─ WS-E  Manager + chat        (reads bench[exists] + research[C] + memory[D]; stub C/D if needed)
  ├─ WS-H  Stock-requests+notes  (extends approval_queue[exists] + notification center; needs universe from A)
  └─ WS-G2 Cockpit wiring rest   (Research / Manager chat / Notifications / Settings → live routers)
```

Dependency rule: a Wave-2 stream may start early by coding against the **stub** its upstream left in
Wave 0 (everything has a 501 stub + typed interface), then integrate when the upstream lands.

---

## Firing sequence (copy-paste prompts)

Fire **Wave 0 alone first.** When it reports green, fire all of **Wave 1** together (one message, 4
agents). When Wave 1's deps are in, fire **Wave 2** together. Each prompt is self-contained.

> **Common preamble** (prepend to every prompt): *"Repo: /home/hayai/projects/trading-agent-build-a-python-trading. Use the `.venv`. Read `design/handoff/CONTRACTS.md` first. Stay strictly within your workstream's 'Owns' file list. Before finishing run `.venv/bin/pytest`, `.venv/bin/ruff check .`, `.venv/bin/mypy src` and report results. Do not touch the cockpit mock unless your brief says so."*

**Wave 0**
> Read `design/handoff/workstreams/00-foundation.md` and implement Workstream 0 (foundation:
> scaffold, SQLite bootstrap, users + local auth, per-user settings, endpoint registry, FastAPI app
> with per-stream router stubs returning 501). This blocks all other work — make the interfaces in
> CONTRACTS.md real and importable.

**Wave 1 (send as one message, 4 parallel agents)**
> A: Read `design/handoff/workstreams/A-data-history.md` and implement Workstream A.
> B: Read `design/handoff/workstreams/B-ingestion.md` and implement Workstream B.
> D: Read `design/handoff/workstreams/D-memory.md` and implement Workstream D.
> G1: Read `design/handoff/workstreams/G-cockpit-wiring.md` and implement **Phase 1 (read-only surfaces)** only.

**Wave 2 (send as one message, 4 parallel agents)**
> C: Read `design/handoff/workstreams/C-research.md` and implement Workstream C.
> E: Read `design/handoff/workstreams/E-manager-chat.md` and implement Workstream E.
> H: Read `design/handoff/workstreams/H-requests-notes.md` and implement Workstream H.
> G2: Read `design/handoff/workstreams/G-cockpit-wiring.md` and implement **Phase 2 (Research/Manager/Notifications/Settings)**.

---

## Definition of done (every stream)

- Implements its CONTRACTS.md interface(s); no hardcoded providers/keys.
- New routes live in the stream's own router and are mounted by `web/app.py` (Workstream 0 wires the include).
- `pytest` / `ruff` / `mypy` green; new behavior has tests.
- Touched only files in its "Owns" list.
- Updated CONTRACTS.md if (and only if) a shared interface genuinely had to change — and flagged it.
