# WS-I · Engine wiring (Wave 3 — the gap Waves 0–2 left)

**Why this exists:** Waves 0–2 built the agents, stores, and the whole cockpit, but **no workstream
was ever assigned to wire the core trading routers to the live engine.** Today `bench.py`, `risk.py`,
and `approvals.py` still return **501**, so the cockpit's main surfaces (Accounts / Leaderboard /
Positions / Activity / Risk / Approvals) render the **mock fallback**, not live data. This stream
closes that, plus the small coordination items flagged in the Wave-2 review.

**Depends on:** everything (Waves 0–2 merged). **Blocks:** a genuinely-live cockpit.

**Owns (edit):**
- `web/routers/bench.py` — implement `GET /api/accounts`, `/api/leaderboard`, `/api/positions`,
  `/api/activity`, and **`POST /api/accounts`** (the add-trader create route — wizard) from a live
  `app.state.bench` (`Bench.leaderboard()/snapshot()/recent_decisions()`, `BenchController.add_model`).
  Match the cockpit's expected shapes (`ACCOUNTS`/`POSITIONS`/leaderboard rows/activity log).
- `web/routers/risk.py` — `GET /api/risk`, `PUT /api/risk/limits`, `POST /api/risk/kill` from the live
  `RiskManager` (+ persist edited limits per `user_settings`).
- `web/routers/approvals.py` — `GET /api/approvals`, approve/reject from the live `ApprovalQueue`.
- **The serve entrypoint** (`web/app.py` `create_cockpit_app` and/or `scripts/serve.py`) — attach
  `app.state.bench`, `app.state.market_watch` (+ optionally `research`/`memory`) so the manager grounds
  replies, notifications show live alerts/fills, and the stock-request `universe_listener` can update a
  live trader's `.symbols`. (See CONTRACTS §"Runtime wiring via app.state".)
- `tests/test_foundation.py` — remove accounts/leaderboard/positions/activity/risk/approvals from
  `STUB_ROUTES` as you implement them (shed-your-own-rows, like WS-C/E/H did).

**Also reconcile (small, flagged in Wave-2 review):**
- **Manager model picker is cosmetic.** The left-rail model selector doesn't drive the chat —
  `WS-E resolve_manager_ref` reads the model from `user_settings` server-side and `ChatIn` has no model
  field. Pick one: (a) have the cockpit persist the selector to the settings key `resolve_manager_ref`
  reads, or (b) add an optional `model`/`endpoint_id` to `ChatIn`. (a) is less coupling.
- (optional) Centralize `trader_universe` + `notification_reads` DDL into `config/db.py` SCHEMA.

**Acceptance:**
- With a live `Bench`/`RiskManager`/`ApprovalQueue` attached to `app.state`, the six surfaces return
  real data; the add-trader wizard creates a real competitor; STUB_ROUTES shrinks accordingly.
- Manager grounds replies on the live book when `app.state.bench` is set.
- `pytest`/`ruff`/`mypy` green; new route tests added.

**Out of scope:** the data/research/memory engines themselves (built in Waves 1–2). Pure wiring.
