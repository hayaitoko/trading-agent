# WS-G · Cockpit wiring (two phases)

**Goal:** make the finished cockpit real — copy `design/cockpit.html` to `web/static/cockpit.html`,
serve it from FastAPI, and replace its mock JS data with live `fetch()` calls to the routers. **Keep
the look and every interaction identical** — this is data plumbing, not a redesign.

**Depends on:** WS-0 (app serves static + auth). Phase 1 also needs live bench/risk/approvals routes
(can land alongside Wave 1). Phase 2 needs WS-C/E/H routers.

**Owns (edit):** `web/static/cockpit.html` (the copy — NOT the `design/` mock, which stays the spec),
and the static-serving line in `web/app.py` if WS-0 didn't add it.

### Phase 1 (Wave 1) — read-only surfaces + auth + settings
- Wire the **login** screen to `POST /api/auth/login` + `/api/me` (replace the localStorage stub).
- Load **settings from the server** (`GET /api/settings`) and persist via `PUT` — theme, limits,
  embed model, endpoints (`/api/endpoints`), sources (`/api/sources`) — instead of localStorage.
- Replace mock arrays with fetches: **Accounts** (`/api/accounts`), **Positions** (`/api/positions`),
  **Leaderboard** (`/api/leaderboard`), **Activity** (`/api/activity`), **Risk** (`/api/risk` +
  `PUT /api/risk/limits` + `POST /api/risk/kill`), **Approvals** (`/api/approvals` + approve/reject).
- Keep the compare chart / fleet strip computing from the live leaderboard payload.

### Phase 2 (Wave 2) — agent surfaces
- **Research tab** ← `GET /api/research` (+ a gated "run now" → `POST /api/research/run`).
- **Manager chat + saved chats** ← `POST /api/chat`, `GET/POST/DELETE /api/chats`.
- **Notification center** ← `GET /api/notifications`, request allow/decline → `/api/requests/...`,
  mark-read → `POST /api/notifications/read`.
- **Advisor notes** (account window + ticker cards) ← `GET/PUT /api/notes`.
- **Add-a-trader wizard** → real create (bench `add_model`/controller) via its route.

**Steps:** copy to static + serve → introduce a tiny `api()` fetch helper + loading/error states (the
mock already has skeletons) → swap each surface from mock→fetch, surface by surface → keep render
functions + shapes; only the data source changes.

**Acceptance:**
- Cockpit served at the app's `/` (or `/cockpit.html`), reachable on `0.0.0.0`.
- Each wired surface shows live data; unwired ones (in Phase 1) still render from mock without errors.
- No visual/interaction regressions vs the mock; palettes, ⌘K, drawers, wizard still work.
- Response shapes consumed match `CONTRACTS.md` / the mock's data shapes.

**Out of scope:** changing the design. Inventing routes — consume exactly the CONTRACTS table.
