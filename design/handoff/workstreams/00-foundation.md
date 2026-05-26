# WS-0 · Foundation (Wave 0 — blocks everything)

**Goal:** stand up the shared spine — real per-user accounts, per-user settings, the endpoint
registry, the SQLite layer, and a FastAPI app that mounts one stub router per stream. After this,
every other stream has a stable, importable interface to build behind.

**Blocks:** all streams. **Depends on:** nothing.

**Owns (create):**
- `config/db.py` — `connect()` (SQLite, WAL, busy_timeout), `bootstrap()` running the DDL in
  `CONTRACTS.md §Per-user model` (users, sessions, user_settings, endpoints, sources, conversations,
  turns, notes, stock_requests). Idempotent.
- `config/users.py` — create user, verify password (hash with `hashlib.scrypt`/`bcrypt`-style stdlib),
  issue/validate session tokens; `current_user` FastAPI dependency → `user_id`.
- `config/settings_store.py` — `get(user_id, key, default)`, `set(user_id, key, value)` (JSON values).
- `config/endpoints.py` — `EndpointRegistry` (CRUD + `client_for` + `chat`) per `CONTRACTS.md
  §Endpoint resolution`. Generalize `llm/openrouter.py`'s client to take base_url+key from an Endpoint;
  add an `anthropic` adapter behind the same `ChatClient` interface. Seed a default OpenRouter
  endpoint from env on first run.
- `web/app.py` (rewrite) — create the FastAPI app, wire auth/session, and `include_router` for ALL
  stream routers below.
- `web/routers/{config,bench,risk,approvals,research,manager,notifications,requests,notes}.py` — each
  with its routes from `CONTRACTS.md §HTTP route table`. **config.py: fully implement** (auth,
  settings, endpoints, sources CRUD). **All others: stub** every route to `raise HTTPException(501)`
  with the correct path+method+`current_user` dependency, so owners just fill the body.

**Steps:** DDL+db → users/auth+session dep → settings_store → endpoint registry (+anthropic adapter)
→ app.py with router includes → fully implement `config.py`, stub the rest → tests.

**Acceptance:**
- `pytest` green incl. new tests: signup→login→`/api/me`; settings round-trip per-user; endpoint
  CRUD + `client_for` returns a working client for openrouter & a local (OpenAI-compatible) base_url;
  two users have isolated settings/endpoints.
- App boots; every route in the table exists (200 for config, 501 elsewhere).
- ruff + mypy green.

**Out of scope:** business logic of other streams (leave their routes 501). Don't build the cockpit
wiring (that's G). Don't call paid models in tests — mock the client.
