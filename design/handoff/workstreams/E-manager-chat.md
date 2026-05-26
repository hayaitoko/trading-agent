# WS-E · Manager / overseer + chat (Wave 2, parallel)

**Goal:** the left-rail chat made real — one overseer model that watches all books and talks to the
operator. It **advises, summarizes, flags — never trades.**

**Depends on:** `bench.snapshot()`/`leaderboard()`/`recent_decisions()` (EXIST), WS-C research store,
WS-D memory, WS-0 endpoint registry. Stub C/D behind their interfaces to start early.
**Blocks:** WS-G2 (chat + saved chats UI).

**Owns (create/edit):**
- `manager/agent.py` — `ManagerAgent.chat(user_id, conversation_id, message, ref)`: build context from
  the **live bench snapshot** (all books, recent decisions, risk state) + recent research briefs +
  (optionally) relevant trader memories → call the manager model via `EndpointRegistry.chat`
  (model from `user_settings.manager_model`, cheap default `google/gemini-3.5-flash`). Returns reply
  text. `flags(user_id)` → notifications worth raising (e.g. a book breaching a soft limit).
- `manager/chat.py` — conversation persistence using the `conversations`/`turns` tables (WS-0). Backs
  the cockpit's saved-chats.
- Fill `web/routers/manager.py`: `GET /api/chats`, `POST /api/chat` (send→reply, persists turns),
  `POST /api/chats` (save current), `DELETE /api/chats/{id}`. Shapes match the cockpit chat + saved-chat
  list. The manager may push items via the notification center (coordinate shape with WS-H/notifications).

**Steps:** conversation store (reuse Artoo's turns/sessions pattern) → context assembler (bench +
research + memory, each optional/guarded) → manager.chat via registry → flags() → wire manager router →
tests with mocked model + fake bench snapshot.

**Acceptance:**
- `POST /api/chat` returns a reply grounded in a fake bench snapshot; turns persist; `GET /api/chats`
  lists saved conversations (per user).
- Manager has **no trading capability** — it can only read + raise notifications/flags (assert it
  never calls the broker).
- Model is configurable via settings; cheap default; cost-gated (1 call per message).
- ruff + mypy green.

**Out of scope:** building research/memory themselves (consume C/D). Trade execution.
