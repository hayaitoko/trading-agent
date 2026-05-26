# CONTRACTS — the shared seams

Everything builds against this. If you must change a shared interface, change it *here* and flag it in
your handoff so dependents update. Signatures are Python-ish pseudocode; match the spirit, type it
properly with the existing style (dataclasses / Pydantic as the repo already uses).

---

## Package layout (new modules)

```
src/trading_agent/
  web/
    app.py                 # EXISTS → rewritten by WS-0 to mount routers below
    routers/               # NEW — one router file per stream, no shared route file
      config.py  bench.py  risk.py  approvals.py  research.py
      manager.py  notifications.py  requests.py  notes.py
    static/cockpit.html    # WS-G copies design/cockpit.html here, swaps mock→fetch
  config/                  # WS-0
    db.py                  # SQLite connection (WAL), migrations/bootstrap
    users.py               # users + local auth (hashed pw), session→user_id
    settings_store.py      # per-user settings get/set
    endpoints.py           # provider/endpoint registry + model resolution
  data/                    # WS-A
    history.py             # historical bars / fundamentals service
    providers/             # alpaca.py, finnhub.py, ... (adapters)
  ingest/                  # WS-B
    registry.py  worker.py  store.py
    fetchers/              # reddit.py, rss.py, stocktwits.py, browser.py (adapters)
  research/                # WS-C
    agent.py  store.py
  memory/                  # WS-D
    store.py  reflect.py  hygiene.py  vector/   (sqlite_vec.py, qdrant.py)
  manager/                 # WS-E
    agent.py  chat.py
  requests.py  notes.py    # WS-H (or under a small package)
```
Existing, reuse as-is: `llm/trader.py`, `llm/openrouter.py`, `bench/bench.py`, `bench/controller.py`,
`paper_broker.py`, `risk_manager.py`, `approval_queue.py`, `web/notifications.py`, `web/market_watch.py`.

---

## Per-user model (WS-0 owns; everyone honors)

Real local accounts. Session cookie/token → `user_id`. **All per-user state keys on `user_id`**; state
that is also per-trader keys on `(user_id, trader_id)`.

```sql
users(         id TEXT PK, username TEXT UNIQUE, pw_hash TEXT, created_at REAL )
sessions(      token TEXT PK, user_id TEXT, created_at REAL, expires_at REAL )
user_settings( user_id TEXT, key TEXT, value TEXT_JSON, PRIMARY KEY(user_id,key) )  -- theme, limits, embed_model, vstore, research_model, research_cadence, ...
endpoints(     id TEXT PK, user_id TEXT, type TEXT, name TEXT, base_url TEXT, api_key TEXT, enabled INT )
sources(       id TEXT PK, user_id TEXT, kind TEXT, name TEXT, config_json TEXT, enabled INT )  -- WS-B
conversations( id TEXT PK, user_id TEXT, title TEXT, started_at REAL )               -- WS-E
turns(         id INTEGER PK, conversation_id TEXT, role TEXT, content TEXT, created_at REAL )
notes(         id TEXT PK, user_id TEXT, scope TEXT, ref TEXT, text TEXT, updated_at REAL )  -- WS-H; scope in {trader,ticker}
stock_requests(id TEXT PK, user_id TEXT, trader_id TEXT, symbol TEXT, reason TEXT, status TEXT, created_at REAL ) -- WS-H
```
`config.db` reuses the SQLite + WAL pattern Artoo uses. WS-0 provides `db.connect()` + bootstrap.

---

## Endpoint resolution (WS-0 owns; WS-A/C/E consume)

No agent constructs its own HTTP client. It asks the registry.

```python
@dataclass
class ModelRef: endpoint_id: str; model: str        # how every agent names "which model"

class EndpointRegistry:
    def list(user_id) -> list[Endpoint]
    def add/update/remove/toggle(...)
    def client_for(user_id, endpoint_id) -> ChatClient   # OpenAI-compatible client (OpenRouter/OpenAI/Anthropic/local)
    def chat(user_id, ref: ModelRef, messages, **opts) -> ChatResult   # convenience: resolve + call

# Endpoint.type ∈ {openrouter, openai, anthropic, local}. local = OpenAI-compatible base_url (Ollama/llama.cpp).
```
`llm/openrouter.py`'s client is the reference adapter; generalize it to honor a base_url + key from an
Endpoint. Anthropic type may need its own adapter (different wire format) behind the same `ChatClient`.

---

## Stores & agent interfaces

```python
# WS-D — vector + memory
class VectorStore(Protocol):                       # impls: sqlite_vec (default), qdrant
    def upsert(collection, id, vector, payload)
    def search(collection, vector, k, flt=None) -> list[Hit]
    def delete(collection, id)
# memory namespacing: collection per (user, kind); private lessons filtered by trader_id; research shared per user
class MemoryStore:
    def remember(user_id, trader_id, lesson, tags) ; def recall(user_id, trader_id, query, k) -> list[Lesson]
def embed(text) -> list[float]                     # local embedder (Ollama mxbai/bge-small), model from user_settings

# WS-C — research
@dataclass
class Brief: ticker:str; summary:str; sentiment:float; catalysts:list[str]; sources:list[str]; ts:str
class ResearchStore:
    def put(user_id, brief: Brief) ; def get(user_id, ticker) -> list[Brief] ; def recent(user_id, n) -> list[Brief]
class ResearchAgent:
    def run(user_id, tickers, ref: ModelRef) -> list[Brief]      # one batched pass; cost-gated

# WS-B — ingestion
@dataclass
class RawItem: source_id:str; ticker:str|None; text:str; url:str; ts:str
class Source(Protocol):                            # reddit, rss, stocktwits, browser
    kind: str
    async def fetch(config) -> list[RawItem]
class IngestStore:
    def append(user_id, items: list[RawItem]) ; def drain(user_id, since) -> list[RawItem]
# Worker runs enabled Sources concurrently (asyncio.gather) on a cadence; LOCATION-AGNOSTIC:
# talks only to IngestStore, so it can run in-process on the Pi or as a remote worker over LAN.

# WS-A — data/history
class HistoryService:
    def bars(symbol, timeframe, lookback) -> list[Bar] ; def fundamentals(symbol) -> dict | None
    def context_block(symbols, account) -> str      # the richer replacement for trader._build_context

# WS-E — manager
class ManagerAgent:
    def chat(user_id, conversation_id, message, ref: ModelRef) -> str   # reads bench snapshot + research + memory
    def flags(user_id) -> list[Notification]                            # things to raise to the operator
```

---

## HTTP route table (each row → the named router; WS-0 stubs all as 501, owners fill in)

```
config.py        GET /api/endpoints · POST /api/endpoints · DELETE /api/endpoints/{id}
                 GET/PUT /api/settings · GET /api/sources · POST /api/sources · DELETE /api/sources/{id}
                 POST /api/auth/signup · POST /api/auth/login · POST /api/auth/logout · GET /api/me
bench.py         GET /api/accounts · GET /api/leaderboard · GET /api/positions · GET /api/activity
research.py      GET /api/research · POST /api/research/run         (run = cost-gated, explicit)
manager.py       GET /api/chats · POST /api/chat · POST /api/chats (save) · DELETE /api/chats/{id}
risk.py          GET /api/risk · PUT /api/risk/limits · POST /api/risk/kill
approvals.py     GET /api/approvals · POST /api/approvals/{id}/approve · /reject
notifications.py GET /api/notifications · POST /api/notifications/read
requests.py      GET /api/requests · POST /api/requests/{id}/allow · /decline
notes.py         GET/PUT /api/notes?scope=&ref=
```
All routes resolve `user_id` from the session (WS-0 dependency `current_user`). The cockpit (WS-G)
calls exactly these; keep response shapes matching what `cockpit.html`'s render functions expect
(see the mock's `ACCOUNTS`, `POSITIONS`, `APPROVALS`, `NOTIFS`, `RESEARCH`, `MEMORY` shapes).

### As-built notes from Wave 0 (foundation) — honor these
- **Auth:** `POST /api/auth/signup` exists (signup→login→/me). `login`/`signup` return the session
  token in the **response body** *and* set a cookie (bearer for non-browser clients).
- **Endpoint object shape (server, keys never returned raw):**
  `{ id, type, name, base_url, key_preview (last-4), has_key (bool), enabled }`. `POST` accepts a raw
  `key`; the registry stores it server-side and only ever reads it internally.
  → **WS-G (cockpit) must map** server fields ↔ the mock's field names: `base_url`↔`url`,
  `enabled`↔`on`, show `key_preview`/`has_key` instead of `key`. Same for `sources`.
- **Deletes are path-id:** `DELETE /api/endpoints/{id}`, `DELETE /api/sources/{id}`.

---

## Cost-gating contract

Any path that calls a paid model takes an explicit trigger or a cadence+budget. Research `run` and
bench rounds are never uncapped loops. Surface estimated spend the way the cockpit's Research-agent
card does (model price × cadence). A per-user **daily $ ceiling** lives in `user_settings`.
