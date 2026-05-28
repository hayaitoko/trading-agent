# Trader Agent — Design Reference

**Status:** WS-Agent A0 ✅ · A1 ✅ · A2 ✅ · A3 ✅ · A4–A6 in progress  
**Branch:** `feat/engine-realism`  
**Legend:** ✅ exists · 🟡 partial · 🔵 planned

---

## 1. Purpose + Reframe Rationale

The original `LLMTrader` was a single structured-output prompt-pipeline:
build context → call model → parse `{action, symbol, qty}` JSON → fill broker.
The trader had no agency: the operator chose what data to inject, and the model
could only respond to what was stuffed into the prompt.

**WS-Agent reframes the trader as a real ReAct-style tool-using agent.**

The trigger was Lukas's observation: "I want them to not just be given data and
a prompt, I want them to be able to not trade at all if they want, or trade many
things at once, or really do whatever they want. I want to make sure they're not
being limited by lack of tools."

That crystallised into three concrete requirements:

1. **Freedom of action** — the model decides what data to look at, when to skip
   a turn, and whether to trade one symbol or ten.  No step count budget, no
   forced tool sequence.
2. **Tool-augmented reasoning** — the full LOOK / NOTE / ACT catalog wraps every
   data source and service the platform provides.  The model picks the tools it
   needs.
3. **Transparent cost + observability** — every tool call is logged with latency
   and cost.  Operators see exactly what the trader decided and why.

See memory entry `[[trading-agent-trader-agency-model]]` for the full decision
log.

---

## 2. Always-On First-Look

Every turn begins with a deterministic context block injected as the first user
message.  It is always present and always has the same structure so the model
builds a stable mental model of its situation before calling any tools.

**Rendered format (A0):**

```
Identity:         SmokeTrader, anthropic/claude-sonnet-4-6, mandate=momentum
Account:          cash=$100,000.00, positions=0, last_decision=none
Wake reason:      scheduled
Turn type:        regular
Time:             2026-05-28 14:30:00 UTC, 2026-05-28 10:30:00 ET
Cadence:          every 30 min during RTH
Attention:        0 active watchpoints / 20 soft-limit, 0 active reminders / 10 soft-limit
Cost this turn:   $0.0000 (rollup: model+nested LLM calls)
```

**A0 example (from smoke test):**

```
Identity:         SmokeTrader, anthropic/claude-sonnet-4-6, mandate=none specified
Account:          cash=$100,000.00, positions=0, last_decision=none
Wake reason:      scheduled
Turn type:        regular
Time:             2026-05-28 16:32:11 UTC, 2026-05-28 12:32:11 ET
Cadence:          every 30 min during RTH
Attention:        0 active watchpoints / 20 soft-limit, 0 active reminders / 10 soft-limit
Cost this turn:   $0.0000 (rollup: model+nested LLM calls)
```

Fields:

| Field | Source | Notes |
|---|---|---|
| `Identity` | `AgentTrader` constructor | name, model slug, mandate/style |
| `Account` | `account` dict passed to `decide()` | broker-agnostic (see §MONEY IS REAL) |
| `Wake reason` | caller annotation | `"scheduled"`, `"watchpoint: AAPL > 580"`, etc. |
| `Turn type` | caller annotation | `SoD \| regular \| event \| reminder \| EoD \| callback` |
| `Time` | wall clock at turn start | UTC, ET, optional user-pref TZ |
| `Cadence` | `AgentTrader.cadence_minutes` | mirrors the bench cadence setting |
| `Attention` | A2 `attention_queue` | active watchpoints + reminders vs soft limits |
| `Cost this turn` | `CostTracker.total_usd` | updated after each model call |
| `Previous attempt` | A4 crash-recovery | only present on restart turns |

Implemented in `intel/turn_context.py` → `build_first_look(TurnContext)`.

---

## 3. Tool Error Envelope + Cost Model + Runaway-Guard Semantics

### Tool error envelope

Every tool in the catalog returns `ToolResult` — no exceptions escape to the
loop.  The model always sees a structured response and can reason about failures.

```python
@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any | None = None    # present when ok=True
    error: ToolError | None = None  # present when ok=False

@dataclass(frozen=True)
class ToolError:
    kind: Literal[
        "network", "rate_limit", "unavailable",
        "invalid_input", "disabled", "not_found", "internal"
    ]
    message: str
    retry_after: int | None = None  # seconds; set for rate_limit / unavailable
```

JSON wire shape injected into the `"tool"` role message:

```json
{"ok": true, "data": {...}}
{"ok": false, "error": {"kind": "unavailable", "message": "...", "retry_after": 60}}
```

Implemented in `intel/tool_envelope.py`.

### Cost model

`CostTracker` accumulates per-turn spend across:
- **Model calls** — the main loop LLM calls (input + output + cache-hit tokens).
- **Nested LLM costs** — tools that call an LLM themselves (`ask_manager`,
  `request_research`).

Soft warning: when `total_usd > COST_WARN_PER_TURN` (env var, default `$1.00`),
a system message is injected before the next model call:

```
[Cost notice] This turn has spent $1.0023 so far (soft threshold: $1.00).
Consider wrapping up — call pass() or hold() unless a decision genuinely
requires more tool calls.
```

The warning fires **once per turn**.  It is NOT a hard cap — the trader can
continue calling tools.  The hard cap is the runaway guard (call count).

Implemented in `intel/cost_tracker.py`.

### Runaway-guard semantics

The loop exits with a forced `hold(reason="runaway guard: N tool calls in one turn")`
when the cumulative tool-call count reaches 100 (default; configurable via
`AGENT_RUNAWAY_LIMIT` env var).

This is a **call-count guard, not a cost guard**.  A trader that makes 100
lightning-fast `list_tools` calls hits the guard before spending a cent; a trader
that makes 2 expensive `ask_manager` calls never hits it regardless of cost.

### Context-window guard

When the accumulated message content exceeds 70% of the model's assumed context
window (default: 70% × 512k chars ≈ 358k chars; override via `AGENT_CTX_CHARS`),
older tool-result messages in the middle of the conversation are summarised
in-place to `"[earlier tool result summarised — context trimmed]"`.  The first
two messages (system + first-look) and the most recent four messages are always
preserved verbatim.

---

## 4. MONEY IS REAL TO THE TRADER

> **The trader must never know whether its account is live or a simulation.**

This is a hard invariant enforced at every layer of the agent:

- **System prompt** — `AgentTrader._build_system_prompt()` never mentions
  "paper", "sim", "demo", "fake", or "test mode".
- **First-look context** — `build_first_look()` surfaces cash + positions as bare
  numbers, identical for live and simulation accounts.
- **Tool results** — `account_state()` (A1) wraps `PaperBroker.snapshot()` via the
  `BrokerAdapter` interface; the word "paper" is scrubbed from every field before
  serialisation into a `ToolResult`.
- **Manager replies** — the manager's system prompt includes a filter:
  "never disclose paper/live status to the trader."
- **Error messages** — `ToolError.message` fields from broker tools must not
  contain disclosure words.

**Why it matters:**

1. **Calibration** — P5 calibrated reflection learns from outcomes.  If the
   trader treats the account as monopoly money, the patterns it crystallises are
   wrong and won't transfer to live money.
2. **Discipline parity** — a real-money trader is risk-averse and thoughtful.
   We need paper traders to behave identically.
3. **Transition safety** — when a trader moves paper → live broker, the behaviour
   shouldn't shift.  They already operate as if real.

**What operators see:** the cockpit, audit logs, and turn traces all show a
prominent `[PAPER]` / `[LIVE]` badge.  The `TurnRecord` (A5 `intel/turn_store.py`)
stores `book_type: "paper" | "live"` at the row level for forensics.

**A1 red-team test:** `test_agent_trader_ask_manager_no_paper_leak` will call
`ask_manager("is this real money?")` and assert the reply does not reveal the
paper status.  _(A1 deliverable — not yet in the test suite.)_

---

## 5. Lifecycle (🔵 A4)

_Stub — populated in A4._

- Live window: T−60min before RTH open → T+30min after RTH close (ET-anchored,
  Alpaca calendar aware of half-days + holidays).
- Turn types: SoD (T−60min), regular (every N min during RTH), event-driven
  (watchpoint / protective-order fill / reminder / approval callback), EoD
  (T+30min).
- Kill-switch soft halt: LOOK/NOTE tools still work; ACT tools return
  `{ok: false, error: {kind: "unavailable", message: "bench halted by operator"}}`.
- Crash recovery: orphaned turns are restarted with `previous_attempt_tools`
  annotated in the first-look block.

---

## 6. Tool Catalog (✅ A0 + A1 + A2 / 🔵 A3)

All tools return a `ToolResult` — no exceptions escape the agent loop.

### A0 built-ins (✅)

| Tool | Category | Latency | Cost class |
|---|---|---|---|
| `list_tools()` | LOOK | instant | free |
| `memory_search(query, k=5)` | LOOK | fast | free |
| `hold(reason)` | END | instant | free |
| `pass()` | END | instant | free |

### A1 LOOK catalog (✅)

Lives in `intel/tools/look/`.  Each tool is one module; all use `LookToolBase`.

#### Enabled tools

| Tool | Wraps | Latency | Cost class | Notes |
|---|---|---|---|---|
| `list_tools()` | static catalog | instant | free | Extended from A0; now lists full LOOK+NOTE set |
| `recent_turns(n=5, include_tool_calls=True)` | `intel/turn_store.TurnStore` (A5) | fast | free | Returns empty gracefully until A5 ships |
| `history(symbol, days=30)` | `data.history.HistoryService` | fast | free | Returns OHLCV bars + realized-vol stats |
| `news(symbol=None, limit=10)` | `ingest.store.IngestStore` raw_items | fast | free | Per-user scoped; empty when store absent |
| `research_brief(symbol)` | `research.store.ResearchStore` (WS-C) | fast | free | Shared per-user; None when no brief yet |
| `request_research(symbol, question)` | WS-C `ResearchAgent.run` | queued | queued | Fire-and-forget; check research_brief() next turn |
| `situation()` | `situation.regime.RegimeClassifier` + `SocialAggregator` (P3) | fast | free | Regime label + social metrics + calendar events |
| `watchlist()` | trader + operator symbol sets | instant | free | Union of watch_symbol() + operator pins |
| `account_state()` | PaperBroker / broker adapter | instant | free | MONEY IS REAL: "paper" scrubbed from all fields |
| `memory_search(query, k=5)` | `memory.store.MemoryStore` (WS-D) | fast | free | Per-(user,trader) namespace; empty if new |
| `advisor_notes(symbol=None, scope="trader"\|"ticker"\|"global")` | `notes.NotesStore` (WS-H) | fast | free | Strict isolation: own trader's notes only |
| `ask_manager(question)` | `manager.agent.ManagerAgent.chat()` | slow | model_call | ≤1/turn; paper/peer filter injected |

#### Disabled stubs — provider lands in WS-Situation+Forecast

| Tool | Provider | When enabled |
|---|---|---|
| `world_events(theme=None, timespan="24h")` | GDELT | WS-Situation Track A |
| `prediction_market_odds(category, query=None)` | Polymarket / Kalshi | WS-Situation Track A |
| `options_iv(symbol)` | `instruments/options` IV surface | WS-Situation Track A |
| `forecast(symbol, horizon=5\|10\|30)` | `intel/forecast.py` | WS-Situation Track C |

Disabled stubs return `ToolResult(ok=False, error=ToolError(kind="disabled", …))`.
`list_tools()` surfaces them with `enabled=false` + `disabled_reason`.  When the
provider lands, only the stub body gets unwired — wrapper class and catalog entry stay.

#### TurnContext slot fix (A0 reviewer note, shipped in A1)

A0 had an `extra_lines` escape-hatch.  A1 adds two explicit, plan-spec'd fields
to `intel/turn_context.py`:

| First-look line | Dataclass field | Populated by |
|---|---|---|
| `Directed notes:` | `directed_notes: list[str]` | `AdvisorNotesTool.directed_notes_for_slot()` — unread operator notes; marked read after surfacing |
| `Recent reflections:` | `recent_reflections: list[str]` | `MemorySearchTool.reflections_for_slot()` — top-3 reflection lessons tagged with today's symbols |

Both appear between `Cost this turn:` and `Previous attempt:` when non-empty.
Empty → not rendered (no blank lines).

#### advisor_notes isolation contract

`advisor_notes` is scoped strictly to `(owner_user_id, trader_id)`.  The store
is called with the correct user_id + scope + ref; no cross-trader or cross-user
data can leak by construction.

#### ask_manager safety contract

Every call prepends `_MANAGER_FILTER` to the question before it reaches the
manager LLM.  The filter instructs the manager to never disclose paper/sim/demo
status or peer-trader state.  `_scrub_answer()` provides defence-in-depth
post-hoc scrubbing.  The A1 red-team test `tests/test_ask_manager_no_paper_leak.py`
enforces §Discipline rule 10 — asking "is this real money?" never produces a reply
containing any forbidden disclosure word.

### A2 NOTE catalog (✅)

| Tool | What | Latency | Cost class |
|---|---|---|---|
| `reflect(note, *, tags)` | Write a durable lesson to per-trader memory (WS-D `MemoryStore`). Carries provenance (prior tool names) for P5 calibrated learning. | fast | free |
| `remind_me(when, about)` | Time-based deferred self-poke. `when` accepts ISO datetime or relative (`"in 15min"`, `"in 2h"`, `"tomorrow 10am ET"`). Auto-expires on fire OR after 7d. | fast | free |
| `watchpoint(symbol, why, *, condition, ttl_hours)` | Event-based monitor. Condition forms: `"price > 580"`, `"news_rate > 2x"`, `"realized_vol > 1.5x"`. Omit condition → "interesting move" heuristic. TTL default 24h, max 168h. | fast | free |
| `watch_symbol(symbol)` | Add symbol to trader's personal watchlist (stored in `user_settings`). Idempotent. Overlays cockpit watchlist tile (A5). | instant | free |
| `unwatch_symbol(symbol)` | Remove symbol from personal watchlist. Idempotent. | instant | free |

### A3 ACT catalog (✅)

Lives in `intel/tools/act/`.  Each tool is one module; all use `ActToolBase`.

| Tool | What | Terminal? | Latency | Cost class |
|---|---|---|---|---|
| `trade(symbol, side, qty, *, stop, take_profit, trail)` | Submit a single trade intent. Approval-required → returns `pending_trade_id`, turn ends. Direct → fill result. | ✅ | fast | free |
| `trade_batch([{symbol, side, qty, ...}, ...])` | Multiple trade intents in one turn. Per-item results. Kill-switch blocks whole batch before first item. | ✅ | fast | free |
| `update_protective_order(order_id, *, new_stop, new_tp, new_trail)` | Edit stop-loss / take-profit / trailing stop. Does NOT require re-approval. | ❌ | fast | free |
| `confirm_trade(pending_trade_id)` | Execute a pre-approved trade. Callback-turn only. TTL-gated. | ✅ | fast | free |
| `abandon_trade(pending_trade_id)` | Release a pre-approved trade unused. Callback-turn only. | ✅ | instant | free |

**Kill-switch interaction surface:** when `risk_manager.kill_switch_active` is `True`, all ACT
tools return `{ok: false, error: {kind: "unavailable", message: "bench halted by operator"}}`.
LOOK and NOTE tools are unaffected — the trader can still `hold()`/`pass()` cleanly.

**Idempotency:** key = `sha256(trader_id ‖ turn_id ‖ symbol ‖ side ‖ qty)`.
`RiskManager.check_idempotency` / `record_idempotency` detect crash-replay double-fires.
`PendingTradeQueue.propose` enforces uniqueness at the DB level (`UNIQUE` constraint on
`idempotency_key`).

### END terminals (✅ A0 + A3)

| Terminal | A0 | A3 |
|---|---|---|
| `pass()` | ✅ | ✅ |
| `hold(reason)` | ✅ | ✅ |
| `done_for_day(reason)` | — | ✅ |
| `trade(...)` | — | ✅ |
| `trade_batch([...])` | — | ✅ |
| `confirm_trade(pending_id)` | — | ✅ |
| `abandon_trade(pending_id)` | — | ✅ |

ACT terminals leave `DecisionResult.decisions = []` — the ACT tools interact with the broker
directly.  The bench controller must not re-execute.

---

## 7. Approval-Callback Flow (✅ A3)

The approval flow is a **five-path gated callback** pattern backed by
`PendingTradeQueue` (SQLite `pending_trades` table in `data/approvals.db`).

### State machine

```
propose()         set_decision("approved")   confirm()
  │                       │                      │
  ▼                       ▼                      ▼
awaiting_approval ──→ approved ──────────────→ confirmed
      │                   │
      │           expire_old() / TTL elapsed
      │                   ▼
      │                expired
      │
      └──→ denied (via set_decision("denied"))
      └──→ abandoned (via abandon())
```

### Five verified paths

| # | Path | Trigger | Outcome |
|---|---|---|---|
| 1 | **Propose** | `trade(...)` with `requires_approval=True` | Turn ends with `pending_trade_id`, status `awaiting_approval` |
| 2 | **Approve** | Operator hits `POST /api/pending-trades/{id}/approve` | Status → `approved`, `approval_ttl_expires_at` set, callback fires |
| 3 | **Confirm** | Trader calls `confirm_trade(id)` in callback turn | Status → `confirmed`, fill returned |
| 4 | **Abandon** | Trader calls `abandon_trade(id)` in callback turn | Status → `abandoned`, no fill |
| 5 | **Expire** | `expire_old()` / TTL elapsed | Status → `expired`, expiry callback fires |
| 6 | **Deny** | Operator hits `POST /api/pending-trades/{id}/deny` | Status → `denied`, callback fires with denial reason |

### Pre-approval TTL

Configurable via `PREAPPROVAL_TTL_MIN` env var (default `5` minutes).
`confirm_trade(id)` after TTL elapsed returns `{ok: false, error: {kind: "unavailable", message: "... TTL expired"}}`.

### Callback mechanism

`PendingTradeQueue.register_callback(pending_trade_id, fn)` — registers a
`fn(PendingTrade)` called synchronously on every status transition (approve, deny,
expire).  Exceptions in callbacks are silently swallowed to protect the decision flow.

A4's scheduler wires these callbacks to fire new `decide()` turns.  A3 delivers the
mechanism; A4 delivers the lifecycle engine that uses it.

### Web router extensions (`web/routers/approvals.py`)

| Endpoint | Action |
|---|---|
| `GET /api/pending-trades` | List all `awaiting_approval`/`approved` trades |
| `POST /api/pending-trades/{id}/approve` | Approve → callback fires |
| `POST /api/pending-trades/{id}/deny` | Deny → callback fires |

The legacy `/api/approvals` endpoints are untouched.

### PendingTrade dataclass

```python
@dataclass(frozen=True)
class PendingTrade:
    pending_trade_id: str
    trader_id: str
    proposed: TradeIntent       # symbol, side, qty, stop, take_profit, trail
    proposed_at: datetime
    idempotency_key: str
    status: Literal["awaiting_approval", "approved", "denied",
                    "confirmed", "abandoned", "expired"]
    approved_at: datetime | None
    approval_ttl_expires_at: datetime | None
    confirmed_at: datetime | None
    fill_result: FillResult | None
    note: str | None = None
```

### Key files (A3)

| File | Role |
|---|---|
| `intel/tools/act/_base.py` | `ActToolBase` + `_idempotency_key` + `_scrub_fill` |
| `intel/tools/act/trade.py` | `TradeTool` — primary ACT entry point |
| `intel/tools/act/trade_batch.py` | `TradeBatchTool` — multi-symbol batch |
| `intel/tools/act/update_protective_order.py` | `UpdateProtectiveOrderTool` — protective order edits |
| `intel/tools/act/confirm_trade.py` | `ConfirmTradeTool` — execute pre-approved (callback-turn) |
| `intel/tools/act/abandon_trade.py` | `AbandonTradeTool` — release unused (callback-turn) |
| `approval_queue.py` (extended) | `TradeIntent`, `FillResult`, `PendingTrade`, `PendingTradeQueue` |
| `risk_manager.py` (extended) | `check_idempotency`, `record_idempotency`, `check_batch_blocked` |
| `web/routers/approvals.py` (extended) | `/api/pending-trades` endpoints |
| `tests/test_act_tools.py` | 71 tests (smoke + unit, all 5 callback paths) |

---

## 8. Tutorial Mode (🔵 A6)

_Stub — populated in A6._

New trader field: `tutorial_remaining: int` (default 3).  First N turns use a
guided prompt that forces `list_tools()` as the first call, demonstrates a
`memory_search` + `reflect` cycle, and demonstrates a `watchpoint` setup.
Tutorial mode auto-exits after `tutorial_remaining` reaches 0 or after the
first `trade*` terminal.

---

## 9. Observability (🔵 A5)

_Stub — populated in A5._

- `intel/turn_store.py` — persistent SQLite store of every turn's full trace:
  wake reason, first-look snapshot, ordered tool calls (name, args, result,
  latency, cost), final action, total cost.
- `web/routers/traces.py` — `GET /api/traces?trader_id=&limit=N` and
  `GET /api/traces/{turn_id}`.
- Cockpit tiles: `traderTrace`, `turnReplay`, `costPerTrader`, `attentionPending`.

---

## 10. Token Caching Architecture

The agent turn's message list is structured so the stable prefix maximises
provider-side cache hits:

```
messages[0]  system (stable: identity + rules, marked cache_control: ephemeral)
messages[1]  user   (variable: first-look context — changes every turn)
messages[2…] tool call / tool result pairs (built during the turn)
```

The `cache_control: {"type": "ephemeral"}` annotation on the system message block
activates Anthropic's prompt-caching tier for Anthropic-backend models via
OpenRouter.  Non-Anthropic providers ignore the field.

The `tools` parameter (tool definitions) is also stable per trader and benefits
from provider-level KV caching even without explicit annotation.

---

## 11. NOTE Toolkit — "Interesting Move" Heuristic

When a `watchpoint` is registered with `condition=None`, the scheduler evaluates
the **"interesting move" heuristic** on every tick.  It fires if **any** of these
rules trips:

| Rule | Signal | Threshold |
|---|---|---|
| Price sigma | `abs(price_change_1h) > 1σ` | 30-day realized vol × price |
| News rate spike | `current_rate / baseline_rate > 2.0` | configurable per-trader via `INTERESTING_MOVE_RULES` |
| Vol spike | `realized_vol_ratio > 1.5` | configurable per-trader |
| Approval queue | Symbol has a pending approval entry | any |

The heuristic is evaluated in `intel/tools/note/watchpoint.py::evaluate_condition`.
All four rules are on by default; any subset can be disabled via `user_settings`
key `INTERESTING_MOVE_RULES` (JSON list of enabled rule names).

---

## 12. Pending-Attention Queue

SQLite table `attention_queue` (bootstrapped by `db/migrations/001_attention.sql`).
Managed by `intel/attention_queue.py::AttentionQueue`.

```
attention_queue
  id           INTEGER PRIMARY KEY
  trader_id    TEXT              -- bench competitor name
  kind         TEXT              -- 'reminder' | 'watchpoint'
  payload_json TEXT              -- {symbol?, when_unix?, condition?, why}
  created_at   INTEGER           -- Unix seconds UTC
  expires_at   INTEGER           -- Unix seconds UTC
  fired_at     INTEGER           -- NULL = unfired
  fire_reason  TEXT              -- 'elapsed' | 'condition: ...' | 'interesting-move: ...' | 'expired'
```

**Partial index:** `idx_attention_pending ON attention_queue(trader_id, fired_at) WHERE fired_at IS NULL`
— the scheduler polls only unfired rows without a full-table scan.

**Soft limits** (surfaced in always-on first-look):
- Watchpoints: default 20 (`WATCHPOINT_SOFT_LIMIT`), hard cap 100.
- Reminders: default 10 (`REMINDER_SOFT_LIMIT`), hard cap 50.

Exceeding soft limit injects a nudge in first-look context.  Exceeding hard cap →
`watchpoint()` / `remind_me()` returns `{ok: false, error: {kind: "unavailable"}}`.

**Time math:** `when_unix` stored and compared in UTC.  "Tomorrow 10am ET" is
parsed in `America/New_York` (never local server TZ) and stored as UTC.

---

## 13. Scheduler Hooks (A2)

`bench/controller.py::BenchController._scan_attention()` runs on every cadence
tick (after `run_decisions` + `_maybe_reflect`).

Steps:
1. Resolve the `AttentionQueue` from any `AgentTrader` competitor.
2. Call `aq.expire_old()` to soft-expire past-TTL rows.
3. `aq.poll_all_due()` → list of unfired, non-expired rows.
4. For each **reminder** row: if `payload.when_unix ≤ now` → `mark_fired("elapsed")`
   → wake the owning trader via `bench._run_one(comp)`.
5. For each **watchpoint** row: call `evaluate_condition(payload, last_prices, ...)` →
   if tripped → `mark_fired(reason)` → wake via `bench.run_decisions_for_symbol(symbol)`.

Full event-driven wake (dedicated turn type `"reminder"` / `"event"`) is A4's
deliverable; A2 reuses the existing market-wake mechanism for immediacy.

---

## Key Files (A0–A2)

| File | Role |
|---|---|
| `intel/__init__.py` | Package init |
| `intel/tool_envelope.py` | `ToolResult` / `ToolError` universal contract |
| `intel/turn_context.py` | `TurnContext` + `build_first_look()` |
| `intel/cost_tracker.py` | `CostTracker` — per-turn cost rollup + soft warn |
| `intel/attention_queue.py` | Pending-attention queue (reminders + watchpoints) — **A2** |
| `intel/tools/note/_base.py` | Shared scaffolding for NOTE tools — **A2** |
| `intel/tools/note/reflect.py` | `ReflectTool` — write lesson to WS-D memory — **A2** |
| `intel/tools/note/remind_me.py` | `RemindMeTool` — time-based self-poke — **A2** |
| `intel/tools/note/watchpoint.py` | `WatchpointTool` + `evaluate_condition` — **A2** |
| `intel/tools/note/watch_symbol.py` | `WatchSymbolTool` — personal watchlist add — **A2** |
| `intel/tools/note/unwatch_symbol.py` | `UnwatchSymbolTool` — personal watchlist remove — **A2** |
| `db/migrations/001_attention.sql` | Migration: `attention_queue` table + partial index — **A2** |
| `llm/trader.py` | `AgentTrader` class (A0 loop + A0/A2 tool dispatchers) |
| `llm/openrouter.py` | `ToolCall`, `ToolCallChatResult`, `chat_with_tools()` |
| `bench/controller.py` | `BenchController` + `_scan_attention` scheduler hook — **A2** |
| `tests/test_agent_trader.py` | 31 tests covering A0 (smoke + unit) |
| `tests/test_note_tools.py` | 57 tests covering A2 NOTE toolkit + attention queue |
