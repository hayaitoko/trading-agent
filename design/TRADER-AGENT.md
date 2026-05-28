# Trader Agent — Design Reference

**Status:** WS-Agent A0 landed · A1–A6 in progress  
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

## 6. Tool Catalog (🔵 A1 / A2 / A3)

_Stub — populated incrementally per wave.  See plan §A1–A3 for full tables._

### A0 built-ins (✅)

| Tool | Category | Latency | Cost class |
|---|---|---|---|
| `list_tools()` | LOOK | instant | free |
| `memory_search(query, k=5)` | LOOK | fast | free |
| `hold(reason)` | END | instant | free |
| `pass()` | END | instant | free |

### A1 LOOK catalog (🔵)

`recent_turns`, `history`, `news`, `research_brief`, `request_research`,
`situation`, `world_events`, `prediction_market_odds`, `options_iv`, `forecast`,
`watchlist`, `account_state`, `ask_manager`

### A2 NOTE catalog (🔵)

`reflect`, `remind_me`, `watchpoint`, `watch_symbol`, `unwatch_symbol`

### A3 ACT catalog (🔵)

`trade`, `trade_batch`, `update_protective_order`, `confirm_trade`, `abandon_trade`

### END terminals (A0 partial, A3 complete)

| Terminal | A0 | A3 |
|---|---|---|
| `pass()` | ✅ | ✅ |
| `hold(reason)` | ✅ | ✅ |
| `done_for_day(reason)` | — | 🔵 |
| `trade(...)` | — | 🔵 |
| `confirm_trade(pending_id)` | — | 🔵 |
| `abandon_trade(pending_id)` | — | 🔵 |

---

## 7. Approval-Callback Flow (🔵 A3)

_Stub — populated in A3._

The approval flow is a two-step gated callback:

1. Trader calls `trade(symbol, side, qty)`.
2. If approval required → enqueue, return `{ok: true, data: {pending_trade_id, status: "awaiting_approval"}}`.  Trader's turn ends.
3. Human approves/denies via cockpit.  State change fires an event-driven
   callback turn for the trader.
4. On approval: `Wake reason: trade_id=X was approved at HH:MM, TTL 5min`.
5. Trader calls `confirm_trade(X)` to execute, or `abandon_trade(X)`, or any
   other tools before terminating.
6. On denial: callback is informational; trader reassesses.

Pre-approval TTL: configurable (`PREAPPROVAL_TTL_MIN=5`).  Expiry fires another
callback.

Idempotency key: `hash(trader_id, turn_id, symbol, side, qty)` — risk manager
rejects duplicate trade attempts within the same `turn_id`.

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

## Key Files (A0)

| File | Role |
|---|---|
| `intel/__init__.py` | Package init |
| `intel/tool_envelope.py` | `ToolResult` / `ToolError` universal contract |
| `intel/turn_context.py` | `TurnContext` + `build_first_look()` |
| `intel/cost_tracker.py` | `CostTracker` — per-turn cost rollup + soft warn |
| `llm/trader.py` | `AgentTrader` class (A0 loop + A0 built-in tools) |
| `llm/openrouter.py` | `ToolCall`, `ToolCallChatResult`, `chat_with_tools()` |
| `tests/test_agent_trader.py` | 31 tests covering A0 (smoke + unit) |
