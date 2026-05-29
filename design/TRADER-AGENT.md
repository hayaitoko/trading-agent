# Trader Agent — Design Reference

**Status:** WS-Agent A0 ✅ · A1 ✅ · A2 ✅ · A3 ✅ · A4 ✅ · A5 ✅ · A6 ✅ (complete) ·
Bench wiring ✅ (WS-Bench-Migration — `AgentTrader` is the bench's default trader)  
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

**Bench wiring (WS-Bench-Migration).** The reframe is now live end-to-end:
`bench/controller.py::add_model()` instantiates `AgentTrader` (not the legacy
structured-output trader), and `BenchController` threads the shared agent
infrastructure — attention queue (NOTE tools), `PendingTradeQueue` (ACT approval
flow), `TurnStore` (A5 trace), and the WS-Situation providers — into every trader
it builds.  The bench mints each competitor's isolated `PaperBroker`;
`AgentTrader.bind_execution()` then attaches that broker + risk + approval queue so
the ACT tools settle trades on the very book the leaderboard values.  `serve.py`
constructs one set of this infrastructure per controller and attaches the trace +
pending-trade stores to `app.state` for the cockpit read routers.  The ET-anchored
lifecycle scheduler is opt-in via `TRADING_AGENT_SCHEDULER=1` (default off keeps the
bench on its plain cadence for any-time-of-day testing).

**`LLMTrader` retired (WS-Bench-Migration M2).** The legacy structured-output
trader is deleted from `llm/trader.py` (along with `_SYSTEM_PROMPT`). The A6
prompt-scrub of `LLMTrader._SYSTEM_PROMPT` (commit `8c2b9a8`, "managing a paper
account" → "managing a financial account") was a temporary bridge while the bench
still constructed the legacy class; M2 completes the move, so that scrub is now
moot — the class is gone and the bench builds `AgentTrader` exclusively.
`StrategyTrader` (the deterministic baseline) and the `Trader` protocol remain.
The only money-is-real-relevant prompt is now `AgentTrader._build_system_prompt()`,
which never names paper/sim/demo/fake.

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

## 5. Lifecycle (✅ A4)

The lifecycle engine ET-anchors every trader's decision loop to the Alpaca
market calendar.  Internal scheduling is UTC; market window boundaries are
converted to UTC from America/New_York via the Alpaca calendar API (half-days,
holidays, DST all handled).  Server local TZ is never used.

### State diagram

```
                  T-60min before NYSE open
  DORMANT ─────────────────────────────────→ LIVE (SoD turn fires)
                                               │
                            regular cadence    │
                         every N min during RTH│
                       event/reminder/callback │
                                               ▼
  DORMANT ←─────────────────────────────────── LIVE
                  T+30min after NYSE close
                         (EoD turn fires)

  DORMANT → DORMANT (research agent still runs; AH fills queued for SoD)
  LIVE → DORMANT (done_for_day() skips remaining regular turns)
```

### Live window

```
  sod_utc  = open_utc  − 60 min   (configurable: SOD_LEAD_MINUTES env var)
  eod_utc  = close_utc + 30 min   (configurable: EOD_TRAIL_MINUTES env var)
  open/close from Alpaca calendar (half-days + holidays respected)
```

Dormant outside `sod_utc → eod_utc`.  Research agent continues independently.

**Pre-SoD research hydration (deferred).**  Plan §A4 calls for the research
agent's overnight batch to be hydrated into the `research_brief()` cache *before*
SoD fires.  An explicit scheduler-triggered hydration handshake is **not**
implemented: the lifecycle engine holds no research-service reference, and
triggering a (cost-gated) research pass from the scheduler is out of A4's scope.
Functionally the trader still gets overnight research on the SoD turn — the
research agent runs on its own background schedule, and the SoD special-prompt
guidance directs the trader to call `research_brief()` / `news` / `situation` to
absorb it.  The guaranteed-fresh pre-SoD trigger is deferred to the Situation+
Forecast Track C integration (which wires providers/research as scheduled passes).

### Turn types

| Turn type | When | Prompt note |
|---|---|---|
| `SoD` | At T-60min before open | Absorb overnight intel; seed watchpoints; reset done_for_day |
| `regular` | Every `cadence_minutes` during live window | Standard decision loop |
| `event` | Watchpoint trip / AH fill / reminder / market move | Carries event description in wake_reason |
| `callback` | Approval-state change (approve / deny / expire) | Carries pending_trade_id + status in wake_reason |
| `EoD` | At T+30min after close | Reflect; lock overnight protections; no new positions (default) |
| `reminder` | `remind_me()` elapsed (via attention queue) | Forwarded as event |
| `tutorial` | First N turns for new traders (A6) | Tutorial prompt injected |

### Turn-type special-prompt guidance

`SoD` and `EoD` turns carry turn-type-conditional guidance text so the trader
knows what that turn is *for* (the "Prompt note" column above):

- **SoD**: absorb overnight developments (news / `research_brief` / `situation`),
  set posture for the day, seed watchpoints; no rush to trade before the open.
- **EoD**: reflect (`reflect`), review and lock protective orders, queue
  tomorrow's watchpoints/reminders; do not open new positions (default-strict).

This guidance is injected into the **per-turn first-look** (the variable
user-message suffix via `TurnContext.extra_lines` → `build_first_look`), **not**
the cached system prefix — so the stable system prompt stays cacheable across
every turn (Discipline #6 token caching).  See
`AgentTrader._turn_type_guidance()` in `llm/trader.py`.  Tutorial-turn guidance
is deferred to A6.

**EoD no-new-positions control (`_eod_no_new_positions`).**  The EoD
"do not open new positions" directive is gated on the `_eod_no_new_positions`
flag on the trader.  `MarketScheduler._fire_one` sets it `True` before firing an
EoD turn (default-strict) and resets it to `False` afterward;
`_turn_type_guidance` reads it when composing the EoD guidance.  This is
instruction-level (the trader retains agency, consistent with the kill-switch
soft-halt philosophy) — configuring EoD as non-strict (scheduler leaving the
flag `False`) omits the directive.  Hard broker/risk-layer enforcement of
no-new-positions is a possible future hardening, not implemented here.

### Per-trader lifecycle config

| Field | Default | Source |
|---|---|---|
| `cadence_minutes` | 30 | `BenchController.add_model(cadence_minutes=…)` |
| `extended_hours` | `False` | `BenchController.add_model(extended_hours=…)` |

`extended_hours=True` adds wake windows: 04:00–09:30 ET (pre-market) and
16:00–20:00 ET (after-hours).  The `trade()` tool must pass
`extended_hours=True` to the broker for fills during these windows.

### Kill-switch soft halt

When `risk_manager.kill_switch_active` is `True`:
- **ACT tools** (`trade`, `trade_batch`, `update_protective_order`, etc.) return
  `{ok: false, error: {kind: "unavailable", message: "bench halted by operator"}}`.
- **LOOK and NOTE tools** are unaffected — the trader can still gather
  information and call `hold()` or `pass()` cleanly.
- **Turns are NOT suppressed.**  SoD / EoD / regular / event / callback turns
  all continue to fire normally — the scheduler holds no risk-manager reference
  and does not gate turn firing on the kill switch.  The halt is *soft*: it
  blocks trading at the ACT tool layer, not thinking.  The trader still wakes,
  gathers information, and reaches `hold()` / `pass()`, preserving its state of
  mind for forensics.

Enforced at the ACT tool layer (A3).  The soft-halt behaviour — ACT tools
return `unavailable`, LOOK/NOTE tools work, `hold()`/`pass()` stay reachable —
is tested in `tests/test_lifecycle.py` (`test_smoke_4_*`).

### Crash recovery (carry-over A4-a)

**Turn-id reuse invariant:** when a turn is orphaned (started but did not reach
a terminal action), the crash-recovery turn fires with the **original turn_id**,
not a fresh UUID.

Why this matters:
- `idempotency_key = sha256(trader_id, turn_id, symbol, side, qty)` is
  identical in both the original and recovery turn.
- `PendingTradeQueue.propose` enforces `UNIQUE(idempotency_key)` at the DB
  level, catching approval-path double-fires.
- For direct-execution trades (no PendingTradeQueue), turn_id reuse relies on
  the risk manager's in-memory idempotency set.  A crash between "broker filled"
  and "turn completed" could theoretically double-fire a direct trade — accepted
  limitation; DB-UNIQUE dedup for direct trades is deferred as future hardening.

**Mechanism:**
1. At turn start, `AgentTrader.decide()` registers the turn in `OrphanTurnStore`
   (disk-backed JSON at `data/orphan_turns.json`).
2. At turn end (terminal reached), the turn is completed (removed from store).
3. On restart, `MarketScheduler.recover_orphans()` reads all orphan records.
4. For each orphan, the scheduler injects:
   - `trader._current_turn_id = orphan.turn_id` (REUSE the original UUID)
   - `trader._recovery_previous_attempt = orphan.tool_names_called`
5. A recovery turn fires with `turn_type="event"` and `wake_reason` describing
   the crash.  The first-look block shows:
   ```
   Previous attempt: history, account_state, news
   ```
   (Tool names only — no stale results.)
6. The orphan record is cleared after the recovery turn fires.

### After-hours protective-order fills

When a stop/TP/trail fires during dormancy:
1. The fill event is NOT merged into SoD.
2. `MarketScheduler.queue_ah_fill()` queues the event in `TraderLifecycle.pending_ah_fills`.
3. At the next SoD, the queued events fire as dedicated `"event"` turns
   **before** SoD fires, so the trader gets context-aware callbacks:
   ```
   Wake reason: Your stop on AAPL hit at 03:42 ET while you were dormant.
   ```

### Key files (A4)

| File | Role |
|---|---|
| `intel/lifecycle.py` | `AlpacaCalendar`, `LiveWindow`, `LifecycleEngine`, `OrphanTurnStore` |
| `bench/scheduler.py` | `MarketScheduler` — tick, fire_turns, callback wiring, crash recovery |
| `bench/controller.py` | `BenchController` — scheduler integration, per-trader cadence/extended_hours |
| `llm/trader.py` | `AgentTrader` — A4 scaffolding: turn type/wake reason injection, crash-recovery turn_id reuse, done_for_day flag |
| `tests/test_lifecycle.py` | 24 tests — 6 required smoke tests + 18 unit tests |

---

## 6. Tool Catalog (✅ A0 + A1 + A2 + A3)

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

## 8. Tutorial Mode (✅ A6)

New traders arrive with no reflections, no watchpoints, and no intuition
about which tools to use.  Tutorial mode gives them three structured turns
before they operate freely.

### New trader field: `tutorial_remaining`

```python
AgentTrader(..., tutorial_remaining=3)   # default — 3 guided turns
AgentTrader(..., tutorial_remaining=0)   # tutorial disabled (legacy/test contexts)
```

`tutorial_remaining` starts at the configured value and counts down.  The
original total is stored in `_tutorial_total` so guidance can render
"turn N of M" correctly.

### Activation and override

In `AgentTrader.decide()`, before building the turn context, the tutorial
override fires:

```python
if self.tutorial_remaining > 0:
    turn_type = "tutorial"
```

This unconditionally replaces whatever turn type the scheduler injected
(regular, SoD, EoD, etc.) — new traders always get tutorial guidance for
their first `tutorial_remaining` turns.

Existing code that tests SoD/EoD turn types must pass `tutorial_remaining=0`
to avoid the override.

### Tutorial prompt template (`prompts/tutorial.py`)

`tutorial_extra_lines(turn_number, total_turns)` returns extra_lines that
are appended to `TurnContext.extra_lines` via `_turn_type_guidance("tutorial")`.
Each turn has a specific focus:

| Turn | Focus | Step instruction |
|---|---|---|
| 1 | Tool discovery | Call `list_tools()` to see LOOK / NOTE / ACT / END catalog |
| 2 | Memory | Call `memory_search()` (empty) then `reflect()` to write first lesson |
| 3 | Watchpoints | Call `watchpoint()` to set a standing monitor on a universe symbol |
| 4+ | Free exploration | Generic "try any tools you haven't used" message |

Every rendered block begins with a header:

```
Tutorial — turn 2 of 3: 1 guided turn remains after this one.
STEP 2 — Explore memory.  Call memory_search(query='first session') ...
```

The last tutorial turn says "after this turn you decide freely — no more
guided prompts."

**MONEY IS REAL invariant:** `prompts/tutorial.py` is scanned by
`test_tutorial_mode.py::test_tutorial_templates_money_is_real` — none of
the strings "paper", "sim", "demo", "fake", or "test mode" appear anywhere
in the templates.

### Empty-state handling (`intel/turn_context.py`)

`TurnContext` has a new field `no_prior_context_hint: bool = False`.  When
`tutorial_remaining > 0`, `AgentTrader._build_turn_context()` sets it
`True`.  In `build_first_look()`, when `no_prior_context_hint=True` and
`recent_reflections` is empty, a help line renders in place of the silent
omission:

```
Context hint:     no prior context — you are new here; call list_tools() to see your full capability set.
```

If `recent_reflections` is non-empty (even for tutorial traders who already
reflected), the normal `Recent reflections:` line renders instead — the hint
disappears once the trader has memories.

### Auto-exit

At the end of each `decide()` call:

```python
if self.tutorial_remaining > 0:
    if terminal_action in {"trade", "trade_batch", "confirm_trade"}:
        self.tutorial_remaining = 0   # auto-exit on first trade
    else:
        self.tutorial_remaining = max(0, self.tutorial_remaining - 1)
```

- `pass()` / `hold()` / `done_for_day()` → decrement by 1.
- Any `trade*` terminal → zero immediately (the trader demonstrated real agency).
- After `tutorial_remaining == 0`: next `decide()` uses `turn_type="regular"`,
  no tutorial guidance in extra_lines, no context hint in first-look.

### Key files (A6)

| File | Role |
|---|---|
| `prompts/__init__.py` | Package init |
| `prompts/tutorial.py` | `tutorial_extra_lines(turn_number, total_turns)` — turn-specific guidance strings |
| `llm/trader.py` | `tutorial_remaining` / `_tutorial_total` fields; tutorial override in `decide()`; `_turn_type_guidance("tutorial")`; bookkeeping after terminal |
| `intel/turn_context.py` | `no_prior_context_hint` field + context-hint render in `build_first_look()` |
| `tests/test_tutorial_mode.py` | 29 tests — template units, MONEY IS REAL scan, TurnContext hint, remaining-decrement, auto-exit, normal-after-exhausted |

---

## 9. Observability (✅ A5)

Full turn-trace store + cockpit tiles.  Shipped in A5.

### Trace store schema (`intel/turn_store.py`)

SQLite table `turn_records` (bootstrapped idempotently at `TurnStore.__init__`):

```
turn_records
  turn_id                   TEXT PRIMARY KEY
  trader_id                 TEXT NOT NULL
  started_at                REAL (Unix UTC)
  ended_at                  REAL (NULL until turn completes — orphan detection)
  wake_reason               TEXT
  turn_type                 TEXT  -- SoD | regular | event | reminder | callback | EoD | tutorial
  book_type                 TEXT  -- 'paper' | 'live'  ← OPERATOR ONLY
  first_look_json           TEXT  -- full structured context block
  tool_calls_json           TEXT  -- ordered list of ToolCallRecord dicts
  final_action              TEXT
  final_action_args_json    TEXT
  total_cost_usd            REAL
  tokens_input              INTEGER
  tokens_output             INTEGER
  tokens_cached             INTEGER
  previous_attempt_turn_id  TEXT  -- set on crash-recovery turns
```

Index: `idx_turns_trader ON turn_records(trader_id, started_at DESC)` — fast
descending lookup per trader.

**`TurnRecord` dataclass** (`intel/turn_store.py`):

| Method | Path | Includes `book_type`? |
|---|---|---|
| `to_trader_dict()` | Trader-facing (A1 `recent_turns()` tool) | **No** — MONEY IS REAL |
| `to_operator_dict()` | Cockpit GET /api/traces/{id} | Yes + `book_badge` |
| `to_summary_dict()` | Cockpit GET /api/traces list | Yes + `book_badge` |

**`ToolCallRecord` dataclass:**

```python
@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    args: dict
    result: dict      # ToolResult.to_dict() — what the agent received
    latency_ms: int
    cost_usd: float   # non-zero for model_call / queued tools only
```

**`TurnStore` methods:**

| Method | Used by |
|---|---|
| `record(rec)` | AgentTrader loop (silently swallows failures) |
| `open_turn(...)` | AgentTrader at turn start — writes interrupted row for crash recovery |
| `close_turn(turn_id, ...)` | AgentTrader at turn end — finalises the row |
| `recent(trader_id, n)` | A1 `RecentTurnsTool` — returns `list[TurnRecord]` newest-first |
| `summaries(trader_id, limit)` | `GET /api/traces` router — operator summary list |
| `get(turn_id)` | `GET /api/traces/{id}` router — full record |
| `cost_rollup(trader_id)` | `GET /api/traces/cost` router — today/week/lifetime USD |
| `orphaned_turns()` | A4 crash-recovery scanner — `ended_at IS NULL` rows >5min old |

### Replay flow

1. Operator opens the `traderTrace` tile → polls `GET /api/traces?trader_id=<id>&limit=N`.
2. Each row shows: wake_reason, turn_type, final_action, total_cost_usd, tool_call_count, book_badge.
3. Operator clicks a row → JS calls `GET /api/traces/{turn_id}`.
4. Full record renders: first_look_snapshot (the exact context the trader saw), ordered
   tool calls (name → args → result → latency → cost), final action.
5. Operator can also open the `turnReplay` tile: paste any `turn_id`, load full trace in one view.

### Cost rollup

`GET /api/traces/cost?trader_id=<id>` returns `{today, week, lifetime}` in USD.
Aggregated by SQL `SUM(total_cost_usd)` over time windows (last 24h / 7d / all time).
The `costPerTrader` tile polls this endpoint on a 60-second refresh interval.

**No LLM is invoked by the cost endpoint — pure SQLite aggregation.**

### Per-trader watchlist overlay UX

The existing `TILES.watchlist` is extended (monkey-patched mount/refresh) by A5:

1. After `wlLoad()` resolves, `wlTraderOverlay()` fetches
   `GET /api/traces/attention?trader_id=<id>` for each trader in `ACCOUNTS`.
2. Symbols found in trader watchpoints are annotated on each watchlist row:
   `"AAPL — + Trader Eta, Trader Alpha"`.
3. A **Sync button** appears if any trader-watched symbols are not in the operator's
   watchlist: clicking adds them to `opts.symbols` and persists the layout.

**No LLM call from the overlay — reads `/api/traces/attention` only.**

### Pending-approvals tile extension

`TILES.approvals` is extended with `loadPendingTradeLineage()`, which reads
`GET /api/pending-trades` and surfaces approved-but-unconfirmed trades separately:

```
Approved — awaiting confirm_trade()
  AAPL · Buy 5 shares · trader: Eta · id: pt-abc · TTL: 2026-05-28 10:47:00 UTC
```

This shows the operator exactly which trades are in the `approved` state waiting
for the trader's callback turn to call `confirm_trade()`.

### Cockpit tiles (A5 — group: Observability)

| Tile | Description | Refresh |
|---|---|---|
| `traderTrace` | Last N turns for one trader, expandable per turn | 30s |
| `turnReplay` | Modal: paste turn_id → full first-look + tool calls + final action | on-demand |
| `costPerTrader` | Rolling spend (today / week / lifetime) for one trader | 60s |
| `attentionPending` | Active watchpoints + reminders for one trader, with prune hint | 30s |

All tiles are in the **Observability** group in the add-tile drawer.
No LLM is invoked by any tile render, mount, or refresh handler.

### MONEY IS REAL — observability enforcement

- `TurnRecord.to_trader_dict()` is the only serialiser used by the trader-facing path.
  It contains no `book_type`, `book_badge`, `paper`, `sim`, or `demo` strings.
- `TurnRecord.to_operator_dict()` and `to_summary_dict()` include `book_type` /
  `book_badge` — they are used **only** by the `/api/traces` router (operator path).
- `tests/test_turn_store.py::test_recent_turns_tool_money_is_real_grepped` and
  `test_turn_record_trader_dict_no_book_type` red-team this invariant by serialising to
  JSON and grepping for forbidden strings.

### Key files (A5)

| File | Role |
|---|---|
| `intel/turn_store.py` | `TurnStore` + `TurnRecord` + `ToolCallRecord` — store + dataclasses |
| `web/routers/traces.py` | `GET /api/traces`, `/api/traces/{id}`, `/api/traces/cost`, `/api/traces/attention` |
| `web/app.py` | `traces_router` wired into `_COCKPIT_ROUTERS` |
| `web/static/cockpit.html` | `TILES.traderTrace`, `.turnReplay`, `.costPerTrader`, `.attentionPending` + watchlist/approvals extensions |
| `tests/test_turn_store.py` | 33 tests — store, router, recent_turns integration, MONEY IS REAL red-team |

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

## 14. Bench Launch + Migration Verification (WS-Bench-Migration)

The bench now boots end-to-end on `AgentTrader` through the real entry point
(`scripts/serve.py::build_cockpit`, exposed as the `trading-agent-serve --cockpit`
factory). On construction `build_cockpit` attaches the shared agent infrastructure
to `app.state` (`turn_store`, `pending_trades`, `attention_queue`) and threads it +
an opt-in `MarketScheduler` into the `BenchController`; `add_model()` then builds an
`AgentTrader` and `bind_execution()` wires the bench's per-competitor `PaperBroker`
into its ACT toolkit.

**Verified (M3, mocked LLM + bench PaperBroker — no outbound network):**

- One `AgentTrader` competitor instantiates through `build_cockpit` → `add_model`
  without crash; its broker is the bench's tracked book.
- SoD / regular / EoD turns fire through `MarketScheduler.fire_turns()` and each
  writes a `TurnRecord` (+ `ToolCallRecord` rows) to the `TurnStore` with the
  correct `turn_type` (`tests/test_bench_migration_e2e.py`).
- Tool dispatch reaches the LOOK wrappers: with a provider wired + `SITUATION_*`
  flag on, `world_events` returns `ok=True`; with no provider/flag it returns
  `ToolError(kind="disabled")` — a structured "off", never a crash.
- `cost_tracker` accumulates a nonzero per-turn cost into the trace, with the
  input/output/cached token rollup.

Live boot trace (regular turn, scripted `world_events` → `trade`):

```
wake_reason : cadence tick (smoke)
turn_type   : regular
tool_calls  : [('world_events', 'err:disabled'), ('trade', 'ok')]
final_action: trade {'symbol': 'AAPL', 'side': 'BUY', 'qty': 1}
total_cost  : $0.0038
tokens      : {'input': 280, 'output': 70, 'cached': 128}
book        : trades=1 cash=$99808.50
```

**Live runway (post-push, needs real credentials).** The M3 verification used a
scripted `chat_with_tools` client and the bench's `PaperBroker`, so no real
OpenRouter or Alpaca calls were made. To validate the live path, boot with
`OPENROUTER_API_KEY` (ZDR) + `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` set and confirm:
a real model drives a turn, the ACT `trade` fills against **paper Alpaca**, the
`world_events`/`forecast` tools fire when `SITUATION_*` providers are wired, and
`/api/traces` renders the turn. Set `TRADING_AGENT_SCHEDULER=1` to exercise the
ET-anchored lifecycle (otherwise the bench runs its plain cadence).

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
| `intel/lifecycle.py` | `AlpacaCalendar`, `LiveWindow`, `LifecycleEngine`, `OrphanTurnStore` — **A4** |
| `bench/scheduler.py` | `MarketScheduler` — tick, fire_turns, callback wiring, orphan recovery — **A4** |
| `tests/test_lifecycle.py` | 24 tests covering A4 lifecycle + scheduler (6 smoke + 18 unit) |
