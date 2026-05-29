"""Bench competitors: a uniform ``Trader`` interface with two implementations.

* :class:`AgentTrader` — the bench's default trader: a ReAct-style tool-calling
  agent (WS-Agent A0–A6).  Runs a multi-step decision loop: build first-look
  context → call model with tools → execute tools → repeat until a terminal
  action or runaway guard.
  Terminals: ``hold(reason)``, ``pass()``, ``done_for_day(reason)``.
  NOTE tools: ``reflect``, ``remind_me``, ``watchpoint``,
  ``watch_symbol``, ``unwatch_symbol``.
  ACT tools (when a broker is bound): ``trade``, ``trade_batch``,
  ``update_protective_order``, ``confirm_trade``, ``abandon_trade``.

* :class:`StrategyTrader` — wraps any deterministic :class:`Strategy` (e.g.
  mean-reversion) so it can compete in the same bench as a baseline.

Both expose ``observe(bar)`` and ``decide(account) -> DecisionResult`` so they
are drop-in substitutes from the bench/controller's perspective.

The legacy structured-output ``LLMTrader`` was retired in WS-Bench-Migration M2
(the A6 prompt-scrub was a temporary bridge; this migration completed the move).
The bench now instantiates ``AgentTrader`` exclusively — see
``design/TRADER-AGENT.md`` §1.

``DecisionResult.decisions`` stays an empty list for AgentTrader — the ACT tools
interact with the broker directly, so the bench controller does not re-execute.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..intel.cost_tracker import CostTracker
from ..intel.tool_envelope import ToolError, ToolResult
from ..intel.turn_context import TurnContext, TurnType, build_first_look
from ..intel.turn_store import ToolCallRecord
from .openrouter import OpenRouterError, ToolCall, ToolCallChatResult

if TYPE_CHECKING:
    from ..strategy import Strategy

# WS-A default: how many private memory lessons to recall per decision (the
# memory_search tool's default k). Kept small — recall embeds the query per turn.
_MEMORY_RECALL_K = 5


@dataclass
class TradeDecision:
    symbol: str
    action: str  # BUY | SELL | HOLD
    quantity: float
    reason: str = ""


@dataclass
class DecisionResult:
    decisions: list[TradeDecision] = field(default_factory=list)
    comment: str = ""
    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class Trader(Protocol):
    name: str

    def observe(self, bar: dict[str, Any]) -> None: ...
    def decide(self, account: dict[str, Any]) -> DecisionResult: ...


def decision_to_signal(d: TradeDecision) -> dict[str, Any] | None:
    """Map a decision to the canonical strategy-signal dict, or None for HOLD."""
    action = d.action.upper()
    if action == "HOLD" or d.quantity <= 0:
        return None
    side = "BUY" if action == "BUY" else "SELL"
    return {
        "asset": d.symbol,
        "side": side,
        "type": "market",
        "amount": float(d.quantity),
        "reason": d.reason,
    }


# --- Strategy baseline trader ----------------------------------------------


class StrategyTrader:
    """Adapt a deterministic :class:`Strategy` into a bench competitor.

    The wrapped strategy emits a signal per bar (LONG/SHORT/NEUTRAL); we hold the
    most recent non-NEUTRAL signal and surface it on the next ``decide`` tick so
    every competitor trades on the same cadence.
    """

    def __init__(self, strategy: Strategy, *, name: str, default_quantity: float = 1.0) -> None:
        self.strategy = strategy
        self.name = name
        self.default_quantity = default_quantity
        self._pending: TradeDecision | None = None

    def observe(self, bar: dict[str, Any]) -> None:
        signal = self.strategy.on_data(bar)
        side = str(signal.get("side", "NEUTRAL")).upper()
        if side in ("LONG", "BUY"):
            action = "BUY"
        elif side in ("SHORT", "SELL"):
            action = "SELL"
        else:
            return
        self._pending = TradeDecision(
            symbol=str(signal.get("asset", bar.get("symbol", ""))),
            action=action,
            quantity=float(signal.get("amount", self.default_quantity) or self.default_quantity),
            reason=str(signal.get("reason", "strategy signal")),
        )

    def decide(self, account: dict[str, Any]) -> DecisionResult:
        if self._pending is None:
            return DecisionResult(comment="no signal")
        decision, self._pending = self._pending, None
        return DecisionResult(decisions=[decision], comment="strategy signal")


# ---------------------------------------------------------------------------
# AgentTrader — WS-Agent A0: ReAct-style tool-calling foundation
# ---------------------------------------------------------------------------

# Runaway guard: maximum tool-invocation calls per turn before forced hold.
_RUNAWAY_LIMIT = int(os.environ.get("AGENT_RUNAWAY_LIMIT", 100))

# Context-window guard: when accumulated message content (rough char count) exceeds
# this fraction of the model's assumed context, summarise older tool results.
_CTX_WARN_FRAC = 0.70
# Assumed context in chars (4 chars ≈ 1 token; 128k tokens = 512k chars).
_CTX_CHARS = int(os.environ.get("AGENT_CTX_CHARS", 512_000))
_CTX_WARN_CHARS = int(_CTX_CHARS * _CTX_WARN_FRAC)

# Terminal tool names — when the model calls any of these the loop exits.
# A0: hold, pass, done_for_day (done_for_day wired in A3 tool defs)
# A3: trade, trade_batch, confirm_trade, abandon_trade
_TERMINALS: frozenset[str] = frozenset(
    {
        "pass",
        "hold",
        "done_for_day",
        "trade",
        "trade_batch",
        "confirm_trade",
        "abandon_trade",
    }
)


class AgentTrader:
    """ReAct-style tool-calling trader agent (WS-Agent A0 foundation).

    This is the bench's canonical :class:`Trader` (WS-Bench-Migration): it
    satisfies the runtime-checkable ``Trader`` protocol (``name`` + ``observe``
    + ``decide``) so :class:`~trading_agent.bench.bench.Bench` and the
    controller hold it anywhere they once held the legacy structured-output
    trader.  Every terminal turn returns ``DecisionResult(decisions=[])`` — the
    ACT tools settle against the broker directly, so the bench does not
    re-execute decisions (see :meth:`_to_decision_result`); the bench's
    ``_run_one`` accounting handles the empty-decisions path as a clean no-op.

    The multi-step decision loop:

      1. Build always-on first-look context (:mod:`intel.turn_context`).
      2. Call the model with tool definitions (OpenAI-compatible via OpenRouter).
      3. On ``tool_calls`` response: execute each tool, append result to the
         message list, repeat.
      4. Exit on: terminal tool call (``hold`` / ``pass``) **or** model returns
         plain text (implicit hold) **or** runaway guard (100 calls) **or**
         context-window guard (summarise-and-trim).

    A0 built-in tools: ``list_tools``, ``memory_search``, ``pass``, ``hold``.
    A2 NOTE tools: ``reflect``, ``remind_me``, ``watchpoint``, ``watch_symbol``,
    ``unwatch_symbol``.
    Later waves add the full LOOK/ACT catalogs.

    **MONEY IS REAL invariant:** nothing in the system prompt, first-look
    context, or any tool result may mention ``"paper"``, ``"sim"``, ``"demo"``,
    or ``"fake"``.  The broker abstraction hides live vs. paper status; only
    operator-facing surfaces (cockpit, audit log) badge it.

    The ``decide()`` return is a :class:`DecisionResult` with an empty
    ``decisions`` list in A0 (no ACT tools yet) and ``comment`` set to the
    terminal reason.  The bench/controller sees this as a hold — correct
    behaviour until A3 adds trade execution.

    Failure modes:
    - ``OpenRouterError`` from the model call → ``DecisionResult(error=...)``
    - Malformed tool-call JSON from model → tool dispatched with empty args
    - Tool raises unexpectedly → wrapped in ``ToolResult(ok=False, ...)``
    """

    RUNAWAY_LIMIT: int = _RUNAWAY_LIMIT
    CTX_WARN_CHARS: int = _CTX_WARN_CHARS

    def __init__(
        self,
        model: str,
        client: Any,  # must expose chat_with_tools(); duck-typed to avoid cycle
        *,
        symbols: list[str],
        name: str | None = None,
        style: str | None = None,
        cadence_minutes: int = 30,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        # WS-D memory (optional; A1 wires fully)
        memory: Any = None,
        owner_user_id: str | None = None,
        memory_k: int = 5,
        # A2: attention queue + per-user settings (both optional)
        attention_queue: Any = None,
        settings_store: Any = None,
        # A3: ACT toolkit dependencies (all optional; absent → hold behaviour)
        broker: Any = None,
        risk_manager: Any = None,
        pending_trade_queue: Any = None,
        requires_approval: bool = False,
        # A6 tutorial mode: first N turns use guided prompts; 0 = disabled.
        tutorial_remaining: int = 3,
        # C0: Situation Track A providers (all optional; absent → tool returns disabled)
        gdelt_provider: Any = None,
        pm_provider: Any = None,
        chain_provider: Any = None,
        spot_prices: dict[str, float] | None = None,
        # A5 observability: persistent turn-trace store (optional; None → no trace
        # write, agent still runs).  Populated by the bench controller.
        turn_store: Any = None,
    ) -> None:
        self.model = model
        self.client = client
        self.symbols = list(symbols)
        self.name = name or model
        self.style = style
        self.cadence_minutes = cadence_minutes
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory = memory
        self.owner_user_id = owner_user_id
        self.memory_k = memory_k
        # A2: attention queue (AttentionQueue instance, may be None → degrade)
        self.attention_queue = attention_queue
        self.settings_store = settings_store
        # A3: ACT dependencies
        self.broker = broker
        self.risk_manager = risk_manager
        self.pending_trade_queue = pending_trade_queue
        self.requires_approval = requires_approval
        # A6 tutorial: remaining guided turns (decremented after each tutorial turn;
        # zeroed immediately on first trade* terminal so auto-exit works).
        self.tutorial_remaining: int = max(0, tutorial_remaining)
        # C0: Situation Track A providers
        self._gdelt_provider = gdelt_provider
        self._pm_provider = pm_provider
        self._chain_provider = chain_provider
        self._spot_prices: dict[str, float] = dict(spot_prices or {})
        # A5 observability: turn-trace store (None → trace writes are skipped).
        self._turn_store = turn_store
        # Store original total so tutorial_extra_lines() can compute "turn N of M".
        self._tutorial_total: int = self.tutorial_remaining
        # Price bar buffer (rolling per-symbol window fed by observe())
        self._bars: dict[str, deque[dict[str, Any]]] = {
            s: deque(maxlen=30) for s in self.symbols
        }
        # Stable system message content (built once at construction)
        self._stable_system_content: str = self._build_system_prompt()
        # Per-turn tool-call name accumulator (used by reflect provenance)
        self._turn_tool_names: list[str] = []
        # Per-turn UUID — scopes idempotency keys so crash-replay is safe.
        self._current_turn_id: str = ""
        # A4 crash-recovery scaffolding:
        # When the scheduler injects a previous-attempt annotation (carry-over A4-a),
        # it sets _recovery_previous_attempt to the tool names from the orphaned turn.
        # decide() reads this once and clears it so it only fires on one turn.
        self._recovery_previous_attempt: list[str] = []
        # A4 scheduler hooks: turn type and wake reason injected by MarketScheduler
        # before _run_one() is called.  Cleared to defaults after each turn.
        self._current_turn_type: str = "regular"
        self._current_wake_reason: str = "scheduled"
        # A4 EoD no-new-positions flag (injected by scheduler for EoD turns).
        self._eod_no_new_positions: bool = False
        # A4 done_for_day flag: set to True when done_for_day() terminal fires;
        # the scheduler reads this to update lifecycle state.
        self._done_for_day_this_turn: bool = False

    # --- Execution binding (WS-Bench-Migration) ------------------------------

    def bind_execution(
        self,
        *,
        broker: Any,
        risk_manager: Any = None,
        pending_trade_queue: Any = None,
        requires_approval: bool = False,
    ) -> None:
        """Attach the bench's per-competitor broker + risk + approval queue.

        The bench owns broker creation (one isolated book per competitor), and a
        competitor's broker only exists *after* ``Bench.add_competitor`` returns —
        but the trader is constructed first.  The controller therefore builds the
        AgentTrader, registers it, then calls this to wire the ACT toolkit into the
        very book the leaderboard values (so trades land on the tracked book rather
        than a detached broker).

        Rebuilds the cached system prompt so the ACT terminals (``trade`` /
        ``trade_batch`` / ``confirm_trade`` / ``abandon_trade``) are advertised in
        the stable prefix.  Runs once, before the first ``decide()``, so the prefix
        stays cacheable across every subsequent turn (Discipline #5).
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.pending_trade_queue = pending_trade_queue
        self.requires_approval = requires_approval
        self._stable_system_content = self._build_system_prompt()

    # --- Trader Protocol interface -------------------------------------------

    def observe(self, bar: dict[str, Any]) -> None:
        """Accumulate price bars; cheap call every market tick."""
        symbol = bar.get("symbol")
        if symbol in self._bars:
            self._bars[symbol].append(bar)

    def decide(self, account: dict[str, Any]) -> DecisionResult:
        """Run one full agent turn and return the terminal action as a DecisionResult."""
        # A4 crash-recovery: if the scheduler has injected a previous-attempt (orphan
        # turn), reuse its turn_id and populate previous_attempt_tools.  Clear both
        # fields so they only fire for this one turn.
        recovery_tools = list(self._recovery_previous_attempt)
        self._recovery_previous_attempt = []
        if not recovery_tools:
            # Normal path: fresh UUID per turn.
            self._current_turn_id = str(uuid.uuid4())
        # else: scheduler already set self._current_turn_id = orphan.turn_id

        # Reset per-turn tool name accumulator (used by reflect provenance).
        self._turn_tool_names = []
        # Reset done_for_day flag.
        self._done_for_day_this_turn = False

        # Determine turn type and wake reason (set by scheduler or defaults).
        turn_type_raw = self._current_turn_type
        wake_reason_raw = self._current_wake_reason
        # Normalise turn_type to the TurnType literal set.
        _valid_tt = {"SoD", "regular", "event", "reminder", "EoD", "callback", "tutorial"}
        turn_type: TurnType = turn_type_raw if turn_type_raw in _valid_tt else "regular"  # type: ignore[assignment]
        # Reset after reading so next turn reverts to defaults.
        self._current_turn_type = "regular"
        self._current_wake_reason = "scheduled"

        # A6: tutorial override — new traders always run tutorial turns until
        # tutorial_remaining is exhausted (or a trade* auto-exits them).
        if self.tutorial_remaining > 0:
            turn_type = "tutorial"

        ctx = self._build_turn_context(
            account,
            wake_reason=wake_reason_raw,
            turn_type=turn_type,
            previous_attempt_tools=recovery_tools if recovery_tools else None,
        )
        cost_tracker = CostTracker()

        # A5: open a trace row so an interrupted turn is detectable and the
        # cockpit/recent_turns can read it.  Every TurnStore write swallows its
        # own errors — a trace failure never affects the decision.
        tool_records: list[ToolCallRecord] = []
        if self._turn_store is not None:
            self._turn_store.open_turn(
                self._current_turn_id,
                self.name,
                wake_reason_raw,
                turn_type,
                first_look_snapshot=asdict(ctx),
            )

        # Build initial message list: stable system + variable first-look.
        # The system message gets cache_control for Anthropic-backend caching;
        # non-Anthropic providers ignore the field.
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self._stable_system_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": build_first_look(ctx)},
        ]

        tool_call_count = 0
        terminal_action = "hold"
        terminal_args: dict[str, Any] = {"reason": "no decision reached"}

        try:
            while True:
                # Inject cost-warn nudge if threshold crossed (once per turn).
                warn = cost_tracker.check_warn()
                if warn:
                    messages.append({"role": "system", "content": warn})

                result: ToolCallChatResult = self.client.chat_with_tools(
                    self.model,
                    messages,
                    tools=self._tool_definitions(),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                usage = result.usage or {}
                cost_tracker.add_model_call(
                    cost_usd=result.cost or 0.0,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    cached_tokens=usage.get("cached_tokens", 0),
                )

                # Model returned plain text → implicit hold.
                if not result.tool_calls:
                    terminal_action = "hold"
                    terminal_args = {"reason": result.content or "(no response)"}
                    break

                # Append the model's tool-call turn to the message list.
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in result.tool_calls
                        ],
                    }
                )

                hit_terminal = False
                for tc in result.tool_calls:
                    tool_call_count += 1
                    _t0 = time.monotonic()
                    tool_result = self._execute_tool(tc, cost_tracker)
                    # A5: capture this call in the turn trace (name, args, result,
                    # latency).  cost_usd stays 0.0 here — turn-level cost rollup
                    # carries nested-LLM spend; per-tool attribution is future work.
                    tool_records.append(
                        ToolCallRecord(
                            tool_name=tc.name,
                            args=dict(tc.arguments) if isinstance(tc.arguments, dict) else {},
                            result=tool_result.to_dict(),
                            latency_ms=int((time.monotonic() - _t0) * 1000),
                        )
                    )

                    if tc.name in _TERMINALS:
                        terminal_action = tc.name
                        terminal_args = tc.arguments
                        hit_terminal = True
                        # A4: flag done_for_day for the scheduler to read.
                        if tc.name == "done_for_day":
                            self._done_for_day_this_turn = True
                        # Append a stub tool result so the message list is valid.
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(tool_result.to_dict()),
                            }
                        )
                        break

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(tool_result.to_dict()),
                        }
                    )

                if hit_terminal:
                    break

                # Runaway guard.
                if tool_call_count >= self.RUNAWAY_LIMIT:
                    terminal_action = "hold"
                    terminal_args = {
                        "reason": f"runaway guard: {tool_call_count} tool calls in one turn"
                    }
                    break

                # Context-window guard: summarise old tool results if too large.
                self._maybe_trim_context(messages)

        except OpenRouterError as exc:
            # A5: close the trace as an errored turn so it isn't left dangling.
            self._record_turn_close(tool_records, "error", {"error": str(exc)}, cost_tracker)
            return DecisionResult(error=str(exc))

        # A6: tutorial bookkeeping — must happen before return so the next
        # call to decide() sees the updated count.
        if self.tutorial_remaining > 0:
            _trade_terminals = {"trade", "trade_batch", "confirm_trade"}
            if terminal_action in _trade_terminals:
                self.tutorial_remaining = 0  # auto-exit on first trade
            else:
                self.tutorial_remaining = max(0, self.tutorial_remaining - 1)

        # A5: finalise the trace row with the terminal action + cost/token rollup.
        self._record_turn_close(
            tool_records,
            terminal_action,
            terminal_args if isinstance(terminal_args, dict) else {},
            cost_tracker,
        )
        return self._to_decision_result(
            terminal_action, terminal_args, cost_tracker
        )

    def _record_turn_close(
        self,
        tool_records: list[ToolCallRecord],
        final_action: str,
        final_action_args: dict[str, Any],
        cost_tracker: CostTracker,
    ) -> None:
        """Finalise this turn's A5 trace row (no-op when no turn_store is wired)."""
        if self._turn_store is None:
            return
        self._turn_store.close_turn(
            self._current_turn_id,
            tool_calls=tool_records,
            final_action=final_action,
            final_action_args=final_action_args,
            total_cost_usd=cost_tracker.total_usd,
            total_tokens=cost_tracker.token_totals(),
        )

    # --- System prompt (stable; built at construction time) ------------------

    def _build_system_prompt(self) -> str:
        universe_str = ", ".join(self.symbols) if self.symbols else "(to be determined)"
        mandate_str = f"\nYour mandate: {self.style.strip()}." if self.style else ""
        has_broker = self.broker is not None
        trade_terminals = (
            "  • trade(symbol, side, qty, ...) — execute a trade (or queue for approval).\n"
            "  • trade_batch([...]) — execute multiple trades in one turn.\n"
            "  • confirm_trade(pending_trade_id) — execute a pre-approved trade (callback turns).\n"
            "  • abandon_trade(pending_trade_id) — release a pre-approved trade unused (callback turns).\n"
            "  • done_for_day(reason) — skip remaining cadence ticks today.\n"
        ) if has_broker else (
            "  • done_for_day(reason) — skip remaining cadence ticks today.\n"
        )
        return (
            f"You are {self.name}, an autonomous trading agent managing a "
            f"financial account investing in US equities and other instruments.{mandate_str}\n"
            f"Your current tradable universe: {universe_str}.\n"
            f"You are called every {self.cadence_minutes} minutes during regular trading hours.\n\n"
            "Use the tools available to gather information, analyse your positions, "
            "and make trading decisions.  When you are done, call one of the terminal tools:\n"
            + trade_terminals +
            "  • hold(reason) — you considered the situation and chose not to trade; "
            "reason is logged for later reflection.\n"
            "  • pass() — nothing warranted your attention; no log entry.\n\n"
            "Rules:\n"
            "  • Trade only symbols in your universe.\n"
            "  • Quantities in whole shares unless the instrument supports fractions.\n"
            "  • Risk limits and position caps are enforced by the system — "
            "focus on *what* to trade, not limit arithmetic.\n"
            "  • When in doubt, hold rather than trade blindly.\n"
        )

    # --- Turn context (variable; rebuilt each turn) --------------------------

    def _build_turn_context(
        self,
        account: dict[str, Any],
        wake_reason: str = "scheduled",
        turn_type: TurnType = "regular",
        previous_attempt_tools: list[str] | None = None,
    ) -> TurnContext:
        positions = account.get("positions", [])
        pos_count = len(positions) if isinstance(positions, list) else 0

        # A2: populate attention-queue counts (defaults to 0/soft-limit when absent).
        aq = self.attention_queue
        active_wp = aq.count_active(self.name, "watchpoint") if aq is not None else 0
        active_rm = aq.count_active(self.name, "reminder") if aq is not None else 0
        wp_soft = int(getattr(aq, "watchpoint_soft_limit", 20)) if aq is not None else 20
        rm_soft = int(getattr(aq, "reminder_soft_limit", 10)) if aq is not None else 10

        return TurnContext(
            trader_name=self.name,
            model=self.model,
            mandate=self.style,
            cash=float(account.get("cash", 0)),
            position_count=pos_count,
            last_decision=account.get("last_decision"),
            wake_reason=wake_reason,
            turn_type=turn_type,
            cadence_minutes=self.cadence_minutes,
            active_watchpoints=active_wp,
            watchpoint_soft_limit=wp_soft,
            active_reminders=active_rm,
            reminder_soft_limit=rm_soft,
            previous_attempt_tools=list(previous_attempt_tools or []),
            extra_lines=self._turn_type_guidance(turn_type),
            # A6: hint shown in place of the reflections slot for new traders.
            no_prior_context_hint=self.tutorial_remaining > 0,
        )

    def _turn_type_guidance(self, turn_type: TurnType) -> list[str]:
        """Turn-type-conditional special-prompt guidance for the first-look block.

        SoD and EoD turns carry extra guidance describing what that turn is for.
        Kept OUT of the cached system prefix (Discipline #6 token caching): these
        lines render in the variable user-message suffix (``TurnContext.extra_lines``
        → :func:`build_first_look`), so the stable system prompt stays cacheable
        across every turn.  All other turn types add no extra lines.

        The EoD "do not open new positions" directive is gated on
        ``self._eod_no_new_positions``, which the scheduler sets True for EoD
        turns by default (default-strict).  Configuring EoD as non-strict (the
        scheduler leaving the flag False) omits the directive — this is what
        makes the flag the live control point for EoD position policy.
        """
        if turn_type == "SoD":
            return [
                "",
                "Start-of-day guidance: the market has not opened yet. Absorb "
                "overnight developments (call news, research_brief, and situation "
                "as needed), set your posture for the day, and seed watchpoints for "
                "the symbols you want to track. There is no need to trade before the "
                "open.",
            ]
        if turn_type == "EoD":
            lines = [
                "",
                "End-of-day guidance: the session has closed. Reflect on today's "
                "decisions (use reflect to capture what you learned), review and lock "
                "in protective orders on your open positions, and queue watchpoints "
                "or reminders for tomorrow.",
            ]
            if self._eod_no_new_positions:
                lines.append(
                    "Do not open new positions on this turn — the end-of-day turn "
                    "is for housekeeping, not new entries."
                )
            return lines
        if turn_type == "tutorial":
            from ..prompts.tutorial import tutorial_extra_lines
            # tutorial_remaining is still pre-decrement here (decremented at end of
            # decide()), so turn_num = total - remaining + 1 gives the correct
            # 1-based index for this turn.
            turn_num = max(1, self._tutorial_total - self.tutorial_remaining + 1)
            return tutorial_extra_lines(turn_num, self._tutorial_total)
        return []

    # --- Tool definitions (stable; re-built per turn but content is constant) -

    def _tool_definitions(self) -> list[dict[str, Any]]:
        """OpenAI-compatible tool list: A0 built-ins + A2 NOTE + A3 ACT catalogs."""
        from ..intel.tools.note import (
            REFLECT_DEF,
            REMIND_DEF,
            UNWATCH_DEF,
            WATCH_DEF,
            WATCHPOINT_DEF,
        )

        defs: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "list_tools",
                    "description": (
                        "List every tool available to you with name, description, "
                        "argument schema, latency tier, and cost class.  Call this "
                        "first on unfamiliar turns to discover what you can do."
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": (
                        "Search your private memory for lessons and reflections relevant "
                        "to a query.  Returns up to k items; empty result means no prior "
                        "context — you may be new here."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Semantic search query, e.g. 'AAPL momentum strategy'",
                            },
                            "k": {
                                "type": "integer",
                                "description": "Max results to return (default 5)",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            # A2 NOTE catalog
            REFLECT_DEF,
            REMIND_DEF,
            WATCHPOINT_DEF,
            WATCH_DEF,
            UNWATCH_DEF,
        ]

        # C0: Situation Track A LOOK tools
        defs += [
            {
                "type": "function",
                "function": {
                    "name": "world_events",
                    "description": (
                        "GDELT-based global macro and geopolitical event feed filtered "
                        "by theme. Returns mention-volume timeline and recent headlines. "
                        "Enable via SITUATION_GDELT in settings."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "theme": {
                                "type": "string",
                                "description": (
                                    "GKG theme string (e.g. 'WAR', 'ELECTION', "
                                    "'EPU_POLICY_*'). Omit to query WAR+ELECTION+EPU defaults."
                                ),
                            },
                            "timespan": {
                                "type": "string",
                                "description": "Rolling lookback window (e.g. '24h', '48h', '7d'). Default '24h'.",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "prediction_market_odds",
                    "description": (
                        "Polymarket + Kalshi implied probabilities for macro events by "
                        "category. Use to check crowd expectations for Fed decisions, "
                        "elections, or economic prints. Enable via "
                        "SITUATION_PREDICTION_MARKETS in settings."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": (
                                    "Category/topic filter, e.g. 'economics', "
                                    "'politics', 'fed_rate', 'crypto'."
                                ),
                            },
                            "query": {
                                "type": "string",
                                "description": "Optional substring filter on event titles (case-insensitive).",
                            },
                            "min_liquidity": {
                                "type": "number",
                                "description": "Minimum USD liquidity for Polymarket markets (default 1000.0).",
                            },
                        },
                        "required": ["category"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "options_iv",
                    "description": (
                        "Implied volatility and Greeks for near-the-money options on a "
                        "symbol. Use to gauge forward vol or check gamma exposure. "
                        "Enable via SITUATION_OPTIONS_IV in settings."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Equity ticker, e.g. 'AAPL', 'SPY'.",
                            },
                            "expiry": {
                                "type": "string",
                                "description": "ISO 'YYYY-MM-DD' expiry filter. Omit for nearest expiry.",
                            },
                        },
                        "required": ["symbol"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "forecast",
                    "description": (
                        "Forward 1σ price-cone forecast for a symbol over 5/10/30 day "
                        "horizon. Combines realized vol, options IV, and prediction-market "
                        "implied move. This is an *envelope*, not a point forecast — mid "
                        "line is flat. Enable via SITUATION_FORECAST in settings."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Ticker or instrument (e.g. 'AAPL', 'SPY', 'BTC/USD').",
                            },
                            "horizon": {
                                "type": "integer",
                                "description": "Forward horizon in trading days: 5, 10, or 30. Default 30.",
                                "enum": [5, 10, 30],
                            },
                        },
                        "required": ["symbol"],
                    },
                },
            },
        ]

        # A3 ACT catalog — only injected when broker is wired.
        if self.broker is not None:
            from ..intel.tools.act import (
                ABANDON_DEF,
                CONFIRM_DEF,
                TRADE_BATCH_DEF,
                TRADE_DEF,
                UPDATE_PROTECTIVE_DEF,
            )

            defs += [
                TRADE_DEF,
                TRADE_BATCH_DEF,
                UPDATE_PROTECTIVE_DEF,
                CONFIRM_DEF,
                ABANDON_DEF,
            ]

        defs += [
            {
                "type": "function",
                "function": {
                    "name": "done_for_day",
                    "description": (
                        "End all regular-cadence turns for today.  Use when you've locked "
                        "in a good day, circuit-broken a losing streak, or the market regime "
                        "doesn't suit your strategy.  EoD reflection and protective-order fill "
                        "wakes are NOT suppressed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Why you are done for the day.",
                            }
                        },
                        "required": ["reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "hold",
                    "description": (
                        "End this turn having considered the situation.  Use when you "
                        "decided not to trade.  The reason is logged for reflection — "
                        "be honest about why you are holding."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Why you chose to hold.",
                            }
                        },
                        "required": ["reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "pass",
                    "description": (
                        "End this turn when there is genuinely nothing interesting. "
                        "No reason is logged — use when you didn't even consider trading."
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

        return defs

    # --- Tool executor -------------------------------------------------------

    def _execute_tool(
        self, tc: ToolCall, cost_tracker: CostTracker
    ) -> ToolResult:
        """Dispatch a tool call and return its :class:`ToolResult`.

        Unknown tools return a ``not_found`` error rather than raising, so the
        model sees a structured error and can reason about it.
        """
        # Track tool name for reflect() provenance (exclude terminals + list_tools).
        if tc.name not in _TERMINALS and tc.name not in ("list_tools",):
            self._turn_tool_names.append(tc.name)

        try:
            if tc.name == "list_tools":
                return self._tool_list_tools()
            if tc.name == "memory_search":
                return self._tool_memory_search(
                    tc.arguments.get("query", ""),
                    int(tc.arguments.get("k", self.memory_k)),
                    cost_tracker,
                )
            # A2 NOTE tools
            if tc.name == "reflect":
                return self._tool_reflect(
                    tc.arguments.get("note", ""),
                    tags=tc.arguments.get("tags"),
                )
            if tc.name == "remind_me":
                return self._tool_remind_me(
                    tc.arguments.get("when", ""),
                    tc.arguments.get("about", ""),
                )
            if tc.name == "watchpoint":
                return self._tool_watchpoint(
                    tc.arguments.get("symbol", ""),
                    tc.arguments.get("why", ""),
                    condition=tc.arguments.get("condition"),
                    ttl_hours=tc.arguments.get("ttl_hours"),
                )
            if tc.name == "watch_symbol":
                return self._tool_watch_symbol(tc.arguments.get("symbol", ""))
            if tc.name == "unwatch_symbol":
                return self._tool_unwatch_symbol(tc.arguments.get("symbol", ""))
            # C0: Situation Track A LOOK tools
            if tc.name == "world_events":
                return self._tool_world_events(
                    theme=tc.arguments.get("theme"),
                    timespan=str(tc.arguments.get("timespan", "24h")),
                )
            if tc.name == "prediction_market_odds":
                return self._tool_prediction_market_odds(
                    category=str(tc.arguments.get("category", "")),
                    query=tc.arguments.get("query"),
                    min_liquidity=float(tc.arguments.get("min_liquidity", 1000.0)),
                )
            if tc.name == "options_iv":
                return self._tool_options_iv(
                    symbol=str(tc.arguments.get("symbol", "")),
                    expiry=tc.arguments.get("expiry"),
                )
            if tc.name == "forecast":
                return self._tool_forecast(
                    symbol=str(tc.arguments.get("symbol", "")),
                    horizon=int(tc.arguments.get("horizon", 30)),
                )
            # A3 ACT tools (also terminals — loop exits after these)
            if tc.name == "trade":
                return self._tool_trade(
                    tc.arguments.get("symbol", ""),
                    tc.arguments.get("side", ""),
                    tc.arguments.get("qty", 0),
                    stop=tc.arguments.get("stop"),
                    take_profit=tc.arguments.get("take_profit"),
                    trail=tc.arguments.get("trail"),
                )
            if tc.name == "trade_batch":
                return self._tool_trade_batch(
                    tc.arguments.get("trades", [])
                )
            if tc.name == "update_protective_order":
                return self._tool_update_protective_order(
                    tc.arguments.get("order_id", ""),
                    new_stop=tc.arguments.get("new_stop"),
                    new_tp=tc.arguments.get("new_tp"),
                    new_trail=tc.arguments.get("new_trail"),
                )
            if tc.name == "confirm_trade":
                return self._tool_confirm_trade(
                    tc.arguments.get("pending_trade_id", "")
                )
            if tc.name == "abandon_trade":
                return self._tool_abandon_trade(
                    tc.arguments.get("pending_trade_id", "")
                )
            if tc.name in _TERMINALS:
                # done_for_day, hold, pass — acknowledge; loop handles exit.
                return ToolResult(ok=True, data={"action": tc.name})
            return ToolResult(
                ok=False,
                error=ToolError(
                    kind="not_found",
                    message=f"Tool '{tc.name}' is not available in this turn. "
                    "Call list_tools() to see what is enabled.",
                ),
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                error=ToolError(kind="internal", message=str(exc)),
            )

    def _tool_list_tools(self) -> ToolResult:
        """Return the full tool catalog, deferring to ListToolsTool as single source of truth.

        ListToolsTool owns the A1 LOOK catalog (including the new Track A tools and
        the builtins list_tools / memory_search / hold / pass).  NOTE tools, ACT tools,
        and done_for_day are injected via extra_entries since they are not part of the
        A1 LOOK set.  Deferring to ListToolsTool (option ii) ensures the catalog
        stays in sync after any future LOOK-tool additions without editing this method.
        """
        from ..intel.tools.look.list_tools import ListToolsTool
        from ..intel.tools.note import (
            REFLECT_CATALOG,
            REMIND_CATALOG,
            UNWATCH_CATALOG,
            WATCH_CATALOG,
            WATCHPOINT_CATALOG,
        )

        extra: list[dict[str, Any]] = [
            REFLECT_CATALOG,
            REMIND_CATALOG,
            WATCHPOINT_CATALOG,
            WATCH_CATALOG,
            UNWATCH_CATALOG,
        ]

        if self.broker is not None:
            from ..intel.tools.act import (
                ABANDON_CATALOG,
                CONFIRM_CATALOG,
                TRADE_BATCH_CATALOG,
                TRADE_CATALOG,
                UPDATE_PROTECTIVE_CATALOG,
            )
            extra += [
                TRADE_CATALOG,
                TRADE_BATCH_CATALOG,
                UPDATE_PROTECTIVE_CATALOG,
                CONFIRM_CATALOG,
                ABANDON_CATALOG,
            ]

        # done_for_day is not in ListToolsTool's LOOK catalog; add it here.
        # hold and pass are already in ListToolsTool's built-in catalog.
        extra.append({
            "name": "done_for_day",
            "description": "Terminal: skip remaining cadence ticks today.",
            "args": {"reason": "str"},
            "latency": "instant",
            "cost_class": "free",
            "enabled": True,
            "disabled_reason": None,
        })

        tool = ListToolsTool(
            owner_user_id=self.owner_user_id,
            trader_id=self.name,
            extra_entries=extra,
        )
        return tool()

    # --- C0: Situation Track A LOOK tool dispatchers -------------------------

    def _tool_world_events(
        self, theme: str | None = None, timespan: str = "24h"
    ) -> ToolResult:
        """Dispatch the world_events LOOK tool (GDELT macro event feed)."""
        from ..intel.tools.look.world_events import WorldEventsTool

        tool = WorldEventsTool(
            owner_user_id=self.owner_user_id,
            trader_id=self.name,
            settings_store=self.settings_store,
            gdelt_provider=self._gdelt_provider,
        )
        return tool(theme=theme, timespan=timespan)

    def _tool_prediction_market_odds(
        self,
        category: str,
        query: str | None = None,
        min_liquidity: float = 1000.0,
    ) -> ToolResult:
        """Dispatch the prediction_market_odds LOOK tool (Polymarket + Kalshi)."""
        from ..intel.tools.look.prediction_market_odds import PredictionMarketOddsTool

        tool = PredictionMarketOddsTool(
            owner_user_id=self.owner_user_id,
            trader_id=self.name,
            settings_store=self.settings_store,
            pm_provider=self._pm_provider,
        )
        return tool(category, query=query, min_liquidity=min_liquidity)

    def _tool_options_iv(
        self, symbol: str, expiry: str | None = None
    ) -> ToolResult:
        """Dispatch the options_iv LOOK tool (Alpaca IV passthrough)."""
        from ..intel.tools.look.options_iv import OptionsIVTool

        tool = OptionsIVTool(
            owner_user_id=self.owner_user_id,
            trader_id=self.name,
            settings_store=self.settings_store,
            chain_provider=self._chain_provider,
            spot_prices=dict(self._spot_prices),
        )
        return tool(symbol, expiry=expiry)

    def _tool_forecast(self, symbol: str, horizon: int = 30) -> ToolResult:
        """Dispatch the forecast LOOK tool (1σ price-cone)."""
        from ..intel.tools.look.forecast import ForecastTool

        tool = ForecastTool(
            owner_user_id=self.owner_user_id,
            trader_id=self.name,
            settings_store=self.settings_store,
            chain_provider=self._chain_provider,
            pm_provider=self._pm_provider,
            spot_prices=dict(self._spot_prices),
        )
        return tool(symbol, horizon=horizon)

    # --- A2 NOTE tool dispatchers -------------------------------------------

    def _note_base_kwargs(self) -> dict[str, Any]:
        """Common kwargs for NoteToolBase constructors."""
        return {
            "attention_queue": self.attention_queue,
            "memory": self.memory,
            "owner_user_id": self.owner_user_id,
            "trader_id": self.name,
        }

    def _tool_reflect(self, note: str, *, tags: Any = None) -> ToolResult:
        """Write a durable lesson to the trader's private memory (WS-D MemoryStore)."""
        from ..intel.tools.note.reflect import ReflectTool

        tool = ReflectTool(**self._note_base_kwargs())
        tag_list = list(tags) if isinstance(tags, (list, tuple)) else None
        return tool.run(
            note,
            tags=tag_list,
            tool_call_names=list(self._turn_tool_names),
        )

    def _tool_remind_me(self, when: str, about: str) -> ToolResult:
        """Schedule a time-based deferred self-poke."""
        from ..intel.tools.note.remind_me import RemindMeTool

        tool = RemindMeTool(**self._note_base_kwargs())
        return tool.run(when, about)

    def _tool_watchpoint(
        self,
        symbol: str,
        why: str,
        *,
        condition: str | None = None,
        ttl_hours: Any = None,
    ) -> ToolResult:
        """Register an event-based symbol monitor."""
        from ..intel.tools.note.watchpoint import WatchpointTool

        tool = WatchpointTool(**self._note_base_kwargs())
        ttl = float(ttl_hours) if ttl_hours is not None else None
        return tool.run(symbol, why, condition=condition, ttl_hours=ttl)

    def _tool_watch_symbol(self, symbol: str) -> ToolResult:
        """Add a symbol to the trader's personal watchlist."""
        from ..intel.tools.note.watch_symbol import WatchSymbolTool

        tool = WatchSymbolTool(
            **self._note_base_kwargs(),
            settings_store=self.settings_store,
        )
        return tool.run(symbol)

    def _tool_unwatch_symbol(self, symbol: str) -> ToolResult:
        """Remove a symbol from the trader's personal watchlist."""
        from ..intel.tools.note.unwatch_symbol import UnwatchSymbolTool

        tool = UnwatchSymbolTool(
            **self._note_base_kwargs(),
            settings_store=self.settings_store,
        )
        return tool.run(symbol)

    # --- A3 ACT tool dispatchers --------------------------------------------

    def _act_base_kwargs(self) -> dict[str, Any]:
        """Common kwargs for ActToolBase constructors."""
        return {
            "broker": self.broker,
            "risk_manager": self.risk_manager,
            "pending_trade_queue": self.pending_trade_queue,
            "trader_id": self.name,
            "turn_id": self._current_turn_id,
            "requires_approval": self.requires_approval,
        }

    def _tool_trade(
        self,
        symbol: str,
        side: str,
        qty: Any,
        *,
        stop: float | None = None,
        take_profit: float | None = None,
        trail: float | None = None,
    ) -> ToolResult:
        from ..intel.tools.act.trade import TradeTool

        tool = TradeTool(**self._act_base_kwargs())
        try:
            qty_f = float(qty)
        except (TypeError, ValueError):
            qty_f = 0.0
        return tool.run(symbol, side, qty_f, stop=stop, take_profit=take_profit, trail=trail)

    def _tool_trade_batch(self, trades: Any) -> ToolResult:
        from ..intel.tools.act.trade_batch import TradeBatchTool

        tool = TradeBatchTool(**self._act_base_kwargs())
        return tool.run(list(trades) if isinstance(trades, list) else [])

    def _tool_update_protective_order(
        self,
        order_id: str,
        *,
        new_stop: float | None = None,
        new_tp: float | None = None,
        new_trail: float | None = None,
    ) -> ToolResult:
        from ..intel.tools.act.update_protective_order import UpdateProtectiveOrderTool

        tool = UpdateProtectiveOrderTool(**self._act_base_kwargs())
        return tool.run(order_id, new_stop=new_stop, new_tp=new_tp, new_trail=new_trail)

    def _tool_confirm_trade(self, pending_trade_id: str) -> ToolResult:
        from ..intel.tools.act.confirm_trade import ConfirmTradeTool

        tool = ConfirmTradeTool(**self._act_base_kwargs())
        return tool.run(pending_trade_id)

    def _tool_abandon_trade(self, pending_trade_id: str) -> ToolResult:
        from ..intel.tools.act.abandon_trade import AbandonTradeTool

        tool = AbandonTradeTool(**self._act_base_kwargs())
        return tool.run(pending_trade_id)

    def _tool_memory_search(
        self,
        query: str,
        k: int,
        cost_tracker: CostTracker,
    ) -> ToolResult:
        """Query the trader's private memory namespace; empty when store absent."""
        if not query:
            return ToolResult(
                ok=False,
                error=ToolError(kind="invalid_input", message="query must not be empty"),
            )
        if self.memory is None or self.owner_user_id is None:
            return ToolResult(
                ok=True,
                data={"memories": [], "note": "memory store not yet available"},
            )
        try:
            lessons = self.memory.recall(
                self.owner_user_id, self.name, query, k
            )
            memories = [
                {"text": str(getattr(lesson, "text", lesson)), "trader_id": self.name}
                for lesson in lessons
            ]
            return ToolResult(ok=True, data={"memories": memories})
        except Exception as exc:
            return ToolResult(
                ok=False,
                error=ToolError(kind="internal", message=str(exc)),
            )

    # --- Context-window guard ------------------------------------------------

    def _maybe_trim_context(self, messages: list[dict[str, Any]]) -> None:
        """If the message list grows too large, summarise older tool results in-place.

        Keeps the first two messages (system + first-look) and the most recent
        4 messages intact; everything in between has its tool-result content
        replaced with a one-line summary.
        """
        total_chars = sum(
            len(json.dumps(m)) for m in messages
        )
        if total_chars <= self.CTX_WARN_CHARS:
            return
        # Summarise tool result messages in the middle section.
        keep_tail = 4
        boundary = len(messages) - keep_tail
        for i in range(2, boundary):
            m = messages[i]
            if m.get("role") == "tool":
                messages[i] = {
                    "role": "tool",
                    "tool_call_id": m.get("tool_call_id", ""),
                    "content": "[earlier tool result summarised — context trimmed]",
                }

    # --- Result mapping ------------------------------------------------------

    def _to_decision_result(
        self,
        action: str,
        args: dict[str, Any],
        cost_tracker: CostTracker,
    ) -> DecisionResult:
        """Map the terminal action to a bench-compatible DecisionResult.

        ACT terminals (trade, confirm_trade, etc.) leave ``decisions=[]`` because
        the ACT tools already interacted with the broker directly.  The bench
        controller must not re-execute them.  The ``comment`` carries a summary
        for the decision log.
        """
        if action == "pass":
            comment = "pass"
        elif action == "hold":
            comment = str(args.get("reason", "hold"))
        elif action == "done_for_day":
            comment = f"done_for_day: {args.get('reason', '')}"
        elif action == "trade":
            sym = args.get("symbol", "?")
            side = args.get("side", "?")
            qty = args.get("qty", "?")
            comment = f"trade: {side} {qty} {sym}"
        elif action == "trade_batch":
            n = len(args.get("trades", []))
            comment = f"trade_batch: {n} item(s)"
        elif action == "confirm_trade":
            comment = f"confirm_trade: {args.get('pending_trade_id', '?')}"
        elif action == "abandon_trade":
            comment = f"abandon_trade: {args.get('pending_trade_id', '?')}"
        else:
            comment = action
        return DecisionResult(
            decisions=[],
            comment=comment,
            usage=cost_tracker.rollup(),  # type: ignore[arg-type]
        )
