"""Bench competitors: a uniform ``Trader`` interface with three implementations.

* :class:`LLMTrader` — prompts an OpenRouter model with recent price context +
  the current portfolio and parses a structured BUY/SELL/HOLD decision.
  Legacy structured-output path; kept for bench backward-compat.

* :class:`StrategyTrader` — wraps any deterministic :class:`Strategy` (e.g.
  mean-reversion) so it can compete in the same bench as a baseline.

* :class:`AgentTrader` — ReAct-style tool-calling agent (WS-Agent A0–A3).
  Runs a multi-step decision loop: build first-look context → call model with
  tools → execute tools → repeat until terminal action or runaway guard.
  A0 terminals: ``hold(reason)``, ``pass()``.
  A2 NOTE tools: ``reflect``, ``remind_me``, ``watchpoint``,
  ``watch_symbol``, ``unwatch_symbol``.
  A3 ACT tools: ``trade``, ``trade_batch``, ``update_protective_order``.
  A3 END terminals added: ``trade``, ``trade_batch``, ``confirm_trade``,
  ``abandon_trade``, ``done_for_day``.

All three expose ``observe(bar)`` and ``decide(account) -> DecisionResult`` so
they are drop-in substitutes from the bench/controller's perspective.

Backward compat: :class:`ManagerAgent.chat()` return shape unchanged.
``DecisionResult.decisions`` stays an empty list for AgentTrader — the ACT tools
interact with the broker directly, so the bench controller does not re-execute.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..intel.cost_tracker import CostTracker
from ..intel.tool_envelope import ToolError, ToolResult
from ..intel.turn_context import TurnContext, TurnType, build_first_look
from .openrouter import OpenRouterError, ToolCall, ToolCallChatResult, parse_json_object

if TYPE_CHECKING:
    from ..data.history import HistoryService
    from ..situation.regime import RegimeClassifier
    from ..situation.social import SocialAggregator, SocialItem
    from ..strategy import Strategy
    from .openrouter import OpenRouterClient

_VALID_ACTIONS = {"BUY", "SELL", "HOLD"}

# WS-A defaults: how many research briefs / memory lessons to pull into a single
# decision. Kept small — recall embeds the query per trader per round.
_RESEARCH_K = 5
_MEMORY_RECALL_K = 5

# P3: how many pattern KB matches to inject per decision.
_PATTERN_K = 3


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


# --- LLM trader -------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an autonomous trading agent managing a paper account of US equities. "
    "Given recent price bars and your current portfolio, decide what to trade to "
    "maximize risk-adjusted return. You may only trade the listed symbols, in whole "
    "shares, and must not spend more than your available cash. Respond with ONLY a "
    "JSON object of this exact shape:\n"
    '{"decisions": [{"symbol": "AAPL", "action": "BUY|SELL|HOLD", '
    '"quantity": <int>, "reason": "<short>"}], "comment": "<one line>"}\n'
    "Use HOLD with quantity 0 when you want no change for a symbol. Be decisive but "
    "manage risk. Do not include any text outside the JSON object."
)


def _system_prompt(style: str | None) -> str:
    """Base trading prompt, optionally given a mandated trading style."""
    if not style or not style.strip():
        return _SYSTEM_PROMPT
    return (
        f"{_SYSTEM_PROMPT}\n\nYour mandated trading style is: {style.strip()}. "
        "Let this style shape which symbols you trade, your position sizing, and how "
        "often you turn the book over — while still respecting your cash and the risk rules."
    )


class LLMTrader:
    """A model that trades by reasoning over recent bars + its portfolio."""

    def __init__(
        self,
        model: str,
        client: OpenRouterClient,
        *,
        symbols: list[str],
        name: str | None = None,
        lookback: int = 30,
        temperature: float = 0.3,
        max_tokens: int = 800,
        history: HistoryService | None = None,
        research: Any = None,
        memory: Any = None,
        owner_user_id: str | None = None,
        research_k: int = _RESEARCH_K,
        memory_k: int = _MEMORY_RECALL_K,
        style: str | None = None,
        # P3: situation layer (all optional; absent → block omitted gracefully)
        regime_classifier: RegimeClassifier | None = None,
        social_aggregator: SocialAggregator | None = None,
        social_items: list[SocialItem] | None = None,
        calendar_events: list[dict[str, Any]] | None = None,
        # P4: pattern KB (optional; absent → block omitted)
        pattern_store: Any = None,
        pattern_k: int = _PATTERN_K,
        # P6: per-trader intelligence override flags (None = use owner defaults)
        intelligence_flags: dict[str, bool] | None = None,
    ) -> None:
        self.model = model
        self.client = client
        self.symbols = list(symbols)
        self.name = name or model
        self.lookback = lookback
        self.temperature = temperature
        self.max_tokens = max_tokens
        # An optional mandated trading style (from the add-trader wizard) folded
        # into the system prompt once at construction.
        self.style = style
        self.system_prompt = _system_prompt(style)
        # WS-A: when injected, the trader sees a richer historical + fundamentals
        # context block instead of just the last `lookback` closes. Optional so
        # the bench/back-compat path (no history) is unchanged.
        self.history = history
        # WS-A intelligence (all duck-typed so there's no import cycle): the
        # shared research store, the trader's private memory, and the owner the
        # two are namespaced by. Any of them None → that block is simply omitted
        # and the decision is still made — the manager's defensive pattern.
        self.research = research
        self.memory = memory
        self.owner_user_id = owner_user_id
        self.research_k = research_k
        self.memory_k = memory_k
        # P3: situation layer
        self.regime_classifier = regime_classifier
        self.social_aggregator = social_aggregator
        self.social_items: list[SocialItem] = list(social_items or [])
        self.calendar_events: list[dict[str, Any]] = list(calendar_events or [])
        # P4: pattern KB
        self.pattern_store = pattern_store
        self.pattern_k = pattern_k
        # P6: per-trader intelligence flags (override owner-level settings)
        self._intel_flags: dict[str, bool] = dict(intelligence_flags or {})
        self._bars: dict[str, deque[dict[str, Any]]] = {
            s: deque(maxlen=lookback) for s in self.symbols
        }

    def observe(self, bar: dict[str, Any]) -> None:
        symbol = bar.get("symbol")
        if symbol in self._bars:
            self._bars[symbol].append(bar)

    def decide(self, account: dict[str, Any]) -> DecisionResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_context(account)},
        ]
        try:
            res = self.client.chat(
                self.model,
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                json_mode=True,
            )
        except OpenRouterError as exc:
            return DecisionResult(error=str(exc))

        try:
            payload = parse_json_object(res.content)
        except ValueError as exc:
            return DecisionResult(raw=res.content, usage=res.usage, error=f"parse: {exc}")

        return DecisionResult(
            decisions=self._coerce_decisions(payload.get("decisions", [])),
            comment=str(payload.get("comment", ""))[:200],
            raw=res.content,
            usage=res.usage,
        )

    # --- internals ----------------------------------------------------------

    def _build_context(self, account: dict[str, Any]) -> str:
        """Layered, gracefully-degrading context.

        Body (rich history if injected, else the last-``lookback`` closes) +
        situation/regime block (P3) + pattern KB (P4) + research briefs +
        the trader's own past lessons; each block dropped if its source is
        absent or errors. A single JSON-decision trailer closes the context.
        """
        body = (
            self.history.context_block(self.symbols, account, include_trailer=False)
            if self.history is not None
            else self._fallback_body(account)
        )
        parts = [
            body,
            self._situation_block(),   # P3
            self._pattern_block(),     # P4
            self._research_block(),
            self._memory_block(),
        ]
        return "\n\n".join(p for p in parts if p) + "\n\nReturn your JSON decision now."

    def _fallback_body(self, account: dict[str, Any]) -> str:
        """The original cash/positions + last-``lookback`` closes body (no trailer)."""
        lines = [
            f"Cash available: {account.get('cash', 0):,.2f}",
            f"Positions: {account.get('positions', [])}",
            f"Tradable symbols: {', '.join(self.symbols)}",
            "",
            "Recent bars (oldest first) — close prices:",
        ]
        for symbol, bars in self._bars.items():
            closes = [round(float(b["close"]), 2) for b in bars if b.get("close") is not None]
            lines.append(f"  {symbol}: {closes[-self.lookback:]}")
        return "\n".join(lines)

    # --- P3: situation / regime block ----------------------------------------

    def _intel_enabled(self, flag: str) -> bool:
        """Check a per-trader intelligence flag (P6). Defaults True if unset."""
        return self._intel_flags.get(flag, True)

    def _situation_block(self) -> str:
        """Regime + social situation block (~10 lines). Omitted if no classifier."""
        if not self._intel_enabled("situation"):
            return ""
        clf = self.regime_classifier
        if clf is None:
            return ""
        try:
            # Build a closes list from observed bars (all symbols, latest close).
            closes: list[float] = []
            for bars in self._bars.values():
                for b in bars:
                    c = b.get("close")
                    if c is not None:
                        closes.append(float(c))
            if len(closes) < 2:
                return ""
            regime = clf.classify(closes, events=self.calendar_events)
            lines = ["## Situation"]
            lines.extend(ln for ln in regime.to_context_lines() if ln)
            # Social metrics per symbol
            agg = self.social_aggregator
            if agg is not None and self.social_items:
                for symbol in self.symbols:
                    metrics = agg.aggregate(self.social_items, ticker=symbol)
                    lines.extend(metrics.to_context_lines())
            return "\n".join(lines)
        except Exception:
            return ""

    # --- P4: pattern KB block ------------------------------------------------

    def _pattern_block(self) -> str:
        """Recent matching patterns with regime-conditioned outcome stats."""
        if not self._intel_enabled("patterns"):
            return ""
        store = self.pattern_store
        if store is None:
            return ""
        try:
            query = self._recall_query()
            matches = store.recall(query, k=self.pattern_k)
            if not matches:
                return ""
            lines = ["## Pattern KB (similar past setups)"]
            for m in matches:
                lines.append(self._format_pattern(m))
            return "\n".join(lines)
        except Exception:
            return ""

    @staticmethod
    def _format_pattern(match: Any) -> str:
        label = getattr(match, "label", "?")
        regime = getattr(match, "regime", "?")
        stats = getattr(match, "stats", {}) or {}
        hit = stats.get("hit_rate")
        n = stats.get("n", 0)
        hit_str = f"{hit:.0%}" if hit is not None else "?"
        return f"  {label} (regime={regime}, n={n}, forward-hit={hit_str})"

    # -------------------------------------------------------------------------

    def _recall_query(self) -> str:
        """The semantic query for research/memory recall: this round's universe."""
        return f"trading decision for {', '.join(self.symbols)}"

    def _research_block(self) -> str:
        """Shared per-user briefs relevant to this round (semantic → per-symbol →
        recent). Omitted when there's no owner/store or anything errors."""
        if not self._intel_enabled("research"):
            return ""
        research = self.research
        if research is None or self.owner_user_id is None:
            return ""
        owner, k = self.owner_user_id, self.research_k
        try:
            briefs = list(research.search(owner, self._recall_query(), k))
            if not briefs:
                briefs = self._briefs_by_symbol(research, k)
            if not briefs:
                briefs = list(research.recent(owner, k))
        except Exception:
            return ""
        # Lazy import breaks the config.endpoints → llm → research import cycle.
        from ..research.format import format_briefs

        return format_briefs(briefs, header="## Research briefs (most relevant)")

    def _briefs_by_symbol(self, research: Any, k: int) -> list[Any]:
        """Structured per-ticker briefs (the no-embedder fallback), deduped to ``k``."""
        out: list[Any] = []
        seen: set[str] = set()
        for symbol in self.symbols:
            for brief in research.get(self.owner_user_id, symbol):
                bid = str(getattr(brief, "id", "") or id(brief))
                if bid in seen:
                    continue
                seen.add(bid)
                out.append(brief)
                if len(out) >= k:
                    return out
        return out

    def _memory_block(self) -> str:
        """This trader's own past lessons (recalled under ``self.name``). Omitted
        on no owner/store, or when embeddings are unavailable (EmbedError)."""
        if not self._intel_enabled("memory"):
            return ""
        memory = self.memory
        if memory is None or self.owner_user_id is None:
            return ""
        try:
            lessons = memory.recall(self.owner_user_id, self.name, self._recall_query(), self.memory_k)
        except Exception:
            # No local embed endpoint (EmbedError) or a flaky store: still decide.
            return ""
        from ..memory.format import format_lessons  # lazy: avoid import cycle

        return format_lessons(lessons, header="## Your past lessons", show_trader=False)

    def _coerce_decisions(self, raw: Any) -> list[TradeDecision]:
        out: list[TradeDecision] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).strip()
            action = str(item.get("action", "HOLD")).strip().upper()
            if symbol not in self.symbols or action not in _VALID_ACTIONS:
                continue
            try:
                qty = max(0.0, float(item.get("quantity", 0) or 0))
            except (TypeError, ValueError):
                qty = 0.0
            out.append(
                TradeDecision(
                    symbol=symbol,
                    action=action,
                    quantity=qty,
                    reason=str(item.get("reason", ""))[:200],
                )
            )
        return out


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

    Replaces the structured-output :class:`LLMTrader` pipeline with a proper
    multi-step decision loop:

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
        # Price bar buffer (same shape as LLMTrader for observe() compat)
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

        ctx = self._build_turn_context(
            account,
            wake_reason=wake_reason_raw,
            turn_type=turn_type,
            previous_attempt_tools=recovery_tools if recovery_tools else None,
        )
        cost_tracker = CostTracker()

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
                    tool_result = self._execute_tool(tc, cost_tracker)

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
            return DecisionResult(error=str(exc))

        return self._to_decision_result(
            terminal_action, terminal_args, cost_tracker
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
        """Return the full A0 + A2 NOTE + A3 ACT tool catalog."""
        from ..intel.tools.note import (
            REFLECT_CATALOG,
            REMIND_CATALOG,
            UNWATCH_CATALOG,
            WATCH_CATALOG,
            WATCHPOINT_CATALOG,
        )

        catalog: list[dict[str, Any]] = [
            {
                "name": "list_tools",
                "description": "List all available tools.",
                "args": {},
                "latency": "instant",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "memory_search",
                "description": "Search your private memory for relevant lessons.",
                "args": {"query": "str", "k": "int (default 5)"},
                "latency": "fast",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            # A2 NOTE catalog entries
            REFLECT_CATALOG,
            REMIND_CATALOG,
            WATCHPOINT_CATALOG,
            WATCH_CATALOG,
            UNWATCH_CATALOG,
        ]

        # A3 ACT catalog — only shown when broker is wired.
        if self.broker is not None:
            from ..intel.tools.act import (
                ABANDON_CATALOG,
                CONFIRM_CATALOG,
                TRADE_BATCH_CATALOG,
                TRADE_CATALOG,
                UPDATE_PROTECTIVE_CATALOG,
            )

            catalog += [
                TRADE_CATALOG,
                TRADE_BATCH_CATALOG,
                UPDATE_PROTECTIVE_CATALOG,
                CONFIRM_CATALOG,
                ABANDON_CATALOG,
            ]

        catalog += [
            {
                "name": "done_for_day",
                "description": "Terminal: skip remaining cadence ticks today.",
                "args": {"reason": "str"},
                "latency": "instant",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "hold",
                "description": "Terminal: end this turn having considered the situation.",
                "args": {"reason": "str"},
                "latency": "instant",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
            {
                "name": "pass",
                "description": "Terminal: end this turn — nothing interesting.",
                "args": {},
                "latency": "instant",
                "cost_class": "free",
                "enabled": True,
                "disabled_reason": None,
            },
        ]
        return ToolResult(ok=True, data={"tools": catalog})

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
