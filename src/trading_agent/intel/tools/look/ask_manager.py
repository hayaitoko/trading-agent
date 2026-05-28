"""``ask_manager`` — ask the overseer manager a question.

Tool name:      ask_manager
Args:           question (str)
ToolResult:     ok=True, data={"answer": str}
Latency tier:   slow (live LLM call)
Cost class:     model_call
Gating flag:    cost-gated ≤1/turn; CostTracker.add_nested_llm() records the spend
Example use:    ask_manager("Should I hold or reduce my AAPL position given today's macro?")

**Cost gate:** the tool enforces at most one ask_manager call per trader turn.
Subsequent calls within the same turn return a gating error — spend on `ask_manager`
is recorded against the per-turn CostTracker as a nested LLM cost.

**Paper/live isolation (MONEY IS REAL):** the manager's system prompt is augmented
with a filter that prevents disclosure of paper/sim/demo status to the trader.
This module adds the filter sentence to whatever system prompt the ManagerAgent uses.

**Peer-trader state isolation:** the manager must not disclose the decisions,
positions, or P&L of other traders in this stable.  This is enforced via the
SYSTEM_ADDENDUM injected by this tool before the question is sent.

Wraps :class:`~trading_agent.manager.agent.ManagerAgent`.
"""

from __future__ import annotations

from typing import Any

from ....intel.cost_tracker import CostTracker
from ._base import LookToolBase

# Injected as a prefix to the question sent to the manager.  It is position-
# invariant: the manager sees this regardless of the manager's own system prompt.
_MANAGER_FILTER = (
    "You are speaking with one of the trading agents you oversee. "
    "Rules for this reply:\n"
    "1. NEVER disclose whether any account is a simulation, test, paper trade, "
    "   demo, or fake account. The agent's account is to be treated as real.\n"
    "2. NEVER disclose the current positions, P&L, or trading decisions of any "
    "   OTHER trading agent — peer-trader information is confidential.\n"
    "3. Answer the agent's question using only market context, your general "
    "   observations, and publicly available information.\n"
    "Now the agent asks:\n\n"
)


class AskManagerTool(LookToolBase):
    """Cost-gated, isolation-enforced bridge to the ManagerAgent.

    Parameters
    ----------
    manager_agent:
        Duck-typed: must expose ``chat(user_id, conversation_id, message, ref) -> str``.
        ``None`` → graceful unavailable error.
    model_ref:
        A :class:`~trading_agent.config.endpoints.ModelRef` passed to the manager.
    cost_tracker:
        The :class:`~trading_agent.intel.cost_tracker.CostTracker` for the current turn.
        Nested LLM spend is recorded here.
    conversation_id:
        Stable identifier for the trader-manager conversation thread.
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "ask_manager",
        "description": (
            "Ask the overseer manager a question. Cost-gated: at most once per turn. "
            "The manager cannot disclose peer-trader state or account simulation status. "
            "Use for strategic guidance only."
        ),
        "args": {"question": "str"},
        "latency": "slow",
        "cost_class": "model_call",
        "enabled": True,
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        manager_agent: Any = None,
        model_ref: Any = None,
        cost_tracker: CostTracker | None = None,
        conversation_id: str | None = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._manager = manager_agent
        self._model_ref = model_ref
        self._cost_tracker = cost_tracker
        self._conversation_id = conversation_id or f"trader:{trader_id}"
        self._called_this_turn: bool = False

    def reset_for_turn(self) -> None:
        """Reset the per-turn call gate.  Called at the start of each turn."""
        self._called_this_turn = False

    def __call__(self, question: str) -> Any:
        """Ask the manager a question; enforce at-most-once-per-turn gate.

        The question is prefixed with :data:`_MANAGER_FILTER` to prevent
        disclosure of paper/sim/demo status and peer-trader state.

        Returns
        -------
        ToolResult
            ok=True,  data={"answer": str}          — manager replied
            ok=False, error={kind:"rate_limit", …}  — already called this turn

        Example
        -------
        >>> tool = AskManagerTool(trader_id="Alpha")
        >>> result = tool("What's your view on AAPL today?")
        >>> result.ok
        False  # manager not wired
        """
        question = (question or "").strip()
        if not question:
            return self._err("invalid_input", "question must not be empty")

        # Cost gate: at most 1 ask_manager per turn.
        if self._called_this_turn:
            return self._err(
                "rate_limit",
                "ask_manager may only be called once per turn. "
                "Use the information from the earlier call or terminal the turn.",
                retry_after=None,
            )

        if self._manager is None or self.owner_user_id is None:
            return self._err(
                "unavailable",
                "manager agent not wired — ManagerAgent required for ask_manager()",
            )

        # Inject the filter prefix so isolation holds regardless of manager's
        # own system prompt configuration.
        filtered_question = _MANAGER_FILTER + question

        self._called_this_turn = True
        try:
            raw_ref = self._model_ref
            answer = self._manager.chat(
                self.owner_user_id,
                self._conversation_id,
                filtered_question,
                raw_ref,
            )
            # Record nested LLM cost if the cost tracker is present.
            # Manager.chat doesn't expose cost directly; we record $0 if unavailable
            # and the CostTracker's soft-warn will fire on model_call tools.
            if self._cost_tracker is not None:
                self._cost_tracker.add_nested_llm("ask_manager", cost_usd=0.0)
        except Exception as exc:
            self._called_this_turn = False  # allow retry on transient error
            return self._err("internal", f"ask_manager failed: {exc}")

        # Scrub forbidden words from the answer before returning to the trader.
        answer = _scrub_answer(str(answer))
        return self._ok({"answer": answer})


# --------------------------------------------------------------------------- helpers

_FORBIDDEN_WORDS = ("paper", "sim", "demo", "fake", "test mode", "monopoly")


def _scrub_answer(text: str) -> str:
    """Replace forbidden disclosure words in the manager's answer.

    This is a last-resort safety net.  The manager's prompt filter should prevent
    these words from appearing; this scrub catches any that slip through.
    """
    lower = text.lower()
    for word in _FORBIDDEN_WORDS:
        while word in lower:
            idx = lower.find(word)
            text = text[:idx] + ("*" * len(word)) + text[idx + len(word):]
            lower = text.lower()
    return text
