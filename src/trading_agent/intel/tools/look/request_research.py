"""``request_research`` — queue a research pass for a symbol+question.

Tool name:      request_research
Args:           symbol (str), question (str)
ToolResult:     ok=True, data={"request_id": str, "symbol": str, "status": "queued"}
Latency tier:   queued (returns immediately; result available next turn)
Cost class:     queued (incurs a cheap-model call asynchronously via WS-C)
Gating flag:    always enabled when research_store + run callable are wired
Example use:    request_research("NVDA", "What are the near-term earnings risks?")

Does NOT block.  Returns immediately with a request_id.  Check research_brief()
on the next turn — or any subsequent turn — to see the result.

Wraps WS-C :class:`~trading_agent.research.agent.ResearchAgent` by calling
a provided ``run_fn(user_id, tickers)`` callable non-blocking (fire-and-forget).
The run_fn is typically a background-queue post; the exact dispatch is injected
by the AgentTrader wiring layer so this tool has no direct asyncio dependency.
"""

from __future__ import annotations

import uuid
from typing import Any

from ._base import LookToolBase


class RequestResearchTool(LookToolBase):
    """Queue a research pass for a symbol; return a request_id immediately.

    Parameters
    ----------
    run_fn:
        ``Callable[[str, list[str]], None]`` — (user_id, tickers) → None.
        Called once to queue the research run.  Must not block.  ``None`` →
        tool returns a graceful "unavailable" error.
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "request_research",
        "description": (
            "Queue a new research pass for a symbol+question. Returns immediately "
            "with a request_id; check research_brief() on the next turn for the result."
        ),
        "args": {"symbol": "str", "question": "str"},
        "latency": "queued",
        "cost_class": "queued",
        "enabled": True,
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        run_fn: Any = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._run_fn = run_fn

    def __call__(self, symbol: str, question: str) -> Any:
        """Queue a research pass and return a request_id immediately.

        Returns
        -------
        ToolResult
            ok=True, data={"request_id": …, "symbol": …, "status": "queued",
                           "note": "check research_brief() next turn"}

        Example
        -------
        >>> tool = RequestResearchTool(trader_id="Alpha")
        >>> result = tool("AAPL", "What are catalysts for next quarter?")
        >>> result.ok
        False  # no run_fn wired
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return self._err("invalid_input", "symbol must not be empty")
        question = (question or "").strip()
        if not question:
            return self._err("invalid_input", "question must not be empty")

        if self._run_fn is None or self.owner_user_id is None:
            return self._err(
                "unavailable",
                "research agent not wired — ResearchAgent.run required for request_research()",
            )

        request_id = uuid.uuid4().hex[:12]
        try:
            self._run_fn(self.owner_user_id, [symbol])
        except Exception as exc:
            return self._err("internal", f"research queue failed: {exc}")

        return self._ok(
            {
                "request_id": request_id,
                "symbol": symbol,
                "question_preview": question[:120],
                "status": "queued",
                "note": "check research_brief() on the next turn for the result",
            }
        )
