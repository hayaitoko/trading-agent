"""``research_brief`` — retrieve the latest distilled research brief for a symbol.

Tool name:      research_brief
Args:           symbol (str)
ToolResult:     ok=True, data={"symbol": str, "brief": Brief|None}
Latency tier:   fast
Cost class:     free
Gating flag:    always enabled (None brief when no research available)
Example use:    research_brief("NVDA") to get the latest NVDA research summary.

data.brief includes: summary, sentiment (-1..1), catalysts, sources, ts.
Returns None when no brief exists yet for the symbol (queue one with
request_research).

Wraps :class:`~trading_agent.research.store.ResearchStore` (WS-C, read-only).
Research is shared per user — all traders under the same user see the same briefs.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase


class ResearchBriefTool(LookToolBase):
    """Read-only access to the shared per-user research brief store.

    Parameters
    ----------
    research_store:
        Duck-typed: must expose ``get(user_id, ticker) -> list[Brief]`` where
        each Brief has .summary/.sentiment/.catalysts/.sources/.ts attributes.
        ``None`` → empty result with a note.
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "research_brief",
        "description": (
            "Retrieve the latest distilled research brief for a symbol "
            "(LLM summary of recent news/social items, sentiment, catalysts)."
        ),
        "args": {"symbol": "str"},
        "latency": "fast",
        "cost_class": "free",
        "enabled": True,
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        research_store: Any = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._research = research_store

    def __call__(self, symbol: str) -> Any:
        """Return the latest brief for ``symbol``, or None if none available.

        Returns
        -------
        ToolResult
            ok=True, data={"symbol": …, "brief": {…}|None}

        Example
        -------
        >>> tool = ResearchBriefTool(trader_id="Alpha")
        >>> result = tool("AAPL")
        >>> result.ok
        True
        >>> result.data["brief"] is None
        True
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return self._err("invalid_input", "symbol must not be empty")

        if self._research is None or self.owner_user_id is None:
            return self._ok(
                {
                    "symbol": symbol,
                    "brief": None,
                    "note": "research store not wired",
                }
            )

        try:
            briefs = self._research.get(self.owner_user_id, symbol)
        except Exception as exc:
            return self._err("internal", f"research_brief fetch failed: {exc}")

        if not briefs:
            return self._ok(
                {
                    "symbol": symbol,
                    "brief": None,
                    "note": (
                        f"no brief yet for {symbol}. "
                        "Queue one with request_research(symbol, question)."
                    ),
                }
            )

        latest = briefs[0]
        brief_dict = {
            "id": getattr(latest, "id", ""),
            "summary": getattr(latest, "summary", ""),
            "sentiment": float(getattr(latest, "sentiment", 0.0)),
            "catalysts": list(getattr(latest, "catalysts", [])),
            "sources": list(getattr(latest, "sources", [])),
            "ts": str(getattr(latest, "ts", "")),
        }
        return self._ok({"symbol": symbol, "brief": brief_dict})
