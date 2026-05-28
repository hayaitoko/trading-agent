"""``prediction_market_odds`` — Polymarket/Kalshi implied probabilities (DISABLED stub).

Tool name:      prediction_market_odds
Args:           category (str), query=None (str|None)
ToolResult:     ok=False, error=ToolError(kind="disabled", …)
Latency tier:   medium
Cost class:     free
Gating flag:    enabled=False — provider lands in WS-Situation Track A

This is an intentional stub.  When WS-Situation Track A (prediction-markets provider)
lands, only the stub gets unwired — this wrapper class and its slot in list_tools() stay.
``list_tools()`` surfaces this tool with ``enabled=false`` so the model knows it's coming.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase


class PredictionMarketOddsTool(LookToolBase):
    """Disabled stub for Polymarket / Kalshi prediction-market odds.

    Parameters
    ----------
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "prediction_market_odds",
        "description": (
            "Polymarket / Kalshi implied probabilities for a category/query. "
            "Provider lands in WS-Situation Track A."
        ),
        "args": {
            "category": "str",
            "query": "str|None (default None)",
        },
        "latency": "medium",
        "cost_class": "free",
        "enabled": False,
        "disabled_reason": "provider lands in WS-Situation+Forecast",
    }

    def __call__(
        self,
        category: str,
        query: str | None = None,
    ) -> Any:
        """Return a disabled error.

        Returns
        -------
        ToolResult
            ok=False, error={kind: "disabled", message: "…provider lands in WS-Situation+Forecast"}

        Example
        -------
        >>> tool = PredictionMarketOddsTool(trader_id="Alpha")
        >>> result = tool("fed_rate_decision")
        >>> result.ok
        False
        >>> result.error.kind
        'disabled'
        """
        return self._disabled("prediction_market_odds")
