"""``forecast`` — forward price-cone forecast for a symbol (DISABLED stub).

Tool name:      forecast
Args:           symbol (str), horizon=5 (5|10|30)
ToolResult:     ok=False, error=ToolError(kind="disabled", …)
Latency tier:   medium
Cost class:     free
Gating flag:    enabled=False — provider lands in WS-Situation Track C

When WS-Situation Track C (intel/forecast.py) lands, the stub gets unwired and
this wrapper calls the real 1σ price-cone implementation.  The wrapper class and
its slot in list_tools() remain unchanged so the agent's tool catalog is stable
before and after the provider lands.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase


class ForecastTool(LookToolBase):
    """Disabled stub for the forward price-cone forecast.

    Parameters
    ----------
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "forecast",
        "description": (
            "Forward 1σ price-cone forecast for a symbol over 5/10/30 day horizon "
            "combining realized vol, options IV, and prediction-market implied move. "
            "Provider lands in WS-Situation Track C."
        ),
        "args": {"symbol": "str", "horizon": "5 | 10 | 30"},
        "latency": "medium",
        "cost_class": "free",
        "enabled": False,
        "disabled_reason": "provider lands in WS-Situation+Forecast",
    }

    def __call__(self, symbol: str, horizon: int = 5) -> Any:
        """Return a disabled error.

        Returns
        -------
        ToolResult
            ok=False, error={kind: "disabled", message: "…provider lands in WS-Situation+Forecast"}

        Example
        -------
        >>> tool = ForecastTool(trader_id="Alpha")
        >>> result = tool("AAPL", horizon=10)
        >>> result.ok
        False
        >>> result.error.kind
        'disabled'
        """
        return self._disabled("forecast")
