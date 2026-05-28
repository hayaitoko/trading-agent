"""``options_iv`` — implied volatility and Greeks for a symbol (DISABLED stub).

Tool name:      options_iv
Args:           symbol (str)
ToolResult:     ok=False, error=ToolError(kind="disabled", …)
Latency tier:   fast
Cost class:     free
Gating flag:    enabled=False — provider lands in WS-Situation Track A

This is an intentional stub.  When WS-Situation Track A (options-IV provider via
OptionQuote.implied_vol/greeks) lands, only the stub gets unwired — this wrapper
class and its slot in list_tools() stay.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase


class OptionsIVTool(LookToolBase):
    """Disabled stub for near-money options implied volatility + Greeks.

    Parameters
    ----------
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "options_iv",
        "description": (
            "Implied volatility and Greeks for a symbol's near-money options. "
            "Provider lands in WS-Situation Track A."
        ),
        "args": {"symbol": "str"},
        "latency": "fast",
        "cost_class": "free",
        "enabled": False,
        "disabled_reason": "provider lands in WS-Situation+Forecast",
    }

    def __call__(self, symbol: str) -> Any:
        """Return a disabled error.

        Returns
        -------
        ToolResult
            ok=False, error={kind: "disabled", message: "…provider lands in WS-Situation+Forecast"}

        Example
        -------
        >>> tool = OptionsIVTool(trader_id="Alpha")
        >>> result = tool("AAPL")
        >>> result.ok
        False
        >>> result.error.kind
        'disabled'
        """
        return self._disabled("options_iv")
