"""``world_events`` — GDELT-based global event feed (DISABLED stub).

Tool name:      world_events
Args:           theme=None (str|None), timespan="24h" (str)
ToolResult:     ok=False, error=ToolError(kind="disabled", …)
Latency tier:   medium (15-min cache when enabled)
Cost class:     free
Gating flag:    enabled=False — provider lands in WS-Situation Track A

This is an intentional stub.  When WS-Situation Track A (GDELT provider) lands,
only the stub get unwired — this wrapper class and its slot in list_tools() stay.
``list_tools()`` surfaces this tool with ``enabled=false`` and the disabled_reason
so the model understands what is coming and can plan around the absence.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase


class WorldEventsTool(LookToolBase):
    """Disabled stub for GDELT-based global event feed.

    Parameters
    ----------
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "world_events",
        "description": (
            "GDELT-based global event feed filtered by theme and timespan. "
            "Provider lands in WS-Situation Track A."
        ),
        "args": {
            "theme": "str|None (default None)",
            "timespan": "str (default '24h')",
        },
        "latency": "medium",
        "cost_class": "free",
        "enabled": False,
        "disabled_reason": "provider lands in WS-Situation+Forecast",
    }

    def __call__(
        self,
        theme: str | None = None,
        timespan: str = "24h",
    ) -> Any:
        """Return a disabled error.

        Returns
        -------
        ToolResult
            ok=False, error={kind: "disabled", message: "…provider lands in WS-Situation+Forecast"}

        Example
        -------
        >>> tool = WorldEventsTool(trader_id="Alpha")
        >>> result = tool()
        >>> result.ok
        False
        >>> result.error.kind
        'disabled'
        """
        return self._disabled("world_events")
