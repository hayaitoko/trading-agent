"""``watchlist`` — retrieve this trader's symbol watchlist.

Tool name:      watchlist
Args:           (none)
ToolResult:     ok=True, data={"symbols": [str, …], "source": "trader"|"operator"|"combined"}
Latency tier:   instant
Cost class:     free
Gating flag:    always enabled
Example use:    watchlist() to see which symbols you are monitoring.

Returns the union of:
  1. Trader-maintained watchlist (symbols added via A2 watch_symbol / unwatch_symbol).
  2. Operator-pinned watchlist (from user settings / cockpit manual additions).

When A2 NOTE tools land, trader_symbols is populated by the attention queue's
watch_symbol set.  In A1 it is read from an injected list (defaulting to the
trader's full universe until A2 wires the real watchlist store).
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase


class WatchlistTool(LookToolBase):
    """Read-only access to the trader's combined watchlist.

    Parameters
    ----------
    trader_symbols:
        Symbols the trader has added via watch_symbol (A2).  May be empty.
    operator_symbols:
        Symbols the operator has pinned in the cockpit for this trader.
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "watchlist",
        "description": (
            "Your own watchlist of symbols (maintained via watch_symbol / unwatch_symbol), "
            "overlaid with any symbols the operator has pinned for you."
        ),
        "args": {},
        "latency": "instant",
        "cost_class": "free",
        "enabled": True,
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        trader_symbols: list[str] | None = None,
        operator_symbols: list[str] | None = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._trader_syms: list[str] = list(trader_symbols or [])
        self._operator_syms: list[str] = list(operator_symbols or [])

    def __call__(self) -> Any:
        """Return the combined watchlist.

        Returns
        -------
        ToolResult
            ok=True, data={"symbols": […], "source": "…",
                           "trader_symbols": […], "operator_symbols": […]}

        Example
        -------
        >>> tool = WatchlistTool(trader_id="Alpha", trader_symbols=["AAPL"])
        >>> result = tool()
        >>> result.ok
        True
        >>> "AAPL" in result.data["symbols"]
        True
        """
        # Union preserving order: trader first, then operator additions.
        seen: set[str] = set()
        combined: list[str] = []
        for sym in self._trader_syms + self._operator_syms:
            upper = sym.strip().upper()
            if upper and upper not in seen:
                seen.add(upper)
                combined.append(upper)

        if self._trader_syms and self._operator_syms:
            source = "combined"
        elif self._trader_syms:
            source = "trader"
        elif self._operator_syms:
            source = "operator"
        else:
            source = "empty"

        return self._ok(
            {
                "symbols": combined,
                "source": source,
                "trader_symbols": [s.upper() for s in self._trader_syms],
                "operator_symbols": [s.upper() for s in self._operator_syms],
            }
        )
