"""NOTE tool: ``unwatch_symbol`` — remove a symbol from the trader's personal watchlist.

**Design role:** complement to :mod:`~.watch_symbol`.  Idempotent — removing a
symbol not on the list returns success without error.

Storage, latency, cost: identical to :mod:`~.watch_symbol`.
"""

from __future__ import annotations

from typing import Any

from ...tool_envelope import ToolResult
from ._base import NoteToolBase
from .watch_symbol import _load_watchlist, _save_watchlist

DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "unwatch_symbol",
        "description": (
            "Remove a symbol from your personal watchlist. Idempotent — "
            "removing a symbol not on the list is fine."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol to remove, e.g. 'MSFT'.",
                }
            },
            "required": ["symbol"],
        },
    },
}

CATALOG_ENTRY: dict[str, Any] = {
    "name": "unwatch_symbol",
    "description": "Remove a symbol from your personal watchlist.",
    "args": {"symbol": "str"},
    "latency": "instant",
    "cost_class": "free",
    "enabled": True,
    "disabled_reason": None,
}


class UnwatchSymbolTool(NoteToolBase):
    """Executes the ``unwatch_symbol(symbol)`` tool call.

    Parameters
    ----------
    settings_store:
        The per-user :class:`~trading_agent.config.settings_store.SettingsStore`
        (may be ``None`` — tool degrades gracefully).
    """

    def __init__(self, *, settings_store: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.settings_store = settings_store

    def run(self, symbol: str) -> ToolResult:
        """Remove ``symbol`` from the trader's watchlist.

        Returns ``{ok: true, data: {symbol, watchlist: [...], removed: bool}}``.
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return self._err("invalid_input", "'symbol' must not be empty")

        watchlist = _load_watchlist(
            self.settings_store, self.owner_user_id, self.trader_id
        )
        removed = symbol in watchlist
        if removed:
            watchlist = [s for s in watchlist if s != symbol]
            _save_watchlist(
                self.settings_store, self.owner_user_id, self.trader_id, watchlist
            )

        return self._ok({"symbol": symbol, "watchlist": watchlist, "removed": removed})
