"""NOTE tool: ``watch_symbol`` — add a symbol to the trader's personal watchlist.

**Design role:** the trader maintains a per-trader watchlist stored in
``user_settings`` under the key ``trader_watchlist:<trader_id>``.  This list
overlays the operator's manual watchlist in the cockpit (A5 tile) and is
available to the trader via the LOOK tool ``watchlist()`` (A1).

This is distinct from a watchpoint (which is a transient price-condition fire);
the watchlist is a durable "I care about these symbols" signal.

**Idempotent:** adding a symbol already on the watchlist returns success without
duplication.

**Storage:** ``user_settings`` table, key = ``trader_watchlist:<trader_id>``,
value = JSON array of uppercase symbol strings.  Absent key ≡ empty list.

**Latency tier:** instant (local DB read/write)
**Cost class:** free
**Gating flag:** (none — always enabled)
"""

from __future__ import annotations

import json
from typing import Any

from ...tool_envelope import ToolResult
from ._base import NoteToolBase

DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "watch_symbol",
        "description": (
            "Add a symbol to your personal watchlist. "
            "Your watchlist overlays the operator's cockpit view and is visible via "
            "the watchlist() LOOK tool. Idempotent — adding a symbol twice is fine."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol to add, e.g. 'MSFT'.",
                }
            },
            "required": ["symbol"],
        },
    },
}

CATALOG_ENTRY: dict[str, Any] = {
    "name": "watch_symbol",
    "description": "Add a symbol to your personal watchlist.",
    "args": {"symbol": "str"},
    "latency": "instant",
    "cost_class": "free",
    "enabled": True,
    "disabled_reason": None,
}


class WatchSymbolTool(NoteToolBase):
    """Executes the ``watch_symbol(symbol)`` tool call.

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
        """Add ``symbol`` to the trader's watchlist.

        Returns ``{ok: true, data: {symbol, watchlist: [...], added: bool}}``.
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return self._err("invalid_input", "'symbol' must not be empty")

        watchlist = _load_watchlist(
            self.settings_store, self.owner_user_id, self.trader_id
        )
        added = symbol not in watchlist
        if added:
            watchlist.append(symbol)
            _save_watchlist(
                self.settings_store, self.owner_user_id, self.trader_id, watchlist
            )

        return self._ok({"symbol": symbol, "watchlist": watchlist, "added": added})


# ── helpers ───────────────────────────────────────────────────────────────────


def _watchlist_key(trader_id: str) -> str:
    return f"trader_watchlist:{trader_id}"


def _load_watchlist(
    settings: Any, owner_user_id: str | None, trader_id: str
) -> list[str]:
    if settings is None or owner_user_id is None:
        return []
    try:
        raw = settings.get(owner_user_id, _watchlist_key(trader_id), None)
        if raw is None:
            return []
        return list(json.loads(raw))
    except Exception:
        return []


def _save_watchlist(
    settings: Any, owner_user_id: str | None, trader_id: str, watchlist: list[str]
) -> None:
    if settings is None or owner_user_id is None:
        return
    try:
        settings.set(owner_user_id, _watchlist_key(trader_id), json.dumps(watchlist))
    except Exception:
        pass
