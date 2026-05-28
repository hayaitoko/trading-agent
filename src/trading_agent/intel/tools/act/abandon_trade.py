"""ACT tool: ``abandon_trade`` — release a pre-approved trade unused.

Design role: callback-turn tool.  After an approval notification, the trader may
decide the opportunity has passed (market moved, changed thesis) and calls
abandon_trade() instead of confirm_trade().  The pending trade is released with no
fill; the slot is freed.

This is a terminal action — calling it ends the turn.

Latency tier: instant
Cost class: free
"""

from __future__ import annotations

from typing import Any

from ...tool_envelope import ToolResult
from ._base import ActToolBase

DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "abandon_trade",
        "description": (
            "Release a pre-approved trade without executing it.  Use in a callback "
            "turn when the situation has changed and you no longer want to take the "
            "position.  No fill is generated.  This is a terminal action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pending_trade_id": {
                    "type": "string",
                    "description": "The pending_trade_id to release.",
                },
            },
            "required": ["pending_trade_id"],
        },
    },
}

CATALOG_ENTRY: dict[str, Any] = {
    "name": "abandon_trade",
    "description": "Release a pre-approved trade unused (terminal, callback-turn only).",
    "args": {"pending_trade_id": "str"},
    "latency": "instant",
    "cost_class": "free",
    "enabled": True,
    "disabled_reason": None,
}


class AbandonTradeTool(ActToolBase):
    """Executes the ``abandon_trade(pending_trade_id)`` tool call."""

    def run(self, pending_trade_id: str) -> ToolResult:
        """Release a pre-approved trade without executing it.

        Returns ok=True, data={pending_trade_id, status='abandoned', symbol, side, qty}.
        Returns ok=False on not_found, wrong status, or missing queue.
        """
        if not pending_trade_id:
            return self._err("invalid_input", "pending_trade_id must not be empty")

        ptq = self.pending_trade_queue
        if ptq is None:
            return self._err("unavailable", "pending trade queue not configured")

        try:
            pt = ptq.abandon(pending_trade_id)
        except KeyError:
            return self._err("not_found", f"pending trade {pending_trade_id!r} not found")
        except ValueError as exc:
            return self._err("invalid_input", str(exc))
        except Exception as exc:
            return self._err("internal", str(exc))

        return self._ok(
            {
                "pending_trade_id": pending_trade_id,
                "status": "abandoned",
                "symbol": pt.proposed.symbol,
                "side": pt.proposed.side,
                "qty": pt.proposed.qty,
            }
        )
