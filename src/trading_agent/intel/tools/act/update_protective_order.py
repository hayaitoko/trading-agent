"""ACT tool: ``update_protective_order`` — edit stop/TP/trail on an open position.

Design role: non-terminal ACT tool that lets the agent tighten a stop or adjust
a take-profit without submitting a new full trade.  Does not require re-approval
(protective order management is risk-neutral on direction).

Kill-switch path: blocked when active — no protective-order changes during a halt.

Latency tier: fast
Cost class: free
"""

from __future__ import annotations

from typing import Any

from ...tool_envelope import ToolResult
from ._base import ActToolBase

DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "update_protective_order",
        "description": (
            "Edit the stop-loss, take-profit, or trailing stop on an existing open "
            "position.  Does not require approval and is NOT a terminal action — "
            "you can continue calling tools afterward."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID of the position to update.",
                },
                "new_stop": {
                    "type": "number",
                    "description": "New stop-loss price (optional).",
                },
                "new_tp": {
                    "type": "number",
                    "description": "New take-profit price (optional).",
                },
                "new_trail": {
                    "type": "number",
                    "description": "New trailing stop distance in dollars (optional).",
                },
            },
            "required": ["order_id"],
        },
    },
}

CATALOG_ENTRY: dict[str, Any] = {
    "name": "update_protective_order",
    "description": "Edit stop-loss, take-profit, or trailing stop on an existing order. Not terminal.",
    "args": {
        "order_id": "str",
        "new_stop": "number (optional)",
        "new_tp": "number (optional)",
        "new_trail": "number (optional)",
    },
    "latency": "fast",
    "cost_class": "free",
    "enabled": True,
    "disabled_reason": None,
}


class UpdateProtectiveOrderTool(ActToolBase):
    """Executes the ``update_protective_order(order_id, ...)`` tool call."""

    def run(
        self,
        order_id: str,
        *,
        new_stop: float | None = None,
        new_tp: float | None = None,
        new_trail: float | None = None,
    ) -> ToolResult:
        """Update protective order parameters.

        Returns ok=True, data={order_id, updated: {...}, broker_result: {...}} on
        success.  Returns ok=False on kill switch, missing broker, not-found, or
        broker error.
        """
        if not order_id:
            return self._err("invalid_input", "order_id must not be empty")
        if new_stop is None and new_tp is None and new_trail is None:
            return self._err(
                "invalid_input",
                "at least one of new_stop, new_tp, new_trail must be provided",
            )

        rm = self.risk_manager
        if rm is not None and getattr(rm, "kill_switch_active", False):
            return self._err("unavailable", "bench halted by operator")

        broker = self.broker
        if broker is None:
            return self._err("unavailable", "broker not configured")

        update: dict[str, Any] = {"order_id": order_id, "order_type": "update_protective"}
        if new_stop is not None:
            update["stop"] = float(new_stop)
        if new_tp is not None:
            update["take_profit"] = float(new_tp)
        if new_trail is not None:
            update["trail"] = float(new_trail)

        try:
            result = broker.place_order(update)
        except Exception as exc:
            return self._err("internal", f"broker error: {exc}")

        if result is None:
            return self._err(
                "not_found", f"order {order_id!r} not found or update rejected by broker"
            )

        return self._ok({"order_id": order_id, "updated": update, "broker_result": result})
