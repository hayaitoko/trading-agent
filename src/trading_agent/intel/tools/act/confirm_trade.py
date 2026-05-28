"""ACT tool: ``confirm_trade`` — execute a pre-approved trade.

Design role: callback-turn tool.  After the operator approves a pending trade,
the AgentTrader is woken with a callback turn.  The trader calls this tool to
actually execute the fill via the broker's pre-approved path.

Pre-approval TTL is ``PREAPPROVAL_TTL_MIN`` minutes (default 5).  If the TTL
elapsed before confirm_trade is called, returns an expiration error so the
trader can reassess.

This is a terminal action — calling it ends the turn.

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
        "name": "confirm_trade",
        "description": (
            "Execute a pre-approved trade.  Call this in your callback turn after "
            "receiving an approval notification.  Returns the fill result, or an "
            "expiration error if the pre-approval TTL elapsed.  "
            "This is a terminal action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pending_trade_id": {
                    "type": "string",
                    "description": (
                        "The pending_trade_id returned by trade() when the trade "
                        "was queued for approval."
                    ),
                },
            },
            "required": ["pending_trade_id"],
        },
    },
}

CATALOG_ENTRY: dict[str, Any] = {
    "name": "confirm_trade",
    "description": (
        "Execute a pre-approved trade (terminal, callback-turn only). "
        "Returns fill result or expiration error."
    ),
    "args": {"pending_trade_id": "str"},
    "latency": "fast",
    "cost_class": "free",
    "enabled": True,
    "disabled_reason": None,
}


class ConfirmTradeTool(ActToolBase):
    """Executes the ``confirm_trade(pending_trade_id)`` tool call."""

    def run(self, pending_trade_id: str) -> ToolResult:
        """Execute a pre-approved trade via the pending_trade_queue.

        Returns ok=True, data={pending_trade_id, fill: {order_id, symbol, side,
        qty_filled, fill_price, status}} on success.

        Returns ok=False on:
        - kill switch active (unavailable)
        - queue not configured (unavailable)
        - broker not configured (unavailable)
        - pending_trade_id not found (not_found)
        - wrong status / TTL expired (unavailable)
        - broker error (internal)
        """
        if not pending_trade_id:
            return self._err("invalid_input", "pending_trade_id must not be empty")

        rm = self.risk_manager
        if rm is not None and getattr(rm, "kill_switch_active", False):
            return self._err("unavailable", "bench halted by operator")

        ptq = self.pending_trade_queue
        if ptq is None:
            return self._err("unavailable", "pending trade queue not configured")

        broker = self.broker
        if broker is None:
            return self._err("unavailable", "broker not configured")

        def _executor(intent: Any) -> Any:
            from ....approval_queue import FillResult

            order_details: dict[str, Any] = {
                "symbol": intent.symbol,
                "side": intent.side,
                "order_type": "market",
                "quantity": intent.qty,
            }
            if intent.stop is not None:
                order_details["stop"] = intent.stop
            if intent.take_profit is not None:
                order_details["take_profit"] = intent.take_profit
            if intent.trail is not None:
                order_details["trail"] = intent.trail

            result = broker.place_order(order_details)
            if result is None:
                raise RuntimeError("broker rejected the order (no result)")

            return FillResult(
                order_id=str(result.get("order_id", "?")),
                symbol=intent.symbol,
                side=intent.side,
                qty_filled=float(result.get("filled_quantity", intent.qty)),
                fill_price=result.get("filled_price"),
                status=str(result.get("status", "filled")),
            )

        try:
            _pt, fill = ptq.confirm(pending_trade_id, _executor)
        except KeyError:
            return self._err("not_found", f"pending trade {pending_trade_id!r} not found")
        except ValueError as exc:
            return self._err("unavailable", str(exc))
        except RuntimeError as exc:
            return self._err("internal", str(exc))
        except Exception as exc:
            return self._err("internal", f"confirm failed: {exc}")

        return self._ok(
            {
                "pending_trade_id": pending_trade_id,
                "fill": {
                    "order_id": fill.order_id,
                    "symbol": fill.symbol,
                    "side": fill.side,
                    "qty_filled": fill.qty_filled,
                    "fill_price": fill.fill_price,
                    "status": fill.status,
                },
            }
        )
