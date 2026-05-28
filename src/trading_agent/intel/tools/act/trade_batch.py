"""ACT tool: ``trade_batch`` — submit multiple trade intents in one terminal turn.

Design role: lets the agent open/close several positions simultaneously without
burning multiple turns.  Each item in the batch is independently routed through
risk + approval; the results list mirrors the input order so the agent can reason
about which items succeeded and which need follow-up.

Kill-switch path: when active, the entire batch is rejected before any item is
submitted — no partial execution.

Idempotency: each item gets its own key via the shared risk_manager, so a
crash-replay will block every duplicate item individually.

Latency tier: fast (items can be approval-queued or filled directly)
Cost class: free
"""

from __future__ import annotations

from typing import Any

from ...tool_envelope import ToolResult
from ._base import ActToolBase

DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "trade_batch",
        "description": (
            "Submit multiple trade orders in a single terminal turn.  "
            "Each item is independently evaluated for risk and approval.  "
            "Returns a per-item result list.  This is a terminal action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trades": {
                    "type": "array",
                    "description": "List of trade intents.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "side": {"type": "string", "enum": ["BUY", "SELL"]},
                            "qty": {"type": "number"},
                            "stop": {"type": "number"},
                            "take_profit": {"type": "number"},
                            "trail": {"type": "number"},
                        },
                        "required": ["symbol", "side", "qty"],
                    },
                }
            },
            "required": ["trades"],
        },
    },
}

CATALOG_ENTRY: dict[str, Any] = {
    "name": "trade_batch",
    "description": (
        "Submit multiple trade orders in one terminal turn. "
        "Returns per-item results (fill or pending_trade_id per item)."
    ),
    "args": {"trades": "list[{symbol, side, qty, stop?, take_profit?, trail?}]"},
    "latency": "fast",
    "cost_class": "free",
    "enabled": True,
    "disabled_reason": None,
}


class TradeBatchTool(ActToolBase):
    """Executes the ``trade_batch([{symbol, side, qty, ...}, ...])`` tool call."""

    def run(self, trades: list[dict[str, Any]]) -> ToolResult:
        """Submit a batch of trade intents.

        Returns ok=True, data={results: [...]} where each element is::

            {"symbol": ..., "side": ..., "qty": ...,
             "result": {"ok": ..., "data": ...} | {"ok": false, "error": ...}}

        Returns ok=False only when the kill switch is active (whole-batch reject)
        or the trades list is empty.  Per-item errors are contained inside
        ``results`` so the batch always completes even when individual items fail.
        """
        if not trades:
            return self._err("invalid_input", "trades list must not be empty")

        rm = self.risk_manager
        if rm is not None and getattr(rm, "kill_switch_active", False):
            return self._err("unavailable", "bench halted by operator", retry_after=None)

        from .trade import TradeTool

        results = []
        for item in trades:
            # Each item shares the risk_manager so idempotency keys accumulate
            # across the batch — duplicate symbols+side+qty are caught.
            single = TradeTool(
                broker=self.broker,
                risk_manager=self.risk_manager,
                pending_trade_queue=self.pending_trade_queue,
                trader_id=self.trader_id,
                turn_id=self.turn_id,
                requires_approval=self.requires_approval,
            )
            symbol = str(item.get("symbol", "")).upper().strip()
            side = str(item.get("side", "")).upper().strip()
            try:
                qty = float(item.get("qty", 0))
            except (TypeError, ValueError):
                qty = 0.0

            res = single.run(
                symbol,
                side,
                qty,
                stop=item.get("stop"),
                take_profit=item.get("take_profit"),
                trail=item.get("trail"),
            )
            results.append(
                {"symbol": symbol, "side": side, "qty": qty, "result": res.to_dict()}
            )

        return self._ok({"results": results})
