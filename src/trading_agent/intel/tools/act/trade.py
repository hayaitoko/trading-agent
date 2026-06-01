"""ACT tool: ``trade`` — submit a single trade intent.

Design role: primary ACT entry point.  Routes through the pre-approval queue
when ``requires_approval=True`` (returns pending_trade_id, turn ends) or
executes directly via the broker otherwise.

MONEY IS REAL invariant: no "paper", "sim", "demo", or "fake" strings appear
in any response field visible to the trader.  ``_scrub_fill`` strips them from
broker result values before they reach the message list.

Kill-switch path: when the risk manager's kill switch is active, returns a clean
``unavailable`` error so the trader can ``pass()``/``hold()`` gracefully.

Idempotency: key = hash(trader_id, turn_id, symbol, side, qty).  The risk
manager rejects duplicate keys so a crash-restart cannot double-fire a trade
that was already submitted in the same turn.

Latency tier: fast (queued) / medium (direct execution)
Cost class: free
Gating flag: (none — trade is always in the ACT catalog when broker is wired)
"""

from __future__ import annotations

from typing import Any

from ...tool_envelope import ToolResult
from ._base import ActToolBase, _idempotency_key, _scrub_fill

DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "trade",
        "description": (
            "Submit a trade order for a single symbol.  If approval is required, "
            "returns a pending_trade_id and ends your turn — you will be called back "
            "when the decision arrives.  If pre-approved or no approval needed, "
            "executes immediately and returns the fill result.  "
            "This is a terminal action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol, e.g. 'AAPL'.",
                },
                "side": {
                    "type": "string",
                    "enum": ["BUY", "SELL"],
                    "description": "Order direction.",
                },
                "qty": {
                    "type": "number",
                    "description": "Quantity in whole shares.",
                },
                "stop": {
                    "type": "number",
                    "description": "Stop-loss price (optional).",
                },
                "take_profit": {
                    "type": "number",
                    "description": "Take-profit price (optional).",
                },
                "trail": {
                    "type": "number",
                    "description": "Trailing stop distance in dollars (optional).",
                },
            },
            "required": ["symbol", "side", "qty"],
        },
    },
}

CATALOG_ENTRY: dict[str, Any] = {
    "name": "trade",
    "description": (
        "Submit a trade order (terminal). Returns pending_trade_id awaiting approval, "
        "or immediate fill result if executed directly."
    ),
    "args": {
        "symbol": "str",
        "side": "'BUY' | 'SELL'",
        "qty": "number",
        "stop": "number (optional)",
        "take_profit": "number (optional)",
        "trail": "number (optional)",
    },
    "latency": "fast",
    "cost_class": "free",
    "enabled": True,
    "disabled_reason": None,
}


class TradeTool(ActToolBase):
    """Executes the ``trade(symbol, side, qty, ...)`` tool call."""

    def run(
        self,
        symbol: str,
        side: str,
        qty: float,
        *,
        stop: float | None = None,
        take_profit: float | None = None,
        trail: float | None = None,
    ) -> ToolResult:
        """Submit a trade intent through risk gating + approval or direct execution.

        Returns:
            ok=True, data={pending_trade_id, status='awaiting_approval', ...}
                when routed through the approval queue.
            ok=True, data={fill: {...}, symbol, side, qty}
                when executed immediately.
            ok=False, error=... on kill switch, invalid input, or broker error.
        """
        symbol = (symbol or "").upper().strip()
        side = (side or "").upper().strip()

        if not symbol:
            return self._err("invalid_input", "symbol must not be empty")
        if side not in ("BUY", "SELL"):
            return self._err("invalid_input", f"side must be 'BUY' or 'SELL', got {side!r}")
        try:
            qty = float(qty)
        except (TypeError, ValueError):
            return self._err("invalid_input", "qty must be a number")
        if qty <= 0:
            return self._err("invalid_input", "qty must be positive")

        # Kill-switch: block new trades; LOOK/NOTE tools still work.
        rm = self.risk_manager
        if rm is not None and getattr(rm, "kill_switch_active", False):
            return self._err("unavailable", "bench halted by operator", retry_after=None)

        # Idempotency guard against crash-replay double-fires.
        idem_key = _idempotency_key(self.trader_id, self.turn_id, symbol, side, qty)
        if rm is not None:
            if rm.check_idempotency(idem_key):
                return self._err(
                    "invalid_input",
                    f"duplicate trade detected (key={idem_key[:8]}…). "
                    "This trade was already submitted in the current turn.",
                )
            rm.record_idempotency(idem_key)

        from ....approval_queue import TradeIntent

        intent = TradeIntent(
            symbol=symbol,
            side=side,
            qty=qty,
            stop=stop,
            take_profit=take_profit,
            trail=trail,
        )

        # Approval-required path: enqueue, register scheduler callback, return id.
        ptq = self.pending_trade_queue
        if self.requires_approval and ptq is not None:
            try:
                pt = ptq.propose(self.trader_id, intent, idem_key)
            except ValueError as exc:
                return self._err("invalid_input", str(exc))
            # A4-b wiring: register a callback so approve / deny / TTL-expire events
            # schedule a callback turn for this trader via the MarketScheduler.  When
            # no scheduler is attached (tests, pre-A4 compat) the approval still lands
            # in the DB — the trader simply won't wake autonomously to act on it.
            if self.scheduler is not None:
                self.scheduler.wire_pending_trade_callbacks(
                    ptq, self.trader_id, pt.pending_trade_id
                )
            return self._ok(
                {
                    "pending_trade_id": pt.pending_trade_id,
                    "status": "awaiting_approval",
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "approval_ttl_min": ptq.PREAPPROVAL_TTL_MIN,
                }
            )

        # Direct execution path.
        broker = self.broker
        if broker is None:
            return self._err("unavailable", "broker not configured")

        order_details: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "order_type": "market",
            "quantity": qty,
        }
        if stop is not None:
            order_details["stop"] = stop
        if take_profit is not None:
            order_details["take_profit"] = take_profit
        if trail is not None:
            order_details["trail"] = trail

        try:
            result = broker.place_order(order_details)
        except Exception as exc:
            return self._err("internal", f"broker error: {exc}")

        if result is None:
            return self._err("internal", "broker rejected the order (no result)")

        return self._ok(
            {"fill": _scrub_fill(result), "symbol": symbol, "side": side, "qty": qty}
        )
