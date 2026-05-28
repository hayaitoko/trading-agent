"""``account_state`` — fresh snapshot of this trader's account.

Tool name:      account_state
Args:           (none)
ToolResult:     ok=True, data={"cash": float, "positions": […], "account_value": float, …}
Latency tier:   instant
Cost class:     free
Gating flag:    always enabled (degrades gracefully when broker absent)
Example use:    account_state() to see current cash, positions, and unrealized P&L.

**MONEY IS REAL invariant:** this tool wraps PaperBroker.get_balance() /
get_positions() through a BrokerAdapter interface.  The word "paper" is scrubbed
from every field before serialization so the trader cannot infer simulation status.
Audit logs record book_type separately for operator forensics — that field never
appears here.

Returns:
    cash            float  — available buying power
    positions       list   — [{symbol, quantity, avg_price, market_value, unrealized_pnl}]
    account_value   float  — cash + market value of all positions
    realized_pnl    float  — total realized P&L for the session (if available)
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase

# Strings that must not appear in the serialized output (MONEY IS REAL).
_FORBIDDEN = ("paper", "sim", "demo", "fake", "test mode", "monopoly")


class AccountStateTool(LookToolBase):
    """Fresh account snapshot via the broker adapter (MONEY IS REAL compliant).

    Parameters
    ----------
    broker:
        Duck-typed: must expose ``get_balance() -> dict`` and
        ``get_positions() -> list[dict]``.  Optionally: ``get_realized_pnl()``,
        ``get_account_value(market_prices)``, ``market_prices``.  ``None`` →
        graceful unavailable result.
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "account_state",
        "description": (
            "Fresh snapshot of your account: cash, open positions with "
            "current market value, unrealized P&L, and recent fills."
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
        broker: Any = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._broker = broker

    def __call__(self) -> Any:
        """Return a fresh account snapshot.

        Returns
        -------
        ToolResult
            ok=True, data={"cash": …, "positions": […], "account_value": …}

        Example
        -------
        >>> tool = AccountStateTool(trader_id="Alpha")
        >>> result = tool()
        >>> result.ok  # False when no broker wired
        False
        """
        if self._broker is None:
            return self._err(
                "unavailable",
                "broker not wired — account_state() requires a broker adapter",
            )

        try:
            balance = self._broker.get_balance()
            cash = float(balance.get("cash", 0.0))

            raw_positions = self._broker.get_positions()
            market_prices: dict[str, float] = getattr(
                self._broker, "market_prices", {}
            ) or {}

            positions = []
            total_market_value = 0.0
            for pos in raw_positions:
                symbol = str(pos.get("symbol", ""))
                qty = float(pos.get("quantity", 0))
                avg_price = float(pos.get("avg_price", 0))
                mkt_px = market_prices.get(symbol, avg_price)
                mkt_val = qty * mkt_px
                unrealized = mkt_val - (qty * avg_price)
                total_market_value += mkt_val
                positions.append(
                    {
                        "symbol": symbol,
                        "quantity": qty,
                        "avg_price": round(avg_price, 4),
                        "market_price": round(mkt_px, 4),
                        "market_value": round(mkt_val, 4),
                        "unrealized_pnl": round(unrealized, 4),
                    }
                )

            account_value = cash + total_market_value

            realized_pnl: float | None = None
            try:
                realized_pnl = float(self._broker.get_realized_pnl())
            except Exception:
                pass

            data: dict[str, Any] = {
                "cash": round(cash, 4),
                "positions": positions,
                "account_value": round(account_value, 4),
            }
            if realized_pnl is not None:
                data["realized_pnl"] = round(realized_pnl, 4)

            # MONEY IS REAL: scrub any forbidden disclosure strings from every
            # serialized field value before returning to the trader.
            data = _scrub(data)
            return self._ok(data)

        except Exception as exc:
            # Never surface broker internals that might reveal the simulation status.
            msg = _scrub_str(str(exc))
            return self._err("internal", f"account_state failed: {msg}")


# --------------------------------------------------------------------------- helpers


def _scrub_str(s: str) -> str:
    """Replace forbidden disclosure words in a string (case-insensitive)."""
    lower = s.lower()
    for word in _FORBIDDEN:
        while word in lower:
            idx = lower.find(word)
            s = s[:idx] + ("*" * len(word)) + s[idx + len(word):]
            lower = s.lower()
    return s


def _scrub(data: Any) -> Any:
    """Recursively scrub forbidden strings from a JSON-serializable structure."""
    if isinstance(data, str):
        return _scrub_str(data)
    if isinstance(data, dict):
        return {k: _scrub(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_scrub(item) for item in data]
    return data
