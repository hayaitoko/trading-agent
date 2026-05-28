"""``history`` — historical OHLCV price data for a symbol.

Tool name:      history
Args:           symbol (str), days=30 (int)
ToolResult:     ok=True, data={"symbol": str, "bars": […], "stats": {…}}
Latency tier:   fast
Cost class:     free
Gating flag:    always enabled (degrades gracefully when history service absent)
Example use:    history("AAPL", days=10) for the last 10 days of AAPL bars.

data.bars is a list of {timestamp, open, high, low, close, volume} dicts,
oldest first.  data.stats includes close_min, close_max, close_mean,
realized_vol_annual (30D annualized log-return std dev, same formula as
the regime classifier).

Wraps :class:`~trading_agent.data.history.HistoryService`.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from ._base import LookToolBase


class HistoryTool(LookToolBase):
    """Historical bars for a tradable symbol.

    Parameters
    ----------
    history_service:
        Duck-typed: must expose ``get_bars(symbol, timeframe, lookback) -> list[Bar]``
        where each Bar has timestamp/open/high/low/close/volume attributes.
        ``None`` → tool returns a graceful "unavailable" error.
    owner_user_id, trader_id:
        Namespace identifiers (not used for the bar fetch; kept for base compat).
    """

    TOOL_META: dict[str, Any] = {
        "name": "history",
        "description": (
            "Historical OHLCV bars for a symbol over the last N days. "
            "Returns close prices, volume, and key stats."
        ),
        "args": {"symbol": "str", "days": "int (default 30)"},
        "latency": "fast",
        "cost_class": "free",
        "enabled": True,
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        history_service: Any = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._history = history_service

    def __call__(self, symbol: str, days: int = 30) -> Any:
        """Fetch historical bars for ``symbol`` over ``days`` calendar days.

        Returns
        -------
        ToolResult
            ok=True, data={"symbol": …, "bars": […], "stats": {…}}

        Example
        -------
        >>> tool = HistoryTool(trader_id="Alpha")
        >>> result = tool("AAPL", days=5)
        >>> result.ok  # False when no history_service wired
        False
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return self._err("invalid_input", "symbol must not be empty")
        days = max(1, min(int(days), 365))

        if self._history is None:
            return self._err(
                "unavailable",
                "history service not wired — HistoryService required for history()",
            )

        try:
            bars = self._history.get_bars(symbol, "1Day", days)
        except Exception as exc:
            return self._err("internal", f"history fetch failed for {symbol}: {exc}")

        if not bars:
            return self._ok(
                {
                    "symbol": symbol,
                    "bars": [],
                    "stats": {},
                    "note": f"no bars returned for {symbol} over {days}d",
                }
            )

        bar_dicts = [
            {
                "timestamp": str(getattr(b, "timestamp", "")),
                "open": float(getattr(b, "open", 0)),
                "high": float(getattr(b, "high", 0)),
                "low": float(getattr(b, "low", 0)),
                "close": float(getattr(b, "close", 0)),
                "volume": float(getattr(b, "volume", 0)),
            }
            for b in bars
        ]

        closes: list[float] = _extract_closes(bar_dicts)
        stats: dict[str, Any] = {}
        if closes:
            stats["close_min"] = round(min(closes), 4)
            stats["close_max"] = round(max(closes), 4)
            stats["close_mean"] = round(sum(closes) / len(closes), 4)
            stats["close_last"] = round(closes[-1], 4)
            vol = _realized_vol(closes)
            if vol is not None:
                stats["realized_vol_annual"] = round(vol, 4)

        return self._ok({"symbol": symbol, "bars": bar_dicts, "stats": stats})


# --------------------------------------------------------------------------- helpers


def _extract_closes(bar_dicts: list[dict[str, Any]]) -> list[float]:
    """Extract positive close prices from bar dicts with explicit float cast."""
    result: list[float] = []
    for d in bar_dicts:
        try:
            c = float(str(d["close"]))
            if c > 0:
                result.append(c)
        except (TypeError, ValueError):
            pass
    return result


def _realized_vol(closes: list[float]) -> float | None:
    """Annualized realized vol from daily closes (same formula as RegimeClassifier)."""
    if len(closes) < 2:
        return None
    log_rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]
    if not log_rets:
        return None
    daily_std = statistics.stdev(log_rets) if len(log_rets) > 1 else abs(log_rets[0])
    return daily_std * math.sqrt(252)
