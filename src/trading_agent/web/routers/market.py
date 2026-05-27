"""Market-data router (WS-J): real price history + fundamentals for the cockpit.

Feeds the interactive crosshair chart (``GET /api/history/{symbol}``) and the
positions financials popup (``GET /api/fundamentals/{symbol}``).

**Data is always real.** "Paper"/"mock" in this product means simulated *money*,
never simulated data — so these routes read the live :class:`HistoryService`
(Alpaca bars + a real fundamentals provider) off ``app.state.history``. If that
service is not attached, or the upstream provider has no API key, the route
fails loud (``503``) rather than inventing numbers.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ...config.users import current_user
from ...data.history import Bar, HistoryProviderError, HistoryService

router = APIRouter(tags=["market"])

# Chart ranges → (Alpaca timeframe, lookback bars). Timeframes must be ones the
# AlpacaBarProvider supports: 1Min / 5Min / 15Min / 1H / 1D. Coarser bars for
# longer ranges keep the payload (and the fetch) sensible. YTD is computed.
_RANGE_MAP: dict[str, tuple[str, int]] = {
    "1D": ("5Min", 78),     # ~one 6.5h session of 5-min bars
    "5D": ("15Min", 130),   # ~5 sessions of 15-min bars
    "1M": ("1H", 150),      # ~22 sessions of hourly bars
    "6M": ("1D", 126),      # ~6 months of daily bars
    "1Y": ("1D", 252),      # ~1 trading year
    "5Y": ("1D", 1260),     # ~5 trading years
    "ALL": ("1D", 2000),    # cap; provider returns whatever it has
}


def _ytd_lookback() -> int:
    """Approximate trading days elapsed since Jan 1 (≈5/7 of calendar days)."""
    elapsed = (date.today() - date(date.today().year, 1, 1)).days
    return max(1, elapsed * 5 // 7)


def _resolve_range(range_: str) -> tuple[str, int]:
    key = (range_ or "1D").strip().upper()
    if key == "YTD":
        return ("1D", _ytd_lookback())
    if key not in _RANGE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"unknown range {range_!r}; use one of "
            f"1D, 5D, 1M, 6M, YTD, 1Y, 5Y, ALL",
        )
    return _RANGE_MAP[key]


def _clean_symbol(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    if not sym or not sym.replace(".", "").replace("-", "").isalnum() or len(sym) > 12:
        raise HTTPException(status_code=400, detail=f"invalid symbol: {symbol!r}")
    return sym


def _history(request: Request) -> HistoryService:
    svc = getattr(request.app.state, "history", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="market data not available (history service not attached)",
        )
    return svc  # type: ignore[no-any-return]


def _bar_public(b: Bar) -> dict[str, Any]:
    """Compact OHLCV row for the chart (t/o/h/l/c/v)."""
    return {"t": b.timestamp, "o": b.open, "h": b.high, "l": b.low, "c": b.close, "v": b.volume}


@router.get("/api/history/{symbol}")
def history(
    request: Request,
    symbol: str,
    range: str = "1D",
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Real OHLCV bars for ``symbol`` over ``range`` (for the crosshair chart)."""
    sym = _clean_symbol(symbol)
    timeframe, lookback = _resolve_range(range)
    svc = _history(request)
    try:
        bars = svc.bars(sym, timeframe, lookback)
    except HistoryProviderError as exc:
        # Missing data key, auth, rate-limit, etc. — fail loud, never fake bars.
        raise HTTPException(status_code=503, detail=str(exc))
    rows = [_bar_public(b) for b in bars]
    first = rows[0]["c"] if rows else None
    last = rows[-1]["c"] if rows else None
    change = (last - first) if (first is not None and last is not None) else None
    change_pct = (change / first * 100.0) if (change is not None and first) else None
    return {
        "symbol": sym,
        "range": range.strip().upper(),
        "timeframe": timeframe,
        "bars": rows,
        "first": first,
        "last": last,
        "change": change,
        "change_pct": change_pct,
    }


@router.get("/api/fundamentals/{symbol}")
def fundamentals(
    request: Request,
    symbol: str,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Real company fundamentals for ``symbol`` (for the positions popup)."""
    sym = _clean_symbol(symbol)
    svc = _history(request)
    if svc.fundamentals_provider is None:
        raise HTTPException(
            status_code=503,
            detail="no fundamentals provider configured "
            "(set FINNHUB_API_KEY or POLYGON_API_KEY / the fundamentals_provider setting)",
        )
    try:
        data = svc.fundamentals(sym)
    except HistoryProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not data:
        raise HTTPException(status_code=404, detail=f"no fundamentals for {sym}")
    return {"symbol": sym, **data}
