"""Alpaca historical-bars adapter (:class:`BarProvider`).

Uses the *same* Alpaca data key that feeds the live books (``ALPACA_API_KEY`` /
``ALPACA_SECRET_KEY`` — resolved from the environment, never hardcoded). The
``StockHistoricalDataClient`` is constructed lazily on first use so importing
this module costs nothing and tests can inject a fake ``data_client`` to stay
fully offline (no live calls in CI).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

from ..history import Bar, HistoryProviderError

# Calendar buffers so a request returns *at least* ``lookback`` bars: daily bars
# skip weekends/holidays (~1.5×), intraday bars only exist ~6.5h of the ~24h day
# and 5/7 weekdays (~5.2× compression). We over-fetch then slice to the tail.
_PER_BAR_MINUTES: dict[str, int] = {
    "1Min": 1,
    "5Min": 5,
    "15Min": 15,
    "1H": 60,
    "1D": 1440,
}


class AlpacaBarProvider:
    """Fetch historical OHLCV bars from Alpaca's market-data API."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        *,
        data_client: Any = None,
        feed: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self._data_client = data_client  # test seam: inject a fake client
        self._feed = feed

    # --- BarProvider --------------------------------------------------------

    def get_bars(self, symbol: str, timeframe: str, lookback: int) -> list[Bar]:
        if timeframe not in _PER_BAR_MINUTES:
            raise HistoryProviderError(f"unsupported timeframe: {timeframe!r}")
        if lookback <= 0:
            return []

        from alpaca.data.requests import StockBarsRequest

        kwargs: dict[str, Any] = {
            "symbol_or_symbols": [symbol],
            "timeframe": self._alpaca_timeframe(timeframe),
            "start": self._lookback_start(timeframe, lookback),
        }
        if self._feed is not None:
            kwargs["feed"] = self._feed
        request = StockBarsRequest(**kwargs)

        try:
            barset = self._client().get_stock_bars(request)
        except HistoryProviderError:
            raise
        except Exception as exc:  # network / auth / rate-limit etc.
            raise HistoryProviderError(f"alpaca bars failed for {symbol}: {exc}")

        raw = getattr(barset, "data", {}) or {}
        items = raw.get(symbol, []) if isinstance(raw, dict) else []
        bars = [self._to_bar(b) for b in items]
        return bars[-lookback:]

    # --- internals ----------------------------------------------------------

    def _client(self) -> Any:
        if self._data_client is None:
            if not (self._api_key and self._secret_key):
                raise HistoryProviderError(
                    "Alpaca data keys missing: set ALPACA_API_KEY / ALPACA_SECRET_KEY."
                )
            from alpaca.data.historical import StockHistoricalDataClient

            self._data_client = StockHistoricalDataClient(self._api_key, self._secret_key)
        return self._data_client

    @staticmethod
    def _alpaca_timeframe(timeframe: str) -> Any:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        return {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1H": TimeFrame(1, TimeFrameUnit.Hour),
            "1D": TimeFrame(1, TimeFrameUnit.Day),
        }[timeframe]

    @staticmethod
    def _lookback_start(timeframe: str, lookback: int) -> datetime:
        now = datetime.now(UTC)
        if timeframe == "1D":
            return now - timedelta(days=lookback * 1.5 + 7)
        per_bar = _PER_BAR_MINUTES[timeframe]
        minutes = lookback * per_bar * 5.2 + per_bar * 50
        return now - timedelta(minutes=minutes)

    @staticmethod
    def _to_bar(raw: Any) -> Bar:
        ts = getattr(raw, "timestamp", None)
        if ts is None:
            ts_str = ""
        elif hasattr(ts, "isoformat"):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)
        return Bar(
            timestamp=ts_str,
            open=float(getattr(raw, "open", 0.0)),
            high=float(getattr(raw, "high", 0.0)),
            low=float(getattr(raw, "low", 0.0)),
            close=float(getattr(raw, "close", 0.0)),
            volume=float(getattr(raw, "volume", 0.0) or 0.0),
        )
