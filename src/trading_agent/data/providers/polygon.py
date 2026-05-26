"""Polygon fundamentals adapter (:class:`FundamentalsProvider`).

A lighter sibling of the Finnhub adapter: Polygon's free tier exposes company
reference data via ``/v3/reference/tickers/{symbol}`` (name, market cap, sector
from the SIC description) — richer valuation ratios (P/E, EPS) live behind paid
financials endpoints, so those come back ``None`` here. Same canonical shape and
the same conventions as :mod:`.finnhub`: ``POLYGON_API_KEY`` from env (never
hardcoded), ``None`` without a key, and a ``transport`` seam for offline tests.
"""

from __future__ import annotations

import os
from typing import Any


class PolygonProvider:
    """Best-effort company reference data from Polygon."""

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        transport: Any = None,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY")
        self._base_url = base_url or self.BASE_URL
        self._transport = transport
        self._timeout = timeout

    def get_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        if not self._api_key:
            return None

        import httpx

        try:
            with httpx.Client(
                base_url=self._base_url, transport=self._transport, timeout=self._timeout
            ) as client:
                resp = client.get(
                    f"/v3/reference/tickers/{symbol}", params={"apiKey": self._api_key}
                )
        except Exception:
            return None

        if resp.status_code != 200:
            return None
        payload = resp.json()
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict):
            return None

        return {
            "symbol": symbol,
            "name": results.get("name"),
            "sector": results.get("sic_description"),
            "market_cap": _f(results.get("market_cap")),
            "pe": None,  # not on the free reference endpoint
            "eps": None,
            "dividend_yield": None,
            "beta": None,
            "week52_high": None,
            "week52_low": None,
            "source": "polygon",
        }


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
