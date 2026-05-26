"""Finnhub fundamentals adapter (:class:`FundamentalsProvider`).

Maps Finnhub's ``/stock/profile2`` + ``/stock/metric`` responses into the
canonical fundamentals dict the rest of WS-A consumes. The API key is resolved
from ``FINNHUB_API_KEY`` (or passed in) — never hardcoded; with no key the
provider returns ``None`` so the context block simply omits fundamentals. A
``transport`` seam lets tests drive it with ``httpx.MockTransport`` (offline).

Canonical shape (any field may be ``None``)::

    {symbol, name, sector, market_cap, pe, eps, dividend_yield,
     beta, week52_high, week52_low, source}
"""

from __future__ import annotations

import os
from typing import Any


class FinnhubProvider:
    """Best-effort company fundamentals from Finnhub's free endpoints."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        transport: Any = None,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        self._base_url = base_url or self.BASE_URL
        self._transport = transport
        self._timeout = timeout

    def get_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        if not self._api_key:
            return None

        import httpx

        params = {"symbol": symbol, "token": self._api_key}
        try:
            with httpx.Client(
                base_url=self._base_url, transport=self._transport, timeout=self._timeout
            ) as client:
                profile = self._get(client, "/stock/profile2", params)
                metric_resp = self._get(client, "/stock/metric", {**params, "metric": "all"})
        except Exception:
            return None

        profile = profile or {}
        metric = (metric_resp or {}).get("metric", {}) or {}
        if not profile and not metric:
            return None

        return {
            "symbol": symbol,
            "name": profile.get("name"),
            "sector": profile.get("finnhubIndustry"),
            "market_cap": _f(profile.get("marketCapitalization")),
            "pe": _f(metric.get("peTTM") or metric.get("peNormalizedAnnual")),
            "eps": _f(metric.get("epsTTM") or metric.get("epsNormalizedAnnual")),
            "dividend_yield": _f(metric.get("dividendYieldIndicatedAnnual")),
            "beta": _f(metric.get("beta")),
            "week52_high": _f(metric.get("52WeekHigh")),
            "week52_low": _f(metric.get("52WeekLow")),
            "source": "finnhub",
        }

    @staticmethod
    def _get(client: Any, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        resp = client.get(path, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
