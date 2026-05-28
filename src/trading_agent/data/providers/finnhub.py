"""Finnhub fundamentals adapter (:class:`FundamentalsProvider`).

Maps Finnhub's ``/stock/profile2`` + ``/stock/metric`` responses into the
canonical fundamentals dict the rest of WS-A consumes. The API key is resolved
from ``FINNHUB_API_KEY`` (or passed in) — never hardcoded; with no key the
provider returns ``None`` so the context block simply omits fundamentals. A
``transport`` seam lets tests drive it with ``httpx.MockTransport`` (offline).

Canonical shape (any field may be ``None``)::

    {symbol, name, sector, market_cap, pe, eps, dividend_yield,
     beta, week52_high, week52_low, source}

Calendar methods (P3) require an API key and FAIL LOUD if one is absent —
calendar data is a hard dependency for the situation layer (no silent stubs).
"""

from __future__ import annotations

import os
from typing import Any


class FinnhubProvider:
    """Best-effort company fundamentals + calendar data from Finnhub's free endpoints."""

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

    # --- P3: Calendar endpoints (fail-loud; no silent stubs) -----------------

    def get_economic_calendar(
        self, from_date: str, to_date: str
    ) -> list[dict[str, Any]]:
        """Fetch economic events between ``from_date`` and ``to_date`` (YYYY-MM-DD).

        Returns a list of event dicts with keys: event, country, impact, time.
        Raises ``ValueError`` if no API key is configured.
        """
        if not self._api_key:
            raise ValueError(
                "FINNHUB_API_KEY is required for the economic calendar; "
                "set it in the environment or pass api_key= to FinnhubProvider"
            )
        import httpx

        params = {"token": self._api_key}
        # Finnhub economic calendar: /calendar/economic
        try:
            with httpx.Client(
                base_url=self._base_url, transport=self._transport, timeout=self._timeout
            ) as client:
                data = self._get(client, "/calendar/economic", params)
        except Exception as exc:
            raise RuntimeError(f"finnhub economic calendar request failed: {exc}") from exc

        if data is None:
            return []
        events_raw = data.get("economicCalendar") or []
        out: list[dict[str, Any]] = []
        for ev in events_raw:
            if not isinstance(ev, dict):
                continue
            ev_time = str(ev.get("time") or ev.get("date") or "")
            # Basic date filter (events come back for a wider range on free tier)
            if ev_time and not (from_date <= ev_time[:10] <= to_date):
                continue
            out.append({
                "event": str(ev.get("event") or ev.get("name") or ""),
                "country": str(ev.get("country") or ""),
                "impact": str(ev.get("impact") or "").lower(),
                "time": ev_time,
            })
        return out

    def get_earnings_calendar(
        self, from_date: str, to_date: str, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch earnings events between ``from_date`` and ``to_date`` (YYYY-MM-DD).

        Optionally filter by ``symbol``. Returns dicts with: symbol, date, eps_estimate.
        Raises ``ValueError`` if no API key is configured.
        """
        if not self._api_key:
            raise ValueError(
                "FINNHUB_API_KEY is required for the earnings calendar; "
                "set it in the environment or pass api_key= to FinnhubProvider"
            )
        import httpx

        params: dict[str, Any] = {
            "from": from_date,
            "to": to_date,
            "token": self._api_key,
        }
        if symbol:
            params["symbol"] = symbol.upper()
        try:
            with httpx.Client(
                base_url=self._base_url, transport=self._transport, timeout=self._timeout
            ) as client:
                data = self._get(client, "/calendar/earnings", params)
        except Exception as exc:
            raise RuntimeError(f"finnhub earnings calendar request failed: {exc}") from exc

        if data is None:
            return []
        raw = data.get("earningsCalendar") or []
        out: list[dict[str, Any]] = []
        for ev in raw:
            if not isinstance(ev, dict):
                continue
            out.append({
                "symbol": str(ev.get("symbol") or ""),
                "date": str(ev.get("date") or ""),
                "eps_estimate": _f(ev.get("epsEstimate")),
                "hour": str(ev.get("hour") or ""),  # bmo | amc | dmh
            })
        return out

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
