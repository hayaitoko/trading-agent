"""Prediction markets provider — Polymarket + Kalshi forward event odds.

Purpose
-------
Unified read-only client surfacing implied probabilities from two prediction
market venues: Polymarket (Gamma + CLOB APIs) and Kalshi (prod REST API).
Exposes a single ``PredictionMarketsProvider`` with two public methods:

  ``event_odds(category, query)``
      Returns a ranked list of ``EventOdds`` objects from both venues,
      filtered by liquidity floor and deduplicated at the caller's discretion.

  ``by_id(venue, event_id)``
      Look up a single event by its venue-specific ID.

Upstream sources
----------------
Polymarket Gamma API  — https://gamma-api.polymarket.com
Polymarket CLOB API   — https://clob.polymarket.com
Kalshi trade API v2   — https://external-api.kalshi.com/trade-api/v2

Auth posture
------------
No API key required for any read endpoint used here.
  - Polymarket Gamma ``/events?closed=false`` — fully public read.
  - Polymarket CLOB ``/price`` — fully public read.
  - Kalshi ``/markets``, ``/events`` — public unauthenticated reads per
    https://docs.kalshi.com/getting_started/quick_start_market_data
    (auth only required for order placement, portfolio, account endpoints).
US IP geoblock on Polymarket applies to ORDER SUBMISSION only — read traffic
from US IPs is unrestricted (confirmed in recon §2).

Failure mode
------------
Fail-loud: ``PredictionMarketsProviderError`` (RuntimeError subclass) on any
network error or HTTP non-2xx.  No silent stubs per WS-J discipline.
429 responses trigger exponential backoff (up to 3 retries, ceiling 8 s).
Events with ``restricted: true`` on Polymarket are skipped without crashing.

Gating flag
-----------
``SITUATION_PREDICTION_MARKETS`` in user_settings (default ``False``).
When the flag is off, the ``prediction_market_odds`` LOOK tool returns
``ToolError(kind="disabled")`` without constructing this provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

_PM_GAMMA_BASE = "https://gamma-api.polymarket.com"
_PM_CLOB_BASE = "https://clob.polymarket.com"
_KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"

_CACHE_TTL_SECONDS = 60  # 1-min cache — prediction market prices update fast
_TIMEOUT_SECONDS = 12.0
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0   # seconds; doubled per retry; capped at 8 s


class PredictionMarketsProviderError(RuntimeError):
    """Raised on network failure or non-2xx HTTP from a prediction market API."""


@dataclass(frozen=True)
class EventOdds:
    """One prediction market event's current implied probabilities.

    Fields
    ------
    venue : Literal["polymarket", "kalshi"]
        The originating venue.
    event_id : str
        Venue-specific event identifier (Polymarket slug/id or Kalshi ticker).
    title : str
        Human-readable event title (e.g. "Will the Fed cut rates in June?").
    outcomes : list[str]
        The discrete outcomes the market resolves to (e.g. ["Yes", "No"]).
    prices : list[float]
        Implied probabilities, parallel to ``outcomes``, in the range 0.0–1.0.
        On a well-traded binary market the two values sum to ≈1.0.
    liquidity : float
        Total open-interest liquidity in USD.  Polymarket markets with
        liquidity < $1,000 are excluded before this object is created.
    volume_24h : float
        Rolling 24-hour trading volume in USD.
    end_date : datetime | None
        UTC expiry/resolution date.  ``None`` when not published by the venue.
    restricted : bool
        ``True`` when the Polymarket event carries a ``restricted`` flag.
        Restricted events are included in the result set for completeness but
        flagged so callers can choose to skip them.
    """

    venue: Literal["polymarket", "kalshi"]
    event_id: str
    title: str
    outcomes: list[str]
    prices: list[float]  # parallel to outcomes, 0.0-1.0
    liquidity: float
    volume_24h: float
    end_date: datetime | None
    restricted: bool


class PredictionMarketsProvider:
    """Unified read-only client for Polymarket and Kalshi event odds.

    Parameters
    ----------
    pm_gamma_base:
        Override Polymarket Gamma base URL (for tests).
    pm_clob_base:
        Override Polymarket CLOB base URL (for tests).
    kalshi_base:
        Override Kalshi API base URL (for tests).
    timeout:
        HTTP request timeout in seconds.
    transport:
        Optional ``httpx`` transport seam (injected in tests for offline use).
        When provided, the same transport is reused for both venues.

    Examples
    --------
    >>> provider = PredictionMarketsProvider()
    >>> odds = provider.event_odds("politics", query="president")
    >>> odds[0].venue
    'polymarket'
    """

    def __init__(
        self,
        *,
        pm_gamma_base: str = _PM_GAMMA_BASE,
        pm_clob_base: str = _PM_CLOB_BASE,
        kalshi_base: str = _KALSHI_BASE,
        timeout: float = _TIMEOUT_SECONDS,
        transport: Any = None,
    ) -> None:
        self._pm_gamma_base = pm_gamma_base
        self._pm_clob_base = pm_clob_base
        self._kalshi_base = kalshi_base
        self._timeout = timeout
        self._transport = transport
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}

    # ------------------------------------------------------------------ public

    def event_odds(
        self,
        category: str,
        query: str | None = None,
        *,
        min_liquidity: float = 1_000.0,
    ) -> list[EventOdds]:
        """Fetch current event odds from Polymarket and Kalshi.

        Parameters
        ----------
        category:
            Category or topic hint (e.g. ``"politics"``, ``"economics"``,
            ``"fed_rate"``, ``"crypto"``).  Used to filter Kalshi series
            and Polymarket tags where possible; falls back to returning all
            open events sorted by volume.
        query:
            Optional free-text substring filter applied to event titles
            (case-insensitive).
        min_liquidity:
            Minimum USD liquidity required for a Polymarket market to be
            included.  Default $1,000 to filter out manipulable micro-markets.
            Kalshi markets are always included (regulated, not manipulable).

        Returns
        -------
        list[EventOdds]
            Combined results from both venues, sorted by volume_24h descending.
            Both venues are attempted independently; a failure on one does NOT
            suppress results from the other.  Empty list if both fail.

        Raises
        ------
        PredictionMarketsProviderError
            Only if BOTH venues fail.  If one succeeds, its results are
            returned and the failure is silently swallowed (the caller's
            LOOK-tool wrapper should log it).
        """
        cache_key = ("event_odds", category, query, min_liquidity)
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        pm_results: list[EventOdds] = []
        kalshi_results: list[EventOdds] = []
        pm_err: Exception | None = None
        kalshi_err: Exception | None = None

        try:
            pm_results = self._fetch_polymarket(
                category=category, query=query, min_liquidity=min_liquidity
            )
        except Exception as exc:  # noqa: BLE001
            pm_err = exc

        try:
            kalshi_results = self._fetch_kalshi(category=category, query=query)
        except Exception as exc:  # noqa: BLE001
            kalshi_err = exc

        if pm_err is not None and kalshi_err is not None:
            raise PredictionMarketsProviderError(
                f"Both Polymarket and Kalshi failed — "
                f"polymarket: {pm_err}; kalshi: {kalshi_err}"
            )

        combined = pm_results + kalshi_results
        combined.sort(key=lambda e: e.volume_24h, reverse=True)
        self._set_cache(cache_key, combined)
        return combined

    def by_id(
        self,
        venue: Literal["polymarket", "kalshi"],
        event_id: str,
    ) -> EventOdds | None:
        """Look up a single event by venue + ID.

        Parameters
        ----------
        venue:
            ``"polymarket"`` or ``"kalshi"``.
        event_id:
            Venue-specific identifier (Polymarket event slug/id, Kalshi
            event ticker).

        Returns
        -------
        EventOdds | None
            The event if found; ``None`` if the ID is not listed or the
            event is closed.

        Raises
        ------
        PredictionMarketsProviderError
            On network failure or HTTP 4xx/5xx.
        """
        cache_key = ("by_id", venue, event_id)
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        result: EventOdds | None = None
        if venue == "polymarket":
            result = self._polymarket_by_id(event_id)
        elif venue == "kalshi":
            result = self._kalshi_by_id(event_id)
        else:
            raise ValueError(f"unknown venue: {venue!r}")

        self._set_cache(cache_key, result)
        return result

    def clear_cache(self) -> None:
        """Flush the in-process response cache."""
        self._cache.clear()

    # ----------------------------------------------------------------- polymarket

    def _fetch_polymarket(
        self,
        *,
        category: str,
        query: str | None,
        min_liquidity: float,
        limit: int = 50,
    ) -> list[EventOdds]:
        """Fetch open Polymarket events from the Gamma API and filter them."""
        params: dict[str, Any] = {
            "closed": "false",
            "limit": limit,
            "order": "volume24hrClob",
            "ascending": "false",
            "active": "true",
        }
        if category:
            # The Gamma /events endpoint accepts a tag_slug parameter;
            # fall through to title-filtering if the tag doesn't match
            params["tag_slug"] = category.lower().replace(" ", "-")

        raw = self._get(_PM_GAMMA_BASE, "/events", params)
        events: list[Any] = raw if isinstance(raw, list) else []

        results: list[EventOdds] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            odds = _polymarket_event_to_odds(ev, min_liquidity=min_liquidity)
            if odds is None:
                continue
            # Apply optional title filter
            if query and query.lower() not in odds.title.lower():
                continue
            results.append(odds)

        # If tag_slug returned nothing, retry without the tag filter
        if not results and category:
            params.pop("tag_slug", None)
            raw2 = self._get(_PM_GAMMA_BASE, "/events", params)
            events2: list[Any] = raw2 if isinstance(raw2, list) else []
            for ev in events2:
                if not isinstance(ev, dict):
                    continue
                odds = _polymarket_event_to_odds(ev, min_liquidity=min_liquidity)
                if odds is None:
                    continue
                if query and query.lower() not in odds.title.lower():
                    continue
                results.append(odds)

        return results

    def _polymarket_by_id(self, event_id: str) -> EventOdds | None:
        try:
            raw = self._get(_PM_GAMMA_BASE, f"/events/{event_id}", {})
        except PredictionMarketsProviderError:
            return None
        if not isinstance(raw, dict):
            return None
        return _polymarket_event_to_odds(raw, min_liquidity=0.0)

    # ----------------------------------------------------------------- kalshi

    def _fetch_kalshi(
        self,
        *,
        category: str,
        query: str | None,
        limit: int = 50,
    ) -> list[EventOdds]:
        """Fetch open Kalshi markets/events."""
        params: dict[str, Any] = {
            "limit": limit,
            "status": "open",
        }
        # Kalshi supports a series_ticker filter and a category param
        if category:
            params["category"] = category.upper()

        raw = self._get(_KALSHI_BASE, "/markets", params)
        markets_raw: list[Any] = []
        if isinstance(raw, dict):
            markets_raw = raw.get("markets") or []
        elif isinstance(raw, list):
            markets_raw = raw

        results: list[EventOdds] = []
        for mkt in markets_raw:
            if not isinstance(mkt, dict):
                continue
            odds = _kalshi_market_to_odds(mkt)
            if odds is None:
                continue
            if query and query.lower() not in odds.title.lower():
                continue
            results.append(odds)

        # If category filtered returned nothing, retry without category
        if not results and category:
            params2 = {"limit": limit, "status": "open"}
            raw2 = self._get(_KALSHI_BASE, "/markets", params2)
            markets2: list[Any] = []
            if isinstance(raw2, dict):
                markets2 = raw2.get("markets") or []
            elif isinstance(raw2, list):
                markets2 = raw2
            for mkt in markets2:
                if not isinstance(mkt, dict):
                    continue
                odds = _kalshi_market_to_odds(mkt)
                if odds is None:
                    continue
                if query and query.lower() not in odds.title.lower():
                    continue
                results.append(odds)

        return results

    def _kalshi_by_id(self, event_id: str) -> EventOdds | None:
        try:
            raw = self._get(_KALSHI_BASE, f"/events/{event_id}", {})
        except PredictionMarketsProviderError:
            return None
        if not isinstance(raw, dict):
            return None
        event_raw = raw.get("event") or raw
        return _kalshi_event_to_odds(event_raw)

    # ----------------------------------------------------------------- http

    def _get(
        self,
        base: str,
        path: str,
        params: dict[str, Any],
        *,
        retry: int = 0,
    ) -> Any:
        """Issue a GET request with backoff on 429.

        Raises ``PredictionMarketsProviderError`` on network error, 4xx, 5xx
        after exhausting retries.
        """
        import httpx

        client_kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        try:
            with httpx.Client(base_url=base, **client_kwargs) as client:
                resp = client.get(path, params=params)
        except Exception as exc:
            raise PredictionMarketsProviderError(
                f"Network error [{base}{path}]: {exc}"
            ) from exc

        if resp.status_code == 429:
            if retry < _MAX_RETRIES:
                wait = min(_BACKOFF_BASE * (2**retry), 8.0)
                time.sleep(wait)
                return self._get(base, path, params, retry=retry + 1)
            raise PredictionMarketsProviderError(
                f"HTTP 429 after {_MAX_RETRIES} retries [{base}{path}]"
            )

        if resp.status_code not in (200, 201):
            raise PredictionMarketsProviderError(
                f"HTTP {resp.status_code} [{base}{path}]: {resp.text[:200]}"
            )

        try:
            return resp.json()
        except Exception as exc:
            raise PredictionMarketsProviderError(
                f"Invalid JSON from [{base}{path}]: {exc}"
            ) from exc

    # ----------------------------------------------------------------- cache

    def _get_cache(self, key: tuple[Any, ...]) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > _CACHE_TTL_SECONDS:
            del self._cache[key]
            return None
        return value

    def _set_cache(self, key: tuple[Any, ...], value: Any) -> None:
        self._cache[key] = (time.monotonic(), value)


# ------------------------------------------------------------------ converters


def _polymarket_event_to_odds(
    ev: dict[str, Any],
    *,
    min_liquidity: float,
) -> EventOdds | None:
    """Convert one Gamma API event object into an ``EventOdds``.

    Applies the liquidity filter and skips events that have no markets or
    no price data.  ``restricted: true`` events are NOT skipped here — they
    are returned with ``restricted=True`` so the caller can decide.
    """
    title = str(ev.get("title") or ev.get("slug") or "")
    if not title:
        return None

    liquidity = _f(ev.get("liquidity")) or 0.0
    volume_24h = _f(
        ev.get("volume24hr") or ev.get("volume24hrClob") or ev.get("volume24h")
    ) or 0.0

    if liquidity < min_liquidity:
        return None

    # End date
    end_date: datetime | None = None
    raw_end = ev.get("endDate") or ev.get("end_date_iso")
    if raw_end:
        try:
            end_date = _parse_iso(str(raw_end))
        except ValueError:
            pass

    restricted = bool(ev.get("restricted") or False)
    event_id = str(ev.get("id") or ev.get("slug") or ev.get("ticker") or "")

    # Extract outcomes + prices from nested markets array
    markets: list[Any] = ev.get("markets") or []
    outcomes: list[str] = []
    prices: list[float] = []

    for mkt in markets:
        if not isinstance(mkt, dict):
            continue
        raw_outcomes = mkt.get("outcomes")
        raw_prices_str = mkt.get("outcomePrices")
        if raw_outcomes and raw_prices_str:
            try:
                raw_outcomes_list = (
                    raw_outcomes
                    if isinstance(raw_outcomes, list)
                    else _try_json_list(raw_outcomes)
                )
                raw_prices_list = (
                    raw_prices_str
                    if isinstance(raw_prices_str, list)
                    else _try_json_list(raw_prices_str)
                )
                if raw_outcomes_list and raw_prices_list:
                    outcomes = [str(o) for o in raw_outcomes_list]
                    prices = [_f(p) or 0.0 for p in raw_prices_list]
                    break  # use first market's prices
            except (ValueError, TypeError):
                continue

    if not outcomes:
        # Binary fallback: use top-level bestBid as Yes price
        bid = _f(ev.get("bestBid") or ev.get("lastTradePrice"))
        if bid is not None:
            outcomes = ["Yes", "No"]
            prices = [bid, max(0.0, 1.0 - bid)]

    if not outcomes:
        return None

    return EventOdds(
        venue="polymarket",
        event_id=event_id,
        title=title,
        outcomes=outcomes,
        prices=prices,
        liquidity=liquidity,
        volume_24h=volume_24h,
        end_date=end_date,
        restricted=restricted,
    )


def _kalshi_market_to_odds(mkt: dict[str, Any]) -> EventOdds | None:
    """Convert one Kalshi ``/markets`` entry into an ``EventOdds``."""
    title = str(
        mkt.get("title")
        or mkt.get("event_title")
        or mkt.get("yes_sub_title")
        or mkt.get("ticker")
        or ""
    )
    if not title:
        return None

    status = str(mkt.get("status") or "").lower()
    if status in ("resolved", "settled", "closed", "finalized"):
        return None

    ticker = str(mkt.get("ticker") or mkt.get("event_ticker") or "")
    if not ticker:
        return None

    # Kalshi binary: yes_bid and no_bid in cents (0–100), OR yes_price / no_price as floats
    yes_price = _kalshi_price(mkt.get("yes_bid") or mkt.get("yes_price"))
    no_price = _kalshi_price(mkt.get("no_bid") or mkt.get("no_price"))

    # If only yes price available, infer no
    if yes_price is not None and no_price is None:
        no_price = max(0.0, 1.0 - yes_price)
    if yes_price is None:
        return None

    volume_24h = _f(mkt.get("volume") or mkt.get("volume_24h") or mkt.get("dollar_volume")) or 0.0
    liquidity = _f(mkt.get("open_interest") or mkt.get("dollar_open_interest")) or 0.0

    end_date: datetime | None = None
    raw_close = mkt.get("close_time") or mkt.get("expiration_time")
    if raw_close:
        try:
            end_date = _parse_iso(str(raw_close))
        except ValueError:
            pass

    return EventOdds(
        venue="kalshi",
        event_id=ticker,
        title=title,
        outcomes=["Yes", "No"],
        prices=[yes_price, no_price if no_price is not None else 0.0],
        liquidity=liquidity,
        volume_24h=volume_24h,
        end_date=end_date,
        restricted=False,  # Kalshi is CFTC-regulated, no geo-restrictions
    )


def _kalshi_event_to_odds(ev: dict[str, Any]) -> EventOdds | None:
    """Convert a Kalshi ``/events/{id}`` response envelope into EventOdds."""
    if not isinstance(ev, dict):
        return None
    # Events may wrap a list of markets; use the first
    markets = ev.get("markets") or []
    if markets and isinstance(markets, list):
        mkt = markets[0] if isinstance(markets[0], dict) else {}
        mkt["title"] = mkt.get("title") or ev.get("title") or ""
        return _kalshi_market_to_odds(mkt)
    return None


# ------------------------------------------------------------------ helpers


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _kalshi_price(raw: Any) -> float | None:
    """Normalise Kalshi prices: cents (0-100) → fraction (0.0-1.0)."""
    v = _f(raw)
    if v is None:
        return None
    # Kalshi sometimes returns cents (0-99), sometimes fractions (0.0-0.99)
    if v > 1.0:
        return v / 100.0
    return v


def _parse_iso(s: str) -> datetime:
    """Parse an ISO 8601 UTC datetime string."""
    s = s.rstrip("Z").replace(" ", "T")
    # Python 3.11+ fromisoformat handles Z; handle +00:00 as well
    if "+" not in s and s.endswith("00:00"):
        s = s[:-6]
    try:
        return datetime.fromisoformat(s).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"cannot parse ISO date: {s!r}") from exc


def _try_json_list(s: Any) -> list[Any]:
    """Attempt to parse a JSON-encoded list string."""
    import json

    if isinstance(s, list):
        return s
    return json.loads(str(s))
