"""GDELT DOC 2.0 API provider — macro/geopolitical regime feed.

Purpose
-------
Thin ``httpx``-only client against the GDELT Project's Document 2.0 API
(https://api.gdeltproject.org/api/v2/doc/doc).  Returns timeline-volume bins,
timeline-tone bins, and top-article headlines for a given GKG theme query.
Used by the ``world_events`` LOOK tool (gated by ``SITUATION_GDELT`` in
user_settings).

Upstream source
---------------
GDELT Project — https://www.gdeltproject.org  (DOC 2.0 API)
  - Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
  - Updated every 15 minutes; rolling window = last 3 months
  - Codebook: http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf

Auth posture
------------
No API key required.  All endpoints used here are fully public.

Failure mode
------------
Fail-loud on any network error or non-2xx HTTP status (``GDELTProviderError``).
No silent stubs per WS-J discipline.  The ``world_events`` tool caller catches
this and returns a structured ``ToolError``.

Caching
-------
Responses are in-process cached for ``_CACHE_TTL_SECONDS`` (900 s / 15 min),
matching GDELT's own update cadence.  The cache is keyed on ``(method, theme,
timespan, n)``.  Use ``provider.clear_cache()`` in tests.

Gating flag
-----------
``SITUATION_GDELT`` in user_settings (default ``False``).  When the flag is
off, the ``world_events`` LOOK tool returns ``ToolError(kind="disabled")``
without constructing this provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

_CACHE_TTL_SECONDS = 900  # 15 min — GDELT update cadence
_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT_SECONDS = 15.0


class GDELTProviderError(RuntimeError):
    """Raised on network failure or HTTP 4xx/5xx from the GDELT DOC 2.0 API."""


@dataclass(frozen=True)
class GDELTBin:
    """One time bucket from a GDELT timeline query.

    Fields
    ------
    bucket_start : datetime
        UTC timestamp marking the start of this 15-minute bucket.
    value : float
        The metric value for this bucket.  ``unit="mentions"`` → raw article
        count; ``unit="tone"`` → average sentiment tone (−100 .. +100, typically
        within ±10; positive = cooperative, negative = conflictual).
    unit : str
        ``"mentions"`` for volume timelines; ``"tone"`` for tone timelines.
    """

    bucket_start: datetime  # UTC
    value: float  # volume count or tone average
    unit: str  # "mentions" | "tone"


@dataclass(frozen=True)
class GDELTArticle:
    """One article from a GDELT ``artlist`` result.

    Fields
    ------
    title : str
        Article headline as captured by GDELT.
    url : str
        Canonical article URL.
    published : datetime
        UTC publication timestamp (best-effort from GDELT crawl time).
    source_domain : str
        Domain of the originating publication (e.g. ``"reuters.com"``).
    tone : float
        Per-article GDELT AvgTone (−100 .. +100).
    """

    title: str
    url: str
    published: datetime  # UTC
    source_domain: str
    tone: float


class GDELTProvider:
    """Pure-httpx client for the GDELT DOC 2.0 API.

    Parameters
    ----------
    base_url:
        Override the API base URL (useful for tests with a local mock server).
    timeout:
        Request timeout in seconds.
    transport:
        Optional ``httpx`` transport seam (injected in tests for offline use).

    Examples
    --------
    >>> provider = GDELTProvider()
    >>> bins = provider.timeline_volume("WAR", timespan="24h")
    >>> bins[0].unit
    'mentions'
    """

    def __init__(
        self,
        *,
        base_url: str = _BASE_URL,
        timeout: float = _TIMEOUT_SECONDS,
        transport: Any = None,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._transport = transport
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}

    # ------------------------------------------------------------------ public

    def timeline_volume(
        self,
        theme: str,
        timespan: str = "24h",
    ) -> list[GDELTBin]:
        """Fetch per-15-min mention-volume curve for ``theme``.

        Parameters
        ----------
        theme:
            A GDELT GKG theme string (e.g. ``"WAR"``, ``"ELECTION"``,
            ``"EPU_POLICY_GOVERNMENT_SPENDING"``).  The DOC 2.0 API supports
            compound queries such as ``"theme:WAR OR theme:ELECTION"``.
        timespan:
            Rolling lookback window.  GDELT accepts ``"24h"``, ``"48h"``,
            ``"7d"``, ``"30d"``, ``"3m"`` etc.

        Returns
        -------
        list[GDELTBin]
            Bins in chronological order.  Empty list if no articles matched.

        Raises
        ------
        GDELTProviderError
            On network failure or HTTP 4xx/5xx.
        """
        cache_key = ("timeline_volume", theme, timespan)
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        raw = self._fetch(
            mode="timelinevolraw",
            query=f"theme:{theme}",
            timespan=timespan,
            format="json",
        )
        bins = _parse_timeline(raw, unit="mentions")
        self._set_cache(cache_key, bins)
        return bins

    def timeline_tone(
        self,
        theme: str,
        timespan: str = "24h",
    ) -> list[GDELTBin]:
        """Fetch per-15-min average-tone curve for ``theme``.

        Parameters
        ----------
        theme:
            A GDELT GKG theme string.
        timespan:
            Rolling lookback window.

        Returns
        -------
        list[GDELTBin]
            Bins in chronological order.  Empty list if no articles matched.

        Raises
        ------
        GDELTProviderError
            On network failure or HTTP 4xx/5xx.
        """
        cache_key = ("timeline_tone", theme, timespan)
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        raw = self._fetch(
            mode="timelinetone",
            query=f"theme:{theme}",
            timespan=timespan,
            format="json",
        )
        bins = _parse_timeline(raw, unit="tone")
        self._set_cache(cache_key, bins)
        return bins

    def top_articles(
        self,
        theme: str,
        n: int = 10,
    ) -> list[GDELTArticle]:
        """Fetch the most recent articles matching ``theme``.

        Parameters
        ----------
        theme:
            A GDELT GKG theme string.
        n:
            Maximum number of articles to return (1 – 250).

        Returns
        -------
        list[GDELTArticle]
            Articles in reverse-chronological order (most recent first).

        Raises
        ------
        GDELTProviderError
            On network failure or HTTP 4xx/5xx.
        """
        n = max(1, min(250, n))
        cache_key = ("top_articles", theme, n)
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        raw = self._fetch(
            mode="artlist",
            query=f"theme:{theme}",
            maxrecords=str(n),
            sort="DateDesc",
            format="json",
        )
        articles = _parse_artlist(raw)
        self._set_cache(cache_key, articles)
        return articles

    def clear_cache(self) -> None:
        """Flush the in-process response cache (useful in tests)."""
        self._cache.clear()

    # ----------------------------------------------------------------- internal

    def _fetch(self, *, mode: str, query: str, **extra: str) -> dict[str, Any]:
        """Issue a GET request and return the parsed JSON body.

        Raises ``GDELTProviderError`` on any network or HTTP error.
        """
        import httpx

        params: dict[str, str] = {"query": query, "mode": mode, **extra}
        try:
            client_kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
                client_kwargs["base_url"] = self._base_url
            else:
                client_kwargs["base_url"] = self._base_url

            with httpx.Client(**client_kwargs) as client:
                resp = client.get("", params=params)
        except Exception as exc:
            raise GDELTProviderError(
                f"GDELT DOC 2.0 network error (mode={mode}, query={query!r}): {exc}"
            ) from exc

        if resp.status_code != 200:
            raise GDELTProviderError(
                f"GDELT DOC 2.0 HTTP {resp.status_code} "
                f"(mode={mode}, query={query!r}): {resp.text[:200]}"
            )

        try:
            return resp.json()
        except Exception as exc:
            raise GDELTProviderError(
                f"GDELT DOC 2.0 invalid JSON (mode={mode}): {exc}"
            ) from exc

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


# ------------------------------------------------------------------ parsers


def _parse_timeline(raw: dict[str, Any], *, unit: str) -> list[GDELTBin]:
    """Parse a GDELT timeline JSON response into :class:`GDELTBin` objects.

    GDELT DOC 2.0 timeline JSON has shape::

        {"timeline": [{"date": "YYYYMMDDHHMMSS", "value": <float>}, …]}

    or, for ``timelinevolraw``, the outer key may be ``"timeline"`` or
    ``"data"`` depending on the response variant.  We check both.
    """
    bins: list[GDELTBin] = []
    rows: list[Any] = []

    # The API returns {"timeline": [...]} at the top level; the inner list
    # contains series objects like {"series": "...", "data": [{"date":..., "value":...}]}
    # OR flat objects directly.
    timeline_outer = raw.get("timeline") or raw.get("data") or []
    if isinstance(timeline_outer, list):
        for item in timeline_outer:
            if isinstance(item, dict):
                # Series-wrapper shape: {"series": "...", "data": [...]}
                inner = item.get("data") or item.get("bins") or []
                if inner and isinstance(inner, list):
                    rows.extend(inner)
                elif "date" in item or "bin" in item:
                    rows.append(item)
    elif isinstance(timeline_outer, dict):
        # Some modes nest under a key equal to the query string
        for v in timeline_outer.values():
            if isinstance(v, list):
                rows.extend(v)

    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_date = row.get("date") or row.get("bin") or row.get("d")
        raw_value = row.get("value") or row.get("v") or row.get("count") or 0.0
        if not raw_date:
            continue
        try:
            bucket_start = _parse_gdelt_date(str(raw_date))
        except ValueError:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 0.0
        bins.append(GDELTBin(bucket_start=bucket_start, value=value, unit=unit))

    return bins


def _parse_artlist(raw: dict[str, Any]) -> list[GDELTArticle]:
    """Parse a GDELT ``artlist`` JSON response into :class:`GDELTArticle` objects.

    Expected shape::

        {"articles": [{"title": ..., "url": ..., "seendate": ...,
                        "domain": ..., "tone": ...}, …]}
    """
    articles: list[GDELTArticle] = []
    raw_articles = raw.get("articles") or raw.get("data") or []
    if not isinstance(raw_articles, list):
        return articles

    for item in raw_articles:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        url = str(item.get("url") or "")
        domain = str(item.get("domain") or item.get("sourcedomain") or "")
        raw_tone = item.get("tone") or 0.0
        try:
            tone = float(raw_tone)
        except (TypeError, ValueError):
            tone = 0.0

        # GDELT seendate: "YYYYMMDDTHHMMSSZ" or "YYYYMMDDHHMMSS"
        raw_date = item.get("seendate") or item.get("date") or ""
        try:
            published = _parse_gdelt_date(str(raw_date))
        except ValueError:
            published = datetime.now(tz=UTC)

        articles.append(
            GDELTArticle(
                title=title,
                url=url,
                published=published,
                source_domain=domain,
                tone=tone,
            )
        )
    return articles


def _parse_gdelt_date(s: str) -> datetime:
    """Parse GDELT timestamps in ``YYYYMMDDHHMMSS`` or ``YYYYMMDDTHHMMSSZ`` format."""
    s = s.replace("T", "").replace("Z", "").replace("-", "").replace(":", "").replace(" ", "")
    if len(s) < 8:
        raise ValueError(f"date string too short: {s!r}")
    year = int(s[0:4])
    month = int(s[4:6])
    day = int(s[6:8])
    hour = int(s[8:10]) if len(s) >= 10 else 0
    minute = int(s[10:12]) if len(s) >= 12 else 0
    second = int(s[12:14]) if len(s) >= 14 else 0
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)
