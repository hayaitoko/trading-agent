"""Shared types for ingestion adapters.

``RawItem`` and the ``Source`` protocol are the WS-B half of
``design/handoff/CONTRACTS.md``. Concrete HTTP adapters subclass
:class:`HttpSource`, which holds the *injected* shared ``httpx.AsyncClient`` —
the same dependency-injection seam ``llm/openrouter.py`` uses for its transport,
so tests run fully offline (pass an ``httpx.MockTransport`` client).

This module is a leaf: it imports only stdlib + httpx, so ``store.py`` /
``worker.py`` can import ``RawItem`` from here without an import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar, Protocol, runtime_checkable

import httpx

# A polite default UA. Reddit (and others) 429/403 the stdlib/library default.
DEFAULT_USER_AGENT = "trading-agent-ingest/0.1 (+https://github.com/hayaitoko)"
DEFAULT_TIMEOUT = 15.0


class SourceError(Exception):
    """A source could not be fetched/parsed. Caught per-source by the worker so
    one bad source never blocks the others."""


@dataclass(frozen=True, slots=True)
class RawItem:
    """One fetched item, pre-digestion. WS-C turns these into briefs.

    Field set matches ``CONTRACTS.md §WS-B``. ``ts`` is the *source-reported*
    timestamp (ISO-8601 UTC); the store stamps its own ``fetched_at`` separately
    for drain cursoring.
    """

    source_id: str
    text: str
    url: str
    ts: str
    ticker: str | None = None


@runtime_checkable
class Source(Protocol):
    """A fetchable source. ``kind`` matches the ``sources.kind`` column.

    ``fetch(config)`` takes the per-source ``config`` dict (decoded from
    ``sources.config_json``) and returns the items found this pass. Adapters
    stamp ``RawItem.source_id`` from their own id (set at construction).
    """

    kind: ClassVar[str]

    async def fetch(self, config: Mapping[str, Any]) -> list[RawItem]: ...


class HttpSource:
    """Base for async HTTP adapters sharing one injected ``httpx.AsyncClient``.

    The client is owned by the worker (one per fetch cycle / process) and shared
    across every adapter so connections pool — that is what makes ~10 sources
    fetch in ~one round-trip. Adapters never construct their own client.
    """

    kind: ClassVar[str] = ""

    def __init__(self, source_id: str, client: httpx.AsyncClient) -> None:
        self.source_id = source_id
        self._client = client

    async def fetch(self, config: Mapping[str, Any]) -> list[RawItem]:  # pragma: no cover
        raise NotImplementedError

    async def _get(
        self, url: str, *, headers: Mapping[str, str] | None = None, params: Any = None
    ) -> httpx.Response:
        """GET with a friendly UA, raising :class:`SourceError` on HTTP/transport
        failure so the worker isolates it."""
        merged = {"User-Agent": DEFAULT_USER_AGENT}
        if headers:
            merged.update(headers)
        try:
            resp = await self._client.get(url, headers=merged, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SourceError(f"{self.kind} GET {url} failed: {exc}") from exc
        return resp


def now_iso() -> str:
    """Current UTC time as ISO-8601 (the ``ts`` fallback when a source omits one)."""
    return datetime.now(UTC).isoformat()


def to_iso(value: Any) -> str:
    """Best-effort normalize a source timestamp to ISO-8601 UTC.

    Accepts epoch seconds (int/float/str), RFC-822 (RSS ``pubDate``), or an
    already-ISO string. Falls back to :func:`now_iso` on anything unparseable so
    a single odd item never aborts a fetch.
    """
    if value is None or value == "":
        return now_iso()
    # epoch seconds
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    if isinstance(value, str):
        s = value.strip()
        # numeric string -> epoch
        try:
            return datetime.fromtimestamp(float(s), tz=UTC).isoformat()
        except ValueError:
            pass
        # RFC-822 (RSS/Atom pubDate)
        try:
            dt = parsedate_to_datetime(s)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt.astimezone(UTC).isoformat()
        except (TypeError, ValueError):
            pass
        # ISO-8601 (allow trailing Z)
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC).isoformat()
        except ValueError:
            pass
    return now_iso()
