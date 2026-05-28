"""Bluesky fetcher via the AT Protocol public XRPC API.

No API key required. Uses the public ``app.bsky.feed.searchPosts`` endpoint
at ``https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts``.

Social text is adversarial: sanitize everything before any LLM consumption.
This adapter NEVER feeds raw posts to a model — it returns :class:`RawItem`
objects for the ingest pipeline, which later aggregates them into compact
metrics (see :mod:`trading_agent.situation.social`).

CONFIG_SCHEMA (``sources.config_json``):
    {"ticker": "AAPL"}         # cashtag to search (required), OR
    {"query": "AAPL stock"}    # free-form query (overrides ticker if both given)
    {"limit": 20}              # optional: max posts per fetch (default 20)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from .base import HttpSource, RawItem, SourceError, to_iso

_BSKY_API = "https://public.api.bsky.app/xrpc"
_SEARCH_PATH = "/app.bsky.feed.searchPosts"
_MAX_LIMIT = 50


class BlueskySource(HttpSource):
    kind: ClassVar[str] = "bluesky"
    CONFIG_SCHEMA: ClassVar[dict[str, str]] = {
        "ticker": "required (unless 'query' given): cashtag/ticker to search, e.g. AAPL",
        "query": "optional: override search query (ticker becomes the label only)",
        "limit": f"optional: max posts per fetch (default 20, max {_MAX_LIMIT})",
    }

    async def fetch(self, config: Mapping[str, Any]) -> list[RawItem]:
        ticker = str(config.get("ticker") or "").strip().upper()
        query_override = str(config.get("query") or "").strip()
        limit = min(int(config.get("limit", 20)), _MAX_LIMIT)

        if not ticker and not query_override:
            raise SourceError("bluesky: 'ticker' or 'query' is required in config")

        # Build a cashtag query if ticker is given and no override.
        query = query_override if query_override else f"${ticker}"

        url = f"{_BSKY_API}{_SEARCH_PATH}"
        resp = await self._get(url, params={"q": query, "limit": limit})
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceError(f"bluesky: non-JSON response from {url}: {exc}") from exc

        posts = payload.get("posts") or []
        items: list[RawItem] = []
        for post in posts:
            record = post.get("record") or {}
            text = str(record.get("text") or "").strip()
            if not text:
                continue
            created_at = record.get("createdAt") or post.get("indexedAt") or ""
            uri = str(post.get("uri") or "")
            # Convert AT URI to a usable link (best-effort; URI may be opaque).
            link = _uri_to_link(uri, post)
            items.append(
                RawItem(
                    source_id=self.source_id,
                    text=text,
                    url=link,
                    ts=to_iso(created_at),
                    ticker=ticker or None,
                )
            )
            if len(items) >= limit:
                break
        return items


def _uri_to_link(uri: str, post: dict[str, Any]) -> str:
    """Convert an AT Protocol URI to an app.bsky.social link if possible."""
    # at://did:plc:xxx/app.bsky.feed.post/rkey → https://bsky.app/profile/{did}/post/{rkey}
    if uri.startswith("at://"):
        parts = uri[5:].split("/")
        if len(parts) >= 3:
            did = parts[0]
            rkey = parts[-1]
            return f"https://bsky.app/profile/{did}/post/{rkey}"
    # Fallback: use the author handle if available
    author = (post.get("author") or {}).get("handle", "")
    if author:
        return f"https://bsky.app/profile/{author}"
    return "https://bsky.app"
