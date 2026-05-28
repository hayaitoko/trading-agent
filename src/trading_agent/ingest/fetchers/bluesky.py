"""Bluesky fetchers via the AT Protocol public XRPC API.

No API key required.  All endpoints served by the public AppView at
``https://public.api.bsky.app/xrpc/``, which is fully unauthenticated for
graph reads, feed reads, and author feeds.

Three source kinds are supported:

``bluesky`` (existing)
    Cashtag / keyword search via ``app.bsky.feed.searchPosts``.  One config key:
    ``ticker`` (or ``query`` to override the cashtag expansion).

``bluesky_list`` (B1 — new)
    Pull posts from every member of an AT Protocol *list* in one call via
    ``app.bsky.feed.getListFeed``.  Config: ``{"list_uri": "at://..."}``.
    The list URI is a stable AT-URI that backs a Bluesky starter pack or
    manually curated list.  Starter-pack URLs must be resolved to their backing
    list AT-URI exactly once using :func:`resolve_starter_pack`; persist the
    resolved URI in the ``config_json`` row so no re-resolution occurs on every
    fetch tick.

``bluesky_author`` (B1 — new)
    Pull posts from a single Bluesky account via ``app.bsky.feed.getAuthorFeed``.
    Config: ``{"handle": "user.bsky.social"}``.

**Aggregation contract (unchanged from the existing kind):** this adapter returns
:class:`~trading_agent.ingest.fetchers.base.RawItem` objects carrying raw post
text for the ingest pipeline's compact-metrics aggregator.  **Raw posts are never
forwarded to a model.**  The situation layer's social aggregator converts them into
compact ``bluesky_metrics`` dicts (mention count, sentiment distribution, top
cashtags) before they reach any LOOK tool.

**MONEY IS REAL compliance:** The text stored in ``RawItem.text`` is social-post
content, not account-status information.  The aggregator produces float scores and
counts only.  No account-status strings ("paper", "sim", "demo") appear at any
point in the ingest → metrics path.

**Starter-pack helper:** :func:`resolve_starter_pack` accepts a ``bsky.app``
starter-pack URL (``https://bsky.app/starter-pack/{handle}/{rkey}`` or the
underlying ``at://…`` URI) and returns the backing list AT-URI by calling
``app.bsky.graph.getStarterPack``.  The caller should persist the result in the
source's ``config_json``; the helper should not be called on every fetch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from .base import HttpSource, RawItem, SourceError, to_iso

_BSKY_API = "https://public.api.bsky.app/xrpc"
_SEARCH_PATH = "/app.bsky.feed.searchPosts"
_LIST_FEED_PATH = "/app.bsky.feed.getListFeed"
_AUTHOR_FEED_PATH = "/app.bsky.feed.getAuthorFeed"
_GET_STARTER_PACK_PATH = "/app.bsky.graph.getStarterPack"
_RESOLVE_HANDLE_PATH = "/com.atproto.identity.resolveHandle"
_MAX_LIMIT = 50


class BlueskySource(HttpSource):
    """Cashtag / keyword search via ``app.bsky.feed.searchPosts``.

    CONFIG_SCHEMA:
        ``{"ticker": "AAPL"}``        cashtag to search (required unless query given)
        ``{"query": "AAPL stock"}``   free-form override (takes precedence)
        ``{"limit": 20}``             max posts per fetch (default 20, max 50)
    """

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


class BlueskyListSource(HttpSource):
    """Posts from a curated AT Protocol list via ``app.bsky.feed.getListFeed``.

    The list AT-URI backs a Bluesky starter pack or a manually curated list of
    accounts.  Use :func:`resolve_starter_pack` to convert a starter-pack URL
    to its backing list URI exactly once and persist the result in config.

    CONFIG_SCHEMA:
        ``{"list_uri": "at://did:plc:.../app.bsky.graph.list/..."}``
            The AT-URI of the list to follow (required).
        ``{"ticker": "SPY"}``
            Optional: stamp all fetched items with this ticker symbol.
        ``{"limit": 25}``
            Max posts per fetch (default 25, max 50).
    """

    kind: ClassVar[str] = "bluesky_list"
    CONFIG_SCHEMA: ClassVar[dict[str, str]] = {
        "list_uri": "required: AT-URI of the list, e.g. at://did:plc:.../app.bsky.graph.list/...",
        "ticker": "optional: stamp all items with this ticker symbol",
        "limit": f"optional: max posts per fetch (default 25, max {_MAX_LIMIT})",
    }

    async def fetch(self, config: Mapping[str, Any]) -> list[RawItem]:
        list_uri = str(config.get("list_uri") or "").strip()
        if not list_uri:
            raise SourceError("bluesky_list: 'list_uri' is required in config")
        if not list_uri.startswith("at://"):
            raise SourceError(
                f"bluesky_list: list_uri must be an AT-URI (at://...), got {list_uri!r}. "
                "Use resolve_starter_pack() to convert a starter-pack URL to its list AT-URI."
            )

        ticker = str(config.get("ticker") or "").strip().upper() or None
        limit = min(int(config.get("limit", 25)), _MAX_LIMIT)

        url = f"{_BSKY_API}{_LIST_FEED_PATH}"
        resp = await self._get(url, params={"list": list_uri, "limit": limit})
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceError(f"bluesky_list: non-JSON response from {url}: {exc}") from exc

        return _feed_items_to_raw(
            payload.get("feed") or [],
            source_id=self.source_id,
            ticker=ticker,
            limit=limit,
        )


class BlueskyAuthorSource(HttpSource):
    """Posts from a single Bluesky account via ``app.bsky.feed.getAuthorFeed``.

    High-signal finance voices (journalists, economists, analysts) are seeded as
    ``bluesky_author`` rows.  The compact-metrics aggregator converts their posts
    into cashtag mention counts + sentiment scores — raw text never reaches the
    trader.

    CONFIG_SCHEMA:
        ``{"handle": "user.bsky.social"}``
            The Bluesky handle to follow (required).  Accepts full handle
            (``user.bsky.social``) or a bare DID (``did:plc:...``).
        ``{"ticker": "SPY"}``
            Optional: stamp all items with this ticker (useful for sources that
            always post about a specific name).
        ``{"limit": 20}``
            Max posts per fetch (default 20, max 50).
    """

    kind: ClassVar[str] = "bluesky_author"
    CONFIG_SCHEMA: ClassVar[dict[str, str]] = {
        "handle": "required: Bluesky handle (e.g. user.bsky.social) or DID",
        "ticker": "optional: stamp all items with this ticker symbol",
        "limit": f"optional: max posts per fetch (default 20, max {_MAX_LIMIT})",
    }

    async def fetch(self, config: Mapping[str, Any]) -> list[RawItem]:
        handle = str(config.get("handle") or "").strip()
        if not handle:
            raise SourceError("bluesky_author: 'handle' is required in config")

        ticker = str(config.get("ticker") or "").strip().upper() or None
        limit = min(int(config.get("limit", 20)), _MAX_LIMIT)

        url = f"{_BSKY_API}{_AUTHOR_FEED_PATH}"
        resp = await self._get(url, params={"actor": handle, "limit": limit})
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceError(f"bluesky_author: non-JSON response from {url}: {exc}") from exc

        return _feed_items_to_raw(
            payload.get("feed") or [],
            source_id=self.source_id,
            ticker=ticker,
            limit=limit,
        )


# ---------------------------------------------------------------------------
# Starter-pack helper
# ---------------------------------------------------------------------------


async def resolve_starter_pack(client: HttpSource, pack_url: str) -> str:
    """Resolve a starter-pack URL or AT-URI to its backing list AT-URI.

    ``pack_url`` may be:
    - An ``at://`` URI for the starter pack
      (``at://did:plc:…/app.bsky.graph.starterpack/…``)
    - A ``bsky.app/starter-pack/{handle}/{rkey}`` URL (handle is resolved to DID
      automatically via ``com.atproto.identity.resolveHandle``).

    Returns the list AT-URI (``at://…/app.bsky.graph.list/…``) that backs the
    starter pack.

    Raises :class:`~trading_agent.ingest.fetchers.base.SourceError` if the
    resolution fails (network error, 4xx, or unexpected response shape).

    **Persistence contract:** call this once and store the returned AT-URI in
    the source's ``config_json`` as ``list_uri``.  Do not call on every fetch
    tick — the list URI for a starter pack does not change after creation.
    """
    pack_uri = pack_url.strip()

    # Convert bsky.app URL to an AT-URI.
    if pack_uri.startswith("https://bsky.app/starter-pack/"):
        # Pattern: https://bsky.app/starter-pack/{handle}/{rkey}
        path = pack_uri.removeprefix("https://bsky.app/starter-pack/")
        parts = path.split("/")
        if len(parts) < 2:
            raise SourceError(
                f"resolve_starter_pack: cannot parse starter-pack URL {pack_url!r}"
            )
        handle, rkey = parts[0], parts[1]
        # Resolve handle → DID.
        resolve_url = f"{_BSKY_API}{_RESOLVE_HANDLE_PATH}"
        resp = await client._get(resolve_url, params={"handle": handle})
        try:
            did = resp.json()["did"]
        except (ValueError, KeyError) as exc:
            raise SourceError(
                f"resolve_starter_pack: failed to resolve handle {handle!r}: {exc}"
            ) from exc
        pack_uri = f"at://{did}/app.bsky.graph.starterpack/{rkey}"

    if not pack_uri.startswith("at://"):
        raise SourceError(
            f"resolve_starter_pack: expected AT-URI or bsky.app URL, got {pack_url!r}"
        )

    # Fetch the starter pack and extract the list URI.
    sp_url = f"{_BSKY_API}{_GET_STARTER_PACK_PATH}"
    resp = await client._get(sp_url, params={"starterPack": pack_uri})
    try:
        data = resp.json()
        list_uri: str = data["starterPack"]["record"]["list"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SourceError(
            f"resolve_starter_pack: unexpected response shape for {pack_uri!r}: {exc}"
        ) from exc

    if not list_uri.startswith("at://"):
        raise SourceError(
            f"resolve_starter_pack: got non-AT-URI list {list_uri!r} for {pack_uri!r}"
        )
    return list_uri


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _feed_items_to_raw(
    feed: list[dict[str, Any]],
    *,
    source_id: str,
    ticker: str | None,
    limit: int,
) -> list[RawItem]:
    """Convert a Bluesky feed (list/author/generator) to :class:`RawItem` objects."""
    items: list[RawItem] = []
    for feed_item in feed:
        post = feed_item.get("post") or {}
        record = post.get("record") or {}
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        created_at = record.get("createdAt") or post.get("indexedAt") or ""
        uri = str(post.get("uri") or "")
        link = _uri_to_link(uri, post)
        # Extract cashtags from the post text for ticker-less sources.
        effective_ticker = ticker or _extract_first_cashtag(text)
        items.append(
            RawItem(
                source_id=source_id,
                text=text,
                url=link,
                ts=to_iso(created_at),
                ticker=effective_ticker,
            )
        )
        if len(items) >= limit:
            break
    return items


def _extract_first_cashtag(text: str) -> str | None:
    """Return the first ``$TICKER`` cashtag found in ``text``, upper-cased, or ``None``."""
    import re
    m = re.search(r"\$([A-Z]{1,5})\b", text.upper())
    return m.group(1) if m else None


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
