"""Reddit adapter via the public ``.json`` listing endpoint — **no browser**.

WSB and friends expose ``https://www.reddit.com/r/<sub>/<listing>.json`` which
returns post data directly. Reddit blocks default library UAs, so
:meth:`HttpSource._get` sends a descriptive User-Agent.

CONFIG_SCHEMA (``sources.config_json``):
    {"subreddit": "wallstreetbets",  # required
     "listing": "new",               # optional: new|hot|rising|top (default new)
     "limit": 25,                    # optional: posts per fetch (default 25)
     "ticker": "AAPL"}               # optional: stamp all items with this symbol
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from .base import HttpSource, RawItem, SourceError, to_iso

_BASE = "https://www.reddit.com"
_LISTINGS = {"new", "hot", "rising", "top", "controversial"}


class RedditSource(HttpSource):
    kind: ClassVar[str] = "reddit"
    CONFIG_SCHEMA: ClassVar[dict[str, str]] = {
        "subreddit": "required: subreddit name without r/ (e.g. wallstreetbets)",
        "listing": "optional: new|hot|rising|top (default new)",
        "limit": "optional: posts per fetch (default 25)",
        "ticker": "optional: stamp all items with this symbol",
    }

    async def fetch(self, config: Mapping[str, Any]) -> list[RawItem]:
        sub = str(config.get("subreddit") or "").strip().lstrip("r/").strip("/")
        if not sub:
            raise SourceError("reddit: 'subreddit' is required in config")
        listing = str(config.get("listing", "new")).strip().lower()
        if listing not in _LISTINGS:
            listing = "new"
        limit = int(config.get("limit", 25))
        ticker = config.get("ticker")

        url = f"{_BASE}/r/{sub}/{listing}.json"
        resp = await self._get(url, params={"limit": limit})
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceError(f"reddit: non-JSON response from {url}: {exc}") from exc

        children = (payload.get("data") or {}).get("children") or []
        items: list[RawItem] = []
        for child in children:
            data = child.get("data") or {}
            title = (data.get("title") or "").strip()
            body = (data.get("selftext") or "").strip()
            text = f"{title}\n{body}".strip()
            if not text:
                continue
            permalink = data.get("permalink") or ""
            link = f"{_BASE}{permalink}" if permalink else (data.get("url") or url)
            items.append(RawItem(self.source_id, text, link, to_iso(data.get("created_utc")), ticker))
            if len(items) >= limit:
                break
        return items
