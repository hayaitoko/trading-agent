"""RSS / Atom feed adapter (any feed URL).

Parses with the stdlib XML parser — no ``feedparser`` dependency. Handles both
RSS 2.0 (``<rss><channel><item>``) and Atom (``<feed><entry>``).

CONFIG_SCHEMA (``sources.config_json``):
    {"url": "<feed url>",            # required
     "ticker": "AAPL",               # optional: stamp every item with this ticker
     "limit": 50}                    # optional: cap items per fetch (default 50)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any, ClassVar

from .base import HttpSource, RawItem, SourceError, to_iso

_ATOM = "{http://www.w3.org/2005/Atom}"


class RssSource(HttpSource):
    kind: ClassVar[str] = "rss"
    CONFIG_SCHEMA: ClassVar[dict[str, str]] = {
        "url": "required: feed URL (RSS or Atom)",
        "ticker": "optional: stamp all items with this symbol",
        "limit": "optional: max items per fetch (default 50)",
    }

    async def fetch(self, config: Mapping[str, Any]) -> list[RawItem]:
        url = str(config.get("url") or "").strip()
        if not url:
            raise SourceError("rss: 'url' is required in config")
        ticker = config.get("ticker")
        limit = int(config.get("limit", 50))

        resp = await self._get(url)
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            raise SourceError(f"rss: malformed feed at {url}: {exc}") from exc

        items: list[RawItem] = []
        # RSS 2.0
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            desc = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or url).strip()
            pub = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date")
            text = f"{title}\n{desc}".strip()
            if text:
                items.append(
                    RawItem(self.source_id, text, link, to_iso(pub), ticker)
                )
            if len(items) >= limit:
                return items
        # Atom
        for entry in root.iter(f"{_ATOM}entry"):
            title = (entry.findtext(f"{_ATOM}title") or "").strip()
            summary = (
                entry.findtext(f"{_ATOM}summary") or entry.findtext(f"{_ATOM}content") or ""
            ).strip()
            link_el = entry.find(f"{_ATOM}link")
            link = (link_el.get("href") if link_el is not None else None) or url
            pub = entry.findtext(f"{_ATOM}updated") or entry.findtext(f"{_ATOM}published")
            text = f"{title}\n{summary}".strip()
            if text:
                items.append(RawItem(self.source_id, text, link, to_iso(pub), ticker))
            if len(items) >= limit:
                break
        return items
