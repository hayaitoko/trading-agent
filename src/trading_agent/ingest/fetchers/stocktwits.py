"""StockTwits adapter via the public streams API.

Two stream modes:
- per-symbol: ``/streams/symbol/<SYMBOL>.json`` (stamps that ticker), or
- trending:   ``/streams/trending.json`` (per-message symbols, if any).

CONFIG_SCHEMA (``sources.config_json``):
    {"symbol": "AAPL"}               # symbol stream (ticker auto-stamped), OR
    {"stream": "trending"}           # trending stream
    {"limit": 30}                    # optional: max messages per fetch (default 30)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from .base import HttpSource, RawItem, SourceError, to_iso

_BASE = "https://api.stocktwits.com/api/2/streams"


class StockTwitsSource(HttpSource):
    kind: ClassVar[str] = "stocktwits"
    CONFIG_SCHEMA: ClassVar[dict[str, str]] = {
        "symbol": "symbol stream, e.g. AAPL (ticker auto-stamped); OR set 'stream'",
        "stream": "named stream, e.g. trending (used when 'symbol' is absent)",
        "limit": "optional: max messages per fetch (default 30)",
    }

    async def fetch(self, config: Mapping[str, Any]) -> list[RawItem]:
        symbol = str(config.get("symbol") or "").strip().upper()
        stream = str(config.get("stream") or "trending").strip().lower()
        limit = int(config.get("limit", 30))

        if symbol:
            url = f"{_BASE}/symbol/{symbol}.json"
            default_ticker: str | None = symbol
        else:
            url = f"{_BASE}/{stream}.json"
            default_ticker = None

        resp = await self._get(url)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceError(f"stocktwits: non-JSON response from {url}: {exc}") from exc

        messages = payload.get("messages") or []
        items: list[RawItem] = []
        for msg in messages:
            body = (msg.get("body") or "").strip()
            if not body:
                continue
            msg_id = msg.get("id")
            ticker = default_ticker or _first_symbol(msg)
            link = f"https://stocktwits.com/message/{msg_id}" if msg_id else url
            items.append(RawItem(self.source_id, body, link, to_iso(msg.get("created_at")), ticker))
            if len(items) >= limit:
                break
        return items


def _first_symbol(msg: Mapping[str, Any]) -> str | None:
    """First cashtag a message references, if any (trending stream has no fixed ticker)."""
    symbols = msg.get("symbols") or []
    for sym in symbols:
        if isinstance(sym, Mapping) and sym.get("symbol"):
            return str(sym["symbol"]).upper()
    return None
