"""Source adapters: one per ``kind``. Async ``aiohttp``-style HTTP fetchers
(implemented on ``httpx.AsyncClient`` — see :mod:`..fetchers.base`) plus an
isolated, optional headless-browser adapter for JS-walled sites.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import HttpSource, RawItem, Source, SourceError
from .bluesky import BlueskySource
from .browser import BrowserSource, BrowserUnavailable
from .reddit import RedditSource
from .rss import RssSource
from .stocktwits import StockTwitsSource

# kind -> adapter constructor. Typed as Callable (not type[Source]) because the
# Source *protocol* declares no __init__; concrete adapters take (source_id,
# client[, manager]). The registry reads this; config schemas are per adapter
# (see each class' CONFIG_SCHEMA).
ADAPTERS: dict[str, Callable[..., Source]] = {
    RssSource.kind: RssSource,
    RedditSource.kind: RedditSource,
    StockTwitsSource.kind: StockTwitsSource,
    BrowserSource.kind: BrowserSource,
    BlueskySource.kind: BlueskySource,
}

__all__ = [
    "ADAPTERS",
    "BlueskySource",
    "BrowserSource",
    "BrowserUnavailable",
    "HttpSource",
    "RawItem",
    "RedditSource",
    "RssSource",
    "Source",
    "SourceError",
    "StockTwitsSource",
]
