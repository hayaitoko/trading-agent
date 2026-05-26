"""WS-A · Data & history.

Historical bars + fundamentals (:mod:`.history`) and the provider adapters
behind them (:mod:`.providers`). Importing this package is cheap — heavy/optional
deps (alpaca-py, httpx) are pulled lazily by the concrete providers only.
"""

from __future__ import annotations

from .history import (
    Bar,
    BarProvider,
    FundamentalsProvider,
    HistoryDepth,
    HistoryProviderError,
    HistoryService,
    build_history_service,
    resolve_depth,
)

__all__ = [
    "Bar",
    "BarProvider",
    "FundamentalsProvider",
    "HistoryDepth",
    "HistoryProviderError",
    "HistoryService",
    "build_history_service",
    "resolve_depth",
]
