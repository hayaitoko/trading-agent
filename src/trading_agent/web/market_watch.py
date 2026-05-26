"""Threshold watcher for "major market moves" on the bar/quote stream.

Each symbol's first observed price becomes its session **reference**. On every
subsequent observation we compute the signed percent move from that reference
and bucket it into integer *bands* of width ``threshold_pct``. Crossing into a
new, non-zero band emits one :class:`MarketMove` event — so a steady drift fires
once per band stepped through, and a reversal fires when it crosses back.

The watcher is fed by whatever is driving the system: ``handle_bar`` for the
in-process bar bus (``CsvReplayFeed``), or ``observe`` directly from a live quote
poller. It is thread-safe because the feed runs in a background thread while the
web layer reads ``recent()`` from the request thread.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


@dataclass(frozen=True)
class MarketMove:
    """A detected move of a symbol away from its session reference price."""

    symbol: str
    reference_price: float
    current_price: float
    pct_change: float  # signed fraction, e.g. -0.031 == down 3.1%
    direction: str  # "up" | "down"
    timestamp: str = field(default_factory=_utcnow_iso)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "reference_price": self.reference_price,
            "current_price": self.current_price,
            "pct_change": self.pct_change,
            "direction": self.direction,
            "timestamp": self.timestamp,
        }


class MarketMoveWatcher:
    """Detect and retain notable percent moves per symbol.

    Args:
        threshold_pct: band width in **percent** (``2.0`` == 2%). A move fires
            on each new non-zero band crossed relative to the session reference.
        max_events: cap on retained recent events (newest first).
    """

    def __init__(self, threshold_pct: float = 2.0, *, max_events: int = 100) -> None:
        if threshold_pct <= 0:
            raise ValueError("threshold_pct must be positive")
        self.threshold = threshold_pct / 100.0
        self._reference: dict[str, float] = {}
        self._last_band: dict[str, int] = {}
        self._events: deque[MarketMove] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def observe(self, symbol: str | None, price: float | None) -> MarketMove | None:
        """Record a price for ``symbol``; return a :class:`MarketMove` if one fired."""
        if not symbol or price is None or price <= 0:
            return None
        with self._lock:
            reference = self._reference.setdefault(symbol, price)
            pct = (price - reference) / reference
            band = int(pct / self.threshold)  # signed; 0 means inside ±threshold
            if band == 0 or band == self._last_band.get(symbol, 0):
                return None
            self._last_band[symbol] = band
            move = MarketMove(
                symbol=symbol,
                reference_price=reference,
                current_price=price,
                pct_change=pct,
                direction="up" if pct >= 0 else "down",
            )
            self._events.appendleft(move)
            return move

    def handle_bar(self, bar: dict[str, Any]) -> MarketMove | None:
        """Bus subscriber: pull symbol + close out of a bar dict."""
        return self.observe(bar.get("symbol"), bar.get("close"))

    def recent(self, limit: int = 20) -> list[MarketMove]:
        """Most recent moves, newest first."""
        with self._lock:
            return list(self._events)[:limit]

    def reset(self) -> None:
        """Clear references and events (e.g. at a new session)."""
        with self._lock:
            self._reference.clear()
            self._last_band.clear()
            self._events.clear()
