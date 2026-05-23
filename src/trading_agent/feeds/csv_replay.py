"""Replay a CSV (or synthetic) bar series through the MessageBus.

Useful for backtests, demos, and unit tests — exercises the full strategy
→ signal → router → broker pipeline without needing live market access.

CSV format expected (header row required):
    timestamp,symbol,open,high,low,close,volume

If ``symbol`` is omitted from the CSV, ``default_symbol`` is used for every row.
"""

from __future__ import annotations

import asyncio
import csv
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Union

from ..data_feed import DataFeed, MessageBus

PathLike = Union[str, Path]


class CsvReplayFeed(DataFeed):
    """Synchronous replay feed that publishes ``bar.<symbol>`` topics."""

    def __init__(
        self,
        message_bus: MessageBus,
        bars: Iterable[dict[str, Any]] | None = None,
        csv_path: PathLike | None = None,
        default_symbol: str | None = None,
    ) -> None:
        super().__init__(message_bus)
        if (bars is None) == (csv_path is None):
            raise ValueError("Provide exactly one of bars= or csv_path=")
        self._csv_path = Path(csv_path) if csv_path else None
        self._inline_bars = list(bars) if bars is not None else None
        self._default_symbol = default_symbol
        self._connected = False

    # --- DataFeed ABC -------------------------------------------------------

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def subscribe_symbols(self, symbols: list[str]) -> None:
        for s in symbols:
            self._subscriptions.add(s)

    async def unsubscribe_symbols(self, symbols: list[str]) -> None:
        for s in symbols:
            self._subscriptions.discard(s)

    # --- Sync API for backtests and demos ----------------------------------

    def replay(self) -> int:
        """Publish every bar to ``bar.<symbol>`` synchronously. Returns count published."""
        published = 0
        for bar in self._iter_bars():
            symbol = str(bar.get("symbol") or self._default_symbol or "")
            if not symbol:
                continue
            if self._subscriptions and symbol not in self._subscriptions:
                continue
            bar["symbol"] = symbol
            self.message_bus.publish(f"bar.{symbol}", bar)
            published += 1
        return published

    async def run(self) -> int:
        """Async wrapper for :meth:`replay`."""
        await self.connect()
        try:
            return await asyncio.to_thread(self.replay)
        finally:
            await self.disconnect()

    # --- Internals ----------------------------------------------------------

    def _iter_bars(self) -> Iterator[dict[str, Any]]:
        if self._inline_bars is not None:
            for bar in self._inline_bars:
                yield dict(bar)
            return
        assert self._csv_path is not None
        with open(self._csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield _coerce_bar(row)


def _coerce_bar(row: dict[str, str]) -> dict[str, Any]:
    """Coerce CSV string fields to numeric types where appropriate."""
    out: dict[str, Any] = dict(row)
    for key in ("open", "high", "low", "close", "volume"):
        if key in out and out[key] not in ("", None):
            try:
                out[key] = float(out[key])
            except (TypeError, ValueError):
                pass
    return out


def synthetic_mean_reverting_bars(
    symbol: str,
    n: int = 200,
    mean: float = 100.0,
    amplitude: float = 5.0,
    period: int = 40,
    noise: float = 0.5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate ``n`` OHLCV bars that oscillate around ``mean`` with cosine + noise.

    Used by the end-to-end demo so it can run with no internet access.
    """
    import math
    import random
    from datetime import datetime, timedelta

    rng = random.Random(seed)
    bars: list[dict[str, Any]] = []
    start = datetime(2026, 1, 1)
    for i in range(n):
        base = mean + amplitude * math.cos(2 * math.pi * i / period)
        close = base + rng.gauss(0, noise)
        open_ = close + rng.gauss(0, noise / 2)
        high = max(open_, close) + abs(rng.gauss(0, noise / 2))
        low = min(open_, close) - abs(rng.gauss(0, noise / 2))
        bars.append(
            {
                "timestamp": (start + timedelta(minutes=i)).isoformat(),
                "symbol": symbol,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000.0 + rng.uniform(-200, 200),
            }
        )
    return bars
