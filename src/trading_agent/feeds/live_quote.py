"""Poll a real broker's quotes and drive a PaperBroker with live prices.

This is the "Path B" mock-trading mode: real market values, simulated fills.
Any object exposing ``get_quote(symbol) -> dict`` works as the quote source —
in practice an ``AlpacaBroker`` (real US-equity quotes) or ``CCXTBroker``
(public crypto tickers, no API key needed for read-only price data).

On each poll the feed:
  * fetches a quote per symbol,
  * pushes bid/ask/last into the PaperBroker (so its fills use real prices and
    resting limit orders get matched),
  * publishes ``quote.<symbol>`` and (optionally) a synthetic ``bar.<symbol>``
    so existing bar-driven strategies keep working unchanged.

Use :meth:`poll_once` for tests/backtests; :meth:`run` for a live loop.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Protocol

from ..data_feed import DataFeed, MessageBus


class QuoteSource(Protocol):
    def get_quote(self, symbol: str) -> dict[str, Any]: ...


def _extract_price(quote: dict[str, Any]) -> float | None:
    """Pull a usable last/mid price out of a broker quote dict."""
    for key in ("price", "last"):
        val = quote.get(key)
        if val is not None:
            return float(val)
    bid, ask = quote.get("bid"), quote.get("ask")
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0
    return None


class LiveQuoteFeed(DataFeed):
    """Polling feed that mirrors a real quote source into a PaperBroker."""

    def __init__(
        self,
        message_bus: MessageBus,
        quote_source: QuoteSource,
        symbols: list[str],
        paper_broker: Any | None = None,
        poll_interval: float = 5.0,
        emit_bars: bool = True,
    ) -> None:
        super().__init__(message_bus)
        self._quote_source = quote_source
        self._symbols = list(symbols)
        self._paper_broker = paper_broker
        self._poll_interval = poll_interval
        self._emit_bars = emit_bars
        self._connected = False
        self._running = False
        for s in self._symbols:
            self._subscriptions.add(s)

    # --- DataFeed ABC -------------------------------------------------------

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        self._running = False

    async def subscribe_symbols(self, symbols: list[str]) -> None:
        for s in symbols:
            if s not in self._symbols:
                self._symbols.append(s)
            self._subscriptions.add(s)

    async def unsubscribe_symbols(self, symbols: list[str]) -> None:
        for s in symbols:
            if s in self._symbols:
                self._symbols.remove(s)
            self._subscriptions.discard(s)

    # --- Polling ------------------------------------------------------------

    def poll_once(self) -> dict[str, dict[str, Any]]:
        """Fetch one quote per symbol, feed the broker, publish events.

        Returns the raw quotes keyed by symbol. Quote-source errors for a single
        symbol are swallowed so one bad symbol can't stall the whole poll.
        """
        results: dict[str, dict[str, Any]] = {}
        for symbol in list(self._symbols):
            try:
                quote = self._quote_source.get_quote(symbol)
            except Exception:
                continue

            price = _extract_price(quote)
            if price is None:
                continue

            bid = quote.get("bid")
            ask = quote.get("ask")

            if self._paper_broker is not None:
                self._paper_broker.update_quote(
                    symbol,
                    bid=float(bid) if bid is not None else None,
                    ask=float(ask) if ask is not None else None,
                    last=price,
                )

            now = datetime.now(UTC).isoformat()
            self.message_bus.publish(
                f"quote.{symbol}",
                {"symbol": symbol, "bid": bid, "ask": ask, "price": price, "timestamp": now},
            )
            if self._emit_bars:
                self.message_bus.publish(
                    f"bar.{symbol}",
                    {
                        "timestamp": now,
                        "symbol": symbol,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": float(quote.get("volume") or 0.0),
                    },
                )
            results[symbol] = quote
        return results

    async def run(self, max_polls: int | None = None) -> int:
        """Poll on ``poll_interval`` until stopped (or ``max_polls`` reached).

        Returns the number of polls performed. Set ``max_polls`` in tests to
        bound the loop; leave it None for an open-ended live session.
        """
        await self.connect()
        self._running = True
        polls = 0
        try:
            while self._running:
                await asyncio.to_thread(self.poll_once)
                polls += 1
                if max_polls is not None and polls >= max_polls:
                    break
                await asyncio.sleep(self._poll_interval)
        finally:
            await self.disconnect()
        return polls

    def stop(self) -> None:
        self._running = False
