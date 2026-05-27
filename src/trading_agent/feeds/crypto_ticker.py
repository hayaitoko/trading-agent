"""Poll crypto tickers from a ccxt-style quote source and fan them to the bench.

Crypto trades 24/7, so unlike the equity path there is no market-hours gate and
quantities may be fractional. This feed reads a ``get_quote(symbol) -> dict``
source (in practice a :class:`~trading_agent.ccxt_broker.CCXTBroker`, whose
public price data needs no API key) and pushes each tick straight into a bench's
``observe_quote`` / ``observe_bar`` so quote- and bar-driven traders both see it.

Use :meth:`poll_once` for tests/backtests; :meth:`run` for a live loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol


class QuoteSource(Protocol):
    def get_quote(self, symbol: str) -> dict[str, Any]: ...


def _last_or_mid(quote: dict[str, Any]) -> float | None:
    """Pull a usable last/mid price out of a ticker dict."""
    for key in ("last", "price"):
        val = quote.get(key)
        if val is not None:
            return float(val)
    bid, ask = quote.get("bid"), quote.get("ask")
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0
    return None


class CryptoTickerFeed:
    """Fan a crypto quote source into a bench's observe callbacks.

    ``on_quote`` / ``on_bar`` are normally ``bench.observe_quote`` /
    ``bench.observe_bar``. A bad symbol (the source raising, or no usable price)
    is skipped so one bad pair can't stall the poll.
    """

    def __init__(
        self,
        quote_source: QuoteSource,
        symbols: list[str],
        *,
        on_quote: Callable[[dict[str, Any]], None] | None = None,
        on_bar: Callable[[dict[str, Any]], None] | None = None,
        emit_bars: bool = True,
        poll_interval: float = 5.0,
    ) -> None:
        self._quote_source = quote_source
        self._symbols = list(symbols)
        self._on_quote = on_quote
        self._on_bar = on_bar
        self._emit_bars = emit_bars
        self._poll_interval = poll_interval
        self._running = False

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def poll_once(self) -> dict[str, dict[str, Any]]:
        """Fetch one ticker per symbol, push it to the bench, return raw quotes."""
        results: dict[str, dict[str, Any]] = {}
        for symbol in list(self._symbols):
            try:
                quote = self._quote_source.get_quote(symbol)
            except Exception:
                continue
            price = _last_or_mid(quote)
            if price is None:
                continue
            bid, ask = quote.get("bid"), quote.get("ask")
            now = datetime.now(UTC).isoformat()
            if self._on_quote is not None:
                self._on_quote(
                    {"symbol": symbol, "bid": bid, "ask": ask, "price": price, "timestamp": now}
                )
            if self._emit_bars and self._on_bar is not None:
                self._on_bar(
                    {
                        "timestamp": now,
                        "symbol": symbol,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": float(quote.get("volume") or 0.0),
                    }
                )
            results[symbol] = quote
        return results

    async def run(self, max_polls: int | None = None) -> int:
        """Poll on ``poll_interval`` until stopped (or ``max_polls`` reached)."""
        self._running = True
        polls = 0
        while self._running:
            await asyncio.to_thread(self.poll_once)
            polls += 1
            if max_polls is not None and polls >= max_polls:
                break
            await asyncio.sleep(self._poll_interval)
        return polls

    def stop(self) -> None:
        self._running = False
