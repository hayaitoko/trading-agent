"""Market-hours helpers for the PaperBroker ``is_market_open`` hook.

US equities follow regular trading hours (09:30–16:00 America/New_York, Mon–Fri;
holidays are not modelled — for holiday-accurate behaviour run against an Alpaca
paper account). Crypto trades 24/7, so the asset-class-aware :func:`market_clock`
never gates a crypto symbol.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, time
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_CLOSE = time(16, 0)


def is_us_equity_market_open(now: datetime | None = None) -> bool:
    """Return True during US equity regular trading hours (ignores holidays)."""
    dt = now.astimezone(_ET) if now is not None else datetime.now(_ET)
    if dt.weekday() >= 5:  # Saturday/Sunday
        return False
    return _OPEN <= dt.time() < _CLOSE


def is_crypto_symbol(symbol: str, crypto_symbols: Iterable[str] | None = None) -> bool:
    """Classify ``symbol`` as crypto.

    Config-driven: when ``crypto_symbols`` is supplied a symbol is crypto iff it
    is in that set. With no set we fall back to the ``BASE/QUOTE`` pair notation
    convention (e.g. ``BTC/USDT``) that ccxt symbols use.
    """
    if crypto_symbols is not None:
        return symbol in set(crypto_symbols)
    return "/" in symbol


def us_equity_clock() -> Callable[[str], bool]:
    """Return a ``Callable[[str], bool]`` for PaperBroker(is_market_open=...).

    The symbol argument is ignored — all US equities share the same session.
    """
    return lambda _symbol: is_us_equity_market_open()


def market_clock(
    crypto_symbols: Iterable[str] = (),
    *,
    equity_open: Callable[[], bool] = is_us_equity_market_open,
) -> Callable[[str], bool]:
    """Asset-class-aware clock for PaperBroker(is_market_open=...).

    Crypto symbols (members of ``crypto_symbols``, or anything in ``BASE/QUOTE``
    notation) are always open; every other symbol defers to ``equity_open`` (US
    RTH by default). ``equity_open`` is injectable so tests can pin the equity
    session without depending on the wall clock.
    """
    crypto = set(crypto_symbols)

    def _open(symbol: str) -> bool:
        if symbol in crypto or "/" in symbol:
            return True
        return equity_open()

    return _open
