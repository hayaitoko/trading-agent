"""Construct real broker adapters from environment / config.

Keeps credential handling in one place and out of the call sites. Imports of
the concrete adapters are lazy so importing this module never forces the
alpaca-py / ccxt SDKs to load unless you actually build that broker.

Credentials come from env vars (never commit them — use a gitignored .env):

    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER   (paper defaults to true)
    BINANCE_API_KEY, BINANCE_SECRET_KEY               (optional; public data
                                                       needs no key)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .config import ConfigError

if TYPE_CHECKING:
    from .alpaca_broker import AlpacaBroker
    from .ccxt_broker import CCXTBroker

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in _TRUTHY


def build_alpaca_broker(
    *,
    api_key: str | None = None,
    secret_key: str | None = None,
    paper: bool | None = None,
) -> AlpacaBroker:
    """Build an :class:`AlpacaBroker`, defaulting to the paper endpoint.

    Explicit args win over env vars. ``paper`` defaults to ``ALPACA_PAPER`` (or
    True if unset) — the safe, Investopedia-equivalent sandbox.

    Raises:
        ConfigError: if API key/secret are not provided or in the environment.
    """
    key = api_key or os.environ.get("ALPACA_API_KEY")
    secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ConfigError(
            "Alpaca credentials missing: set ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "(get free paper-trading keys at https://alpaca.markets)."
        )
    if paper is None:
        paper = _truthy(os.environ.get("ALPACA_PAPER", "true"))

    from .alpaca_broker import AlpacaBroker

    return AlpacaBroker(api_key=key, secret_key=secret, paper=paper)


def build_ccxt_broker(
    exchange: str = "binance",
    *,
    api_key: str | None = None,
    secret: str | None = None,
    passphrase: str | None = None,
    sandbox: bool = False,
) -> CCXTBroker:
    """Build a :class:`CCXTBroker`. Read-only public price data needs no keys,
    so credentials default to empty strings and may be supplied later for trading.
    """
    prefix = exchange.upper()
    key = api_key if api_key is not None else os.environ.get(f"{prefix}_API_KEY", "")
    sec = secret if secret is not None else os.environ.get(f"{prefix}_SECRET_KEY", "")

    from .ccxt_broker import CCXTBroker

    return CCXTBroker(
        exchange_name=exchange,
        api_key=key,
        secret=sec,
        passphrase=passphrase,
        sandbox=sandbox,
    )
