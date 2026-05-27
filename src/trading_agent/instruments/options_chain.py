"""Options-chain data behind a provider interface — real data only.

Mirrors WS-J's discipline: the default :class:`AlpacaOptionChainProvider` reads
the *same* Alpaca data keys as the live books (``ALPACA_API_KEY`` /
``ALPACA_SECRET_KEY``, from the environment, never hardcoded) and **fails loud**
(:class:`OptionsProviderError`) when they are missing — no silent stub chains. A
``data_client`` seam lets tests inject a fake ``OptionHistoricalDataClient`` so CI
stays fully offline.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from .options import OptionContract, OptionQuote


class OptionsProviderError(RuntimeError):
    """An options data provider is unreachable or missing credentials."""


@runtime_checkable
class OptionChainProvider(Protocol):
    """Source of option-chain quotes for an underlying."""

    def get_chain(self, underlying: str, expiry: str | None = None) -> list[OptionQuote]: ...

    def get_quote(self, contract: OptionContract) -> OptionQuote | None: ...


class AlpacaOptionChainProvider:
    """Fetch option-chain snapshots from Alpaca's options market-data API."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        *,
        data_client: Any = None,
        feed: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self._data_client = data_client  # test seam: inject a fake client
        self._feed = feed

    # --- OptionChainProvider ------------------------------------------------

    def get_chain(self, underlying: str, expiry: str | None = None) -> list[OptionQuote]:
        """Return the live quote snapshot for every listed contract on ``underlying``.

        ``expiry`` (ISO ``YYYY-MM-DD``) narrows to a single expiration when given.
        """
        from alpaca.data.requests import OptionChainRequest

        kwargs: dict[str, Any] = {"underlying_symbol": underlying}
        if expiry is not None:
            kwargs["expiration_date"] = expiry
        if self._feed is not None:
            kwargs["feed"] = self._feed
        request = OptionChainRequest(**kwargs)

        try:
            chain = self._client().get_option_chain(request)
        except OptionsProviderError:
            raise
        except Exception as exc:  # network / auth / rate-limit etc.
            raise OptionsProviderError(f"alpaca option chain failed for {underlying}: {exc}")

        quotes: list[OptionQuote] = []
        for occ, snapshot in (chain or {}).items():
            quote = self._snapshot_to_quote(occ, snapshot)
            if quote is not None:
                quotes.append(quote)
        return quotes

    def get_quote(self, contract: OptionContract) -> OptionQuote | None:
        """Return the quote for a specific contract, or ``None`` if not listed."""
        target = contract.occ_symbol
        for quote in self.get_chain(contract.underlying, contract.expiry.isoformat()):
            if quote.contract.occ_symbol == target:
                return quote
        return None

    # --- internals ----------------------------------------------------------

    def _client(self) -> Any:
        if self._data_client is None:
            if not (self._api_key and self._secret_key):
                raise OptionsProviderError(
                    "Alpaca options keys missing: set ALPACA_API_KEY / ALPACA_SECRET_KEY."
                )
            from alpaca.data.historical.option import OptionHistoricalDataClient

            self._data_client = OptionHistoricalDataClient(self._api_key, self._secret_key)
        return self._data_client

    @staticmethod
    def _snapshot_to_quote(occ: str, snapshot: Any) -> OptionQuote | None:
        try:
            contract = OptionContract.from_occ(occ)
        except ValueError:
            return None
        latest_quote = getattr(snapshot, "latest_quote", None)
        latest_trade = getattr(snapshot, "latest_trade", None)
        return OptionQuote(
            contract=contract,
            bid=_f(getattr(latest_quote, "bid_price", None)),
            ask=_f(getattr(latest_quote, "ask_price", None)),
            last=_f(getattr(latest_trade, "price", None)),
        )


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
