"""Instrument models beyond plain long equities.

Currently single-leg equity options (:mod:`.options`) with a chain-data provider
interface (:mod:`.options_chain`). Short selling and crypto live on the shared
PaperBroker; this package holds models that need their own identity / accounting.
"""

from .options import (
    OptionContract,
    OptionPosition,
    OptionQuote,
    OptionRight,
    OptionsBook,
    black_scholes_price,
    mark_price,
)
from .options_chain import (
    AlpacaOptionChainProvider,
    OptionChainProvider,
    OptionsProviderError,
)

__all__ = [
    "AlpacaOptionChainProvider",
    "OptionChainProvider",
    "OptionContract",
    "OptionPosition",
    "OptionQuote",
    "OptionRight",
    "OptionsBook",
    "OptionsProviderError",
    "black_scholes_price",
    "mark_price",
]
