from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class Mode(Enum):
    AUTONOMOUS = "autonomous"
    APPROVAL = "approval"


class AssetClass(Enum):
    """The kind of instrument a symbol denotes.

    Drives session rules (crypto trades 24/7, equities follow US RTH) and order
    handling (crypto allows fractional quantities). ``OPTION`` is reserved for the
    options instrument model.
    """

    EQUITY = "equity"
    CRYPTO = "crypto"
    OPTION = "option"
