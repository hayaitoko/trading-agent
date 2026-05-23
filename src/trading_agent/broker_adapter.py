from abc import ABC, abstractmethod
from typing import Any


class BrokerAdapter(ABC):
    """Abstract base class for broker adapters.

    The canonical place_order contract is a single dict argument
    (``order_details``) so that SignalRouter and all concrete adapters
    (PaperBroker, AlpacaBroker, CCXTBroker) share one stable call shape.
    The expected keys are:

        {
            'symbol':     str,
            'side':       OrderSide enum or 'BUY'/'SELL' string,
            'order_type': OrderType enum or 'market'/'limit' string,
            'amount' or 'quantity': positive float,
            'price':      Optional[float],          # required for limit orders
            'time_in_force': Optional[str],         # optional broker-specific
        }

    Implementations return a result dict on success (containing at least
    ``order_id`` and ``status``) or ``None`` when the order could not be
    constructed (e.g. missing required keys).

    NOTE: This signature is frozen. Do not add **kwargs and do not split
    it into positional parameters — doing so silently breaks every caller.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Test connection to broker."""
        pass

    @abstractmethod
    def get_balance(self) -> dict[str, Any]:
        """Get account balance. Must include a 'cash' key."""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get current quote for symbol. Must include a 'price' key."""
        pass

    @abstractmethod
    def place_order(self, order_details: dict[str, Any]) -> dict[str, Any] | None:
        """Place an order from a canonical order_details dict.

        See class docstring for the dict shape. Returns a result dict
        (with at least 'order_id' and 'status') or None if required keys
        are missing.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get status of specific order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order. Returns True on success, False otherwise."""
        pass

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions."""
        pass
