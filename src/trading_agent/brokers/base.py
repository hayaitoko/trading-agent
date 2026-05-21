from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal

from trading_agent.models import Order, Position, Trade


class BrokerError(Exception):
    pass


class InsufficientFundsError(BrokerError):
    pass


class UnknownTickerError(BrokerError):
    pass


class Broker(ABC):
    @abstractmethod
    async def get_account_value(self) -> Decimal: ...

    @abstractmethod
    async def get_cash(self) -> Decimal: ...

    @abstractmethod
    async def get_positions(self) -> list[Position]: ...

    @abstractmethod
    async def get_quote(self, ticker: str) -> Decimal: ...

    @abstractmethod
    async def place_order(self, order: Order) -> str: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    async def get_trades(self, since: datetime | None = None) -> list[Trade]: ...
