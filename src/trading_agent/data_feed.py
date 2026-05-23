from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class MessageBus:
    """Simple in-process pub/sub message bus for routing market data."""

    def __init__(self):
        self._subscribers: dict[str, set[Callable]] = {}

    def subscribe(self, topic: str, handler: Callable) -> None:
        if topic not in self._subscribers:
            self._subscribers[topic] = set()
        self._subscribers[topic].add(handler)

    def unsubscribe(self, topic: str, handler: Callable) -> None:
        if topic in self._subscribers:
            self._subscribers[topic].discard(handler)
            if not self._subscribers[topic]:
                del self._subscribers[topic]

    def publish(self, topic: str, message: Any) -> None:
        if topic in self._subscribers:
            for handler in self._subscribers[topic]:
                try:
                    handler(message)
                except Exception:
                    # Suppress subscriber errors to avoid breaking message delivery to others
                    pass


class DataFeed(ABC):
    """Abstract base class for market data feeds."""

    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self._subscriptions: set[str] = set()

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def subscribe_symbols(self, symbols: list[str]) -> None:
        pass

    @abstractmethod
    async def unsubscribe_symbols(self, symbols: list[str]) -> None:
        pass
