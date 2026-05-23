"""Core data models for the trading agent."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Re-export canonical OrderType from enums to avoid shadow class definition.
from .enums import OrderType  # noqa: F401


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class PositionType(Enum):
    """Types of positions."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Signal:
    """Represents a trading signal."""
    symbol: str
    type: SignalType
    strength: float  # Signal strength between 0 and 1
    timestamp: datetime
    price: float | None = None
    metadata: dict | None = None


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    type: OrderType
    quantity: float
    price: float | None = None
    timestamp: datetime | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    type: PositionType
    quantity: float
    entry_price: float
    timestamp: datetime
    profit_loss: float = 0.0
