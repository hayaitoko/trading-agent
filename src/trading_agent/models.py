from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
TimeInForce = Literal["day", "gtc"]


@dataclass(frozen=True)
class Post:
    source: str
    post_id: str
    author: str
    text: str
    url: str
    created_at: datetime
    score: int = 0
    num_comments: int = 0


@dataclass(frozen=True)
class Signal:
    ticker: str
    sentiment: float
    confidence: float
    mentions: int
    window_start: datetime
    window_end: datetime
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Order:
    ticker: str
    side: Side
    qty: int
    order_type: OrderType = "market"
    limit_price: Decimal | None = None
    time_in_force: TimeInForce = "day"


@dataclass(frozen=True)
class Position:
    ticker: str
    qty: int
    avg_cost: Decimal
    current_price: Decimal


@dataclass(frozen=True)
class Trade:
    order_id: str
    ticker: str
    side: Side
    qty: int
    price: Decimal
    executed_at: datetime
