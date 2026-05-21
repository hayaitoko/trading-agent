from dataclasses import dataclass
from decimal import Decimal

from trading_agent.brokers.base import Broker


@dataclass
class Account:
    id: str
    name: str
    broker: Broker
    starting_cash: Decimal
    enabled: bool = True
