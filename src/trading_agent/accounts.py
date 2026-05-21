from dataclasses import dataclass

from trading_agent.brokers.base import Broker


@dataclass
class Account:
    id: str
    name: str
    broker: Broker
    enabled: bool = True
