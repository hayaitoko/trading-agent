from dataclasses import dataclass
from decimal import Decimal

from trading_agent.brokers.base import Broker

# Default model bound to a new account if none is specified. Kept as a string
# constant rather than imported from chat.models to avoid a layering dependency.
DEFAULT_ACCOUNT_MODEL = "anthropic/claude-sonnet-4.6"


@dataclass
class Account:
    id: str
    name: str
    broker: Broker
    starting_cash: Decimal
    enabled: bool = True
    model: str = DEFAULT_ACCOUNT_MODEL
