"""LLM layer: OpenRouter client + LLM-driven traders for the evaluation bench.

The bench (see :mod:`trading_agent.bench`) runs several of these against the
same real market data on isolated paper books and compares P&L.
"""

from .openrouter import ChatResult, OpenRouterClient, OpenRouterError
from .trader import (
    AgentTrader,
    DecisionResult,
    StrategyTrader,
    TradeDecision,
    Trader,
    decision_to_signal,
)

__all__ = [
    "AgentTrader",
    "ChatResult",
    "DecisionResult",
    "OpenRouterClient",
    "OpenRouterError",
    "StrategyTrader",
    "TradeDecision",
    "Trader",
    "decision_to_signal",
]
