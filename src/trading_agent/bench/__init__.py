"""Multi-model evaluation bench.

One shared price stream (synthetic or real Alpaca quotes) is fanned out to N
isolated paper books, each driven by its own :class:`~trading_agent.llm.trader.Trader`
(an LLM model or a deterministic strategy baseline). Competitors trade
autonomously; the bench computes a live P&L leaderboard.
"""

from .bench import Bench, Competitor, DecisionLogEntry

__all__ = ["Bench", "Competitor", "DecisionLogEntry"]
