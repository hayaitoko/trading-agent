"""Tests for StrategyTrader + decision mapping (no network).

LLMTrader was retired in WS-Bench-Migration M2 — the bench runs AgentTrader now
(see test_agent_trader.py for that surface). This file covers the surviving
non-agent pieces of llm/trader.py: the deterministic StrategyTrader baseline and
the decision_to_signal mapping.
"""

from typing import Any

from trading_agent.llm.trader import (
    DecisionResult,
    StrategyTrader,
    TradeDecision,
    Trader,
    decision_to_signal,
)


def test_decision_to_signal_mapping() -> None:
    assert decision_to_signal(TradeDecision("AAPL", "BUY", 2)) == {
        "asset": "AAPL", "side": "BUY", "type": "market", "amount": 2.0, "reason": "",
    }
    assert decision_to_signal(TradeDecision("AAPL", "SELL", 1))["side"] == "SELL"
    assert decision_to_signal(TradeDecision("AAPL", "HOLD", 0)) is None
    assert decision_to_signal(TradeDecision("AAPL", "BUY", 0)) is None  # zero qty


class _FakeStrategy:
    def __init__(self, sides: list[str]) -> None:
        self._sides = iter(sides)

    def on_data(self, bar: dict[str, Any]) -> dict[str, Any]:
        return {"asset": bar["symbol"], "side": next(self._sides, "NEUTRAL"), "amount": 1.0}


def test_strategy_trader_surfaces_latest_signal() -> None:
    strat = _FakeStrategy(["NEUTRAL", "LONG", "NEUTRAL"])
    trader = StrategyTrader(strat, name="mean-reversion")
    trader.observe({"symbol": "AAPL", "close": 100})  # NEUTRAL -> nothing
    trader.observe({"symbol": "AAPL", "close": 99})  # LONG -> pending BUY
    result = trader.decide({"cash": 1000, "positions": []})
    assert len(result.decisions) == 1
    assert result.decisions[0].action == "BUY"
    # consumed: next decide has nothing new
    assert trader.decide({}) == DecisionResult(comment="no signal")


def test_strategy_trader_satisfies_protocol() -> None:
    trader = StrategyTrader(_FakeStrategy([]), name="baseline")
    assert isinstance(trader, Trader)
