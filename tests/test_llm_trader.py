"""Tests for LLMTrader / StrategyTrader / decision mapping (no network)."""

from typing import Any

from trading_agent.llm.openrouter import ChatResult, OpenRouterError
from trading_agent.llm.trader import (
    DecisionResult,
    LLMTrader,
    StrategyTrader,
    TradeDecision,
    Trader,
    decision_to_signal,
)


class FakeClient:
    """Duck-typed OpenRouterClient: returns a canned content or raises."""

    def __init__(self, content: str | None = None, error: str | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat(self, model: str, messages: list[dict[str, str]], **kw: Any) -> ChatResult:
        self.calls.append({"model": model, "messages": messages, "kw": kw})
        if self.error is not None:
            raise OpenRouterError(self.error)
        assert self.content is not None
        return ChatResult(content=self.content, model=model, usage={"total_tokens": 10})


def test_decision_to_signal_mapping() -> None:
    assert decision_to_signal(TradeDecision("AAPL", "BUY", 2)) == {
        "asset": "AAPL", "side": "BUY", "type": "market", "amount": 2.0, "reason": "",
    }
    assert decision_to_signal(TradeDecision("AAPL", "SELL", 1))["side"] == "SELL"
    assert decision_to_signal(TradeDecision("AAPL", "HOLD", 0)) is None
    assert decision_to_signal(TradeDecision("AAPL", "BUY", 0)) is None  # zero qty


def test_llm_trader_parses_decisions() -> None:
    content = (
        '{"decisions": [{"symbol": "AAPL", "action": "BUY", "quantity": 3, '
        '"reason": "dip"}, {"symbol": "MSFT", "action": "HOLD", "quantity": 0}], '
        '"comment": "buying the dip"}'
    )
    trader = LLMTrader("test/model", FakeClient(content=content), symbols=["AAPL", "MSFT"])
    result = trader.decide({"cash": 10000, "positions": []})
    assert result.error is None
    assert result.comment == "buying the dip"
    assert [(d.symbol, d.action, d.quantity) for d in result.decisions] == [
        ("AAPL", "BUY", 3.0),
        ("MSFT", "HOLD", 0.0),
    ]
    # request used json_mode
    assert trader.client.calls[0]["kw"]["json_mode"] is True  # type: ignore[attr-defined]


def test_llm_trader_filters_unknown_symbol_and_bad_action() -> None:
    content = (
        '{"decisions": [{"symbol": "TSLA", "action": "BUY", "quantity": 1}, '
        '{"symbol": "AAPL", "action": "YOLO", "quantity": 1}]}'
    )
    trader = LLMTrader("m", FakeClient(content=content), symbols=["AAPL"])
    result = trader.decide({"cash": 1000, "positions": []})
    assert result.decisions == []  # TSLA not tradable, YOLO invalid


def test_llm_trader_parse_error_is_contained() -> None:
    trader = LLMTrader("m", FakeClient(content="totally not json"), symbols=["AAPL"])
    result = trader.decide({"cash": 1000, "positions": []})
    assert result.error and "parse" in result.error
    assert result.decisions == []


def test_llm_trader_api_error_is_contained() -> None:
    trader = LLMTrader("m", FakeClient(error="429 rate limited"), symbols=["AAPL"])
    result = trader.decide({"cash": 1000, "positions": []})
    assert result.error == "429 rate limited"
    assert result.decisions == []


def test_llm_trader_observe_builds_context() -> None:
    client = FakeClient(content='{"decisions": []}')
    trader = LLMTrader("m", client, symbols=["AAPL"], lookback=5)
    for px in (100, 101, 102):
        trader.observe({"symbol": "AAPL", "close": px})
    trader.observe({"symbol": "IGNORED", "close": 5})  # not a tradable symbol
    trader.decide({"cash": 1000, "positions": []})
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "100" in user_msg and "102" in user_msg
    assert "IGNORED" not in user_msg


def test_llm_trader_satisfies_protocol() -> None:
    trader = LLMTrader("m", FakeClient(content="{}"), symbols=["AAPL"])
    assert isinstance(trader, Trader)


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
    assert trader.decide({})  == DecisionResult(comment="no signal")


def test_strategy_trader_satisfies_protocol() -> None:
    trader = StrategyTrader(_FakeStrategy([]), name="baseline")
    assert isinstance(trader, Trader)


# --- mandated trading style --------------------------------------------------


def test_style_folded_into_system_prompt() -> None:
    client = FakeClient(content='{"decisions": [], "comment": "ok"}')
    trader = LLMTrader("m", client, symbols=["AAPL"], style="aggressive momentum")
    assert "aggressive momentum" in trader.system_prompt
    trader.decide({"cash": 1000, "positions": []})
    system_msg = client.calls[0]["messages"][0]
    assert system_msg["role"] == "system"
    assert "aggressive momentum" in system_msg["content"]


def test_no_style_uses_base_prompt() -> None:
    trader = LLMTrader("m", FakeClient(content="{}"), symbols=["AAPL"])
    assert "mandated trading style" not in trader.system_prompt
