"""Tests for LLMTrader / StrategyTrader / decision mapping (no network)."""

from dataclasses import dataclass, field
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
from trading_agent.memory.embed import EmbedError


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


# --- WS-A P2: layered research + memory context ------------------------------


@dataclass
class _Brief:
    ticker: str
    summary: str
    sentiment: float = 0.0
    catalysts: list[str] = field(default_factory=list)
    id: str = ""


@dataclass
class _Lesson:
    text: str
    trader_id: str = ""


class _FakeResearch:
    """Duck-typed ResearchStore: records recall calls, serves canned briefs."""

    def __init__(
        self,
        *,
        search: list[_Brief] | None = None,
        by_symbol: dict[str, list[_Brief]] | None = None,
        recent: list[_Brief] | None = None,
    ) -> None:
        self._search = search or []
        self._by_symbol = by_symbol or {}
        self._recent = recent or []
        self.calls: list[tuple[str, str]] = []

    def search(self, owner: str, query: str, k: int) -> list[_Brief]:
        self.calls.append(("search", query))
        return self._search[:k]

    def get(self, owner: str, ticker: str) -> list[_Brief]:
        self.calls.append(("get", ticker))
        return self._by_symbol.get(ticker, [])

    def recent(self, owner: str, k: int) -> list[_Brief]:
        self.calls.append(("recent", str(k)))
        return self._recent[:k]


class _FakeMemory:
    """Duck-typed MemoryStore: serves canned lessons or raises on recall."""

    def __init__(self, lessons: list[_Lesson] | None = None, raises: Exception | None = None) -> None:
        self._lessons = lessons or []
        self._raises = raises
        self.recall_calls: list[tuple[str, str]] = []

    def recall(self, owner: str, trader_id: str, query: str, k: int) -> list[_Lesson]:
        self.recall_calls.append((owner, trader_id))
        if self._raises is not None:
            raise self._raises
        return self._lessons[:k]


def _user_msg(client: FakeClient) -> str:
    return client.calls[0]["messages"][1]["content"]


def test_trader_layers_research_and_memory_blocks() -> None:
    client = FakeClient(content='{"decisions": []}')
    research = _FakeResearch(search=[_Brief("AAPL", "buyable dip on volume", 0.4, ["earnings"])])
    memory = _FakeMemory([_Lesson("don't chase gaps on AAPL", "alpha")])
    trader = LLMTrader(
        "m", client, symbols=["AAPL"], name="alpha",
        research=research, memory=memory, owner_user_id="u1",
    )
    result = trader.decide({"cash": 1000, "positions": []})
    assert result.error is None
    msg = _user_msg(client)
    assert "Research briefs" in msg and "buyable dip on volume" in msg
    assert "Your past lessons" in msg and "don't chase gaps on AAPL" in msg
    # exactly one decision trailer, never doubled
    assert msg.count("Return your JSON decision now.") == 1
    # recall keys on the trader's own name (closes the recall/reflect loop)
    assert memory.recall_calls == [("u1", "alpha")]


def test_memory_omitted_on_embed_error_but_decision_made() -> None:
    client = FakeClient(content='{"decisions": [{"symbol": "AAPL", "action": "HOLD", "quantity": 0}]}')
    research = _FakeResearch(search=[_Brief("AAPL", "still constructive", 0.2)])
    memory = _FakeMemory(raises=EmbedError("no local embed endpoint"))
    trader = LLMTrader(
        "m", client, symbols=["AAPL"], name="alpha",
        research=research, memory=memory, owner_user_id="u1",
    )
    result = trader.decide({"cash": 1000, "positions": []})
    assert result.error is None  # the EmbedError never reaches the decision
    msg = _user_msg(client)
    assert "still constructive" in msg  # research still present
    assert "Your past lessons" not in msg  # memory block dropped


def test_owner_none_omits_research_and_memory() -> None:
    client = FakeClient(content='{"decisions": []}')
    research = _FakeResearch(search=[_Brief("AAPL", "should not appear", 0.9)])
    memory = _FakeMemory([_Lesson("should not appear either", "alpha")])
    trader = LLMTrader(
        "m", client, symbols=["AAPL"], name="alpha",
        research=research, memory=memory, owner_user_id=None,
    )
    trader.decide({"cash": 1000, "positions": []})
    msg = _user_msg(client)
    assert "should not appear" not in msg
    assert "Research briefs" not in msg and "Your past lessons" not in msg
    assert memory.recall_calls == []  # never queried without an owner


def test_research_falls_back_search_then_symbol_then_recent() -> None:
    client = FakeClient(content='{"decisions": []}')
    # search empty, per-symbol empty, recent has one → recent is what renders
    research = _FakeResearch(recent=[_Brief("MSFT", "recent fallback brief", -0.1)])
    trader = LLMTrader("m", client, symbols=["AAPL"], research=research, owner_user_id="u1")
    trader.decide({"cash": 1000, "positions": []})
    msg = _user_msg(client)
    assert "recent fallback brief" in msg
    assert [c[0] for c in research.calls] == ["search", "get", "recent"]


def test_research_per_symbol_fallback_when_search_empty() -> None:
    client = FakeClient(content='{"decisions": []}')
    research = _FakeResearch(by_symbol={"AAPL": [_Brief("AAPL", "from per-symbol get", 0.3, id="b1")]})
    trader = LLMTrader("m", client, symbols=["AAPL"], research=research, owner_user_id="u1")
    trader.decide({"cash": 1000, "positions": []})
    msg = _user_msg(client)
    assert "from per-symbol get" in msg
    assert "recent" not in [c[0] for c in research.calls]  # didn't need recent


def test_no_stores_keeps_30_close_fallback_and_single_trailer() -> None:
    client = FakeClient(content='{"decisions": []}')
    trader = LLMTrader("m", client, symbols=["AAPL"], lookback=5)
    for px in (100, 101, 102):
        trader.observe({"symbol": "AAPL", "close": px})
    trader.decide({"cash": 1000, "positions": []})
    msg = _user_msg(client)
    assert "100" in msg and "102" in msg  # close prices, unchanged path
    assert "Research briefs" not in msg and "Your past lessons" not in msg
    assert msg.count("Return your JSON decision now.") == 1


class _FakeHistory:
    """Duck-typed HistoryService: honors include_trailer like the real one."""

    def context_block(self, symbols: list[str], account: dict[str, Any], *, include_trailer: bool = True) -> str:
        body = f"RICH HISTORY for {', '.join(symbols)}"
        return body + "\n\nReturn your JSON decision now." if include_trailer else body


def test_history_block_composes_without_double_trailer() -> None:
    client = FakeClient(content='{"decisions": []}')
    trader = LLMTrader(
        "m", client, symbols=["AAPL"], history=_FakeHistory(),
        research=_FakeResearch(search=[_Brief("AAPL", "rich-path brief", 0.1)]),
        owner_user_id="u1",
    )
    trader.decide({"cash": 1000, "positions": []})
    msg = _user_msg(client)
    assert "RICH HISTORY for AAPL" in msg and "rich-path brief" in msg
    assert msg.count("Return your JSON decision now.") == 1
