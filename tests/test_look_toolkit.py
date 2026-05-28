"""WS-Agent A1 — LOOK toolkit tests.

Tests cover:
  - TurnContext slot fix (directed_notes, recent_reflections)
  - Uniform ToolResult shape for every enabled and disabled tool
  - advisor_notes isolation (only own trader's notes)
  - memory_search reflections_for_slot
  - ask_manager per-turn gate (≤1 call)
  - Disabled tools return kind="disabled"
  - list_tools() includes all enabled and disabled tools with correct flags
  - MONEY IS REAL: account_state and ask_manager scrub forbidden words

The red-team test for ask_manager paper-leak is in a separate file:
    tests/test_ask_manager_no_paper_leak.py
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_agent.intel.cost_tracker import CostTracker
from trading_agent.intel.tool_envelope import ToolError, ToolResult
from trading_agent.intel.tools.look.account_state import AccountStateTool, _scrub_str
from trading_agent.intel.tools.look.advisor_notes import AdvisorNotesTool
from trading_agent.intel.tools.look.ask_manager import AskManagerTool, _scrub_answer
from trading_agent.intel.tools.look.forecast import ForecastTool
from trading_agent.intel.tools.look.history import HistoryTool
from trading_agent.intel.tools.look.list_tools import ListToolsTool
from trading_agent.intel.tools.look.memory_search import MemorySearchTool
from trading_agent.intel.tools.look.news import NewsTool
from trading_agent.intel.tools.look.options_iv import OptionsIVTool
from trading_agent.intel.tools.look.prediction_market_odds import PredictionMarketOddsTool
from trading_agent.intel.tools.look.recent_turns import RecentTurnsTool
from trading_agent.intel.tools.look.request_research import RequestResearchTool
from trading_agent.intel.tools.look.research_brief import ResearchBriefTool
from trading_agent.intel.tools.look.situation import SituationTool
from trading_agent.intel.tools.look.watchlist import WatchlistTool
from trading_agent.intel.tools.look.world_events import WorldEventsTool
from trading_agent.intel.turn_context import TurnContext, build_first_look

# ---------------------------------------------------------------------------
# TurnContext slot fix — directed_notes and recent_reflections
# ---------------------------------------------------------------------------


def test_turn_context_has_directed_notes_field() -> None:
    ctx = TurnContext(trader_name="T", model="m")
    assert hasattr(ctx, "directed_notes")
    assert ctx.directed_notes == []


def test_turn_context_has_recent_reflections_field() -> None:
    ctx = TurnContext(trader_name="T", model="m")
    assert hasattr(ctx, "recent_reflections")
    assert ctx.recent_reflections == []


def test_build_first_look_directed_notes_shown() -> None:
    ctx = TurnContext(
        trader_name="T", model="m",
        directed_notes=["Review NVDA risk before trading"],
    )
    text = build_first_look(ctx)
    assert "Directed notes:" in text
    assert "Review NVDA risk" in text


def test_build_first_look_recent_reflections_shown() -> None:
    ctx = TurnContext(
        trader_name="T", model="m",
        recent_reflections=["AAPL: sold too early last time", "MSFT: overextended"],
    )
    text = build_first_look(ctx)
    assert "Recent reflections:" in text
    assert "sold too early" in text
    assert "overextended" in text


def test_build_first_look_empty_slots_not_shown() -> None:
    """Empty directed_notes and recent_reflections must not add blank lines."""
    ctx = TurnContext(trader_name="T", model="m")
    text = build_first_look(ctx)
    assert "Directed notes:" not in text
    assert "Recent reflections:" not in text


def test_build_first_look_slots_before_previous_attempt() -> None:
    """directed_notes and recent_reflections appear between cost and previous-attempt."""
    ctx = TurnContext(
        trader_name="T", model="m",
        directed_notes=["Check AAPL"],
        recent_reflections=["Lesson 1"],
        previous_attempt_tools=["list_tools"],
    )
    text = build_first_look(ctx)
    cost_pos = text.find("Cost this turn:")
    notes_pos = text.find("Directed notes:")
    refl_pos = text.find("Recent reflections:")
    prev_pos = text.find("Previous attempt:")
    assert cost_pos < notes_pos < refl_pos < prev_pos


def test_turn_context_no_forbidden_words_with_slots() -> None:
    ctx = TurnContext(
        trader_name="T",
        model="m",
        directed_notes=["Monitor NVDA"],
        recent_reflections=["Held too long"],
    )
    text = build_first_look(ctx)
    for word in ("paper", "sim", "demo", "fake", "test mode"):
        assert word not in text.lower(), f"forbidden word '{word}' in first-look with slots"


# ---------------------------------------------------------------------------
# Helper: uniform ToolResult shape assertion
# ---------------------------------------------------------------------------


def _assert_tool_result_shape(result: Any, *, expect_ok: bool | None = None) -> None:
    """Assert that result is a ToolResult with the canonical shape."""
    assert isinstance(result, ToolResult), f"expected ToolResult, got {type(result)}"
    assert isinstance(result.ok, bool)
    if expect_ok is True:
        assert result.ok is True, f"expected ok=True, got error={result.error}"
        assert result.data is not None
    elif expect_ok is False:
        assert result.ok is False
        assert isinstance(result.error, ToolError)
        assert result.error.kind in {
            "network", "rate_limit", "unavailable", "invalid_input",
            "disabled", "not_found", "internal",
        }
    d = result.to_dict()
    assert "ok" in d


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------


def test_list_tools_returns_tool_result() -> None:
    tool = ListToolsTool(trader_id="Alpha")
    result = tool()
    _assert_tool_result_shape(result, expect_ok=True)


def test_list_tools_contains_required_fields() -> None:
    result = ListToolsTool(trader_id="Alpha")()
    tools = result.data["tools"]
    assert isinstance(tools, list)
    assert len(tools) > 0
    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "latency" in t
        assert "cost_class" in t
        assert "enabled" in t
        assert "disabled_reason" in t


def test_list_tools_includes_disabled_with_reason() -> None:
    result = ListToolsTool(trader_id="Alpha")()
    tools = result.data["tools"]
    disabled = [t for t in tools if not t["enabled"]]
    assert len(disabled) >= 4  # world_events, prediction_market_odds, options_iv, forecast
    for t in disabled:
        assert t["disabled_reason"] is not None
        assert len(t["disabled_reason"]) > 0


def test_list_tools_includes_all_look_tools() -> None:
    result = ListToolsTool(trader_id="Alpha")()
    names = {t["name"] for t in result.data["tools"]}
    expected = {
        "list_tools", "memory_search", "hold", "pass",
        "recent_turns", "history", "news", "research_brief", "request_research",
        "situation", "world_events", "prediction_market_odds", "options_iv", "forecast",
        "watchlist", "account_state", "advisor_notes", "ask_manager",
    }
    assert expected.issubset(names), f"missing: {expected - names}"


# ---------------------------------------------------------------------------
# recent_turns
# ---------------------------------------------------------------------------


def test_recent_turns_no_store_returns_empty() -> None:
    tool = RecentTurnsTool(trader_id="Alpha")
    result = tool(n=5)
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["turns"] == []


def test_recent_turns_with_store() -> None:
    record = MagicMock()
    record.turn_id = "t1"
    record.started_at = None
    record.wake_reason = "scheduled"
    record.turn_type = "regular"
    record.final_action = "hold"
    record.total_cost_usd = 0.001
    record.tool_calls = []

    store = MagicMock()
    store.recent.return_value = [record]

    tool = RecentTurnsTool(trader_id="Alpha", turn_store=store)
    result = tool(n=1)
    _assert_tool_result_shape(result, expect_ok=True)
    assert len(result.data["turns"]) == 1
    assert result.data["turns"][0]["turn_id"] == "t1"


def test_recent_turns_store_error_returns_error() -> None:
    store = MagicMock()
    store.recent.side_effect = RuntimeError("db gone")
    tool = RecentTurnsTool(trader_id="Alpha", turn_store=store)
    result = tool()
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "internal"


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def test_history_no_service_returns_error() -> None:
    tool = HistoryTool(trader_id="Alpha")
    result = tool("AAPL")
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "unavailable"


def test_history_empty_symbol_returns_error() -> None:
    tool = HistoryTool(trader_id="Alpha")
    result = tool("")
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "invalid_input"


def test_history_with_service() -> None:
    bar = MagicMock()
    bar.timestamp = "2026-05-27T16:00:00"
    bar.open = 180.0
    bar.high = 185.0
    bar.low = 178.0
    bar.close = 183.0
    bar.volume = 1_000_000.0

    svc = MagicMock()
    svc.get_bars.return_value = [bar]

    tool = HistoryTool(trader_id="Alpha", history_service=svc)
    result = tool("AAPL", days=5)
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["symbol"] == "AAPL"
    assert len(result.data["bars"]) == 1
    assert result.data["stats"]["close_last"] == 183.0


# ---------------------------------------------------------------------------
# news
# ---------------------------------------------------------------------------


def test_news_no_db_returns_empty() -> None:
    tool = NewsTool(trader_id="Alpha")
    result = tool("AAPL")
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["items"] == []


def test_news_no_user_id_returns_empty() -> None:
    tool = NewsTool(trader_id="Alpha", db=MagicMock(), owner_user_id=None)
    result = tool()
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["items"] == []


# ---------------------------------------------------------------------------
# research_brief
# ---------------------------------------------------------------------------


def test_research_brief_no_store_returns_none_brief() -> None:
    tool = ResearchBriefTool(trader_id="Alpha")
    result = tool("AAPL")
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["brief"] is None


def test_research_brief_empty_symbol_returns_error() -> None:
    tool = ResearchBriefTool(trader_id="Alpha")
    result = tool("")
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "invalid_input"


def test_research_brief_with_store() -> None:
    brief = MagicMock()
    brief.id = "br1"
    brief.summary = "AAPL looks bullish."
    brief.sentiment = 0.6
    brief.catalysts = ["iPhone cycle"]
    brief.sources = ["https://example.com"]
    brief.ts = "2026-05-27"

    store = MagicMock()
    store.get.return_value = [brief]

    tool = ResearchBriefTool(
        trader_id="Alpha",
        owner_user_id="user1",
        research_store=store,
    )
    result = tool("AAPL")
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["brief"]["summary"] == "AAPL looks bullish."
    assert result.data["brief"]["sentiment"] == 0.6


# ---------------------------------------------------------------------------
# request_research
# ---------------------------------------------------------------------------


def test_request_research_no_fn_returns_unavailable() -> None:
    tool = RequestResearchTool(trader_id="Alpha")
    result = tool("AAPL", "What are the catalysts?")
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "unavailable"


def test_request_research_empty_symbol_returns_error() -> None:
    tool = RequestResearchTool(trader_id="Alpha", owner_user_id="u1", run_fn=lambda *a: None)
    result = tool("", "question")
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "invalid_input"


def test_request_research_queues_and_returns_request_id() -> None:
    called = []
    tool = RequestResearchTool(
        trader_id="Alpha",
        owner_user_id="u1",
        run_fn=lambda uid, tickers: called.append((uid, tickers)),
    )
    result = tool("NVDA", "Near-term earnings risk?")
    _assert_tool_result_shape(result, expect_ok=True)
    assert "request_id" in result.data
    assert result.data["status"] == "queued"
    assert called == [("u1", ["NVDA"])]


# ---------------------------------------------------------------------------
# situation
# ---------------------------------------------------------------------------


def test_situation_no_classifier_returns_none_regime() -> None:
    tool = SituationTool(trader_id="Alpha")
    result = tool()
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["regime"] is None


def test_situation_with_classifier() -> None:
    state = MagicMock()
    state.label = "calm"
    state.realized_vol_annual = 0.12
    state.event_count = 0
    state.note = "vol below calm threshold"

    clf = MagicMock()
    clf.classify.return_value = state

    tool = SituationTool(
        trader_id="Alpha",
        regime_classifier=clf,
        recent_closes=[100.0, 101.0, 100.5, 102.0],
        symbols=["AAPL"],
    )
    result = tool()
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["regime"]["label"] == "calm"


# ---------------------------------------------------------------------------
# Disabled tool stubs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ToolCls,args", [
    (WorldEventsTool, {}),
    (PredictionMarketOddsTool, {"category": "fed_rates"}),
    (OptionsIVTool, {"symbol": "AAPL"}),
    (ForecastTool, {"symbol": "AAPL", "horizon": 10}),
])
def test_disabled_tools_return_disabled_error(ToolCls: Any, args: dict[str, Any]) -> None:
    """All four tools return kind="disabled" when no settings/provider is supplied.

    WorldEventsTool, PredictionMarketOddsTool, OptionsIVTool are wired (Track A)
    but flag-gated — they return disabled when settings_store is None (default).
    ForecastTool is still a stub (Track C) and returns the stub disabled message.
    All four must return ok=False, error.kind="disabled".
    """
    tool = ToolCls(trader_id="Alpha")
    result = tool(**args)
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "disabled"
    # Message is non-empty; exact wording varies between stub and wired tools
    assert result.error.message


# ---------------------------------------------------------------------------
# watchlist
# ---------------------------------------------------------------------------


def test_watchlist_empty() -> None:
    tool = WatchlistTool(trader_id="Alpha")
    result = tool()
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["symbols"] == []


def test_watchlist_combined_deduped() -> None:
    tool = WatchlistTool(
        trader_id="Alpha",
        trader_symbols=["AAPL", "MSFT"],
        operator_symbols=["MSFT", "NVDA"],
    )
    result = tool()
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["symbols"] == ["AAPL", "MSFT", "NVDA"]
    assert result.data["source"] == "combined"


# ---------------------------------------------------------------------------
# account_state
# ---------------------------------------------------------------------------


def test_account_state_no_broker_returns_error() -> None:
    tool = AccountStateTool(trader_id="Alpha")
    result = tool()
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "unavailable"


def test_account_state_with_broker() -> None:
    broker = MagicMock()
    broker.get_balance.return_value = {"cash": 50_000.0}
    broker.get_positions.return_value = [
        {"symbol": "AAPL", "quantity": 10.0, "avg_price": 175.0}
    ]
    broker.market_prices = {"AAPL": 180.0}
    broker.get_realized_pnl.return_value = 250.0

    tool = AccountStateTool(trader_id="Alpha", broker=broker)
    result = tool()
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["cash"] == 50_000.0
    assert len(result.data["positions"]) == 1
    assert result.data["positions"][0]["unrealized_pnl"] == 50.0  # (180-175)*10
    assert result.data["realized_pnl"] == 250.0


def test_account_state_scrubs_forbidden_words() -> None:
    """MONEY IS REAL: account_state must not leak paper/sim/demo in position data."""
    broker = MagicMock()
    broker.get_balance.return_value = {"cash": 1_000.0}
    broker.get_positions.return_value = [
        {"symbol": "DEMO-ETF", "quantity": 1.0, "avg_price": 10.0}
    ]
    broker.market_prices = {"DEMO-ETF": 11.0}
    broker.get_realized_pnl.return_value = 0.0

    tool = AccountStateTool(trader_id="Alpha", broker=broker)
    result = tool()
    _assert_tool_result_shape(result, expect_ok=True)
    result_str = str(result.data).lower()
    # "demo" in the symbol name — after scrubbing it should be replaced
    assert "demo" not in result_str


def test_scrub_str_replaces_forbidden_words() -> None:
    assert "demo" not in _scrub_str("demo account").lower()
    assert "paper" not in _scrub_str("paper trading").lower()
    assert "sim" not in _scrub_str("sim mode").lower()


# ---------------------------------------------------------------------------
# advisor_notes
# ---------------------------------------------------------------------------


def test_advisor_notes_no_store_returns_empty() -> None:
    tool = AdvisorNotesTool(trader_id="Alpha")
    result = tool(scope="trader")
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["notes"] == []


def test_advisor_notes_invalid_scope_returns_error() -> None:
    tool = AdvisorNotesTool(trader_id="Alpha")
    result = tool(scope="everything")
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "invalid_input"


def test_advisor_notes_ticker_scope_without_symbol_returns_error() -> None:
    tool = AdvisorNotesTool(trader_id="Alpha")
    result = tool(scope="ticker")
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "invalid_input"


def test_advisor_notes_with_store() -> None:
    note = MagicMock()
    note.id = "n1"
    note.text = "Watch AAPL closely this week."
    note.updated_at = 1748300000.0

    store = MagicMock()
    store.get.return_value = note

    tool = AdvisorNotesTool(
        trader_id="Alpha",
        owner_user_id="user1",
        notes_store=store,
    )
    result = tool(scope="trader")
    _assert_tool_result_shape(result, expect_ok=True)
    assert len(result.data["notes"]) == 1
    assert result.data["notes"][0]["text"] == "Watch AAPL closely this week."


def test_advisor_notes_directed_notes_slot_marks_read() -> None:
    note = MagicMock()
    note.id = "n42"
    note.text = "Reduce risk today."
    note.updated_at = 1748300000.0

    store = MagicMock()
    store.get.return_value = note

    marked: list[str] = []
    tool = AdvisorNotesTool(
        trader_id="Alpha",
        owner_user_id="user1",
        notes_store=store,
        mark_read_fn=lambda nid: marked.append(nid),
    )

    # First call surfaces the note.
    texts = tool.directed_notes_for_slot()
    assert "Reduce risk today." in texts
    assert "n42" in marked

    # Second call (same session) should return empty — already read.
    texts2 = tool.directed_notes_for_slot()
    assert texts2 == []


def test_advisor_notes_isolation_uses_own_trader_id() -> None:
    """Ensure the store is called with the correct (user_id, scope, trader_id)."""
    store = MagicMock()
    store.get.return_value = None

    tool = AdvisorNotesTool(
        trader_id="TraderBeta",
        owner_user_id="user7",
        notes_store=store,
    )
    tool(scope="trader")
    store.get.assert_called_once_with("user7", "trader", "TraderBeta")


# ---------------------------------------------------------------------------
# memory_search
# ---------------------------------------------------------------------------


def test_memory_search_no_store_returns_empty() -> None:
    tool = MemorySearchTool(trader_id="Alpha")
    result = tool("AAPL momentum")
    _assert_tool_result_shape(result, expect_ok=True)
    assert result.data["memories"] == []


def test_memory_search_empty_query_returns_error() -> None:
    tool = MemorySearchTool(trader_id="Alpha")
    result = tool("")
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "invalid_input"


def test_memory_search_with_store() -> None:
    lesson = MagicMock()
    lesson.text = "AAPL: sold too early on volume spike."
    lesson.score = 0.92
    lesson.tags = ["AAPL", "momentum"]

    store = MagicMock()
    store.recall.return_value = [lesson]

    tool = MemorySearchTool(
        trader_id="Alpha",
        owner_user_id="u1",
        memory_store=store,
    )
    result = tool("AAPL momentum", k=3)
    _assert_tool_result_shape(result, expect_ok=True)
    assert len(result.data["memories"]) == 1
    assert result.data["memories"][0]["score"] == 0.92


def test_memory_search_reflections_for_slot_top3() -> None:
    lessons = []
    for i in range(5):
        lesson_mock = MagicMock()
        lesson_mock.text = f"Lesson {i}: observation about AAPL."
        lesson_mock.score = 0.9 - i * 0.05
        lesson_mock.tags = ["AAPL"] if i < 3 else ["OTHER"]
        lessons.append(lesson_mock)

    store = MagicMock()
    store.recall.return_value = lessons

    tool = MemorySearchTool(
        trader_id="Alpha",
        owner_user_id="u1",
        memory_store=store,
    )
    reflections = tool.reflections_for_slot("AAPL momentum", symbols=["AAPL"])
    assert len(reflections) <= 3
    # All returned should mention AAPL (tag-prioritized)
    for r in reflections:
        assert "Lesson" in r


def test_memory_search_reflections_no_store_returns_empty() -> None:
    tool = MemorySearchTool(trader_id="Alpha")
    assert tool.reflections_for_slot("anything") == []


# ---------------------------------------------------------------------------
# ask_manager
# ---------------------------------------------------------------------------


def test_ask_manager_no_manager_returns_unavailable() -> None:
    tool = AskManagerTool(trader_id="Alpha")
    result = tool("How are things?")
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "unavailable"


def test_ask_manager_empty_question_returns_error() -> None:
    tool = AskManagerTool(trader_id="Alpha")
    result = tool("")
    _assert_tool_result_shape(result, expect_ok=False)
    assert result.error.kind == "invalid_input"


def test_ask_manager_per_turn_gate() -> None:
    """At most one ask_manager per turn — second call returns rate_limit."""
    manager = MagicMock()
    manager.chat.return_value = "I think you should hold."

    tool = AskManagerTool(
        trader_id="Alpha",
        owner_user_id="u1",
        manager_agent=manager,
        model_ref=MagicMock(),
    )
    first = tool("First question?")
    _assert_tool_result_shape(first, expect_ok=True)

    second = tool("Second question?")
    _assert_tool_result_shape(second, expect_ok=False)
    assert second.error.kind == "rate_limit"

    # After reset, gate opens again.
    tool.reset_for_turn()
    third = tool("Third question after reset?")
    _assert_tool_result_shape(third, expect_ok=True)


def test_ask_manager_records_nested_cost() -> None:
    manager = MagicMock()
    manager.chat.return_value = "Hold."

    ct = CostTracker()
    tool = AskManagerTool(
        trader_id="Alpha",
        owner_user_id="u1",
        manager_agent=manager,
        model_ref=MagicMock(),
        cost_tracker=ct,
    )
    tool("What's your view?")
    rollup = ct.rollup()
    assert rollup["nested_llm_calls"] == 1


def test_ask_manager_injects_filter_prefix() -> None:
    """The filter prefix must be prepended to the question."""
    captured: list[str] = []

    def fake_chat(user_id: str, conv_id: str, msg: str, ref: Any) -> str:
        captured.append(msg)
        return "I can advise."

    manager = MagicMock()
    manager.chat.side_effect = fake_chat

    tool = AskManagerTool(
        trader_id="Alpha",
        owner_user_id="u1",
        manager_agent=manager,
        model_ref=MagicMock(),
    )
    tool("Should I buy AAPL?")
    assert len(captured) == 1
    assert "NEVER disclose" in captured[0]
    assert "Should I buy AAPL?" in captured[0]


def test_ask_manager_scrubs_forbidden_words_from_answer() -> None:
    """MONEY IS REAL: forbidden words in manager reply must be scrubbed."""
    for word in ("paper", "sim", "demo", "fake", "monopoly"):
        scrubbed = _scrub_answer(f"Your account is a {word} account.")
        assert word not in scrubbed.lower(), f"'{word}' leaked through scrub"


# ---------------------------------------------------------------------------
# Smoke: all enabled tools return uniform ToolResult shape
# ---------------------------------------------------------------------------


def test_all_enabled_tools_return_tool_result_shape() -> None:
    """Smoke: call every enabled LOOK tool once; assert uniform ToolResult shape."""
    tools_and_calls: list[tuple[str, Any]] = [
        ("list_tools", lambda: ListToolsTool(trader_id="Smoke")()),
        ("recent_turns", lambda: RecentTurnsTool(trader_id="Smoke")(n=3)),
        ("history", lambda: HistoryTool(trader_id="Smoke")("AAPL")),
        ("news", lambda: NewsTool(trader_id="Smoke")("AAPL")),
        ("research_brief", lambda: ResearchBriefTool(trader_id="Smoke")("AAPL")),
        ("request_research", lambda: RequestResearchTool(trader_id="Smoke")("AAPL", "Q?")),
        ("situation", lambda: SituationTool(trader_id="Smoke")()),
        ("world_events", lambda: WorldEventsTool(trader_id="Smoke")()),
        ("prediction_market_odds", lambda: PredictionMarketOddsTool(trader_id="Smoke")("fed")),
        ("options_iv", lambda: OptionsIVTool(trader_id="Smoke")("AAPL")),
        ("forecast", lambda: ForecastTool(trader_id="Smoke")("AAPL")),
        ("watchlist", lambda: WatchlistTool(trader_id="Smoke")()),
        ("account_state", lambda: AccountStateTool(trader_id="Smoke")()),
        ("advisor_notes", lambda: AdvisorNotesTool(trader_id="Smoke")()),
        ("memory_search", lambda: MemorySearchTool(trader_id="Smoke")("test query")),
        ("ask_manager", lambda: AskManagerTool(trader_id="Smoke")("test q")),
    ]

    for name, call in tools_and_calls:
        result = call()
        assert isinstance(result, ToolResult), f"{name}: not a ToolResult"
        assert isinstance(result.ok, bool), f"{name}: ok not bool"
        d = result.to_dict()
        assert "ok" in d, f"{name}: to_dict missing 'ok'"
