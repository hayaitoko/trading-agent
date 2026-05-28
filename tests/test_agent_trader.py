"""WS-Agent A0 — tests for tool envelope, turn context, cost tracker, and AgentTrader.

Live smoke (test_agent_trader_smoke_list_tools_memory_search_hold): a stub LLM
emits list_tools → memory_search (empty) → hold("nothing interesting"); asserts
that first-look renders correctly, ToolResult envelopes are uniform, cost rollup
is nonzero, and the loop terminates cleanly.

MONEY IS REAL: tests verify that neither the AgentTrader system prompt nor the
first-look output contains any of the forbidden disclosure words.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from trading_agent.intel.cost_tracker import CostTracker
from trading_agent.intel.tool_envelope import ToolError, ToolResult
from trading_agent.intel.turn_context import TurnContext, build_first_look
from trading_agent.llm.openrouter import ToolCall, ToolCallChatResult
from trading_agent.llm.trader import AgentTrader, Trader

# ---------------------------------------------------------------------------
# Tool envelope
# ---------------------------------------------------------------------------


def test_tool_result_ok_to_dict() -> None:
    r = ToolResult(ok=True, data={"value": 42})
    d = r.to_dict()
    assert d == {"ok": True, "data": {"value": 42}}


def test_tool_result_error_to_dict() -> None:
    r = ToolResult(ok=False, error=ToolError(kind="not_found", message="no such tool"))
    d = r.to_dict()
    assert d["ok"] is False
    assert d["error"]["kind"] == "not_found"
    assert "retry_after" not in d["error"]


def test_tool_error_retry_after_present() -> None:
    r = ToolResult(
        ok=False,
        error=ToolError(kind="rate_limit", message="slow down", retry_after=30),
    )
    d = r.to_dict()
    assert d["error"]["retry_after"] == 30


def test_tool_result_frozen() -> None:
    r = ToolResult(ok=True, data="x")
    with pytest.raises(Exception):
        r.ok = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Turn context
# ---------------------------------------------------------------------------


def _ctx(**kw: Any) -> TurnContext:
    return TurnContext(
        trader_name="Alpha",
        model="anthropic/claude-sonnet-4-6",
        utc_now=datetime(2026, 5, 28, 14, 30, 0, tzinfo=UTC),
        **kw,
    )


def test_build_first_look_has_required_fields() -> None:
    text = build_first_look(_ctx(cash=50_000.0, position_count=2))
    assert "Identity:" in text
    assert "Alpha" in text
    assert "Account:" in text
    assert "cash=$50,000.00" in text
    assert "positions=2" in text
    assert "Wake reason:" in text
    assert "Turn type:" in text
    assert "Time:" in text
    assert "Cadence:" in text
    assert "Attention:" in text
    assert "Cost this turn:" in text


def test_build_first_look_no_previous_attempt_line() -> None:
    text = build_first_look(_ctx())
    assert "Previous attempt:" not in text


def test_build_first_look_previous_attempt_present() -> None:
    text = build_first_look(_ctx(previous_attempt_tools=["list_tools", "memory_search"]))
    assert "Previous attempt:" in text
    assert "list_tools" in text
    assert "memory_search" in text


def test_build_first_look_et_time_present() -> None:
    text = build_first_look(_ctx())
    # US/Eastern is UTC-4 in May: 14:30 UTC → 10:30 ET
    assert "ET" in text


def test_build_first_look_mandate_shows() -> None:
    text = build_first_look(_ctx(mandate="aggressive momentum"))
    assert "aggressive momentum" in text


def test_build_first_look_no_forbidden_disclosure_words() -> None:
    """MONEY IS REAL: first-look must not disclose paper/sim/demo status."""
    text = build_first_look(
        _ctx(
            cash=100_000.0,
            position_count=0,
            mandate="value investing",
        )
    )
    forbidden = ("paper", "sim", "demo", "fake", "test mode")
    for word in forbidden:
        assert word not in text.lower(), f"forbidden word '{word}' found in first-look"


def test_build_first_look_extra_lines() -> None:
    text = build_first_look(_ctx(extra_lines=["SoD: markets open in 60min"]))
    assert "SoD: markets open in 60min" in text


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------


def test_cost_tracker_accumulates() -> None:
    t = CostTracker()
    t.add_model_call(cost_usd=0.002, input_tokens=500, output_tokens=100)
    t.add_model_call(cost_usd=0.003, input_tokens=300, output_tokens=80)
    assert abs(t.total_usd - 0.005) < 1e-9
    assert t.call_count == 2


def test_cost_tracker_nested_llm() -> None:
    t = CostTracker()
    t.add_model_call(cost_usd=0.001)
    t.add_nested_llm("ask_manager", cost_usd=0.010)
    assert abs(t.total_usd - 0.011) < 1e-9


def test_cost_tracker_warn_returns_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COST_WARN_PER_TURN", "0.001")
    t = CostTracker()
    t.add_model_call(cost_usd=0.002)
    warn = t.check_warn()
    assert warn is not None
    assert "Cost notice" in warn
    assert "$0.002" in warn or "0.002" in warn


def test_cost_tracker_warn_only_fires_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COST_WARN_PER_TURN", "0.001")
    t = CostTracker()
    t.add_model_call(cost_usd=0.002)
    assert t.check_warn() is not None
    assert t.check_warn() is None  # second check → None


def test_cost_tracker_no_warn_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COST_WARN_PER_TURN", "1.00")
    t = CostTracker()
    t.add_model_call(cost_usd=0.001)
    assert t.check_warn() is None


def test_cost_tracker_rollup() -> None:
    t = CostTracker()
    t.add_model_call(cost_usd=0.002)
    t.add_nested_llm("ask_manager", cost_usd=0.010)
    r = t.rollup()
    assert r["model_calls"] == 1
    assert r["nested_llm_calls"] == 1
    assert abs(float(str(r["total_usd"])) - 0.012) < 1e-6


# ---------------------------------------------------------------------------
# AgentTrader helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolResponse:
    """One scripted response for FakeToolClient."""
    tool_calls: list[ToolCall]
    content: str | None = None
    cost: float = 0.001


class FakeToolClient:
    """Stub chat_with_tools client that returns scripted ToolCallChatResult objects."""

    def __init__(self, responses: list[_FakeToolResponse]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tool_choice: str = "auto",
    ) -> ToolCallChatResult:
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        if self._idx >= len(self._responses):
            # Fallback: return hold to prevent infinite loops in bad tests.
            return ToolCallChatResult(
                content=None,
                tool_calls=[ToolCall(id="tc_fallback", name="hold", arguments={"reason": "stub exhausted"})],
                model=model,
                usage={"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001},
                cost=0.001,
                finish_reason="tool_calls",
            )
        resp = self._responses[self._idx]
        self._idx += 1
        return ToolCallChatResult(
            content=resp.content,
            tool_calls=resp.tool_calls,
            model=model,
            usage={"prompt_tokens": 100, "completion_tokens": 20, "cost": resp.cost},
            cost=resp.cost,
            finish_reason="tool_calls" if resp.tool_calls else "stop",
        )


def _make_trader(**kw: Any) -> AgentTrader:
    defaults: dict[str, Any] = {
        "symbols": ["AAPL", "MSFT"],
        "name": "TestTrader",
        "cadence_minutes": 30,
    }
    defaults.update(kw)
    model = defaults.pop("model", "anthropic/claude-sonnet-4-6")
    client = defaults.pop("client", FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="hold", arguments={"reason": "default"})]),
    ]))
    return AgentTrader(model, client, **defaults)


# ---------------------------------------------------------------------------
# AgentTrader — smoke test (A0 live smoke)
# ---------------------------------------------------------------------------


def test_agent_trader_smoke_list_tools_memory_search_hold() -> None:
    """Smoke: stub LLM emits list_tools → memory_search (empty) → hold.

    Verifies:
    - first-look renders (Identity / Account / Wake reason present)
    - ToolResult envelopes are uniform (all have ok + data/error)
    - cost rollup is nonzero (3 model calls × 0.001 = 0.003)
    - loop terminates cleanly with hold("nothing interesting")
    """
    client = FakeToolClient([
        _FakeToolResponse(
            tool_calls=[ToolCall(id="tc1", name="list_tools", arguments={})],
            cost=0.001,
        ),
        _FakeToolResponse(
            tool_calls=[
                ToolCall(id="tc2", name="memory_search", arguments={"query": "AAPL momentum", "k": 5})
            ],
            cost=0.001,
        ),
        _FakeToolResponse(
            tool_calls=[ToolCall(id="tc3", name="hold", arguments={"reason": "nothing interesting"})],
            cost=0.001,
        ),
    ])
    trader = AgentTrader(
        "anthropic/claude-sonnet-4-6",
        client,
        symbols=["AAPL", "MSFT"],
        name="SmokeTrader",
    )

    result = trader.decide({"cash": 100_000.0, "positions": []})

    # Loop terminated cleanly
    assert result.error is None
    assert result.comment == "nothing interesting"
    assert result.decisions == []

    # 3 model calls made
    assert len(client.calls) == 3

    # First-look in the first call's messages
    first_messages = client.calls[0]["messages"]
    # messages[0] is system (list form for cache_control); messages[1] is user (first-look)
    first_look = first_messages[1]["content"]
    assert "Identity:" in first_look
    assert "SmokeTrader" in first_look
    assert "Wake reason:" in first_look
    assert "Account:" in first_look

    # All tool results in the message history have uniform ToolResult shape
    tool_msgs = [m for m in client.calls[-1]["messages"] if m.get("role") == "tool"]
    for m in tool_msgs:
        payload = json.loads(m["content"])
        assert "ok" in payload

    # Cost rollup nonzero (3 calls × $0.001)
    rollup = result.usage
    assert isinstance(rollup, dict)
    assert float(str(rollup["total_usd"])) > 0.0


def test_agent_trader_pass_terminates_cleanly() -> None:
    client = FakeToolClient([
        _FakeToolResponse(
            tool_calls=[ToolCall(id="t1", name="pass", arguments={})],
            cost=0.001,
        ),
    ])
    trader = AgentTrader("m", client, symbols=["AAPL"], name="PassTrader")
    result = trader.decide({"cash": 10_000.0, "positions": []})
    assert result.error is None
    assert result.comment == "pass"
    assert result.decisions == []
    assert len(client.calls) == 1


def test_agent_trader_implicit_hold_on_text_response() -> None:
    """Model returns plain text (no tool calls) → implicit hold."""
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[], content="I'll hold for now.", cost=0.001),
    ])
    trader = AgentTrader("m", client, symbols=["AAPL"], name="TextTrader")
    result = trader.decide({"cash": 5_000.0, "positions": []})
    assert result.error is None
    assert "hold" in result.comment.lower() or "hold for now" in result.comment


def test_agent_trader_runaway_guard_triggers() -> None:
    """100 list_tools calls should trip the runaway guard → forced hold."""
    # Return list_tools forever (client falls back after scripted responses exhausted)
    responses = [
        _FakeToolResponse(tool_calls=[ToolCall(id=f"t{i}", name="list_tools", arguments={})], cost=0.0)
        for i in range(105)
    ]
    client = FakeToolClient(responses)
    trader = AgentTrader("m", client, symbols=["AAPL"], name="RunawayTrader")
    trader.RUNAWAY_LIMIT = 5  # override for test speed
    result = trader.decide({"cash": 1_000.0, "positions": []})
    assert result.error is None
    assert "runaway" in result.comment.lower()


def test_agent_trader_cost_warn_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    """When cost crosses threshold, a warning system message appears before next call."""
    monkeypatch.setenv("COST_WARN_PER_TURN", "0.0005")
    # list_tools costs $0.001 (> threshold) → warning injected before hold call
    client = FakeToolClient([
        _FakeToolResponse(
            tool_calls=[ToolCall(id="t1", name="list_tools", arguments={})],
            cost=0.001,
        ),
        _FakeToolResponse(
            tool_calls=[ToolCall(id="t2", name="hold", arguments={"reason": "done"})],
            cost=0.0,
        ),
    ])
    trader = AgentTrader("m", client, symbols=["AAPL"], name="CostTrader")
    trader.decide({"cash": 1_000.0, "positions": []})

    # Second model call should have a cost-warning system message injected
    second_messages = client.calls[1]["messages"]
    system_msgs = [m for m in second_messages if m.get("role") == "system"]
    assert any("Cost notice" in (m.get("content") or "") for m in system_msgs)


def test_agent_trader_satisfies_trader_protocol() -> None:
    trader = _make_trader()
    assert isinstance(trader, Trader)


def test_agent_trader_no_forbidden_words_in_system_prompt() -> None:
    """MONEY IS REAL: system prompt must not disclose paper/sim/demo status."""
    trader = _make_trader(style="momentum")
    prompt = trader._stable_system_content.lower()
    for word in ("paper", "sim", "demo", "fake", "test mode"):
        assert word not in prompt, f"forbidden word '{word}' in AgentTrader system prompt"


def test_agent_trader_observe_accumulates_bars() -> None:
    trader = _make_trader()
    for px in (100, 101, 102):
        trader.observe({"symbol": "AAPL", "close": px})
    assert len(trader._bars["AAPL"]) == 3


def test_agent_trader_tool_definitions_include_terminals() -> None:
    trader = _make_trader()
    defs = trader._tool_definitions()
    names = {d["function"]["name"] for d in defs}
    assert "hold" in names
    assert "pass" in names
    assert "list_tools" in names
    assert "memory_search" in names


def test_agent_trader_unknown_tool_returns_not_found_error() -> None:
    trader = _make_trader()
    tc = ToolCall(id="x", name="nonexistent_tool", arguments={})
    result = trader._execute_tool(tc, CostTracker())
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind == "not_found"


def test_agent_trader_memory_search_no_store_returns_empty() -> None:
    trader = _make_trader(memory=None, owner_user_id=None)
    tc = ToolCall(id="x", name="memory_search", arguments={"query": "AAPL"})
    result = trader._execute_tool(tc, CostTracker())
    assert result.ok is True
    assert result.data["memories"] == []


def test_agent_trader_memory_search_empty_query_returns_error() -> None:
    trader = _make_trader()
    tc = ToolCall(id="x", name="memory_search", arguments={"query": ""})
    result = trader._execute_tool(tc, CostTracker())
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind == "invalid_input"


def test_agent_trader_list_tools_result_shape() -> None:
    trader = _make_trader()
    tc = ToolCall(id="x", name="list_tools", arguments={})
    result = trader._execute_tool(tc, CostTracker())
    assert result.ok is True
    tools = result.data["tools"]
    assert isinstance(tools, list)
    for tool in tools:
        assert "name" in tool
        assert "cost_class" in tool
        assert "enabled" in tool


def test_agent_trader_stable_system_message_has_cache_control() -> None:
    """System message must use list-of-blocks format with cache_control for caching."""
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
    ])
    trader = AgentTrader("m", client, symbols=["AAPL"], name="CacheTrader")
    trader.decide({"cash": 1_000.0, "positions": []})
    sys_msg = client.calls[0]["messages"][0]
    assert sys_msg["role"] == "system"
    # Content is a list (block format) for cache_control support
    assert isinstance(sys_msg["content"], list)
    block = sys_msg["content"][0]
    assert block["type"] == "text"
    assert "cache_control" in block


# ---------------------------------------------------------------------------
# A4 fixer — SoD/EoD turn-type special-prompt guidance (finding #1)
# ---------------------------------------------------------------------------


def _first_look_and_system(client: FakeToolClient) -> tuple[str, str]:
    """Return (first_look_user_message, stable_system_text) from the first call."""
    msgs = client.calls[0]["messages"]
    system_text = msgs[0]["content"][0]["text"]
    first_look = msgs[1]["content"]
    return first_look, system_text


def test_sod_turn_injects_start_of_day_guidance() -> None:
    """A SoD turn appends start-of-day guidance to the per-turn first-look."""
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
    ])
    trader = AgentTrader("m", client, symbols=["AAPL"], name="SoDTrader", tutorial_remaining=0)
    trader._current_turn_type = "SoD"
    trader.decide({"cash": 1_000.0, "positions": []})

    first_look, system_text = _first_look_and_system(client)
    assert "Start-of-day guidance:" in first_look
    assert "seed watchpoints" in first_look
    # Discipline #6: guidance must NOT leak into the cached system prefix.
    assert "Start-of-day guidance:" not in system_text


def test_eod_turn_injects_end_of_day_guidance_no_new_positions() -> None:
    """An EoD turn (default-strict flag set, as the scheduler does) appends
    end-of-day guidance including the no-new-positions directive."""
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="hold", arguments={"reason": "eod"})]),
    ])
    trader = AgentTrader("m", client, symbols=["AAPL"], name="EoDTrader", tutorial_remaining=0)
    trader._current_turn_type = "EoD"
    # Mirror MarketScheduler._fire_one: it sets this True before an EoD decide().
    trader._eod_no_new_positions = True
    trader.decide({"cash": 1_000.0, "positions": []})

    first_look, system_text = _first_look_and_system(client)
    assert "End-of-day guidance:" in first_look
    assert "Do not open new positions" in first_look
    # Discipline #6: guidance must NOT leak into the cached system prefix.
    assert "End-of-day guidance:" not in system_text


def test_eod_turn_without_strict_flag_omits_no_new_positions() -> None:
    """When EoD is configured non-strict (_eod_no_new_positions left False), the
    EoD guidance still appears but omits the no-new-positions directive — proving
    the flag is the live control point, not dead code."""
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="hold", arguments={"reason": "eod"})]),
    ])
    trader = AgentTrader("m", client, symbols=["AAPL"], name="EoDLooseTrader", tutorial_remaining=0)
    trader._current_turn_type = "EoD"
    # Flag deliberately left False (non-strict EoD).
    trader.decide({"cash": 1_000.0, "positions": []})

    first_look, _ = _first_look_and_system(client)
    assert "End-of-day guidance:" in first_look
    assert "Do not open new positions" not in first_look


def test_regular_turn_has_no_turn_type_guidance() -> None:
    """A regular turn carries no SoD/EoD special-prompt guidance."""
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
    ])
    trader = AgentTrader("m", client, symbols=["AAPL"], name="RegularTrader")
    # turn_type defaults to "regular"
    trader.decide({"cash": 1_000.0, "positions": []})

    first_look, _ = _first_look_and_system(client)
    assert "Start-of-day guidance:" not in first_look
    assert "End-of-day guidance:" not in first_look
