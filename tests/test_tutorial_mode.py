"""WS-Agent A6 — Tutorial mode tests.

Covers:
- tutorial_remaining field defaults (3), zero, custom
- turn_type override to "tutorial" when tutorial_remaining > 0
- first-look renders "Tutorial — turn N of M" header in extra_lines
- tutorial_remaining decrements correctly across turns
- auto-exit on first trade* terminal (tutorial_remaining → 0)
- normal turn type after tutorial exhausted
- no_prior_context_hint renders context-hint line when reflections empty
- context hint suppressed when recent_reflections populated
- MONEY IS REAL: no forbidden strings in tutorial templates
- prompts/tutorial.py: all three focus turns + generic fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from trading_agent.intel.turn_context import TurnContext, build_first_look
from trading_agent.llm.openrouter import ToolCall, ToolCallChatResult
from trading_agent.llm.trader import AgentTrader
from trading_agent.prompts.tutorial import tutorial_extra_lines

# ---------------------------------------------------------------------------
# Helpers (shared with test_agent_trader.py style)
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolResponse:
    tool_calls: list[ToolCall]
    content: str | None = None
    cost: float = 0.001


class FakeToolClient:
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
            return ToolCallChatResult(
                content=None,
                tool_calls=[ToolCall(id="tc_fb", name="hold", arguments={"reason": "stub exhausted"})],
                model=model,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
                cost=0.0,
                finish_reason="tool_calls",
            )
        resp = self._responses[self._idx]
        self._idx += 1
        return ToolCallChatResult(
            content=resp.content,
            tool_calls=resp.tool_calls,
            model=model,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": resp.cost},
            cost=resp.cost,
            finish_reason="tool_calls" if resp.tool_calls else "stop",
        )


def _hold_client() -> FakeToolClient:
    return FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="hold", arguments={"reason": "ok"})]),
    ])


def _make_trader(tutorial_remaining: int = 3, **kw: Any) -> AgentTrader:
    defaults: dict[str, Any] = {
        "symbols": ["AAPL", "MSFT"],
        "name": "TutTrader",
        "tutorial_remaining": tutorial_remaining,
    }
    defaults.update(kw)
    model = defaults.pop("model", "test/model")
    client = defaults.pop("client", _hold_client())
    return AgentTrader(model, client, **defaults)


# ---------------------------------------------------------------------------
# prompts/tutorial.py unit tests
# ---------------------------------------------------------------------------


def test_tutorial_extra_lines_turn1_contains_step_1() -> None:
    lines = tutorial_extra_lines(1, 3)
    combined = " ".join(lines)
    assert "STEP 1" in combined
    assert "list_tools" in combined


def test_tutorial_extra_lines_turn2_contains_step_2() -> None:
    lines = tutorial_extra_lines(2, 3)
    combined = " ".join(lines)
    assert "STEP 2" in combined
    assert "memory_search" in combined
    assert "reflect" in combined


def test_tutorial_extra_lines_turn3_contains_step_3() -> None:
    lines = tutorial_extra_lines(3, 3)
    combined = " ".join(lines)
    assert "STEP 3" in combined
    assert "watchpoint" in combined


def test_tutorial_extra_lines_turn4_generic() -> None:
    lines = tutorial_extra_lines(4, 5)
    combined = " ".join(lines)
    assert "explore" in combined.lower() or "try" in combined.lower()


def test_tutorial_extra_lines_last_turn_says_freely() -> None:
    lines = tutorial_extra_lines(3, 3)
    combined = " ".join(lines)
    assert "freely" in combined


def test_tutorial_extra_lines_non_last_shows_remaining() -> None:
    lines = tutorial_extra_lines(1, 3)
    combined = " ".join(lines)
    # "2 guided turns remain after this one." or similar
    assert "2" in combined


def test_tutorial_extra_lines_starts_with_newline() -> None:
    lines = tutorial_extra_lines(1, 3)
    # First element is blank line separator "\n..." — starts with newline
    assert lines[0].startswith("\n")


def test_tutorial_header_format() -> None:
    lines = tutorial_extra_lines(2, 4)
    header = lines[0]
    assert "Tutorial" in header
    assert "turn 2 of 4" in header


# ---------------------------------------------------------------------------
# MONEY IS REAL: scan tutorial templates for forbidden strings
# ---------------------------------------------------------------------------

_FORBIDDEN = {"paper", "sim", "demo", "fake", "test mode"}


def test_tutorial_templates_money_is_real() -> None:
    all_text = ""
    for turn in range(1, 6):
        for total in range(turn, turn + 3):
            all_text += " ".join(tutorial_extra_lines(turn, total))
    lower = all_text.lower()
    for word in _FORBIDDEN:
        assert word not in lower, (
            f"Tutorial template contains forbidden disclosure word '{word}'. "
            "MONEY IS REAL invariant violated."
        )


# ---------------------------------------------------------------------------
# no_prior_context_hint in TurnContext
# ---------------------------------------------------------------------------


def _base_ctx(**kw: Any) -> TurnContext:
    return TurnContext(
        trader_name="New",
        model="test/model",
        utc_now=datetime(2026, 5, 28, 14, 0, 0, tzinfo=UTC),
        **kw,
    )


def test_no_prior_context_hint_renders_when_set_and_reflections_empty() -> None:
    ctx = _base_ctx(no_prior_context_hint=True, recent_reflections=[])
    text = build_first_look(ctx)
    assert "Context hint:" in text
    assert "new here" in text
    assert "list_tools" in text


def test_no_prior_context_hint_suppressed_when_reflections_present() -> None:
    ctx = _base_ctx(no_prior_context_hint=True, recent_reflections=["lesson 1"])
    text = build_first_look(ctx)
    assert "Context hint:" not in text
    assert "Recent reflections:" in text
    assert "lesson 1" in text


def test_no_prior_context_hint_off_by_default_no_hint_rendered() -> None:
    ctx = _base_ctx(no_prior_context_hint=False, recent_reflections=[])
    text = build_first_look(ctx)
    assert "Context hint:" not in text


def test_no_prior_context_hint_no_forbidden_strings() -> None:
    ctx = _base_ctx(no_prior_context_hint=True, recent_reflections=[])
    text = build_first_look(ctx).lower()
    for word in _FORBIDDEN:
        assert word not in text, f"Context hint contains forbidden word '{word}'"


# ---------------------------------------------------------------------------
# AgentTrader: tutorial_remaining field defaults
# ---------------------------------------------------------------------------


def test_agent_trader_tutorial_remaining_default_3() -> None:
    trader = _make_trader()
    assert trader.tutorial_remaining == 3
    assert trader._tutorial_total == 3


def test_agent_trader_tutorial_remaining_zero_disabled() -> None:
    trader = _make_trader(tutorial_remaining=0)
    assert trader.tutorial_remaining == 0
    assert trader._tutorial_total == 0


def test_agent_trader_tutorial_remaining_negative_clamped_to_zero() -> None:
    trader = _make_trader(tutorial_remaining=-5)
    assert trader.tutorial_remaining == 0


def test_agent_trader_tutorial_remaining_custom() -> None:
    trader = _make_trader(tutorial_remaining=5)
    assert trader.tutorial_remaining == 5
    assert trader._tutorial_total == 5


# ---------------------------------------------------------------------------
# AgentTrader: turn_type override to "tutorial"
# ---------------------------------------------------------------------------


def test_tutorial_turn_type_in_first_look_when_remaining() -> None:
    """First-look block shows turn_type=tutorial when tutorial_remaining > 0."""
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
    ])
    trader = _make_trader(tutorial_remaining=2, client=client)
    trader.decide({"cash": 100_000.0, "positions": []})

    first_look = client.calls[0]["messages"][1]["content"]
    assert "Turn type:        tutorial" in first_look


def test_normal_turn_type_when_tutorial_remaining_zero() -> None:
    """turn_type=regular when tutorial_remaining=0."""
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
    ])
    trader = _make_trader(tutorial_remaining=0, client=client)
    trader.decide({"cash": 100_000.0, "positions": []})

    first_look = client.calls[0]["messages"][1]["content"]
    assert "Turn type:        regular" in first_look


# ---------------------------------------------------------------------------
# AgentTrader: tutorial guidance in extra_lines / first-look
# ---------------------------------------------------------------------------


def test_tutorial_step1_guidance_in_first_look() -> None:
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
    ])
    trader = _make_trader(tutorial_remaining=3, client=client)
    trader.decide({"cash": 100_000.0, "positions": []})

    first_look = client.calls[0]["messages"][1]["content"]
    assert "Tutorial" in first_look
    assert "turn 1 of 3" in first_look
    assert "STEP 1" in first_look


def test_tutorial_step2_guidance_on_second_turn() -> None:
    def _pass_client() -> FakeToolClient:
        return FakeToolClient([
            _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
        ])

    trader = _make_trader(tutorial_remaining=3)
    trader.client = _pass_client()
    trader.decide({"cash": 100_000.0, "positions": []})
    assert trader.tutorial_remaining == 2

    # Second turn
    trader.client = _pass_client()
    trader.decide({"cash": 100_000.0, "positions": []})
    first_look = trader.client.calls[0]["messages"][1]["content"]
    assert "turn 2 of 3" in first_look
    assert "STEP 2" in first_look


def test_tutorial_context_hint_in_first_look_during_tutorial() -> None:
    """Context hint renders in first-look when no reflections and tutorial active."""
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
    ])
    trader = _make_trader(tutorial_remaining=1, client=client)
    trader.decide({"cash": 100_000.0, "positions": []})

    first_look = client.calls[0]["messages"][1]["content"]
    assert "Context hint:" in first_look


# ---------------------------------------------------------------------------
# AgentTrader: tutorial_remaining decrements
# ---------------------------------------------------------------------------


def test_tutorial_remaining_decrements_on_pass() -> None:
    def _pass_client() -> FakeToolClient:
        return FakeToolClient([
            _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
        ])

    trader = _make_trader(tutorial_remaining=3)
    for expected_remaining in [2, 1, 0]:
        trader.client = _pass_client()
        trader.decide({"cash": 100_000.0, "positions": []})
        assert trader.tutorial_remaining == expected_remaining


def test_tutorial_remaining_decrements_on_hold() -> None:
    def _hold_c() -> FakeToolClient:
        return FakeToolClient([
            _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="hold", arguments={"reason": "x"})]),
        ])

    trader = _make_trader(tutorial_remaining=2)
    trader.client = _hold_c()
    trader.decide({"cash": 100_000.0, "positions": []})
    assert trader.tutorial_remaining == 1

    trader.client = _hold_c()
    trader.decide({"cash": 100_000.0, "positions": []})
    assert trader.tutorial_remaining == 0


def test_tutorial_remaining_does_not_go_below_zero() -> None:
    def _pass_c() -> FakeToolClient:
        return FakeToolClient([
            _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
        ])

    trader = _make_trader(tutorial_remaining=1)
    trader.client = _pass_c()
    trader.decide({"cash": 100_000.0, "positions": []})
    assert trader.tutorial_remaining == 0

    # Extra turn beyond tutorial — should stay at 0, not go negative.
    trader.client = _pass_c()
    trader.decide({"cash": 100_000.0, "positions": []})
    assert trader.tutorial_remaining == 0


# ---------------------------------------------------------------------------
# AgentTrader: auto-exit on first trade* terminal
# ---------------------------------------------------------------------------


class _FakeBroker:
    """Minimal broker stub that returns a successful fill."""

    def execute(self, intent: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "filled",
            "symbol": intent.get("symbol", "AAPL"),
            "side": intent.get("side", "BUY"),
            "qty": intent.get("qty", 1),
            "fill_price": 150.0,
        }

    def snapshot(self) -> dict[str, Any]:
        return {"cash": 100_000.0, "positions": [], "equity": 100_000.0}


def test_tutorial_auto_exits_on_trade_terminal() -> None:
    """tutorial_remaining is zeroed immediately when a trade* terminal fires."""
    class _AlwaysFillBroker:
        def execute(self, intent: dict[str, Any]) -> dict[str, Any]:
            return {"status": "filled", "symbol": "AAPL", "side": "BUY", "qty": 1, "fill_price": 150.0}

        def snapshot(self) -> dict[str, Any]:
            return {"cash": 98_500.0, "positions": [{"symbol": "AAPL", "qty": 1}], "equity": 98_650.0}

    # Client that emits a trade terminal directly.
    client = FakeToolClient([
        _FakeToolResponse(
            tool_calls=[ToolCall(id="t1", name="trade", arguments={"symbol": "AAPL", "side": "BUY", "qty": 1})],
        ),
    ])
    from trading_agent.risk_manager import RiskManager
    rm = RiskManager()
    trader = _make_trader(
        tutorial_remaining=3,
        client=client,
        broker=_AlwaysFillBroker(),
        risk_manager=rm,
    )
    trader.decide({"cash": 100_000.0, "positions": []})

    # tutorial_remaining must be 0 regardless of how many guided turns were left
    assert trader.tutorial_remaining == 0


def test_tutorial_no_auto_exit_on_hold() -> None:
    """hold terminal does NOT zero tutorial_remaining — only trade* does."""
    client = FakeToolClient([
        _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="hold", arguments={"reason": "waiting"})]),
    ])
    trader = _make_trader(tutorial_remaining=3, client=client)
    trader.decide({"cash": 100_000.0, "positions": []})
    assert trader.tutorial_remaining == 2  # decremented by 1, not zeroed


# ---------------------------------------------------------------------------
# AgentTrader: normal turn after tutorial exhausted
# ---------------------------------------------------------------------------


def test_normal_first_look_after_tutorial_exhausted() -> None:
    """After tutorial_remaining reaches 0, turn_type reverts to regular."""
    def _pass_c() -> FakeToolClient:
        return FakeToolClient([
            _FakeToolResponse(tool_calls=[ToolCall(id="t1", name="pass", arguments={})]),
        ])

    trader = _make_trader(tutorial_remaining=1)

    # Exhaust tutorial
    trader.client = _pass_c()
    trader.decide({"cash": 100_000.0, "positions": []})
    assert trader.tutorial_remaining == 0

    # Next turn — normal
    trader.client = _pass_c()
    trader.decide({"cash": 100_000.0, "positions": []})
    first_look = trader.client.calls[0]["messages"][1]["content"]
    assert "Turn type:        regular" in first_look
    assert "Tutorial" not in first_look
    assert "Context hint:" not in first_look


# ---------------------------------------------------------------------------
# Carry-over: LLMTrader._SYSTEM_PROMPT no longer contains "paper account"
# ---------------------------------------------------------------------------


def test_llm_trader_system_prompt_no_paper_string() -> None:
    """Verify the carry-over scrub: 'paper account' removed from LLMTrader._SYSTEM_PROMPT."""
    from trading_agent.llm.trader import _SYSTEM_PROMPT
    assert "paper account" not in _SYSTEM_PROMPT.lower()
    assert "financial account" in _SYSTEM_PROMPT.lower()
