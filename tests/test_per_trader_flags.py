"""Tests for per-trader requires_approval flag overrides and per-tool cost attribution.

Covers:
  - Resolution order: per-trader param → per-user setting → default (False)
  - A trader created with requires_approval=True routes to the approval queue
  - A trader with requires_approval=False executes directly even when user default is True
  - Per-tool cost_usd in ToolCallRecord is non-zero for tools that incur LLM cost
  - Pure-read LOOK tools have cost_usd == 0.0 in their ToolCallRecord
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_agent.approval_queue import PendingTradeQueue
from trading_agent.bench.bench import Bench
from trading_agent.bench.controller import BenchController
from trading_agent.intel.cost_tracker import CostTracker
from trading_agent.intel.turn_store import ToolCallRecord, TurnStore
from trading_agent.llm.openrouter import ToolCall, ToolCallChatResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedClient:
    """Scripted LLM client: each call pops the next (tool_name, args) or None → text."""

    def __init__(self, scripts: list[Any]) -> None:
        self._scripts = list(scripts)
        self._i = 0

    def chat_with_tools(self, model: str, messages: Any, tools: Any, **kw: Any) -> ToolCallChatResult:
        spec = self._scripts[min(self._i, len(self._scripts) - 1)]
        self._i += 1
        usage = {"prompt_tokens": 80, "completion_tokens": 20, "cached_tokens": 0}
        if spec is None:
            return ToolCallChatResult(
                content="holding", tool_calls=[], model=model,
                usage=usage, cost=0.001, finish_reason="stop",
            )
        name, args = spec
        return ToolCallChatResult(
            content=None,
            tool_calls=[ToolCall(id=f"c{self._i}", name=name, arguments=args)],
            model=model, usage=usage, cost=0.001, finish_reason="tool_calls",
        )


def _make_controller(
    bench: Bench | None = None,
    client: Any = None,
    ptq: PendingTradeQueue | None = None,
    **kw: Any,
) -> BenchController:
    if bench is None:
        bench = Bench(["AAPL"], initial_balance=10_000.0)
    if client is None:
        client = _ScriptedClient([("hold", {"reason": "default"})])
    return BenchController(
        bench, client, symbols=["AAPL"],
        pending_trade_queue=ptq,
        **kw,
    )


# ---------------------------------------------------------------------------
# Resolution order: requires_approval
# ---------------------------------------------------------------------------


def test_default_requires_approval_is_false() -> None:
    """No settings, no override → autonomous (requires_approval=False)."""
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    ctl = _make_controller(bench)
    name = ctl.add_model("m", "Alpha")
    trader = bench._competitors[name].trader
    assert trader.requires_approval is False


def test_per_trader_override_true() -> None:
    """Explicit requires_approval=True on add_model overrides the default."""
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    ctl = _make_controller(bench)
    name = ctl.add_model("m", "Alpha", requires_approval=True)
    trader = bench._competitors[name].trader
    assert trader.requires_approval is True


def test_per_trader_override_false_beats_user_setting() -> None:
    """Per-trader False beats a per-user True setting."""
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    settings = MagicMock()
    settings.get = lambda uid, key, default=None: (True if key == "requires_approval" else default)
    ctl = _make_controller(bench)
    ctl._settings_store = settings
    ctl._cached_owner_id = "user1"
    name = ctl.add_model("m", "Alpha", requires_approval=False)
    trader = bench._competitors[name].trader
    assert trader.requires_approval is False


def test_per_user_setting_propagates_when_no_override() -> None:
    """Per-user setting True propagates to trader when no per-trader override is given."""
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    settings = MagicMock()
    settings.get = lambda uid, key, default=None: (True if key == "requires_approval" else default)
    ctl = _make_controller(bench)
    ctl._settings_store = settings
    ctl._cached_owner_id = "user1"
    # No requires_approval kwarg → falls through to user setting
    name = ctl.add_model("m", "Alpha")
    trader = bench._competitors[name].trader
    assert trader.requires_approval is True


def test_two_traders_different_approval_modes() -> None:
    """Two traders on the same controller can have different requires_approval values."""
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    ctl = _make_controller(bench)
    name_auto = ctl.add_model("m", "Auto", requires_approval=False)
    name_gated = ctl.add_model("m", "Gated", requires_approval=True)
    assert bench._competitors[name_auto].trader.requires_approval is False
    assert bench._competitors[name_gated].trader.requires_approval is True


# ---------------------------------------------------------------------------
# requires_approval=True → trade routes to approval queue
# ---------------------------------------------------------------------------


def test_approval_true_routes_to_pending_queue(tmp_path: Any) -> None:
    """A trader with requires_approval=True puts trades in the PendingTradeQueue
    instead of executing directly on the broker.
    """
    ptq = PendingTradeQueue(db_path=str(tmp_path / "approvals.db"))
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    # Script: trade BUY 1 AAPL → terminal
    client = _ScriptedClient([("trade", {"symbol": "AAPL", "side": "BUY", "qty": 1})])
    ctl = _make_controller(bench, client=client, ptq=ptq)
    name = ctl.add_model("m", "Gated", requires_approval=True)
    comp = bench._competitors[name]
    comp.trader.tutorial_remaining = 0  # skip tutorial mode

    # Price needed so trade() does not reject on missing market data.
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})

    # Run one turn — the trade should land in the pending queue, not the broker.
    result = comp.trader.decide({"cash": 10_000.0, "positions": []})
    assert result.error is None

    pending = ptq.pending_for_trader(name)
    # At least one pending record for this trader
    assert len(pending) > 0, "Expected trade in approval queue"

    # Broker should NOT have executed the trade directly (no filled history yet).
    history = comp.broker.get_trade_history()
    assert len(history) == 0


def test_approval_false_executes_directly(tmp_path: Any) -> None:
    """A trader with requires_approval=False executes trades directly on the broker."""
    ptq = PendingTradeQueue(db_path=str(tmp_path / "approvals.db"))
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    client = _ScriptedClient([("trade", {"symbol": "AAPL", "side": "BUY", "qty": 1})])
    ctl = _make_controller(bench, client=client, ptq=ptq)
    name = ctl.add_model("m", "Auto", requires_approval=False)
    comp = bench._competitors[name]
    comp.trader.tutorial_remaining = 0

    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    comp.trader.decide({"cash": 10_000.0, "positions": []})

    # Trade should have settled on the broker (no pending approval).
    history = comp.broker.get_trade_history()
    assert len(history) == 1
    # Trade history is tuple (symbol, fill_price, quantity, side)
    assert history[0][3].upper() == "BUY"
    pending = ptq.pending_for_trader(name)
    assert len(pending) == 0


# ---------------------------------------------------------------------------
# Per-tool cost attribution (Problem B)
# ---------------------------------------------------------------------------


def test_tool_call_record_has_zero_cost_for_pure_read_tool() -> None:
    """Pure-read LOOK tools (e.g. memory_search when memory is absent) → cost_usd=0.0."""
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    ts = TurnStore(db_path=":memory:")
    client = _ScriptedClient([
        ("memory_search", {"query": "AAPL momentum"}),
        ("hold", {"reason": "nothing to do"}),
    ])
    ctl = _make_controller(bench, client=client)
    ctl._turn_store = ts
    name = ctl.add_model("m", "Alpha")
    comp = bench._competitors[name]
    comp.trader.tutorial_remaining = 0
    comp.trader._turn_store = ts

    comp.trader.decide({"cash": 10_000.0, "positions": []})

    traces = ts.recent(name, 5)
    assert len(traces) == 1
    tr = traces[0]
    # memory_search record (first tool call) should have zero cost
    memory_calls = [tc for tc in tr.tool_calls if tc.tool_name == "memory_search"]
    assert memory_calls, "Expected a memory_search tool call in the trace"
    assert memory_calls[0].cost_usd == 0.0


def test_tool_call_record_nonzero_cost_for_nested_llm_tool() -> None:
    """A tool that calls add_nested_llm() inside _execute_tool should carry non-zero cost_usd."""
    # We patch ask_manager to call cost_tracker.add_nested_llm, simulating a real cost.
    from trading_agent.intel.tool_envelope import ToolResult

    bench = Bench(["AAPL"], initial_balance=10_000.0)
    ts = TurnStore(db_path=":memory:")
    client = _ScriptedClient([
        ("ask_manager", {"question": "What do you think about AAPL?"}),
        ("hold", {"reason": "done"}),
    ])
    ctl = _make_controller(bench, client=client)
    ctl._turn_store = ts
    name = ctl.add_model("m", "Alpha")
    comp = bench._competitors[name]
    comp.trader.tutorial_remaining = 0
    comp.trader._turn_store = ts

    # Inject a fake manager that records a cost.
    fake_manager = MagicMock()
    fake_manager.chat.return_value = "Hold for now."
    comp.trader._manager_agent = fake_manager
    comp.trader.owner_user_id = "test_user"
    comp.trader._manager_ref_fn = lambda: None

    # Patch AskManagerTool so it records a nested LLM cost.
    def _patched_ask_manager(question: str, cost_tracker: CostTracker) -> ToolResult:
        cost_tracker.add_nested_llm("ask_manager", cost_usd=0.015)
        return ToolResult(ok=True, data={"answer": "Hold AAPL."})

    comp.trader._tool_ask_manager = _patched_ask_manager  # type: ignore[method-assign]

    comp.trader.decide({"cash": 10_000.0, "positions": []})

    traces = ts.recent(name, 5)
    assert len(traces) == 1
    ask_calls = [tc for tc in traces[0].tool_calls if tc.tool_name == "ask_manager"]
    assert ask_calls, "Expected an ask_manager tool call in the trace"
    assert ask_calls[0].cost_usd > 0.0, f"Expected non-zero cost_usd, got {ask_calls[0].cost_usd}"


def test_cost_tracker_snapshot_total() -> None:
    """CostTracker.snapshot_total() returns the running total at that moment."""
    ct = CostTracker()
    assert ct.snapshot_total() == 0.0
    ct.add_model_call(cost_usd=0.010, input_tokens=100, output_tokens=50)
    snap = ct.snapshot_total()
    assert snap == pytest.approx(0.010)
    ct.add_nested_llm("ask_manager", cost_usd=0.005)
    # Delta between current total and snapshot = nested call cost
    assert ct.total_usd - snap == pytest.approx(0.005)


def test_tool_call_record_cost_usd_default_zero() -> None:
    """ToolCallRecord.cost_usd defaults to 0.0 (backward-compat)."""
    rec = ToolCallRecord(
        tool_name="hold",
        args={},
        result={"ok": True},
        latency_ms=5,
    )
    assert rec.cost_usd == 0.0


def test_tool_call_record_cost_usd_roundtrip() -> None:
    """cost_usd survives to_dict / from_dict round-trip."""
    rec = ToolCallRecord(
        tool_name="ask_manager",
        args={"question": "Q"},
        result={"ok": True, "data": {"answer": "A"}},
        latency_ms=300,
        cost_usd=0.0123456789,
    )
    d = rec.to_dict()
    assert d["cost_usd"] == 0.0123456789
    rec2 = ToolCallRecord.from_dict(d)
    assert rec2.cost_usd == 0.0123456789
