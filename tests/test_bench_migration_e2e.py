"""WS-Bench-Migration M3 — end-to-end bench-on-AgentTrader verification.

Boots the bench through the same controller wiring serve.py uses (BenchController
with the shared agent infra: attention queue, PendingTradeQueue, TurnStore,
SITUATION providers), drives SoD / regular / EoD turns through the real
MarketScheduler, exercises a SITUATION LOOK tool on/off through the
controller-built trader, and asserts the turn_store trace + accumulated cost.

No outbound network: the LLM is a scripted duck-typed client (chat_with_tools) and
the broker is the bench's own PaperBroker.  This locks in the migration's
acceptance criteria as a regression guard — see the M3 commit for the live-boot
runway Lukas still needs to exercise against real OpenRouter + paper Alpaca.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from trading_agent.approval_queue import PendingTradeQueue
from trading_agent.bench.bench import Bench
from trading_agent.bench.controller import BenchController
from trading_agent.bench.scheduler import MarketScheduler
from trading_agent.intel.attention_queue import AttentionQueue
from trading_agent.intel.cost_tracker import CostTracker
from trading_agent.intel.lifecycle import OrphanTurnStore
from trading_agent.intel.turn_store import TurnStore
from trading_agent.llm.openrouter import ToolCall, ToolCallChatResult


class _ScriptedClient:
    """Duck-typed OpenRouter client: emits scripted ToolCallChatResults in order.

    Each script entry is ``(tool_name, args)`` for a tool call, or ``None`` for a
    plain-text response (→ implicit hold).  The last entry repeats once exhausted.
    """

    def __init__(self, scripts: list[Any]) -> None:
        self._scripts = list(scripts)
        self._i = 0

    def chat_with_tools(
        self, model: str, messages: Any, tools: Any, **kw: Any
    ) -> ToolCallChatResult:
        spec = self._scripts[min(self._i, len(self._scripts) - 1)]
        self._i += 1
        usage = {"prompt_tokens": 80, "completion_tokens": 20, "cached_tokens": 0}
        if spec is None:
            return ToolCallChatResult(
                content="standing pat", tool_calls=[], model=model,
                usage=usage, cost=0.0007, finish_reason="stop",
            )
        name, args = spec
        return ToolCallChatResult(
            content=None,
            tool_calls=[ToolCall(id=f"c{self._i}", name=name, arguments=args)],
            model=model, usage=usage, cost=0.0015, finish_reason="tool_calls",
        )


def test_bench_agenttrader_turn_flow_trace_and_cost(tmp_path: Any) -> None:
    """SoD / regular / EoD turns flow through the scheduler into a controller-built
    AgentTrader; each writes a turn_store trace with the right turn_type and a
    nonzero cost; the regular turn's BUY settles on the bench's tracked book.
    """
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    ts = TurnStore(db_path=str(tmp_path / "turns.db"))
    scripts = [
        ("hold", {"reason": "SoD posture set"}),
        ("trade", {"symbol": "AAPL", "side": "BUY", "qty": 2}),
        ("hold", {"reason": "EoD housekeeping"}),
    ]
    sched = MarketScheduler(bench, orphan_store=OrphanTurnStore(tmp_path / "orphans.db"))
    ctl = BenchController(
        bench, _ScriptedClient(scripts), symbols=["AAPL"],
        turn_store=ts,
        attention_queue=AttentionQueue(),
        pending_trade_queue=PendingTradeQueue(db_path=str(tmp_path / "approvals.db")),
        scheduler=sched,
    )

    name = ctl.add_model("anthropic/claude-opus-4.7", "opus")
    comp = bench._competitors[name]
    # Post-tutorial steady state so the scheduler's turn-type classification is
    # honored (the tutorial override is covered by test_tutorial_mode.py).
    comp.trader.tutorial_remaining = 0

    bench.observe_bar({"symbol": "AAPL", "close": 100.0})  # price so the BUY fills
    sched.fire_turns([(name, "SoD", "T-60 before open")])
    sched.fire_turns([(name, "regular", "cadence tick")])
    sched.fire_turns([(name, "EoD", "T+30 after close")])

    traces = ts.recent(name, 5)
    assert len(traces) == 3
    by_type = {t.turn_type: t for t in traces}
    assert set(by_type) == {"SoD", "regular", "EoD"}

    # The regular turn invoked the trade ACT tool (terminal) and recorded it.
    assert by_type["regular"].final_action == "trade"
    assert by_type["regular"].tool_calls[0].tool_name == "trade"
    assert by_type["SoD"].final_action == "hold"
    assert by_type["EoD"].final_action == "hold"

    # cost_tracker accumulated a nonzero per-turn cost on every trace.
    assert all(t.total_cost_usd > 0 for t in traces)
    # Token rollup is captured too (input/output from the scripted usage).
    assert by_type["regular"].total_tokens["input"] == 80

    # The BUY settled on the bench's tracked book: 1 trade, cash 10000 - 2*100.
    assert len(comp.broker.get_trade_history()) == 1
    assert comp.broker.get_balance()["cash"] == 9_800.0


def test_controller_threads_situation_provider_and_tool_fires(tmp_path: Any) -> None:
    """The controller threads the gdelt provider into the AgentTrader; with the
    SITUATION_GDELT flag on, world_events fires through the controller-built trader.
    """
    from datetime import UTC, datetime
    from types import SimpleNamespace

    bench = Bench(["AAPL"], initial_balance=10_000.0)
    gdelt = MagicMock()
    gdelt.timeline_volume.return_value = [
        SimpleNamespace(bucket_start=datetime(2026, 5, 28, 12, tzinfo=UTC), value=50.0, unit="mentions")
    ]
    gdelt.top_articles.return_value = [
        SimpleNamespace(
            title="Conflict news", url="https://reuters.com/x",
            published=datetime(2026, 5, 28, 10, tzinfo=UTC),
            source_domain="reuters.com", tone=-3.0,
        )
    ]

    ctl = BenchController(bench, _ScriptedClient([None]), symbols=["AAPL"], gdelt_provider=gdelt)
    name = ctl.add_model("m", "g")
    trader = bench._competitors[name].trader

    # The controller threaded the provider through to the trader (M1 plumbing).
    assert trader._gdelt_provider is gdelt

    # Simulate the SITUATION_GDELT flag being on (db-backed in production).
    settings = MagicMock()
    settings.get = lambda uid, key, default=None: (True if key == "SITUATION_GDELT" else False)
    trader.settings_store = settings

    tc = ToolCall(id="x", name="world_events", arguments={"theme": "WAR", "timespan": "24h"})
    result = trader._execute_tool(tc, CostTracker())
    assert result.ok is True
    assert result.data["theme"] == "WAR"
    assert len(result.data["bins"]) == 1


def test_controller_situation_tool_disabled_without_provider(tmp_path: Any) -> None:
    """A controller-built trader with no SITUATION provider/flag (the default
    build_cockpit state) returns ToolError(kind='disabled') for world_events —
    the agent sees a structured 'off' rather than a crash.
    """
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    ctl = BenchController(bench, _ScriptedClient([None]), symbols=["AAPL"])  # no provider
    name = ctl.add_model("m", "g")
    trader = bench._competitors[name].trader

    tc = ToolCall(id="x", name="world_events", arguments={"theme": "WAR"})
    result = trader._execute_tool(tc, CostTracker())
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind == "disabled"
