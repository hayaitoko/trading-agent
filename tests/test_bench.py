"""Tests for the multi-model bench orchestrator (deterministic, no network)."""

from typing import Any

import pytest

from trading_agent.bench import Bench
from trading_agent.llm.trader import DecisionResult, TradeDecision, Trader


class ScriptedTrader:
    """Returns a pre-scripted DecisionResult on each decide() call."""

    def __init__(self, name: str, script: list[DecisionResult] | None = None) -> None:
        self.name = name
        self.model = name
        self._script = iter(script or [])
        self.observed: list[dict[str, Any]] = []

    def observe(self, bar: dict[str, Any]) -> None:
        self.observed.append(bar)

    def decide(self, account: dict[str, Any]) -> DecisionResult:
        return next(self._script, DecisionResult())


def _buy(symbol: str, qty: float) -> DecisionResult:
    return DecisionResult(decisions=[TradeDecision(symbol, "BUY", qty)], comment="buy")


def test_add_competitor_isolated_book() -> None:
    bench = Bench(["AAPL"], initial_balance=5000.0)
    comp = bench.add_competitor("opus", ScriptedTrader("opus"))
    assert comp.broker.get_balance()["cash"] == 5000.0
    assert bench.names() == ["opus"]
    with pytest.raises(ValueError):
        bench.add_competitor("opus", ScriptedTrader("opus"))


def test_scripted_trader_satisfies_protocol() -> None:
    assert isinstance(ScriptedTrader("x"), Trader)


def test_observe_bar_sets_price_and_feeds_traders() -> None:
    t = ScriptedTrader("m")
    bench = Bench(["AAPL"])
    bench.add_competitor("m", t)
    bench.observe_bar({"symbol": "AAPL", "close": 150.0})
    assert bench._last_prices["AAPL"] == 150.0
    assert t.observed == [{"symbol": "AAPL", "close": 150.0}]


def test_buy_decision_fills_and_moves_cash() -> None:
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    bench.add_competitor("m", ScriptedTrader("m", [_buy("AAPL", 2)]))
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench.run_decisions()

    row = bench.leaderboard()[0]
    assert row["cash"] == 10_000.0 - 200.0  # 2 @ 100
    assert row["trades"] == 1
    assert row["positions"][0]["symbol"] == "AAPL"
    assert bench.recent_decisions()[0]["status"] == "filled"


def test_hold_does_nothing() -> None:
    hold = DecisionResult(decisions=[TradeDecision("AAPL", "HOLD", 0)])
    bench = Bench(["AAPL"])
    bench.add_competitor("m", ScriptedTrader("m", [hold]))
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench.run_decisions()
    assert bench.leaderboard()[0]["trades"] == 0
    assert bench.recent_decisions() == []


def test_position_size_block_is_logged_not_filled() -> None:
    bench = Bench(["AAPL"], initial_balance=1_000_000.0, max_position_size=10.0)
    bench.add_competitor("m", ScriptedTrader("m", [_buy("AAPL", 50)]))  # 50 > limit 10
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench.run_decisions()
    assert bench.leaderboard()[0]["trades"] == 0
    entry = bench.recent_decisions()[0]
    assert entry["status"] == "blocked" and "position size" in entry["detail"]


def test_trader_error_surfaced() -> None:
    err = DecisionResult(error="429 rate limited")
    bench = Bench(["AAPL"])
    bench.add_competitor("m", ScriptedTrader("m", [err]))
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench.run_decisions()
    row = bench.leaderboard()[0]
    assert row["error"] == "429 rate limited"
    assert bench.recent_decisions()[0]["status"] == "error"


def test_leaderboard_ranks_by_value_and_books_isolated() -> None:
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    # winner buys low then we mark price up; loser sits in cash
    bench.add_competitor("winner", ScriptedTrader("winner", [_buy("AAPL", 50)]))
    bench.add_competitor("loser", ScriptedTrader("loser", [DecisionResult()]))
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench.run_decisions()  # winner buys 50 @ 100 (=5000), loser holds
    bench.observe_bar({"symbol": "AAPL", "close": 120.0})  # AAPL +20%

    board = bench.leaderboard()
    assert [r["name"] for r in board] == ["winner", "loser"]
    assert board[0]["rank"] == 1
    # winner: 5000 cash + 50*120 = 11000 ; loser flat at 10000
    assert board[0]["account_value"] == pytest.approx(11_000.0)
    assert board[1]["account_value"] == pytest.approx(10_000.0)
    assert board[0]["pnl"] == pytest.approx(1_000.0)
    assert board[0]["return_pct"] == pytest.approx(10.0)


def test_observe_quote_enables_bidask_fill() -> None:
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    bench.add_competitor("m", ScriptedTrader("m", [_buy("AAPL", 1)]))
    bench.observe_quote({"symbol": "AAPL", "bid": 99.0, "ask": 101.0, "price": 100.0})
    bench.run_decisions()
    # market buy fills at ask (101), not last (100)
    assert bench.leaderboard()[0]["cash"] == pytest.approx(10_000.0 - 101.0)


def test_snapshot_shape() -> None:
    bench = Bench(["AAPL"])
    bench.add_competitor("m", ScriptedTrader("m"))
    snap = bench.snapshot()
    assert set(snap) >= {
        "generated_at", "symbols", "leaderboard", "recent_decisions", "last_prices",
    }
    assert snap["symbols"] == ["AAPL"]


def test_remove_competitor() -> None:
    bench = Bench(["AAPL"])
    bench.add_competitor("m", ScriptedTrader("m"))
    bench.remove_competitor("m")
    assert bench.names() == []
