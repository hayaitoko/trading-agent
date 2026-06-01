"""Tests for Bench durable broker store wiring.

Verifies that:
- When broker_store is supplied, each competitor's PaperBroker uses it with
  book_id == competitor_name.
- Fills persist across a simulated restart (new Bench instance, same store).
- Two competitors in the same bench share the store but remain isolated
  (book_id isolation).
- When broker_store is None (the default), the broker is still ephemeral.
"""

from __future__ import annotations

from typing import Any

import pytest

from trading_agent.bench.bench import Bench
from trading_agent.llm.trader import DecisionResult, TradeDecision
from trading_agent.paper_broker_store import PaperBrokerStore


class ScriptedTrader:
    """Minimal Trader stub."""

    def __init__(self, name: str, script: list[DecisionResult] | None = None) -> None:
        self.name = name
        self.model = name
        self._script = iter(script or [])

    def observe(self, bar: dict[str, Any]) -> None:
        pass

    def decide(self, account: dict[str, Any]) -> DecisionResult:
        return next(self._script, DecisionResult())


def _buy(symbol: str, qty: float) -> DecisionResult:
    return DecisionResult(decisions=[TradeDecision(symbol, "BUY", qty)])


# ---------------------------------------------------------------------------
# No store (default) — ephemeral, no regression
# ---------------------------------------------------------------------------


def test_bench_without_store_is_ephemeral() -> None:
    """Default bench (no broker_store) works exactly as before."""
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    bench.add_competitor("t", ScriptedTrader("t", [_buy("AAPL", 1)]))
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench.run_decisions()
    row = bench.leaderboard()[0]
    assert row["cash"] == pytest.approx(9_900.0)
    # No store was attached
    assert bench._broker_store is None


# ---------------------------------------------------------------------------
# With store — broker wired correctly
# ---------------------------------------------------------------------------


def test_bench_with_store_wires_book_id(tmp_path: Any) -> None:
    """Each competitor's broker must use book_id == competitor name."""
    store = PaperBrokerStore(tmp_path / "paper.db")
    bench = Bench(["AAPL"], initial_balance=10_000.0, broker_store=store)
    comp = bench.add_competitor("alpha", ScriptedTrader("alpha"))

    # The broker's book_id must equal the competitor name.
    assert comp.broker._book_id == "alpha"
    assert comp.broker._store is store


def test_bench_with_store_fills_persist(tmp_path: Any) -> None:
    """A fill in the first Bench instance is visible to a second instance using the same store."""
    store = PaperBrokerStore(tmp_path / "paper.db")

    # Session 1: place a trade.
    bench1 = Bench(["AAPL"], initial_balance=10_000.0, broker_store=store)
    bench1.add_competitor("alpha", ScriptedTrader("alpha", [_buy("AAPL", 5)]))
    bench1.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench1.run_decisions()
    cash_after = bench1.leaderboard()[0]["cash"]
    assert cash_after == pytest.approx(9_500.0)  # 5 * 100 spent

    # Session 2 (simulated restart): same store, same competitor name.
    bench2 = Bench(["AAPL"], initial_balance=10_000.0, broker_store=store)
    bench2.add_competitor("alpha", ScriptedTrader("alpha"))
    bench2.observe_bar({"symbol": "AAPL", "close": 100.0})  # re-seed prices
    replayed_cash = bench2.leaderboard()[0]["cash"]
    assert replayed_cash == pytest.approx(cash_after), (
        f"Expected cash {cash_after} after replay, got {replayed_cash}"
    )
    # Position must also be restored.
    positions = bench2.leaderboard()[0]["positions"]
    assert len(positions) == 1
    assert positions[0]["symbol"] == "AAPL"
    assert positions[0]["quantity"] == pytest.approx(5.0)


def test_bench_two_competitors_share_store_isolated(tmp_path: Any) -> None:
    """Two competitors in the same bench share the store but their books are isolated."""
    store = PaperBrokerStore(tmp_path / "paper.db")
    bench = Bench(["AAPL"], initial_balance=10_000.0, broker_store=store)
    bench.add_competitor("alpha", ScriptedTrader("alpha", [_buy("AAPL", 3)]))
    bench.add_competitor("beta", ScriptedTrader("beta"))  # no trades

    bench.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench.run_decisions()

    # alpha bought 3 @ 100 → 10_000 - 300 = 9_700
    assert store.fill_count("alpha") == 1
    assert store.fill_count("beta") == 0

    rows = {r["name"]: r for r in bench.leaderboard()}
    assert rows["alpha"]["cash"] == pytest.approx(9_700.0)
    assert rows["beta"]["cash"] == pytest.approx(10_000.0)


def test_bench_restart_with_two_competitors_restores_both(tmp_path: Any) -> None:
    """Both competitor books are restored independently after a restart."""
    store = PaperBrokerStore(tmp_path / "paper.db")

    # Session 1
    bench1 = Bench(["AAPL"], initial_balance=10_000.0, broker_store=store)
    bench1.add_competitor("alpha", ScriptedTrader("alpha", [_buy("AAPL", 2)]))
    bench1.add_competitor("beta", ScriptedTrader("beta", [_buy("AAPL", 4)]))
    bench1.observe_bar({"symbol": "AAPL", "close": 100.0})
    bench1.run_decisions()

    alpha_cash1 = next(r["cash"] for r in bench1.leaderboard() if r["name"] == "alpha")
    beta_cash1 = next(r["cash"] for r in bench1.leaderboard() if r["name"] == "beta")

    # Session 2
    bench2 = Bench(["AAPL"], initial_balance=10_000.0, broker_store=store)
    bench2.add_competitor("alpha", ScriptedTrader("alpha"))
    bench2.add_competitor("beta", ScriptedTrader("beta"))
    bench2.observe_bar({"symbol": "AAPL", "close": 100.0})

    alpha_cash2 = next(r["cash"] for r in bench2.leaderboard() if r["name"] == "alpha")
    beta_cash2 = next(r["cash"] for r in bench2.leaderboard() if r["name"] == "beta")

    assert alpha_cash2 == pytest.approx(alpha_cash1)
    assert beta_cash2 == pytest.approx(beta_cash1)


def test_bench_store_book_id_equals_name_not_model(tmp_path: Any) -> None:
    """book_id uses the competitor name (which the controller sets to model slug)."""
    store = PaperBrokerStore(tmp_path / "paper.db")
    bench = Bench(["AAPL"], broker_store=store)

    # Simulate controller: add_competitor called with a long model slug as name.
    name = "openai/gpt-4o"
    comp = bench.add_competitor(name, ScriptedTrader(name))
    assert comp.broker._book_id == name
    assert store.fill_count(name) == 0  # empty but keyed correctly
