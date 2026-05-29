"""P6 tests: per-trader intelligence flag (memory) override, calibration router,
and the calibration metric from a fixture.

Note (WS-Bench-Migration M2): the legacy `_build_context` A/B tests (research /
memory / pattern / situation prompt-block toggles) were retired alongside
`LLMTrader` — under the agent model that context is tool-mediated, not assembled
into a single prompt block, so there is no `_build_context` to diff. The
surviving per-trader knob is the `memory` flag, exercised through the controller
below.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.memory import FakeEmbedder
from trading_agent.memory.vector.sqlite_vec import SqliteVecStore
from trading_agent.patterns.store import PatternEpisode, PatternStore

# ---- BenchController per-trader flag override ------------------------------


def test_bench_controller_per_trader_flags(tmp_path: Any) -> None:
    """add_model with intelligence_flags={"memory": False} disables the private
    memory layer for that specific trader even when the owner has memory wired.

    Under the agent model `memory` is the context layer carried via the
    AgentTrader constructor (the memory_search + reflect tools wrap it); the
    legacy research/situation/pattern sub-flags are now tool-mediated and no
    longer toggle a constructor attribute.
    """
    from unittest.mock import MagicMock

    from trading_agent.bench.bench import Bench
    from trading_agent.bench.controller import BenchController

    client = MagicMock()
    bench = Bench(["AAPL"])

    controller = BenchController(
        bench,
        client,
        symbols=["AAPL"],
        memory=MagicMock(),  # owner has the private memory layer wired
    )

    # Add model with the memory layer disabled at the per-trader level.
    name = controller.add_model("test/model", "no-memory",
                                intelligence_flags={"memory": False})
    competitor = bench._competitors.get(name)
    assert competitor is not None
    assert competitor.trader.memory is None  # overridden by flag


def test_bench_controller_same_model_ab(tmp_path: Any) -> None:
    """Same model added twice: one with the memory layer, one with none."""
    from unittest.mock import MagicMock

    from trading_agent.bench.bench import Bench
    from trading_agent.bench.controller import BenchController

    client = MagicMock()
    bench = Bench(["NVDA"])

    memory = MagicMock()

    controller = BenchController(
        bench,
        client,
        symbols=["NVDA"],
        memory=memory,
    )

    name_on = controller.add_model("test/model", "intel-on")
    name_off = controller.add_model("test/model", "intel-off",
                                    intelligence_flags={
                                        "research": False,
                                        "memory": False,
                                        "situation": False,
                                        "patterns": False,
                                    })

    comp_on = bench._competitors.get(name_on)
    comp_off = bench._competitors.get(name_off)
    assert comp_on is not None and comp_off is not None
    # ON carries the private memory layer; OFF has it disabled by the per-trader flag.
    assert comp_on.trader.memory is not None
    assert comp_off.trader.memory is None


# ---- Calibration router -----------------------------------------------------


def _authed_client(app: Any, username: str = "u1", password: str = "pw") -> Any:
    """Return a TestClient that is logged in (session cookie)."""
    client = TestClient(app, raise_server_exceptions=False)
    client.post("/api/auth/signup", json={"username": username, "password": password})
    client.post("/api/auth/login", json={"username": username, "password": password})
    return client


def test_calibration_router_no_bench(tmp_path: Any) -> None:
    """Calibration endpoint works even when no bench is attached (returns empty)."""
    from trading_agent.web.app import create_cockpit_app

    db = Database(str(tmp_path / "web.db"))
    app = create_cockpit_app(db=db)
    client = _authed_client(app, "u1")

    resp = client.get("/api/calibration/")
    assert resp.status_code == 200
    data = resp.json()
    assert "traders" in data
    assert "pattern_kb" in data


def test_calibration_router_with_pattern_store(tmp_path: Any) -> None:
    """Calibration endpoint reads from pattern_store on app.state."""
    from trading_agent.web.app import create_cockpit_app

    db = Database(str(tmp_path / "web2.db"))
    vec = SqliteVecStore(str(tmp_path / "pvec2.db"))
    ps = PatternStore(db, vector=vec, embedder=FakeEmbedder())

    app = create_cockpit_app(db=db)
    app.state.pattern_store = ps

    ps.add(PatternEpisode(
        symbol="AAPL", label="bull-flag", event_date="2026-05-01",
        regime="calm", realized_hit=1, outcome=0.03, predicted_prob=0.72,
    ))

    client = _authed_client(app, "u2", "pw2")

    resp = client.get("/api/calibration/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pattern_kb"]["available"] is True


def test_calibration_metric_from_fixture(tmp_path: Any) -> None:
    """Calibration metric (Brier score) computed correctly from a fixture."""
    from trading_agent.memory.reflect import LearningLoop

    db = Database(str(tmp_path / "brier.db"))
    db.connect()
    vec = SqliteVecStore(str(tmp_path / "bvec.db"))
    ps = PatternStore(db, vector=vec, embedder=FakeEmbedder())
    loop = LearningLoop(ps)

    ep1 = ps.add(PatternEpisode(symbol="AAPL", label="gap-up-no-news",
                                event_date="2026-01-01", regime="calm"))
    ep2 = ps.add(PatternEpisode(symbol="AAPL", label="gap-up-no-news",
                                event_date="2026-01-02", regime="calm"))

    # ep1: predicted 0.8, hit → error = (0.8-1)^2 = 0.04
    loop.observe_outcome(ep1.id, entry_price=100, exit_price=103, action="BUY", predicted_prob=0.8)
    # ep2: predicted 0.5, miss → error = (0.5-0)^2 = 0.25
    loop.observe_outcome(ep2.id, entry_price=100, exit_price=97, action="BUY", predicted_prob=0.5)

    s = loop.calibration_summary(label="gap-up-no-news")
    expected_brier = (0.04 + 0.25) / 2
    assert s["calibration_brier"] == pytest.approx(expected_brier, rel=1e-4)
    assert s["n"] == 2
    assert s["hit_rate"] == pytest.approx(0.5)
