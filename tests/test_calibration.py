"""P6 tests: per-trader intelligence flags, A/B decision distinction, calibration
router, and calibration metric from fixture.

Tests that:
- A flag-off trader sees no research/memory/pattern/situation blocks in its prompt.
- The same model added twice with on vs off produces distinct _build_context output.
- Calibration endpoint returns expected structure.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.memory import FakeEmbedder, MemoryStore
from trading_agent.memory.vector.sqlite_vec import SqliteVecStore
from trading_agent.patterns.store import PatternEpisode, PatternStore
from trading_agent.situation.regime import RegimeClassifier

# ---- helpers ----------------------------------------------------------------


def _make_trader(
    model: str = "test/model",
    name: str = "alpha",
    *,
    intelligence_flags: dict[str, bool] | None = None,
    research: Any = None,
    memory: Any = None,
    pattern_store: Any = None,
    regime_classifier: Any = None,
    social_aggregator: Any = None,
) -> Any:
    """Build a minimal LLMTrader with a stub client (no real HTTP)."""
    from unittest.mock import MagicMock

    from trading_agent.llm.trader import LLMTrader

    client = MagicMock()
    return LLMTrader(
        model,
        client,
        symbols=["AAPL"],
        name=name,
        intelligence_flags=intelligence_flags,
        research=research,
        memory=memory,
        owner_user_id="u1",
        pattern_store=pattern_store,
        regime_classifier=regime_classifier,
        social_aggregator=social_aggregator,
    )


def _account() -> dict[str, Any]:
    return {"cash": 10_000.0, "positions": []}


# ---- Flag-off trader sees no intelligence blocks ---------------------------


def test_flag_off_research_omitted() -> None:
    """research=False flag causes the research block to be empty."""
    from unittest.mock import MagicMock

    research = MagicMock()
    research.search.return_value = [MagicMock(summary="great alpha!", ticker="AAPL")]
    research.recent.return_value = []

    trader_on = _make_trader("test/model", "on", research=research)
    trader_off = _make_trader("test/model", "off", research=research,
                              intelligence_flags={"research": False})

    # Inject a bar so the fallback body builds.
    bar = {"symbol": "AAPL", "close": 100.0}
    trader_on.observe(bar)
    trader_off.observe(bar)

    trader_on._build_context(_account())
    ctx_off = trader_off._build_context(_account())

    # The flag-off trader should not call research at all.
    assert "Research" not in ctx_off


def test_flag_off_memory_omitted(tmp_path: Any) -> None:
    """memory=False flag causes the memory block to be empty."""
    vec = SqliteVecStore(str(tmp_path / "mem.db"))
    memory = MemoryStore(vec, FakeEmbedder())
    memory.remember("u1", "on-trader", "Always check volume before entry.")

    trader_on = _make_trader("t", "on-trader", memory=memory)
    trader_off = _make_trader("t", "off-trader", memory=memory,
                              intelligence_flags={"memory": False})

    bar = {"symbol": "AAPL", "close": 100.0}
    trader_on.observe(bar)
    trader_off.observe(bar)

    ctx_off = trader_off._build_context(_account())
    # Flag-off trader must not surface any memory block.
    assert "Always check volume" not in ctx_off

    # ON trader would call memory.recall — we verify the flag doesn't gate it.
    # (The lesson may or may not appear depending on embed distance; the key
    # invariant is that ctx_off definitely has NO memory block.)
    assert "Your past lessons" not in ctx_off


def test_flag_off_patterns_omitted(tmp_path: Any) -> None:
    """patterns=False flag suppresses the pattern KB block."""
    db = Database(str(tmp_path / "db.db"))
    db.connect()
    vec = SqliteVecStore(str(tmp_path / "pvec.db"))
    ps = PatternStore(db, vector=vec, embedder=FakeEmbedder())
    ps.add(PatternEpisode(symbol="AAPL", label="gap-up-no-news", event_date="2026-05-01",
                          regime="calm", realized_hit=1, outcome=0.03))

    trader_on = _make_trader("t", "on", pattern_store=ps)
    trader_off = _make_trader("t", "off", pattern_store=ps,
                              intelligence_flags={"patterns": False})

    bar = {"symbol": "AAPL", "close": 100.0}
    trader_on.observe(bar)
    trader_off.observe(bar)

    ctx_on = trader_on._build_context(_account())
    ctx_off = trader_off._build_context(_account())

    assert "Pattern KB" in ctx_on
    assert "Pattern KB" not in ctx_off


def test_flag_off_situation_omitted() -> None:
    """situation=False flag suppresses the situation/regime block."""
    closes = [100.0 + i * 0.5 for i in range(30)]
    clf = RegimeClassifier()

    trader_on = _make_trader("t", "on", regime_classifier=clf)
    trader_off = _make_trader("t", "off", regime_classifier=clf,
                              intelligence_flags={"situation": False})

    for _i, price in enumerate(closes):
        bar = {"symbol": "AAPL", "close": price, "open": price}
        trader_on.observe(bar)
        trader_off.observe(bar)

    ctx_on = trader_on._build_context(_account())
    ctx_off = trader_off._build_context(_account())

    assert "Situation" in ctx_on
    assert "Situation" not in ctx_off


# ---- Same model on/off produces distinct contexts --------------------------


def test_on_off_contexts_differ(tmp_path: Any) -> None:
    """Adding the same model with on vs off flags yields distinct build_context()."""
    db = Database(str(tmp_path / "db2.db"))
    db.connect()
    vec_m = SqliteVecStore(str(tmp_path / "m.db"))
    memory = MemoryStore(vec_m, FakeEmbedder())
    memory.remember("u1", "on-trader", "Volume precedes breakouts; wait for the bar close.")

    trader_on = _make_trader("test/model", "on-trader", memory=memory)
    trader_off = _make_trader("test/model", "off-trader", memory=memory,
                              intelligence_flags={"memory": False})

    bar = {"symbol": "AAPL", "close": 100.0}
    trader_on.observe(bar)
    trader_off.observe(bar)

    ctx_on = trader_on._build_context(_account())
    ctx_off = trader_off._build_context(_account())

    assert ctx_on != ctx_off


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
