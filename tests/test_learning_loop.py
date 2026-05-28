"""P5 tests: calibrated learning loop, dual-output reflection, decay-prune,
calibration tracking, and walk-forward validity (outcomes from price, not model).

All offline — FakeEmbedder + sqlite-vec.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from trading_agent.config.db import Database
from trading_agent.config.endpoints import EndpointRegistry, ModelRef
from trading_agent.config.settings_store import SettingsStore
from trading_agent.memory import FakeEmbedder, LearningLoop, Reflector
from trading_agent.memory.reflect import DualReflectionOutput, OutcomeRecord  # noqa: F401
from trading_agent.memory.store import MemoryStore
from trading_agent.memory.vector.sqlite_vec import SqliteVecStore
from trading_agent.patterns.store import PatternEpisode, PatternStore

# ---- fixtures ---------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Any) -> Database:
    d = Database(str(tmp_path / "test.db"))
    d.connect()
    return d


@pytest.fixture
def pattern_store(db: Database, tmp_path: Any) -> PatternStore:
    vec = SqliteVecStore(str(tmp_path / "vec.db"))
    return PatternStore(db, vector=vec, embedder=FakeEmbedder())


@pytest.fixture
def learning_loop(pattern_store: PatternStore) -> LearningLoop:
    return LearningLoop(pattern_store)


def _ep(**kw: Any) -> PatternEpisode:
    defaults = {
        "symbol": "AAPL",
        "label": "gap-up-no-news",
        "event_date": "2026-05-01",
        "regime": "calm",
    }
    defaults.update(kw)
    return PatternEpisode(**defaults)


# ---- Outcome scoring from realized prices (not model narrative) -------------


def test_observe_outcome_buy_hit(learning_loop: LearningLoop, pattern_store: PatternStore) -> None:
    ep = pattern_store.add(_ep())
    record = learning_loop.observe_outcome(
        ep.id,
        entry_price=100.0,
        exit_price=105.0,
        action="BUY",
        predicted_prob=0.7,
    )
    assert isinstance(record, OutcomeRecord)
    assert record.realized_hit == 1
    assert record.fwd_pct == pytest.approx(0.05)
    assert record.calibration_error == pytest.approx((0.7 - 1) ** 2)

    # Verify the DB was updated.
    fetched = pattern_store.get(ep.id)
    assert fetched is not None
    assert fetched.realized_hit == 1
    assert fetched.outcome == pytest.approx(0.05)


def test_observe_outcome_sell_miss(learning_loop: LearningLoop, pattern_store: PatternStore) -> None:
    ep = pattern_store.add(_ep())
    record = learning_loop.observe_outcome(
        ep.id,
        entry_price=100.0,
        exit_price=103.0,  # price went UP → SELL missed
        action="SELL",
    )
    assert record.realized_hit == 0
    assert record.fwd_pct > 0


def test_observe_outcome_buy_miss(learning_loop: LearningLoop, pattern_store: PatternStore) -> None:
    ep = pattern_store.add(_ep())
    record = learning_loop.observe_outcome(
        ep.id,
        entry_price=100.0,
        exit_price=98.0,  # price went DOWN → BUY missed
        action="BUY",
    )
    assert record.realized_hit == 0


def test_observe_outcome_invalid_action(learning_loop: LearningLoop, pattern_store: PatternStore) -> None:
    ep = pattern_store.add(_ep())
    with pytest.raises(ValueError, match="BUY or SELL"):
        learning_loop.observe_outcome(ep.id, entry_price=100.0, exit_price=100.0, action="HOLD")


def test_outcome_is_from_price_not_narrative(learning_loop: LearningLoop, pattern_store: PatternStore) -> None:
    """Realized outcome is computed from entry/exit price, not from model text."""
    ep = pattern_store.add(_ep())
    # "Model says it was great" — irrelevant; we score from price movement.
    record = learning_loop.observe_outcome(
        ep.id,
        entry_price=100.0,
        exit_price=95.0,  # objectively bad for a BUY
        action="BUY",
        predicted_prob=0.9,
    )
    # Even if model said 90% confidence, the realized hit is 0 (price dropped).
    assert record.realized_hit == 0
    assert record.calibration_error == pytest.approx((0.9 - 0) ** 2)


# ---- Calibration tracking ---------------------------------------------------


def test_calibration_summary_brier(learning_loop: LearningLoop, pattern_store: PatternStore) -> None:
    """Brier score computed from (predicted_prob - realized_hit)^2 per episode."""
    ep1 = pattern_store.add(_ep(label="gap-up-no-news", regime="calm"))
    ep2 = pattern_store.add(_ep(label="gap-up-no-news", regime="calm"))

    # ep1: predicted 0.8, realized 1 → error = 0.04
    learning_loop.observe_outcome(ep1.id, entry_price=100, exit_price=105, action="BUY", predicted_prob=0.8)
    # ep2: predicted 0.6, realized 0 → error = 0.36
    learning_loop.observe_outcome(ep2.id, entry_price=100, exit_price=97, action="BUY", predicted_prob=0.6)

    s = learning_loop.calibration_summary(label="gap-up-no-news", regime="calm")
    assert s["n"] == 2
    assert s["calibration_brier"] == pytest.approx((0.04 + 0.36) / 2)


def test_calibration_summary_overall(learning_loop: LearningLoop, pattern_store: PatternStore) -> None:
    ep = pattern_store.add(_ep(label="bull-flag", regime="elevated"))
    learning_loop.observe_outcome(ep.id, entry_price=100, exit_price=102, action="BUY", predicted_prob=0.65)
    summary = learning_loop.calibration_summary()
    assert "total_episodes" in summary
    assert summary["total_episodes"] >= 1


# ---- Decay pruning ----------------------------------------------------------


def test_decay_prune_archives_rotting_label(
    learning_loop: LearningLoop, pattern_store: PatternStore
) -> None:
    """A label whose hit-rate < floor (coin-flip) is archived by decay_prune."""
    # Add 6 episodes for "always-miss-label" — all misses (hit_rate = 0.0).
    for i in range(6):
        ep = pattern_store.add(_ep(label="always-miss", event_date=f"2026-0{i // 3 + 1}-{i % 3 + 1:02d}"))
        learning_loop.observe_outcome(ep.id, entry_price=100.0, exit_price=95.0, action="BUY")

    # Prune with a tight floor — this label should be archived.
    archived = learning_loop.decay_prune()
    assert "always-miss" in archived
    # All episodes for that label should now be archived.
    active = pattern_store.by_label("always-miss", status="active")
    assert len(active) == 0


def test_decay_prune_spares_healthy_label(
    learning_loop: LearningLoop, pattern_store: PatternStore
) -> None:
    """A label with good hit-rate is NOT pruned."""
    for i in range(6):
        ep = pattern_store.add(_ep(label="always-hit", event_date=f"2026-01-{i + 1:02d}"))
        learning_loop.observe_outcome(ep.id, entry_price=100.0, exit_price=103.0, action="BUY")

    archived = learning_loop.decay_prune()
    assert "always-hit" not in archived
    active = pattern_store.by_label("always-hit", status="active")
    assert len(active) == 6


def test_decay_prune_needs_min_n(
    learning_loop: LearningLoop, pattern_store: PatternStore
) -> None:
    """Labels with < min_n scored episodes are not pruned regardless of hit-rate."""
    for _ in range(3):  # only 3 episodes — below min_n=5
        ep = pattern_store.add(_ep(label="too-few"))
        learning_loop.observe_outcome(ep.id, entry_price=100, exit_price=95, action="BUY")

    archived = learning_loop.decay_prune()
    assert "too-few" not in archived


# ---- Dual-output reflection -------------------------------------------------


def _dual_transport(pattern_obs: str, private_lessons: list[str]) -> httpx.MockTransport:
    """Mock that returns a dual-output JSON."""
    body = json.dumps({
        "pattern_observation": pattern_obs,
        "private_lessons": private_lessons,
    })

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "model": "test",
                    "choices": [{"message": {"content": body}}],
                    "usage": {"total_tokens": 50, "cost": 0.001},
                },
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _make_reflector(db: Database, vec_path: str, transport: httpx.MockTransport) -> tuple[Reflector, MemoryStore, SettingsStore, str]:
    """Build a Reflector with an endpoint registered. Returns (reflector, memory, settings, ep_id)."""
    vec = SqliteVecStore(vec_path)
    memory = MemoryStore(vec, FakeEmbedder())
    settings = SettingsStore(db)
    registry = EndpointRegistry(db, transport=transport)
    ep = registry.add("u1", "openrouter", "OR", api_key="test-key")
    settings.set("u1", "daily_usd_ceiling", 100.0)
    reflector = Reflector(memory, settings=settings, registry=registry)
    return reflector, memory, settings, ep.id


def test_dual_reflection_outputs_land_correctly(
    db: Database,
    tmp_path: Any,
) -> None:
    """distill_dual_outputs returns pattern_obs + private_lessons as separate outputs."""
    transport = _dual_transport(
        "Gap-down reversed on volume.",
        ["Don't chase; set limit orders."],
    )
    reflector, memory, settings, ep_id = _make_reflector(
        db, str(tmp_path / "mem2.db"), transport
    )

    ref = ModelRef(ep_id, "test/model")
    dual = reflector.distill_dual_outputs("u1", "trader-a", "some context", ref)

    assert isinstance(dual, DualReflectionOutput)
    assert dual.pattern_obs  # non-empty
    assert isinstance(dual.private_lessons, list)
    assert len(dual.private_lessons) >= 1


def test_reflect_dual_private_goes_to_memory(
    db: Database, tmp_path: Any
) -> None:
    """reflect_dual writes private lessons to memory and returns the dual output."""
    transport = _dual_transport(
        "Failed breakout on NVDA: price rejected at resistance.",
        ["Wait for second confirmation before entering breakouts."],
    )
    reflector, memory, settings, ep_id = _make_reflector(
        db, str(tmp_path / "mem3.db"), transport
    )

    ref = ModelRef(ep_id, "test/model")
    dual, result = reflector.reflect_dual("u1", "trader-b", "context", ref)

    # Private lesson was written.
    assert len(result.written) >= 1
    # Pattern obs is separate (not in trader memory).
    assert dual.pattern_obs
    lessons = memory.recall("u1", "trader-b", "confirmation", k=5)
    assert any("confirmation" in ln.text.lower() for ln in lessons)


def test_dual_reflection_pattern_obs_in_global_kb(
    db: Database, tmp_path: Any, pattern_store: PatternStore
) -> None:
    """Caller writes pattern_obs to global KB; private lesson goes to per-trader memory."""
    transport = _dual_transport(
        "Bull flag formed in SPY after 3% consolidation.",
        ["Reduce size on low-conviction setups."],
    )
    reflector, memory, settings, ep_id = _make_reflector(
        db, str(tmp_path / "mem4.db"), transport
    )

    ref = ModelRef(ep_id, "test/model")
    dual, _ = reflector.reflect_dual("u1", "trader-c", "context", ref)

    # Caller writes pattern obs to global KB (as the design specifies).
    ep = PatternEpisode(
        symbol="SPY",
        label="bull-flag",
        event_date="2026-05-27",
        regime="calm",
        notes=dual.pattern_obs,
    )
    stored = pattern_store.add(ep)
    assert stored.notes == dual.pattern_obs
    # Verify global visibility (any "trader" using same store can read it).
    fetched = pattern_store.get(stored.id)
    assert fetched is not None
    assert fetched.notes == dual.pattern_obs
