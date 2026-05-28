"""P4 tests: PatternStore round-trip, date-range query, label-filtered recall,
regime-conditioned stats, cross-trader visibility (alpha writes, beta sees it).

No network; vector store is FakeEmbedder + sqlite-vec in-memory.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from trading_agent.config.db import Database
from trading_agent.memory.embed import FakeEmbedder
from trading_agent.memory.vector.sqlite_vec import SqliteVecStore
from trading_agent.patterns.labels import (
    PatternLabel,
    compute_label,
    extract_features,
)
from trading_agent.patterns.store import (
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    PatternEpisode,
    PatternStore,
)

# ---- fixtures ---------------------------------------------------------------


@pytest.fixture
def db(tmp_path: any) -> Database:
    d = Database(str(tmp_path / "test.db"))
    d.connect()
    return d


@pytest.fixture
def vector_store(tmp_path: any) -> SqliteVecStore:
    return SqliteVecStore(str(tmp_path / "vec.db"))


@pytest.fixture
def store(db: Database, vector_store: SqliteVecStore) -> PatternStore:
    return PatternStore(db, vector=vector_store, embedder=FakeEmbedder())


@pytest.fixture
def store_no_vector(db: Database) -> PatternStore:
    return PatternStore(db)


def _ep(
    symbol: str = "AAPL",
    label: str = PatternLabel.GAP_UP_NO_NEWS,
    event_date: str = "2026-05-01",
    regime: str = "calm",
    outcome: float | None = None,
    realized_hit: int | None = None,
    predicted_prob: float | None = None,
) -> PatternEpisode:
    return PatternEpisode(
        symbol=symbol,
        label=label,
        event_date=event_date,
        regime=regime,
        outcome=outcome,
        realized_hit=realized_hit,
        predicted_prob=predicted_prob,
    )


# ---- label taxonomy ---------------------------------------------------------


def _bars(
    n: int = 20,
    *,
    gap_pct: float = 0.0,
    vol_mult: float = 1.0,
    trend: float = 0.01,
) -> list[dict[str, float]]:
    """Synthetic bar sequence (oldest first)."""
    bars = []
    price = 100.0
    for i in range(n):
        close = price * (1 + trend * i * 0.1)
        bars.append({
            "open": close * (1 + gap_pct if i == n - 1 else 1.0),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1_000_000.0 * vol_mult,
        })
    return bars


def test_label_extract_features_needs_3_bars():
    assert extract_features(_bars(n=2)) is None


def test_label_gap_up_no_news():
    bars = _bars(n=10, gap_pct=0.05)
    label = compute_label(bars, social_velocity=0.0)
    assert label == PatternLabel.GAP_UP_NO_NEWS


def test_label_gap_up_catalyst():
    bars = _bars(n=10, gap_pct=0.05)
    label = compute_label(bars, social_velocity=1.5)
    assert label == PatternLabel.GAP_UP_CATALYST


def test_label_capitulation_volume_spike():
    bars = _bars(n=20, vol_mult=10.0, trend=-0.02)
    label = compute_label(bars)
    assert label in (PatternLabel.CAPITULATION_VOLUME_SPIKE, PatternLabel.VOLUME_SPIKE_UP, PatternLabel.UNKNOWN)


def test_label_earnings_drift():
    bars = _bars(n=10, gap_pct=0.04)
    label = compute_label(bars, has_earnings=True)
    assert label == PatternLabel.EARNINGS_DRIFT


def test_label_unknown_too_few_bars():
    label = compute_label(_bars(n=2))
    assert label == PatternLabel.UNKNOWN


# ---- PatternStore round-trip ------------------------------------------------


def test_store_add_and_get(store: PatternStore) -> None:
    ep = store.add(_ep("AAPL", label=PatternLabel.BULL_FLAG, event_date="2026-05-01"))
    assert ep.id
    fetched = store.get(ep.id)
    assert fetched is not None
    assert fetched.symbol == "AAPL"
    assert fetched.label == PatternLabel.BULL_FLAG


def test_store_round_trip_no_vector(store_no_vector: PatternStore) -> None:
    ep = store_no_vector.add(_ep("NVDA", label=PatternLabel.FAILED_BREAKOUT))
    fetched = store_no_vector.get(ep.id)
    assert fetched is not None
    assert fetched.label == PatternLabel.FAILED_BREAKOUT


def test_store_count(store: PatternStore) -> None:
    assert store.count() == 0
    store.add(_ep())
    store.add(_ep())
    assert store.count() == 2


# ---- Date-range query -------------------------------------------------------


def test_store_date_range(store: PatternStore) -> None:
    store.add(_ep(event_date="2025-01-10", label=PatternLabel.BULL_FLAG))
    store.add(_ep(event_date="2025-01-15", label=PatternLabel.BEAR_FLAG))
    store.add(_ep(event_date="2025-01-20", label=PatternLabel.GAP_UP_NO_NEWS))

    results = store.by_date_range("2025-01-12", "2025-01-18")
    assert len(results) == 1
    assert results[0].label == PatternLabel.BEAR_FLAG


def test_store_date_range_a_year_ago(store: PatternStore) -> None:
    """Simulates the 'a year ago ± window' recall pattern."""
    today = date.today()
    year_ago = today - timedelta(days=365)
    window_start = (year_ago - timedelta(days=7)).isoformat()
    window_end = (year_ago + timedelta(days=7)).isoformat()

    store.add(_ep(event_date=year_ago.isoformat(), label=PatternLabel.MEAN_REVERSION_SETUP))
    store.add(_ep(event_date=today.isoformat(), label=PatternLabel.BULL_FLAG))

    results = store.by_date_range(window_start, window_end)
    assert len(results) == 1
    assert results[0].label == PatternLabel.MEAN_REVERSION_SETUP


# ---- Label-filtered recall --------------------------------------------------


def test_store_by_label(store: PatternStore) -> None:
    store.add(_ep(label=PatternLabel.BULL_FLAG))
    store.add(_ep(label=PatternLabel.BULL_FLAG))
    store.add(_ep(label=PatternLabel.BEAR_FLAG))

    bulls = store.by_label(PatternLabel.BULL_FLAG)
    assert len(bulls) == 2
    assert all(ep.label == PatternLabel.BULL_FLAG for ep in bulls)


def test_store_recall_vector_similarity(store: PatternStore) -> None:
    store.add(_ep(symbol="AAPL", label=PatternLabel.GAP_UP_NO_NEWS))
    store.add(_ep(symbol="NVDA", label=PatternLabel.BEAR_FLAG))

    matches = store.recall("gap up AAPL", k=2)
    assert len(matches) >= 1
    # The "gap-up" episode should score first
    assert any(m.label == PatternLabel.GAP_UP_NO_NEWS for m in matches)


def test_store_recall_no_vector_fallback(store_no_vector: PatternStore) -> None:
    store_no_vector.add(_ep(label=PatternLabel.BULL_FLAG))
    matches = store_no_vector.recall("bull pattern", k=5)
    assert isinstance(matches, list)


# ---- Regime-conditioned stats -----------------------------------------------


def test_stats_regime_conditioned(store: PatternStore) -> None:
    # Add episodes: 2 calm hits, 1 calm miss, 1 risk-off hit.
    store.add(_ep(label=PatternLabel.GAP_UP_NO_NEWS, regime="calm",
                  outcome=0.03, realized_hit=1, predicted_prob=0.7))
    store.add(_ep(label=PatternLabel.GAP_UP_NO_NEWS, regime="calm",
                  outcome=0.02, realized_hit=1, predicted_prob=0.6))
    store.add(_ep(label=PatternLabel.GAP_UP_NO_NEWS, regime="calm",
                  outcome=-0.01, realized_hit=0, predicted_prob=0.65))
    store.add(_ep(label=PatternLabel.GAP_UP_NO_NEWS, regime="risk-off",
                  outcome=0.05, realized_hit=1, predicted_prob=0.8))

    calm_stats = store.stats(PatternLabel.GAP_UP_NO_NEWS, regime="calm")
    riskoff_stats = store.stats(PatternLabel.GAP_UP_NO_NEWS, regime="risk-off")
    all_stats = store.stats(PatternLabel.GAP_UP_NO_NEWS)

    assert calm_stats["n"] == 3
    assert calm_stats["hit_rate"] == pytest.approx(2 / 3)
    assert riskoff_stats["n"] == 1
    assert riskoff_stats["hit_rate"] == pytest.approx(1.0)
    assert all_stats["n"] == 4
    assert calm_stats["calibration_brier"] is not None


def test_stats_no_data(store: PatternStore) -> None:
    s = store.stats("nonexistent-label", regime="calm")
    assert s["n"] == 0
    assert s["hit_rate"] is None
    assert not s["has_sufficient_data"]


# ---- Archive / soft-delete --------------------------------------------------


def test_archive_removes_from_active(store: PatternStore) -> None:
    ep = store.add(_ep(label=PatternLabel.BULL_FLAG))
    assert store.count(STATUS_ACTIVE) == 1
    store.archive(ep.id)
    assert store.count(STATUS_ACTIVE) == 0
    assert store.count(STATUS_ARCHIVED) == 1


# ---- Cross-trader visibility ------------------------------------------------


def test_cross_trader_visibility(tmp_path: any) -> None:
    """Alpha writes a pattern; beta (different trader, same global KB) can see it.

    This is the inverse of WS-A's per-trader memory namespacing — patterns are
    global market truth with no per-user or per-trader boundary.
    """
    db = Database(str(tmp_path / "shared.db"))
    db.connect()
    vec = SqliteVecStore(str(tmp_path / "vec.db"))
    emb = FakeEmbedder()

    # Both traders share the same PatternStore (same DB + vector collection).
    alpha_store = PatternStore(db, vector=vec, embedder=emb)
    beta_store = PatternStore(db, vector=vec, embedder=emb)

    # Alpha writes a pattern.
    ep = alpha_store.add(_ep(symbol="GME", label=PatternLabel.VOLUME_SPIKE_UP, regime="elevated"))
    assert ep.id

    # Beta reads it.
    fetched = beta_store.get(ep.id)
    assert fetched is not None
    assert fetched.symbol == "GME"
    assert beta_store.count() == 1


def test_decaying_labels(store: PatternStore) -> None:
    """Labels with sub-threshold hit-rate appear in decaying_labels()."""
    # Add 6 episodes with all misses (hit_rate = 0.0 < floor).
    for _ in range(6):
        store.add(_ep(label="bad-label", realized_hit=0, outcome=-0.02, predicted_prob=0.6))
    decaying = store.decaying_labels(hit_rate_floor=0.52)
    assert "bad-label" in decaying


def test_healthy_label_not_decaying(store: PatternStore) -> None:
    for _ in range(6):
        store.add(_ep(label="good-label", realized_hit=1, outcome=0.03, predicted_prob=0.7))
    decaying = store.decaying_labels(hit_rate_floor=0.52)
    assert "good-label" not in decaying
