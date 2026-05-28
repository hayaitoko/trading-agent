"""Tests for ingest/seed_sources.py (B0 — Substack + Seeking Alpha RSS seeds).

All tests run fully offline against a temporary SQLite DB — no network calls.
"""

from __future__ import annotations

import json

import pytest

from trading_agent.config.db import Database
from trading_agent.ingest.seed_sources import (
    _BSKY_AUTHOR_SEEDS,
    _BSKY_LIST_SEEDS,
    _SA_DEFAULT_TICKERS,
    _SA_GLOBAL_SEEDS,
    _SUBSTACK_SEEDS,
    seed_finance_sources,
    seed_sa_ticker,
)


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "config.db")


USER = "u1"


# ---------------------------------------------------------------------------
# seed_finance_sources
# ---------------------------------------------------------------------------


def test_seed_inserts_all_expected_rows(db: Database) -> None:
    """Every Substack + SA global + SA per-ticker + Bluesky row is inserted on first call."""
    n = seed_finance_sources(db, USER)
    expected = (
        len(_SUBSTACK_SEEDS)
        + len(_SA_GLOBAL_SEEDS)
        + len(_SA_DEFAULT_TICKERS)
        + len(_BSKY_LIST_SEEDS)
        + len(_BSKY_AUTHOR_SEEDS)
    )
    assert n == expected


def test_seed_is_idempotent(db: Database) -> None:
    """Calling seed_finance_sources twice inserts nothing on the second call."""
    first = seed_finance_sources(db, USER)
    second = seed_finance_sources(db, USER)
    assert first > 0
    assert second == 0


def test_seed_rows_have_known_kinds(db: Database) -> None:
    """All seeded sources use known adapter kinds (rss, bluesky_list, bluesky_author)."""
    seed_finance_sources(db, USER)
    rows = db.query("SELECT DISTINCT kind FROM sources WHERE user_id = ?", (USER,))
    kinds = {r["kind"] for r in rows}
    assert kinds <= {"rss", "bluesky_list", "bluesky_author"}


def test_seed_rss_rows_exist(db: Database) -> None:
    """RSS sources (Substack + SA) are seeded with kind='rss'."""
    seed_finance_sources(db, USER)
    rows = db.query(
        "SELECT * FROM sources WHERE user_id = ? AND kind = 'rss'", (USER,)
    )
    assert len(rows) > 0


def test_substack_urls_are_well_formed(db: Database) -> None:
    """Every Substack source URL follows the expected feed pattern."""
    seed_finance_sources(db, USER)
    rows = db.query(
        "SELECT config_json FROM sources WHERE user_id = ? AND name LIKE 'Substack:%'",
        (USER,),
    )
    for row in rows:
        cfg = json.loads(row["config_json"])
        url = cfg["url"]
        assert url.startswith("https://")
        assert ".substack.com/feed" in url, f"expected substack feed URL, got {url!r}"


def test_sa_global_seeds_have_expected_urls(db: Database) -> None:
    """The three SA global feed URLs are seeded exactly."""
    seed_finance_sources(db, USER)
    expected_urls = {url for _, url, _ in _SA_GLOBAL_SEEDS}
    rows = db.query(
        "SELECT config_json FROM sources WHERE user_id = ? AND name LIKE 'Seeking Alpha: %'",
        (USER,),
    )
    seeded_urls = {json.loads(r["config_json"])["url"] for r in rows}
    for url in expected_urls:
        assert url in seeded_urls, f"SA global URL {url!r} not seeded"


def test_sa_per_ticker_seeds_include_default_symbols(db: Database) -> None:
    """The five default ticker sources (SPY/AAPL/MSFT/NVDA/TSLA) are seeded."""
    seed_finance_sources(db, USER)
    for ticker in _SA_DEFAULT_TICKERS:
        rows = db.query(
            "SELECT config_json FROM sources WHERE user_id = ? AND name = ?",
            (USER, f"Seeking Alpha: {ticker} combined"),
        )
        assert len(rows) == 1, f"Expected exactly one SA source for {ticker}"
        cfg = json.loads(rows[0]["config_json"])
        assert cfg["url"] == f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
        assert cfg["ticker"] == ticker


def test_seeds_do_not_bleed_across_users(db: Database) -> None:
    """Sources seeded for one user do not appear for another."""
    seed_finance_sources(db, "userA")
    rowsB = db.query("SELECT * FROM sources WHERE user_id = ?", ("userB",))
    assert rowsB == []


def test_all_seeded_sources_enabled_by_default(db: Database) -> None:
    """Every seeded source starts enabled."""
    seed_finance_sources(db, USER)
    disabled = db.query(
        "SELECT * FROM sources WHERE user_id = ? AND enabled = 0", (USER,)
    )
    assert disabled == []


# ---------------------------------------------------------------------------
# seed_sa_ticker
# ---------------------------------------------------------------------------


def test_seed_sa_ticker_inserts_new_row(db: Database) -> None:
    """seed_sa_ticker inserts a new row and returns True."""
    result = seed_sa_ticker(db, USER, "AMD")
    assert result is True
    rows = db.query(
        "SELECT config_json FROM sources WHERE user_id = ? AND name = ?",
        (USER, "Seeking Alpha: AMD combined"),
    )
    assert len(rows) == 1
    cfg = json.loads(rows[0]["config_json"])
    assert cfg["url"] == "https://seekingalpha.com/api/sa/combined/AMD.xml"
    assert cfg["ticker"] == "AMD"


def test_seed_sa_ticker_is_idempotent(db: Database) -> None:
    """Calling seed_sa_ticker twice for the same ticker returns False on second call."""
    assert seed_sa_ticker(db, USER, "AMZN") is True
    assert seed_sa_ticker(db, USER, "AMZN") is False


def test_seed_sa_ticker_normalises_case(db: Database) -> None:
    """Ticker is upper-cased regardless of input case."""
    seed_sa_ticker(db, USER, "goog")
    rows = db.query(
        "SELECT config_json FROM sources WHERE user_id = ? AND name = ?",
        (USER, "Seeking Alpha: GOOG combined"),
    )
    assert len(rows) == 1
    cfg = json.loads(rows[0]["config_json"])
    assert cfg["ticker"] == "GOOG"
    assert "GOOG" in cfg["url"]


def test_seed_sa_ticker_disabled_env(monkeypatch: pytest.MonkeyPatch, db: Database) -> None:
    """When INGEST_SEEDS_ENABLED=0, seed_sa_ticker is a no-op."""
    monkeypatch.setenv("INGEST_SEEDS_ENABLED", "0")
    # Re-import to pick up monkeypatched env (module-level constant is set at import;
    # use the function's guard which re-reads the module-level _SEEDS_ENABLED).
    # Since _SEEDS_ENABLED is read at module load, we patch it directly here.
    import trading_agent.ingest.seed_sources as ss
    original = ss._SEEDS_ENABLED
    ss._SEEDS_ENABLED = False
    try:
        result = ss.seed_sa_ticker(db, USER, "META")
        assert result is False
        rows = db.query("SELECT * FROM sources WHERE user_id = ?", (USER,))
        assert rows == []
    finally:
        ss._SEEDS_ENABLED = original


def test_seed_finance_sources_disabled_env(db: Database) -> None:
    """When _SEEDS_ENABLED is False, seed_finance_sources inserts nothing."""
    import trading_agent.ingest.seed_sources as ss
    original = ss._SEEDS_ENABLED
    ss._SEEDS_ENABLED = False
    try:
        n = ss.seed_finance_sources(db, USER)
        assert n == 0
    finally:
        ss._SEEDS_ENABLED = original
