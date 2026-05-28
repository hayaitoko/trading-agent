"""Tests for GET /api/symbols — server-side ticker search.

Exercises the prefix-first / substring match ordering, the 25-result cap, the
empty-query "browse" behavior, the auth requirement, and the graceful fallback
when the universe file is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.web.app import create_cockpit_app
from trading_agent.web.routers import symbols as symbols_module


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "lukas", "password": "pw"})
    return c


def test_symbols_search_prefix_match(client: TestClient) -> None:
    r = client.get("/api/symbols", params={"q": "AAP"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "AAP"
    syms = [row["symbol"] for row in body["results"]]
    # Apple is the only common ticker that prefix-matches "AAP".
    assert "AAPL" in syms
    # Result rows expose symbol + name.
    aapl = next(r for r in body["results"] if r["symbol"] == "AAPL")
    assert "Apple" in aapl["name"]


def test_symbols_search_substring_on_name(client: TestClient) -> None:
    r = client.get("/api/symbols", params={"q": "Microsoft"})
    assert r.status_code == 200
    syms = [row["symbol"] for row in r.json()["results"]]
    assert "MSFT" in syms


def test_symbols_search_exact_symbol_ranks_first(client: TestClient) -> None:
    # "V" is exact for Visa; many names contain a "V". Exact must win.
    r = client.get("/api/symbols", params={"q": "V"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results, "expected non-empty results for 'V'"
    assert results[0]["symbol"] == "V"


def test_symbols_search_cap_25(client: TestClient) -> None:
    # Lowercase "a" matches a huge slice of the universe; cap must hold.
    r = client.get("/api/symbols", params={"q": "a"})
    assert r.status_code == 200
    assert len(r.json()["results"]) <= 25


def test_symbols_search_empty_query_browses(client: TestClient) -> None:
    r = client.get("/api/symbols", params={"q": ""})
    assert r.status_code == 200
    results = r.json()["results"]
    assert 0 < len(results) <= 25
    # Browse default surfaces the top-weight symbols (AAPL leads symbols.json).
    assert results[0]["symbol"] == "AAPL"


def test_symbols_search_case_insensitive(client: TestClient) -> None:
    upper = client.get("/api/symbols", params={"q": "msft"}).json()["results"]
    assert any(row["symbol"] == "MSFT" for row in upper)


def test_symbols_search_requires_auth(tmp_path: Path) -> None:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    c = TestClient(app)  # no signup → no session cookie
    assert c.get("/api/symbols", params={"q": "AAP"}).status_code == 401


def test_symbols_missing_universe_falls_back_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing/unreadable universe file degrades to an empty result set."""
    monkeypatch.setattr(symbols_module, "_SYMBOLS_PATH", tmp_path / "nope.json")
    symbols_module._reset_cache_for_tests()
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "u", "password": "pw"})
    try:
        r = c.get("/api/symbols", params={"q": "AAP"})
        assert r.status_code == 200
        assert r.json()["results"] == []
    finally:
        # Restore the real universe for the rest of the suite.
        symbols_module._reset_cache_for_tests()


def test_symbols_custom_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the on-disk shape we accept (``s``/``n``/``x``)."""
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(
        json.dumps(
            [
                {"s": "FOO", "n": "Foo Industries", "x": "NYSE"},
                {"s": "BAR", "n": "Bar Holdings", "x": "NASDAQ"},
            ]
        )
    )
    monkeypatch.setattr(symbols_module, "_SYMBOLS_PATH", universe_path)
    symbols_module._reset_cache_for_tests()
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "u", "password": "pw"})
    try:
        r = c.get("/api/symbols", params={"q": "foo"})
        assert r.status_code == 200
        results = r.json()["results"]
        assert results == [
            {"symbol": "FOO", "name": "Foo Industries", "exchange": "NYSE"}
        ]
    finally:
        symbols_module._reset_cache_for_tests()
