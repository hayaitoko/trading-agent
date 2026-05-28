"""Tests for GET /api/news — raw ingested headlines surface.

Per-user scoping, newest-first ordering, optional ?symbol= filter, limit
cap, auth, and graceful-empty (no rows yet).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.ingest.fetchers.base import RawItem
from trading_agent.ingest.store import IngestStore
from trading_agent.web.app import create_cockpit_app


def _signup(client: TestClient, username: str = "lukas") -> str:
    """Return the user_id of a freshly-signed-up account."""
    r = client.post("/api/auth/signup", json={"username": username, "password": "pw"})
    assert r.status_code == 200, r.text
    return str(r.json().get("user_id") or r.json().get("id") or username)


def _user_id(db: Database, username: str) -> str:
    row = db.query_one("SELECT id FROM users WHERE username = ?", (username,))
    assert row is not None
    return str(row["id"])


@pytest.fixture
def env(tmp_path: Path) -> tuple[TestClient, Database]:
    db = Database(tmp_path / "c.db")
    app = create_cockpit_app(db)
    c = TestClient(app)
    _signup(c, "lukas")
    return c, db


def _item(source_id: str, text: str, *, ticker: str | None = None, url: str = "") -> RawItem:
    return RawItem(
        source_id=source_id,
        text=text,
        url=url or f"https://example.test/{text.replace(' ', '-')}",
        ts="2026-05-28T15:00:00Z",
        ticker=ticker,
    )


def test_news_returns_newest_first(env: tuple[TestClient, Database]) -> None:
    client, db = env
    store = IngestStore(db)
    user_id = _user_id(db, "lukas")
    store.append(user_id, [_item("rss:bloomberg", "Older headline", ticker="AAPL")])
    time.sleep(0.01)  # fetched_at must strictly differ
    store.append(user_id, [_item("rss:reuters", "Newer headline", ticker="AAPL")])

    r = client.get("/api/news")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] is None
    titles = [it["title"] for it in body["items"]]
    assert titles == ["Newer headline", "Older headline"]
    assert body["items"][0]["source"] == "rss:reuters"
    assert body["items"][0]["ticker"] == "AAPL"
    assert body["items"][0]["url"].startswith("https://")


def test_news_symbol_filter(env: tuple[TestClient, Database]) -> None:
    client, db = env
    store = IngestStore(db)
    user_id = _user_id(db, "lukas")
    store.append(
        user_id,
        [
            _item("rss:a", "Apple thing", ticker="AAPL"),
            _item("rss:b", "Microsoft thing", ticker="MSFT"),
            _item("rss:c", "Generic market wrap", ticker=None),
        ],
    )
    r = client.get("/api/news", params={"symbol": "msft"})
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "MSFT"
    assert [it["title"] for it in body["items"]] == ["Microsoft thing"]


def test_news_limit_cap(env: tuple[TestClient, Database]) -> None:
    client, db = env
    store = IngestStore(db)
    user_id = _user_id(db, "lukas")
    store.append(
        user_id,
        [_item("rss:a", f"Headline {i}", url=f"https://x/{i}") for i in range(30)],
    )
    r = client.get("/api/news", params={"limit": 5})
    assert r.status_code == 200
    assert len(r.json()["items"]) == 5


def test_news_limit_validated(env: tuple[TestClient, Database]) -> None:
    client, _db = env
    assert client.get("/api/news", params={"limit": 0}).status_code == 400
    assert client.get("/api/news", params={"limit": 1000}).status_code == 400


def test_news_invalid_symbol_400(env: tuple[TestClient, Database]) -> None:
    client, _db = env
    assert client.get("/api/news", params={"symbol": "@@@"}).status_code == 400


def test_news_per_user_isolated(tmp_path: Path) -> None:
    db = Database(tmp_path / "c.db")
    app = create_cockpit_app(db)
    c_a = TestClient(app)
    c_b = TestClient(app)
    _signup(c_a, "alice")
    _signup(c_b, "bob")
    store = IngestStore(db)
    store.append(_user_id(db, "alice"), [_item("rss:a", "alice item", ticker="AAPL")])
    store.append(_user_id(db, "bob"), [_item("rss:b", "bob item", ticker="AAPL")])

    a_titles = [it["title"] for it in c_a.get("/api/news").json()["items"]]
    b_titles = [it["title"] for it in c_b.get("/api/news").json()["items"]]
    assert a_titles == ["alice item"]
    assert b_titles == ["bob item"]


def test_news_empty(env: tuple[TestClient, Database]) -> None:
    client, _db = env
    r = client.get("/api/news")
    assert r.status_code == 200
    assert r.json() == {"symbol": None, "items": []}


def test_news_requires_auth(tmp_path: Path) -> None:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    c = TestClient(app)
    assert c.get("/api/news").status_code == 401
