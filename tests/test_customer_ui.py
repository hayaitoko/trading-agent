"""Smoke tests for the customer product UI.

Tests:
* GET /app → 200 HTML
* GET /app/accounts, /app/sources, /app/memory, /app/settings → 200 HTML (SPA shell)
* GET /api/memory (no store) → {"lessons":[], "total":0, "source":"empty"}
* GET /api/memory (with mock store) → returns lessons
* GET /app/* (unknown sub-path) → SPA shell, not 404
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.web.app import create_cockpit_app


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Any, monkeypatch: Any) -> TestClient:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    app = create_cockpit_app(Database(tmp_path / "config.db"))
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def authed_client(tmp_path: Any, monkeypatch: Any) -> TestClient:
    """Client with a live session (signup → cookie auto-set by TestClient)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    app = create_cockpit_app(Database(tmp_path / "config.db"))
    c = TestClient(app, raise_server_exceptions=True)
    r = c.post("/api/auth/signup", json={"username": "tester", "password": "secret99"})
    assert r.status_code == 200
    return c


# ---------------------------------------------------------------------------
# Customer SPA shell routes
# ---------------------------------------------------------------------------


def test_app_root_200(client: TestClient) -> None:
    r = client.get("/app")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


def test_app_root_contains_brand(client: TestClient) -> None:
    r = client.get("/app")
    assert "HELM" in r.text


def test_app_subpaths_return_spa_shell(client: TestClient) -> None:
    """Every /app/* path should return the SPA shell, not 404."""
    for path in ("/app/accounts", "/app/sources", "/app/memory", "/app/settings"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"
        assert r.headers["content-type"].startswith("text/html"), path


def test_app_arbitrary_path_returns_spa(client: TestClient) -> None:
    r = client.get("/app/some/deep/path")
    assert r.status_code == 200
    assert "HELM" in r.text


def test_app_spa_contains_nav_pages(client: TestClient) -> None:
    html = client.get("/app").text
    for page in ("Accounts", "News Sources", "Memory", "Settings"):
        assert page in html, f"Missing nav item: {page}"


def test_app_spa_contains_chat_dock(client: TestClient) -> None:
    html = client.get("/app").text
    assert "Manager Chat" in html
    assert "/api/chat" in html


def test_app_spa_wired_to_api_routes(client: TestClient) -> None:
    """The SPA must reference the CONTRACTS API routes."""
    html = client.get("/app").text
    for route in ("/api/auth/", "/api/me", "/api/accounts", "/api/pending-trades",
                  "/api/sources", "/api/news", "/api/memory", "/api/settings"):
        assert route in html, f"SPA missing reference to route: {route}"


# ---------------------------------------------------------------------------
# /api/memory — no store attached (degrade gracefully)
# ---------------------------------------------------------------------------


def test_memory_no_store_requires_auth(client: TestClient) -> None:
    r = client.get("/api/memory")
    # Without a session, 401 or 403 depending on dependency.
    assert r.status_code in (401, 403)


def test_memory_no_store_returns_empty(authed_client: TestClient) -> None:
    r = authed_client.get("/api/memory")
    assert r.status_code == 200
    data = r.json()
    assert data["lessons"] == []
    assert data["total"] == 0
    assert data["source"] == "empty"


def test_memory_query_params_accepted(authed_client: TestClient) -> None:
    r = authed_client.get("/api/memory?trader_id=t1&q=AAPL&k=5")
    assert r.status_code == 200
    data = r.json()
    assert "lessons" in data
    assert "total" in data
    assert "source" in data


# ---------------------------------------------------------------------------
# /api/memory — with a mock MemoryStore on app.state
# ---------------------------------------------------------------------------


@dataclass
class _FakeLesson:
    id: str
    user_id: str
    trader_id: str
    text: str
    tags: list[str] = field(default_factory=list)
    status: str = "active"
    created_at: float = 0.0
    updated_at: float = 0.0
    score: float | None = None


class _FakeMemoryStore:
    def __init__(self, lessons: list[_FakeLesson]) -> None:
        self._lessons = lessons

    def list(self, user_id: str, trader_id: str | None = None,
             *, include_archived: bool = False) -> list[_FakeLesson]:
        out = [l for l in self._lessons if l.user_id == user_id]
        if trader_id is not None:
            out = [l for l in out if l.trader_id == trader_id]
        return out

    def recall(self, user_id: str, trader_id: str, query: str,
               k: int = 5) -> list[_FakeLesson]:
        # Return lessons tagged with the query term for predictable test results.
        return [l for l in self._lessons
                if l.user_id == user_id and l.trader_id == trader_id
                and (query in l.text or query in " ".join(l.tags))][:k]


@pytest.fixture
def client_with_store(tmp_path: Any, monkeypatch: Any) -> TestClient:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    app = create_cockpit_app(Database(tmp_path / "config.db"))

    # Create user so we can get a session token.
    client = TestClient(app, raise_server_exceptions=True)
    r = client.post("/api/auth/signup", json={"username": "mem_tester", "password": "password1"})
    user_id = r.json()["user_id"]

    lessons = [
        _FakeLesson(id="l1", user_id=user_id, trader_id="t1",
                    text="AAPL tends to run pre-earnings", tags=["AAPL", "momentum"],
                    created_at=1000.0, updated_at=1000.0),
        _FakeLesson(id="l2", user_id=user_id, trader_id="t2",
                    text="TSLA is choppy near key support", tags=["TSLA", "support"],
                    created_at=900.0, updated_at=900.0),
    ]
    app.state.memory = _FakeMemoryStore(lessons)
    return client


def test_memory_with_store_returns_lessons(client_with_store: TestClient) -> None:
    r = client_with_store.get("/api/memory")
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "memory"
    assert data["total"] == 2
    assert len(data["lessons"]) == 2


def test_memory_lesson_shape(client_with_store: TestClient) -> None:
    """Each lesson must carry the expected fields."""
    data = client_with_store.get("/api/memory").json()
    lesson = data["lessons"][0]
    for key in ("id", "trader_id", "text", "tags", "status", "created_at",
                "updated_at", "score"):
        assert key in lesson, f"Lesson missing field: {key}"


def test_memory_filter_by_trader(client_with_store: TestClient) -> None:
    data = client_with_store.get("/api/memory?trader_id=t1").json()
    assert all(l["trader_id"] == "t1" for l in data["lessons"])
    assert data["total"] == 1


def test_memory_semantic_search(client_with_store: TestClient) -> None:
    """When q= + trader_id= given, recall() is invoked."""
    data = client_with_store.get("/api/memory?q=AAPL&trader_id=t1").json()
    assert data["source"] == "memory"
    # At least one lesson contains AAPL in text or tags.
    assert any("AAPL" in l["text"] or "AAPL" in l["tags"] for l in data["lessons"])


def test_memory_k_cap(client_with_store: TestClient) -> None:
    """k= caps the number of returned lessons."""
    data = client_with_store.get("/api/memory?k=1").json()
    assert len(data["lessons"]) <= 1
