"""Tests for the /api/admin/* sysadmin back-office router.

Covers: user provisioning (list/create/reset-password), per-user endpoint
assignment (create/update/delete), per-user settings read/write, system info
endpoint, and storage info endpoint. Auth follows the same session cookie /
Bearer token contract as every other cockpit route.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.web.app import create_cockpit_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Any, monkeypatch: Any) -> TestClient:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return TestClient(create_cockpit_app(Database(tmp_path / "config.db")))


def _signup_and_token(client: TestClient, username: str = "admin", password: str = "pass123") -> str:
    """Sign up a user, return the session token."""
    r = client.post("/api/auth/signup", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------


def test_list_users_unauthenticated(client: TestClient) -> None:
    r = client.get("/api/admin/users")
    assert r.status_code == 401


def test_list_users_empty_then_one(client: TestClient) -> None:
    token = _signup_and_token(client, "alice", "secret")
    r = client.get("/api/admin/users", headers=_auth_headers(token))
    assert r.status_code == 200
    users = r.json()
    assert isinstance(users, list)
    assert len(users) == 1
    assert users[0]["username"] == "alice"
    assert "id" in users[0]
    assert "created_at" in users[0]
    assert isinstance(users[0]["endpoint_count"], int)


def test_create_user_via_admin(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    r = client.post(
        "/api/admin/users",
        json={"username": "bob", "password": "bobpass"},
        headers=_auth_headers(token),
    )
    assert r.status_code == 201
    body = r.json()
    assert body["username"] == "bob"
    assert "id" in body

    # Should appear in the user list now
    r2 = client.get("/api/admin/users", headers=_auth_headers(token))
    usernames = [u["username"] for u in r2.json()]
    assert "bob" in usernames


def test_create_user_duplicate_returns_409(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    client.post("/api/admin/users", json={"username": "dup", "password": "pass"}, headers=_auth_headers(token))
    r = client.post(
        "/api/admin/users",
        json={"username": "dup", "password": "pass2"},
        headers=_auth_headers(token),
    )
    assert r.status_code == 409


def test_reset_password(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    # Create a second user
    r = client.post("/api/admin/users", json={"username": "charlie", "password": "old"}, headers=_auth_headers(token))
    uid = r.json()["id"]

    # Reset password
    r2 = client.post(
        f"/api/admin/users/{uid}/reset-password",
        json={"new_password": "newpass"},
        headers=_auth_headers(token),
    )
    assert r2.status_code == 200
    assert r2.json()["status"] == "ok"

    # Old sessions invalidated — charlie can now log in with new password
    r3 = client.post("/api/auth/login", json={"username": "charlie", "password": "newpass"})
    assert r3.status_code == 200

    # Old password no longer works
    r4 = client.post("/api/auth/login", json={"username": "charlie", "password": "old"})
    assert r4.status_code == 401


def test_reset_password_unknown_user_404(client: TestClient) -> None:
    token = _signup_and_token(client)
    r = client.post(
        "/api/admin/users/doesnotexist/reset-password",
        json={"new_password": "x"},
        headers=_auth_headers(token),
    )
    assert r.status_code == 404


def test_reset_password_empty_new_password(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    r_create = client.post("/api/admin/users", json={"username": "dan", "password": "pass"}, headers=_auth_headers(token))
    uid = r_create.json()["id"]
    r = client.post(
        f"/api/admin/users/{uid}/reset-password",
        json={"new_password": ""},
        headers=_auth_headers(token),
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Per-user endpoint management
# ---------------------------------------------------------------------------


def test_list_user_endpoints_empty(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    r = client.get("/api/admin/users", headers=_auth_headers(token))
    uid = r.json()[0]["id"]

    r2 = client.get(f"/api/admin/users/{uid}/endpoints", headers=_auth_headers(token))
    assert r2.status_code == 200
    assert r2.json() == []


def test_create_and_list_endpoint(client: TestClient, monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    token = _signup_and_token(client, "admin", "admin123")
    r = client.get("/api/admin/users", headers=_auth_headers(token))
    uid = r.json()[0]["id"]

    r2 = client.post(
        f"/api/admin/users/{uid}/endpoints",
        json={"type": "openrouter", "name": "Admin-seeded OR", "api_key": "sk-test1234", "enabled": True},
        headers=_auth_headers(token),
    )
    assert r2.status_code == 201
    body = r2.json()
    assert body["name"] == "Admin-seeded OR"
    assert body["type"] == "openrouter"
    assert body["has_key"] is True
    # Secret key is never returned raw
    assert "api_key" not in body
    assert "sk-test1234" not in str(body)
    # key_preview shows last 4 chars
    assert "1234" in body["key_preview"]

    r3 = client.get(f"/api/admin/users/{uid}/endpoints", headers=_auth_headers(token))
    assert len(r3.json()) == 1


def test_update_endpoint(client: TestClient, monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    token = _signup_and_token(client, "admin", "admin123")
    r_users = client.get("/api/admin/users", headers=_auth_headers(token))
    uid = r_users.json()[0]["id"]

    r_create = client.post(
        f"/api/admin/users/{uid}/endpoints",
        json={"type": "openrouter", "name": "Old name", "enabled": True},
        headers=_auth_headers(token),
    )
    eid = r_create.json()["id"]

    r_update = client.put(
        f"/api/admin/users/{uid}/endpoints/{eid}",
        json={"type": "openrouter", "name": "New name", "enabled": False},
        headers=_auth_headers(token),
    )
    assert r_update.status_code == 200
    assert r_update.json()["name"] == "New name"
    assert r_update.json()["enabled"] is False


def test_delete_endpoint(client: TestClient, monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    token = _signup_and_token(client, "admin", "admin123")
    r_users = client.get("/api/admin/users", headers=_auth_headers(token))
    uid = r_users.json()[0]["id"]

    r_create = client.post(
        f"/api/admin/users/{uid}/endpoints",
        json={"type": "openrouter", "name": "To delete", "enabled": True},
        headers=_auth_headers(token),
    )
    eid = r_create.json()["id"]

    r_del = client.delete(f"/api/admin/users/{uid}/endpoints/{eid}", headers=_auth_headers(token))
    assert r_del.status_code == 200
    assert r_del.json()["status"] == "deleted"

    r_list = client.get(f"/api/admin/users/{uid}/endpoints", headers=_auth_headers(token))
    assert r_list.json() == []


def test_delete_nonexistent_endpoint_404(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    r_users = client.get("/api/admin/users", headers=_auth_headers(token))
    uid = r_users.json()[0]["id"]
    r = client.delete(f"/api/admin/users/{uid}/endpoints/bad-id", headers=_auth_headers(token))
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Per-user settings
# ---------------------------------------------------------------------------


def test_get_user_settings_returns_defaults(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    r_users = client.get("/api/admin/users", headers=_auth_headers(token))
    uid = r_users.json()[0]["id"]

    r = client.get(f"/api/admin/users/{uid}/settings", headers=_auth_headers(token))
    assert r.status_code == 200
    settings = r.json()
    # These keys come from DEFAULTS
    assert "embed_model" in settings
    assert "daily_usd_ceiling" in settings


def test_update_user_settings(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    r_users = client.get("/api/admin/users", headers=_auth_headers(token))
    uid = r_users.json()[0]["id"]

    r = client.put(
        f"/api/admin/users/{uid}/settings",
        json={"values": {"daily_usd_ceiling": 10.0, "theme": "midnight"}},
        headers=_auth_headers(token),
    )
    assert r.status_code == 200
    updated = r.json()
    assert updated["daily_usd_ceiling"] == 10.0
    assert updated["theme"] == "midnight"

    # Persisted: re-fetch returns updated values
    r2 = client.get(f"/api/admin/users/{uid}/settings", headers=_auth_headers(token))
    assert r2.json()["daily_usd_ceiling"] == 10.0


def test_settings_unknown_user_404(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    r = client.get("/api/admin/users/nope/settings", headers=_auth_headers(token))
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def test_system_info(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    r = client.get("/api/admin/system", headers=_auth_headers(token))
    assert r.status_code == 200
    body = r.json()
    assert body["users"] == 1
    assert "active_sessions" in body
    assert "endpoints" in body
    assert "engine" in body
    assert isinstance(body["engine"]["bench"], bool)
    assert "env" in body
    assert isinstance(body["env"]["OPENROUTER_API_KEY"], bool)


def test_system_info_unauthenticated(client: TestClient) -> None:
    r = client.get("/api/admin/system")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Storage info
# ---------------------------------------------------------------------------


def test_storage_info(client: TestClient) -> None:
    token = _signup_and_token(client, "admin", "admin123")
    r = client.get("/api/admin/storage", headers=_auth_headers(token))
    assert r.status_code == 200
    body = r.json()
    assert "config_db" in body
    assert body["config_db"]["exists"] is True
    assert body["config_db"]["size_bytes"] is not None
    assert "data_dir" in body
    assert "other_dbs" in body
    assert isinstance(body["other_dbs"], list)


def test_storage_info_unauthenticated(client: TestClient) -> None:
    r = client.get("/api/admin/storage")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Admin console wiring markers in cockpit.html
# ---------------------------------------------------------------------------


def test_admin_panel_in_cockpit(client: TestClient) -> None:
    """The cockpit SPA should contain markers for the admin console wiring."""
    html = client.get("/").text
    for marker in (
        "/api/admin/users",
        "/api/admin/system",
        "/api/admin/storage",
        "adminLoad(",
        "adminCreateUser(",
        "adminConsolePcardHTML(",
    ):
        assert marker in html, f"missing admin console wiring marker: {marker}"


def test_new_market_tile_wiring_markers(client: TestClient) -> None:
    """Finance tiles should be wired to live backends with LIVE-badge support."""
    html = client.get("/").text
    for marker in (
        "/api/quotes",        # batch watchlist quotes
        "/api/news",          # news headlines
        "/api/movers",        # market movers
        "/api/symbols",       # server-side ticker search
        "/ws/quotes",         # WebSocket live ticks
        "liveBadgeHTML(",     # LIVE/MOCK badge helper
        "wsqConnect(",        # WebSocket connect helper
        "wsqSubscribe(",      # WebSocket subscribe helper
        "LIVE.quotes",        # quotes LIVE flag
        "LIVE.news",          # news LIVE flag
        "LIVE.movers",        # movers LIVE flag
        "LIVE.symbols",       # symbols LIVE flag
    ):
        assert marker in html, f"missing market tile wiring marker: {marker}"


def test_stale_comment_removed(client: TestClient) -> None:
    """The stale 'Missing backend pieces' comment must no longer claim these as missing."""
    html = client.get("/").text
    # The old comment claimed these endpoints don't exist yet — they do now
    assert "Missing backend pieces (ticker-search API, batch /api/quotes, WS stream" not in html
