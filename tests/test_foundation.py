"""WS-0 Foundation tests: db bootstrap, auth/sessions, settings, endpoint
registry (mocked clients), and the FastAPI cockpit app (config real, rest 501,
per-user isolation). No network or live keys — model clients use MockTransport.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from trading_agent.config import users as users_mod
from trading_agent.config.db import Database, bootstrap, connect
from trading_agent.config.endpoints import (
    AnthropicClient,
    EndpointError,
    EndpointRegistry,
    ModelRef,
    OpenAICompatibleClient,
)
from trading_agent.config.settings_store import DEFAULTS, SettingsStore
from trading_agent.web.app import create_cockpit_app

# --- transport that fakes both OpenAI and Anthropic wire formats -------------


def _mock_transport(captured: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        captured.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": body,
            }
        )
        if request.url.path.endswith("/messages"):  # Anthropic
            return httpx.Response(
                200,
                json={
                    "model": body.get("model", "claude"),
                    "content": [{"type": "text", "text": "anthropic-reply"}],
                    "usage": {"input_tokens": 3, "output_tokens": 5},
                },
            )
        return httpx.Response(  # OpenAI-compatible
            200,
            json={
                "model": body.get("model", "x"),
                "choices": [{"message": {"content": "openai-reply"}}],
                "usage": {"total_tokens": 9, "cost": 0.002},
            },
        )

    return httpx.MockTransport(handler)


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Any) -> Database:
    return Database(tmp_path / "config.db")


@pytest.fixture
def captured() -> list[dict[str, Any]]:
    return []


@pytest.fixture
def client(tmp_path: Any, captured: list[dict[str, Any]], monkeypatch: Any) -> TestClient:
    # No env key → signup won't auto-seed, keeping endpoint tests deterministic.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    app = create_cockpit_app(Database(tmp_path / "config.db"), transport=_mock_transport(captured))
    return TestClient(app)


# --- db ----------------------------------------------------------------------


def test_bootstrap_idempotent_and_tables_exist(tmp_path: Any) -> None:
    path = tmp_path / "c.db"
    conn = connect(path)
    bootstrap(conn)
    bootstrap(conn)  # second run must not raise
    names = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    for table in (
        "users",
        "sessions",
        "user_settings",
        "endpoints",
        "sources",
        "conversations",
        "turns",
        "notes",
        "stock_requests",
    ):
        assert table in names
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


# --- users / auth / sessions -------------------------------------------------


def test_password_hash_roundtrip() -> None:
    h = users_mod.hash_password("hunter2")
    assert h.startswith("scrypt$")
    assert users_mod.verify_hash("hunter2", h)
    assert not users_mod.verify_hash("wrong", h)


def test_create_user_and_authenticate(db: Database) -> None:
    user = users_mod.create_user(db, "alice", "pw")
    assert users_mod.get_user(db, user.id).username == "alice"
    assert users_mod.authenticate(db, "alice", "pw").id == user.id
    with pytest.raises(users_mod.AuthError):
        users_mod.authenticate(db, "alice", "nope")


def test_duplicate_username_rejected(db: Database) -> None:
    users_mod.create_user(db, "bob", "pw")
    with pytest.raises(users_mod.AuthError):
        users_mod.create_user(db, "bob", "pw2")


def test_session_lifecycle(db: Database) -> None:
    user = users_mod.create_user(db, "carol", "pw")
    token = users_mod.create_session(db, user.id)
    assert users_mod.resolve_session(db, token) == user.id
    assert users_mod.resolve_session(db, "bogus") is None
    users_mod.delete_session(db, token)
    assert users_mod.resolve_session(db, token) is None


def test_session_expiry(db: Database) -> None:
    user = users_mod.create_user(db, "dave", "pw")
    token = users_mod.create_session(db, user.id, ttl=-1)  # already expired
    assert users_mod.resolve_session(db, token) is None


# --- settings ----------------------------------------------------------------


def test_settings_roundtrip_and_defaults(db: Database) -> None:
    store = SettingsStore(db)
    assert store.get("u1", "theme") == DEFAULTS["theme"]
    store.set("u1", "theme", "light")
    store.set("u1", "risk_limits", {"dailyLoss": 1000})
    assert store.get("u1", "theme") == "light"
    assert store.get("u1", "risk_limits") == {"dailyLoss": 1000}
    allv = store.all("u1")
    assert allv["theme"] == "light"
    assert allv["vstore"] == DEFAULTS["vstore"]  # default merged in


def test_settings_isolated_per_user(db: Database) -> None:
    store = SettingsStore(db)
    store.set("u1", "theme", "light")
    store.set("u2", "theme", "noir")
    assert store.get("u1", "theme") == "light"
    assert store.get("u2", "theme") == "noir"


# --- endpoint registry -------------------------------------------------------


def test_endpoint_crud_and_isolation(db: Database) -> None:
    reg = EndpointRegistry(db)
    ep = reg.add("u1", "openrouter", "OR", api_key="sk-or-secret")
    assert reg.get("u1", ep.id).name == "OR"
    assert reg.get("u2", ep.id) is None  # other user can't see it
    assert [e.id for e in reg.list("u2")] == []
    reg.update("u1", ep.id, name="OR2")
    reg.toggle("u1", ep.id, False)
    refreshed = reg.get("u1", ep.id)
    assert refreshed.name == "OR2" and refreshed.enabled is False
    # public view masks the key
    pub = refreshed.public()
    assert "secret" not in json.dumps(pub) and pub["key_preview"].endswith("cret")
    assert reg.remove("u1", ep.id) is True
    assert reg.get("u1", ep.id) is None


def test_endpoint_unknown_type_rejected(db: Database) -> None:
    reg = EndpointRegistry(db)
    with pytest.raises(EndpointError):
        reg.add("u1", "bogus", "X")


def test_client_for_openrouter_uses_base_url_and_zdr(
    db: Database, captured: list[dict[str, Any]]
) -> None:
    reg = EndpointRegistry(db, transport=_mock_transport(captured))
    ep = reg.add("u1", "openrouter", "OR", api_key="sk-or-key")
    res = reg.chat("u1", ModelRef(ep.id, "some/model"), [{"role": "user", "content": "hi"}])
    assert res.content == "openai-reply"
    assert res.cost == 0.002
    call = captured[-1]
    assert call["url"].endswith("/api/v1/chat/completions")
    assert call["headers"]["authorization"] == "Bearer sk-or-key"
    assert call["body"]["provider"] == {"data_collection": "deny"}  # ZDR on for openrouter


def test_client_for_local_openai_compatible(db: Database, captured: list[dict[str, Any]]) -> None:
    reg = EndpointRegistry(db, transport=_mock_transport(captured))
    ep = reg.add("u1", "local", "Pi-Ollama", base_url="http://10.0.0.26:11434/v1")
    client = reg.client_for("u1", ep.id)
    assert isinstance(client, OpenAICompatibleClient)
    with client:
        res = client.chat("llama4", [{"role": "user", "content": "hi"}])
    assert res.content == "openai-reply"
    call = captured[-1]
    assert call["url"].startswith("http://10.0.0.26:11434/v1/chat/completions")
    assert "provider" not in call["body"]  # ZDR only for openrouter
    assert "authorization" not in {k.lower() for k in call["headers"]}  # no key → no auth header


def test_client_for_anthropic_wire_format(db: Database, captured: list[dict[str, Any]]) -> None:
    reg = EndpointRegistry(db, transport=_mock_transport(captured))
    ep = reg.add("u1", "anthropic", "Claude", api_key="sk-ant-key")
    client = reg.client_for("u1", ep.id)
    assert isinstance(client, AnthropicClient)
    with client:
        res = client.chat(
            "claude-opus-4.7",
            [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
        )
    assert res.content == "anthropic-reply"
    call = captured[-1]
    assert call["url"].endswith("/v1/messages")
    assert call["headers"]["x-api-key"] == "sk-ant-key"
    assert call["headers"]["anthropic-version"] == AnthropicClient.ANTHROPIC_VERSION
    assert call["body"]["system"] == "be terse"  # system pulled out of messages
    assert call["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_refuses_disabled_endpoint(db: Database, captured: list[dict[str, Any]]) -> None:
    reg = EndpointRegistry(db, transport=_mock_transport(captured))
    ep = reg.add("u1", "openrouter", "OR", api_key="k", enabled=False)
    with pytest.raises(EndpointError):
        reg.chat("u1", ModelRef(ep.id, "m"), [{"role": "user", "content": "hi"}])


def test_seed_defaults_from_env(db: Database, monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    reg = EndpointRegistry(db)
    reg.seed_defaults("u1")
    reg.seed_defaults("u1")  # idempotent
    eps = reg.list("u1")
    assert len(eps) == 1 and eps[0].type == "openrouter" and eps[0].api_key == "sk-or-env"


def test_seed_defaults_noop_without_env(db: Database, monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reg = EndpointRegistry(db)
    reg.seed_defaults("u1")
    assert reg.list("u1") == []


# --- HTTP: auth flow ---------------------------------------------------------


def test_signup_login_me_flow(client: TestClient) -> None:
    assert client.get("/api/me").status_code == 401  # unauthenticated
    r = client.post("/api/auth/signup", json={"username": "ada", "password": "pw"})
    assert r.status_code == 200
    uid = r.json()["user_id"]
    me = client.get("/api/me")
    assert me.status_code == 200 and me.json()["user_id"] == uid
    # logout drops the session
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/me").status_code == 401
    # log back in
    r2 = client.post("/api/auth/login", json={"username": "ada", "password": "pw"})
    assert r2.status_code == 200 and r2.json()["user_id"] == uid


def test_signup_duplicate_is_409(client: TestClient) -> None:
    client.post("/api/auth/signup", json={"username": "dup", "password": "pw"})
    r = client.post("/api/auth/signup", json={"username": "dup", "password": "pw"})
    assert r.status_code == 409


def test_login_bad_password_is_401(client: TestClient) -> None:
    client.post("/api/auth/signup", json={"username": "eve", "password": "pw"})
    r = client.post("/api/auth/login", json={"username": "eve", "password": "WRONG"})
    assert r.status_code == 401


# --- HTTP: settings / endpoints / sources ------------------------------------


def test_settings_http_roundtrip(client: TestClient) -> None:
    client.post("/api/auth/signup", json={"username": "fae", "password": "pw"})
    assert client.get("/api/settings").json()["theme"] == DEFAULTS["theme"]
    put = client.put("/api/settings", json={"theme": "light", "daily_usd_ceiling": 2.5})
    assert put.status_code == 200 and put.json()["theme"] == "light"
    assert client.get("/api/settings").json()["daily_usd_ceiling"] == 2.5


def test_endpoints_http_crud_masks_key(client: TestClient) -> None:
    client.post("/api/auth/signup", json={"username": "gus", "password": "pw"})
    created = client.post(
        "/api/endpoints",
        json={"type": "openrouter", "name": "OR", "api_key": "sk-or-supersecret"},
    )
    assert created.status_code == 200
    eid = created.json()["id"]
    listed = client.get("/api/endpoints").json()
    assert len(listed) == 1
    assert "supersecret" not in json.dumps(listed)  # never leak the key
    assert listed[0]["key_preview"].endswith("cret")
    # update via id
    client.post("/api/endpoints", json={"id": eid, "type": "openrouter", "name": "OR-2"})
    assert client.get("/api/endpoints").json()[0]["name"] == "OR-2"
    assert client.delete(f"/api/endpoints/{eid}").status_code == 200
    assert client.get("/api/endpoints").json() == []
    assert client.delete(f"/api/endpoints/{eid}").status_code == 404


def test_sources_http_crud(client: TestClient) -> None:
    client.post("/api/auth/signup", json={"username": "han", "password": "pw"})
    created = client.post(
        "/api/sources",
        json={"kind": "rss", "name": "Reuters", "config": {"url": "http://x/feed"}},
    )
    assert created.status_code == 200
    sid = created.json()["id"]
    assert created.json()["config"]["url"] == "http://x/feed"
    assert len(client.get("/api/sources").json()) == 1
    assert client.delete(f"/api/sources/{sid}").status_code == 200
    assert client.get("/api/sources").json() == []


def test_per_user_isolation_over_http(client: TestClient) -> None:
    app = client.app
    c1, c2 = TestClient(app), TestClient(app)
    c1.post("/api/auth/signup", json={"username": "iris", "password": "pw"})
    c2.post("/api/auth/signup", json={"username": "jane", "password": "pw"})
    c1.put("/api/settings", json={"theme": "light"})
    c1.post("/api/endpoints", json={"type": "openai", "name": "mine"})
    assert c1.get("/api/settings").json()["theme"] == "light"
    assert c2.get("/api/settings").json()["theme"] == DEFAULTS["theme"]  # untouched
    assert len(c1.get("/api/endpoints").json()) == 1
    assert c2.get("/api/endpoints").json() == []  # can't see u1's endpoint


# --- HTTP: stub routers ------------------------------------------------------
#
# There are no 501 stub routes left. Every former stub is now implemented:
#   /api/accounts · /api/leaderboard · /api/positions · /api/activity ·
#   /api/risk · /api/risk/limits · /api/risk/kill · /api/approvals* — WS-I engine
#     wiring (see tests/test_engine_wiring.py);
#   /api/research* — WS-C (tests/test_research.py);
#   /api/chat[s]* — WS-E (tests/test_manager.py);
#   notifications/requests/notes — WS-H (tests/test_requests_notes.py).


def test_health_open_without_auth(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_root_serves_placeholder_when_no_cockpit(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "cockpit" in r.text.lower()


def test_created_at_is_recent(db: Database) -> None:
    user = users_mod.create_user(db, "leo", "pw")
    assert abs(user.created_at - time.time()) < 5
