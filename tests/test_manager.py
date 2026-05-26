"""WS-E Manager tests: conversation persistence, the overseer agent (grounded
context, history, cost-gating, flags, and the no-trading guarantee), and the
manager router over HTTP. No network — the model client uses MockTransport.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.config.endpoints import ModelRef
from trading_agent.config.settings_store import SettingsStore
from trading_agent.llm.openrouter import ChatResult
from trading_agent.manager.agent import (
    DEFAULT_MANAGER_MODEL,
    ManagerAgent,
    ManagerConfigError,
    resolve_manager_ref,
)
from trading_agent.manager.chat import ConversationStore
from trading_agent.memory.reflect import CostGateError
from trading_agent.web.app import create_cockpit_app

# --- fakes -------------------------------------------------------------------


SNAP: dict[str, Any] = {
    "generated_at": "2026-05-26T10:00:00",
    "leaderboard": [
        {
            "rank": 1,
            "name": "opus",
            "model": "anthropic/claude-opus-4.7",
            "account_value": 104_820.0,
            "cash": 50_000.0,
            "pnl": 4_820.0,
            "return_pct": 4.82,
            "trades": 3,
            "decisions": 5,
            "last_comment": "bought AAPL on a dip",
            "error": None,
        },
        {
            "rank": 2,
            "name": "gemini",
            "model": "google/gemini-3.5-flash",
            "account_value": 96_780.0,
            "cash": 40_000.0,
            "pnl": -3_220.0,
            "return_pct": -3.22,
            "trades": 2,
            "decisions": 4,
            "last_comment": "chased TSLA",
            "error": None,
        },
    ],
    "recent_decisions": [
        {
            "timestamp": "2026-05-26T09:59:00",
            "competitor": "opus",
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 10,
            "status": "filled",
            "reason": "dip",
        },
        {
            "timestamp": "2026-05-26T09:58:00",
            "competitor": "gemini",
            "symbol": "TSLA",
            "action": "BUY",
            "quantity": 5,
            "status": "blocked",
            "reason": "size",
            "detail": "exceeds max position size",
        },
    ],
}


class FakeBench:
    def __init__(self, snap: dict[str, Any] | None = None) -> None:
        self._snap = snap if snap is not None else SNAP
        self.calls = 0

    def snapshot(self) -> dict[str, Any]:
        self.calls += 1
        return self._snap


class FakeRegistry:
    """Stand-in for EndpointRegistry.chat: records calls, returns a canned reply."""

    def __init__(self, reply: str = "grounded reply", cost: float | None = 0.002) -> None:
        self.calls: list[dict[str, Any]] = []
        self._reply = reply
        self._cost = cost

    def chat(
        self, user_id: str, ref: ModelRef, messages: list[dict[str, str]], **opts: Any
    ) -> ChatResult:
        self.calls.append(
            {"user_id": user_id, "ref": ref, "messages": messages, "opts": opts}
        )
        return ChatResult(content=self._reply, model=ref.model, usage={}, cost=self._cost)


@dataclass
class FakeBrief:
    ticker: str
    summary: str
    sentiment: float
    catalysts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    ts: str = ""


class FakeResearch:
    def __init__(self, briefs: list[FakeBrief]) -> None:
        self._briefs = briefs

    def recent(self, user_id: str, n: int) -> list[FakeBrief]:
        return self._briefs[:n]


@dataclass
class FakeLesson:
    text: str
    trader_id: str
    score: float = 0.9


class FakeMemory:
    def __init__(self, by_trader: dict[str, list[FakeLesson]]) -> None:
        self._by_trader = by_trader

    def recall(self, user_id: str, trader_id: str, query: str, k: int) -> list[FakeLesson]:
        return self._by_trader.get(trader_id, [])[:k]


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Any) -> Database:
    return Database(tmp_path / "config.db")


@pytest.fixture
def store(db: Database) -> ConversationStore:
    return ConversationStore(db)


def _mock_transport(captured: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({"body": json.loads(request.content) if request.content else {}})
        return httpx.Response(
            200,
            json={
                "model": "x",
                "choices": [{"message": {"content": "openai-reply"}}],
                "usage": {"total_tokens": 9, "cost": 0.001},
            },
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def captured() -> list[dict[str, Any]]:
    return []


@pytest.fixture
def http(tmp_path: Any, captured: list[dict[str, Any]], monkeypatch: Any) -> TestClient:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    app = create_cockpit_app(Database(tmp_path / "c.db"), transport=_mock_transport(captured))
    app.state.bench = FakeBench()  # wire the live-snapshot source the manager reads
    return TestClient(app)


def _signup_with_endpoint(client: TestClient, username: str = "ada") -> None:
    client.post("/api/auth/signup", json={"username": username, "password": "pw"})
    client.post("/api/endpoints", json={"type": "openrouter", "name": "OR", "api_key": "k"})


# --- ConversationStore -------------------------------------------------------


def test_create_and_get_roundtrip(store: ConversationStore) -> None:
    conv = store.create("u1")
    assert conv.title is None and not conv.saved
    store.add_turn(conv.id, "user", "hello")
    store.add_turn(conv.id, "assistant", "hi there")
    fetched = store.get("u1", conv.id)
    assert fetched is not None
    assert [t.content for t in fetched.turns] == ["hello", "hi there"]


def test_get_or_create_unknown_id_makes_fresh(store: ConversationStore) -> None:
    conv = store.get_or_create("u1", "does-not-exist")
    assert conv.id != "does-not-exist"
    assert store.get("u1", conv.id) is not None


def test_history_messages_filters_and_orders(store: ConversationStore) -> None:
    conv = store.create("u1")
    store.add_turn(conv.id, "user", "first")
    store.add_turn(conv.id, "assistant", "reply")
    store.add_turn(conv.id, "user", "second")
    msgs = store.history_messages(conv.id)
    assert msgs == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]


def test_save_derives_title_from_first_user_turn(store: ConversationStore) -> None:
    conv = store.create("u1")
    store.add_turn(conv.id, "user", "Why is gemini down today?")
    saved = store.save("u1", conv.id)
    assert saved.title == "Why is gemini down today?"
    assert [c.id for c in store.list_saved("u1")] == [conv.id]


def test_list_saved_excludes_untitled(store: ConversationStore) -> None:
    titled = store.create("u1")
    store.add_turn(titled.id, "user", "keep me")
    store.save("u1", titled.id, "Kept")
    store.create("u1")  # untitled — must not appear
    saved = store.list_saved("u1")
    assert [c.title for c in saved] == ["Kept"]


def test_delete_removes_conversation_and_turns(store: ConversationStore) -> None:
    conv = store.create("u1")
    store.add_turn(conv.id, "user", "x")
    assert store.delete("u1", conv.id) is True
    assert store.get("u1", conv.id) is None
    assert store.turns(conv.id) == []  # turns gone too
    assert store.delete("u1", conv.id) is False  # idempotent / not found


def test_conversations_isolated_per_user(store: ConversationStore) -> None:
    conv = store.create("u1")
    store.add_turn(conv.id, "user", "mine")
    store.save("u1", conv.id, "U1 chat")
    assert store.get("u2", conv.id) is None  # other user can't read it
    assert store.delete("u2", conv.id) is False  # nor delete it
    assert store.list_saved("u2") == []


# --- ManagerAgent: context + chat -------------------------------------------


def test_chat_grounds_in_bench_snapshot(db: Database) -> None:
    reg = FakeRegistry()
    settings = SettingsStore(db)
    store = ConversationStore(db)
    agent = ManagerAgent(reg, settings, store, bench=FakeBench())
    conv = store.create("u1")

    reply = agent.chat("u1", conv.id, "How are the books doing?", ModelRef("e", "m"))

    assert reply == "grounded reply"
    assert len(reg.calls) == 1  # exactly one model call per message
    system = reg.calls[0]["messages"][0]
    assert system["role"] == "system"
    # the snapshot's books and a recent decision made it into the prompt
    assert "opus" in system["content"] and "gemini" in system["content"]
    assert "AAPL" in system["content"]
    # the user's message is the final turn
    assert reg.calls[0]["messages"][-1] == {
        "role": "user",
        "content": "How are the books doing?",
    }


def test_chat_replays_prior_history(db: Database) -> None:
    reg = FakeRegistry()
    store = ConversationStore(db)
    agent = ManagerAgent(reg, SettingsStore(db), store, bench=FakeBench())
    conv = store.create("u1")
    store.add_turn(conv.id, "user", "earlier question")
    store.add_turn(conv.id, "assistant", "earlier answer")

    agent.chat("u1", conv.id, "follow up", ModelRef("e", "m"))

    contents = [m["content"] for m in reg.calls[0]["messages"]]
    assert "earlier question" in contents and "earlier answer" in contents
    assert contents[-1] == "follow up"


def test_chat_works_without_any_context_sources(db: Database) -> None:
    reg = FakeRegistry()
    store = ConversationStore(db)
    agent = ManagerAgent(reg, SettingsStore(db), store)  # no bench/research/memory
    conv = store.create("u1")
    reply = agent.chat("u1", conv.id, "hello", ModelRef("e", "m"))
    assert reply == "grounded reply"
    assert len(reg.calls) == 1


def test_chat_includes_research_and_memory_when_present(db: Database) -> None:
    reg = FakeRegistry()
    store = ConversationStore(db)
    research = FakeResearch(
        [FakeBrief("NVDA", "earnings beat, strong guidance", 0.7, ["Q1 earnings"])]
    )
    memory = FakeMemory({"opus": [FakeLesson("don't chase gaps on TSLA", "opus")]})
    agent = ManagerAgent(
        reg, SettingsStore(db), store, bench=FakeBench(), research=research, memory=memory
    )
    conv = store.create("u1")
    agent.chat("u1", conv.id, "what should I watch?", ModelRef("e", "m"))
    system = reg.calls[0]["messages"][0]["content"]
    assert "NVDA" in system and "earnings beat" in system
    assert "don't chase gaps on TSLA" in system


def test_chat_cost_gated_by_daily_ceiling(db: Database) -> None:
    reg = FakeRegistry()
    settings = SettingsStore(db)
    settings.set("u1", "daily_usd_ceiling", 0.0)  # nothing left to spend
    store = ConversationStore(db)
    agent = ManagerAgent(reg, settings, store, bench=FakeBench())
    conv = store.create("u1")
    with pytest.raises(CostGateError):
        agent.chat("u1", conv.id, "hi", ModelRef("e", "m"))
    assert reg.calls == []  # refused before any model call


def test_chat_records_spend(db: Database) -> None:
    reg = FakeRegistry(cost=0.0031)
    settings = SettingsStore(db)
    store = ConversationStore(db)
    agent = ManagerAgent(reg, settings, store, bench=FakeBench())
    conv = store.create("u1")
    agent.chat("u1", conv.id, "hi", ModelRef("e", "m"))
    spend = settings.get("u1", "__daily_spend__", {})
    assert sum(spend.values()) == pytest.approx(0.0031)


def test_manager_never_trades(db: Database) -> None:
    """The agent has no broker/order path and never invokes one."""
    reg = FakeRegistry()
    store = ConversationStore(db)
    bench = MagicMock()
    bench.snapshot.return_value = SNAP
    agent = ManagerAgent(reg, SettingsStore(db), store, bench=bench)
    conv = store.create("u1")

    agent.chat("u1", conv.id, "should you buy AAPL?", ModelRef("e", "m"))
    agent.flags("u1")

    # the only thing the agent ever asks the bench for is a read-only snapshot
    bench.snapshot.assert_called()
    assert not bench.place_order.called
    assert not bench.buy.called and not bench.sell.called
    # and the agent exposes no order surface of its own
    for attr in ("broker", "place_order", "buy", "sell", "execute"):
        assert not hasattr(agent, attr)


# --- ManagerAgent: flags -----------------------------------------------------


def test_flags_raises_drawdown_error_and_blocked(db: Database) -> None:
    settings = SettingsStore(db)
    settings.set("u1", "manager_flag_drawdown_pct", 3.0)  # gemini at -3.22% trips this
    snap = json.loads(json.dumps(SNAP))
    snap["leaderboard"][0]["error"] = "rate limited"  # opus erroring
    agent = ManagerAgent(FakeRegistry(), settings, ConversationStore(db), bench=FakeBench(snap))

    flags = agent.flags("u1")
    by_id = {f.id: f for f in flags}
    assert "manager:error:opus" in by_id and by_id["manager:error:opus"].severity == "critical"
    assert "manager:drawdown:gemini" in by_id
    assert any(f.id.startswith("manager:blocked:gemini") for f in flags)
    # advisory only — flags carry no actionable proposal
    assert all(not f.actionable for f in flags)


def test_flags_empty_without_bench(db: Database) -> None:
    agent = ManagerAgent(FakeRegistry(), SettingsStore(db), ConversationStore(db))
    assert agent.flags("u1") == []


# --- resolve_manager_ref -----------------------------------------------------


def test_resolve_ref_defaults_to_cheap_model_on_enabled_endpoint(db: Database) -> None:
    from trading_agent.config.endpoints import EndpointRegistry

    reg = EndpointRegistry(db)
    ep = reg.add("u1", "openrouter", "OR", api_key="k")
    ref = resolve_manager_ref(SettingsStore(db), reg, "u1")
    assert ref.endpoint_id == ep.id and ref.model == DEFAULT_MANAGER_MODEL


def test_resolve_ref_honors_pinned_and_slug_settings(db: Database) -> None:
    from trading_agent.config.endpoints import EndpointRegistry

    reg = EndpointRegistry(db)
    ep = reg.add("u1", "openrouter", "OR", api_key="k")
    settings = SettingsStore(db)
    settings.set("u1", "manager_model", "z-ai/glm-5.1")  # bare slug
    assert resolve_manager_ref(settings, reg, "u1").model == "z-ai/glm-5.1"
    settings.set("u1", "manager_model", {"endpoint_id": ep.id, "model": "x-ai/grok-4.3"})
    pinned = resolve_manager_ref(settings, reg, "u1")
    assert pinned.endpoint_id == ep.id and pinned.model == "x-ai/grok-4.3"


def test_resolve_ref_raises_without_enabled_endpoint(db: Database) -> None:
    from trading_agent.config.endpoints import EndpointRegistry

    reg = EndpointRegistry(db)
    reg.add("u1", "openrouter", "OR", api_key="k", enabled=False)  # disabled only
    with pytest.raises(ManagerConfigError):
        resolve_manager_ref(SettingsStore(db), reg, "u1")


# --- HTTP routes -------------------------------------------------------------


def test_http_chat_persists_and_grounds(http: TestClient, captured: list[dict[str, Any]]) -> None:
    _signup_with_endpoint(http)
    r = http.post("/api/chat", json={"message": "rundown please"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reply"] == "openai-reply"
    cid = body["conversation_id"]
    # grounded in the wired bench snapshot
    system = captured[-1]["body"]["messages"][0]["content"]
    assert "opus" in system and "gemini" in system

    # a second message on the same conversation replays history
    http.post("/api/chat", json={"message": "and gemini?", "conversation_id": cid})
    contents = [m["content"] for m in captured[-1]["body"]["messages"]]
    assert "rundown please" in contents and "openai-reply" in contents


def test_http_save_list_delete_cycle(http: TestClient) -> None:
    _signup_with_endpoint(http)
    cid = http.post("/api/chat", json={"message": "hello manager"}).json()["conversation_id"]
    assert http.get("/api/chats").json() == []  # unsaved chats don't list

    saved = http.post("/api/chats", json={"conversation_id": cid})
    assert saved.status_code == 200 and saved.json()["title"] == "hello manager"

    listed = http.get("/api/chats").json()
    assert len(listed) == 1 and listed[0]["id"] == cid
    assert [t["content"] for t in listed[0]["turns"]] == ["hello manager", "openai-reply"]

    assert http.delete(f"/api/chats/{cid}").status_code == 200
    assert http.get("/api/chats").json() == []
    assert http.delete(f"/api/chats/{cid}").status_code == 404


def test_http_chat_requires_endpoint(http: TestClient) -> None:
    http.post("/api/auth/signup", json={"username": "noep", "password": "pw"})  # no endpoint
    r = http.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 400


def test_http_chat_cost_ceiling_returns_429(http: TestClient) -> None:
    _signup_with_endpoint(http)
    http.put("/api/settings", json={"daily_usd_ceiling": 0.0})
    r = http.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 429


def test_http_chats_isolated_per_user(http: TestClient) -> None:
    app = http.app
    c1, c2 = TestClient(app), TestClient(app)
    _signup_with_endpoint(c1, "u1")
    _signup_with_endpoint(c2, "u2")
    cid = c1.post("/api/chat", json={"message": "u1 secret"}).json()["conversation_id"]
    c1.post("/api/chats", json={"conversation_id": cid})
    assert c2.get("/api/chats").json() == []  # u2 can't see u1's chat
    assert c2.delete(f"/api/chats/{cid}").status_code == 404


def test_http_chat_requires_auth(http: TestClient) -> None:
    assert http.post("/api/chat", json={"message": "hi"}).status_code == 401
    assert http.get("/api/chats").status_code == 401
