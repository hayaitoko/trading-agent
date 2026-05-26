"""WS-C tests: ResearchStore, ResearchAgent (cost-gated batched pass), router.

No network, no live keys: the embedder is the deterministic FakeEmbedder and
the model is faked with an ``httpx.MockTransport`` returning canned briefs JSON.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.config.endpoints import EndpointRegistry, ModelRef
from trading_agent.config.settings_store import SettingsStore
from trading_agent.ingest.fetchers.base import RawItem
from trading_agent.ingest.store import IngestStore
from trading_agent.memory.embed import FakeEmbedder
from trading_agent.memory.reflect import SPEND_KEY, CostGate, CostGateError
from trading_agent.memory.vector.sqlite_vec import SqliteVecStore
from trading_agent.research.agent import CURSOR_KEY, MARKET_TICKER, ResearchAgent
from trading_agent.research.store import Brief, ResearchStore, research_collection_for
from trading_agent.web.app import create_cockpit_app

USER = "u1"

# Default canned model reply: one brief per ticker the agent will ask about.
_BRIEFS_JSON = json.dumps(
    {
        "briefs": [
            {"ticker": "AAPL", "summary": "Apple dip looks buyable on volume.",
             "sentiment": 0.4, "catalysts": ["earnings", "buyback"]},
            {"ticker": "NVDA", "summary": "Chip demand keeps NVDA bid; sharp reversals.",
             "sentiment": 0.6, "catalysts": ["AI demand"]},
        ]
    }
)


def _item(ticker: str | None, url: str, text: str = "noteworthy") -> RawItem:
    return RawItem(source_id="s1", text=text, url=url, ts="2026-05-26T00:00:00+00:00", ticker=ticker)


def _chat_transport(captured: list[dict[str, Any]], content: str = _BRIEFS_JSON) -> httpx.MockTransport:
    """OpenAI-compatible transport whose chat reply body is ``content``."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        captured.append({"url": str(request.url), "body": body})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "model": body.get("model", "x"),
                    "choices": [{"message": {"content": content}}],
                    "usage": {"total_tokens": 100, "cost": 0.001},
                },
            )
        return httpx.Response(404, json={"error": "unexpected path in test"})

    return httpx.MockTransport(handler)


# =============================================================================
# ResearchStore
# =============================================================================


@pytest.fixture
def store(tmp_path: Any) -> ResearchStore:
    return ResearchStore(
        Database(tmp_path / "config.db"),
        SqliteVecStore(tmp_path / "memory.db"),
        FakeEmbedder(),
    )


def _brief(ticker: str, summary: str, sources: list[str] | None = None) -> Brief:
    return Brief(
        ticker=ticker, summary=summary, sentiment=0.1, catalysts=["c1"],
        sources=sources or ["http://a"], ts="2026-05-26T00:00:00+00:00",
    )


def test_brief_positional_matches_contract() -> None:
    # CONTRACTS Brief(ticker, summary, sentiment, catalysts, sources, ts)
    b = Brief("AAPL", "s", 0.2, ["x"], ["http://u"], "2026-05-26T00:00:00+00:00")
    assert b.id == "" and b.created_at == 0.0


def test_put_then_recent_and_id_assigned(store: ResearchStore) -> None:
    saved = store.put(USER, _brief("AAPL", "Apple looks fine"))
    assert saved.id and saved.created_at > 0
    recent = store.recent(USER, 10)
    assert [b.ticker for b in recent] == ["AAPL"]
    assert recent[0].summary == "Apple looks fine"
    assert recent[0].catalysts == ["c1"]


def test_get_by_ticker_newest_first(store: ResearchStore) -> None:
    store.put(USER, _brief("AAPL", "first"))
    store.put(USER, _brief("AAPL", "second"))
    store.put(USER, _brief("NVDA", "other"))
    aapl = store.get(USER, "AAPL")
    assert [b.summary for b in aapl] == ["second", "first"]
    assert store.get(USER, "aapl")  # case-insensitive lookup


def test_recent_limit_and_order(store: ResearchStore) -> None:
    for i in range(5):
        store.put(USER, _brief("AAPL", f"s{i}"))
    assert len(store.recent(USER, 3)) == 3
    assert store.recent(USER, 0) == []
    assert store.count(USER) == 5


def test_briefs_are_per_user_isolated(store: ResearchStore) -> None:
    store.put(USER, _brief("AAPL", "u1 only"))
    assert store.recent("u2", 10) == []
    assert store.get("u2", "AAPL") == []


def test_search_is_semantic_and_scoped(store: ResearchStore) -> None:
    store.put(USER, _brief("AAPL", "apple iphone earnings beat expectations"))
    store.put(USER, _brief("XOM", "oil refinery margins compress sharply"))
    hits = store.search(USER, "iphone earnings", k=2)
    assert hits and hits[0].ticker == "AAPL"
    # vector point landed in the shared per-user collection
    assert store._vector is not None  # noqa: SLF001 - test introspection
    assert store._vector.count(research_collection_for(USER)) == 2  # noqa: SLF001


def test_put_without_vector_still_persists_sql(tmp_path: Any) -> None:
    bare = ResearchStore(Database(tmp_path / "c.db"))  # no vector/embedder
    bare.put(USER, _brief("AAPL", "still stored"))
    assert [b.summary for b in bare.recent(USER, 10)] == ["still stored"]
    assert bare.search(USER, "anything") == []  # no vector store → empty, no crash


# =============================================================================
# ResearchAgent
# =============================================================================


def _make_agent(
    tmp_path: Any, captured: list[dict[str, Any]], content: str = _BRIEFS_JSON
) -> tuple[ResearchAgent, ResearchStore, IngestStore, SettingsStore, ModelRef]:
    db = Database(tmp_path / "config.db")
    settings = SettingsStore(db)
    registry = EndpointRegistry(db, transport=_chat_transport(captured, content))
    ep = registry.add(USER, "openrouter", "OR", api_key="sk-test")
    ingest = IngestStore(db)
    store = ResearchStore(db, SqliteVecStore(tmp_path / "memory.db"), FakeEmbedder())
    agent = ResearchAgent(ingest, store, registry, settings)
    return agent, store, ingest, settings, ModelRef(ep.id, "cheap/model")


def test_run_empty_backlog_is_noop_no_spend(tmp_path: Any) -> None:
    captured: list[dict[str, Any]] = []
    agent, store, _ingest, settings, ref = _make_agent(tmp_path, captured)
    assert agent.run(USER, None, ref) == []
    assert captured == []  # no model call
    assert CostGate(settings).spent_today(USER) == 0.0


def test_run_produces_briefs_and_records_spend(tmp_path: Any) -> None:
    captured: list[dict[str, Any]] = []
    agent, store, ingest, settings, ref = _make_agent(tmp_path, captured)
    ingest.append(USER, [_item("AAPL", "http://a"), _item("NVDA", "http://b")])

    briefs = agent.run(USER, None, ref)

    assert {b.ticker for b in briefs} == {"AAPL", "NVDA"}
    assert len(captured) == 1  # exactly one batched call
    # sources come from the ingested items, never the model
    aapl = next(b for b in briefs if b.ticker == "AAPL")
    assert aapl.sources == ["http://a"]
    assert -1.0 <= aapl.sentiment <= 1.0
    # persisted + spend charged from reported usage cost
    assert store.count(USER) == 2
    assert CostGate(settings).spent_today(USER) == pytest.approx(0.001)


def test_full_pass_advances_cursor_then_drains_clean(tmp_path: Any) -> None:
    captured: list[dict[str, Any]] = []
    agent, _store, ingest, settings, ref = _make_agent(tmp_path, captured)
    ingest.append(USER, [_item("AAPL", "http://a")])

    agent.run(USER, None, ref)
    assert float(settings.get(USER, CURSOR_KEY)) > 0.0
    # nothing new since the cursor → no second model call
    assert agent.run(USER, None, ref) == []
    assert len(captured) == 1


def test_targeted_subset_does_not_consume_backlog(tmp_path: Any) -> None:
    captured: list[dict[str, Any]] = []
    agent, _store, ingest, settings, ref = _make_agent(tmp_path, captured)
    ingest.append(USER, [_item("AAPL", "http://a"), _item("NVDA", "http://b")])

    subset = agent.run(USER, ["AAPL"], ref)
    assert {b.ticker for b in subset} == {"AAPL"}  # NVDA dropped from the reply
    assert float(settings.get(USER, CURSOR_KEY, 0.0)) == 0.0  # cursor untouched

    full = agent.run(USER, None, ref)  # backlog still available
    assert {b.ticker for b in full} == {"AAPL", "NVDA"}


def test_untagged_items_fold_into_market_bucket(tmp_path: Any) -> None:
    captured: list[dict[str, Any]] = []
    content = json.dumps(
        {"briefs": [{"ticker": "MARKET", "summary": "Macro steady.",
                     "sentiment": 0.0, "catalysts": ["rates"]}]}
    )
    agent, store, ingest, _settings, ref = _make_agent(tmp_path, captured, content)
    ingest.append(USER, [_item(None, "http://macro")])

    briefs = agent.run(USER, None, ref)
    assert [b.ticker for b in briefs] == [MARKET_TICKER]


def test_cost_gate_refuses_over_ceiling(tmp_path: Any) -> None:
    captured: list[dict[str, Any]] = []
    agent, _store, ingest, settings, ref = _make_agent(tmp_path, captured)
    ingest.append(USER, [_item("AAPL", "http://a")])
    settings.set(USER, "daily_usd_ceiling", 0.0)  # no budget

    with pytest.raises(CostGateError):
        agent.run(USER, None, ref)
    assert captured == []  # gate fires before the paid call
    # spend ledger untouched
    assert settings.get(USER, SPEND_KEY, {}) == {}


def test_non_json_model_reply_yields_no_briefs(tmp_path: Any) -> None:
    captured: list[dict[str, Any]] = []
    agent, store, ingest, _settings, ref = _make_agent(tmp_path, captured, "not json at all")
    ingest.append(USER, [_item("AAPL", "http://a")])
    assert agent.run(USER, None, ref) == []
    assert store.count(USER) == 0


# =============================================================================
# Research router
# =============================================================================


@pytest.fixture
def web(tmp_path: Any, monkeypatch: Any) -> tuple[TestClient, Database, list[dict[str, Any]]]:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("TRADING_AGENT_MEMORY_DB", str(tmp_path / "memory.db"))
    captured: list[dict[str, Any]] = []
    db = Database(tmp_path / "config.db")
    app = create_cockpit_app(db, transport=_chat_transport(captured))
    return TestClient(app), db, captured


def _signup(client: TestClient, name: str = "ada") -> None:
    r = client.post("/api/auth/signup", json={"username": name, "password": "pw"})
    assert r.status_code == 200


def _uid(db: Database) -> str:
    row = db.query_one("SELECT id FROM users LIMIT 1")
    assert row is not None
    return str(row["id"])


def test_research_requires_auth(web: tuple[TestClient, Database, Any]) -> None:
    client, _db, _cap = web
    assert client.get("/api/research").status_code == 401
    assert client.post("/api/research/run").status_code == 401


def test_research_empty_then_run_then_list(web: tuple[TestClient, Database, Any]) -> None:
    client, db, captured = web
    _signup(client)
    uid = _uid(db)
    assert client.get("/api/research").json() == []

    # configure a chat endpoint + research model the way Settings would
    client.post("/api/endpoints", json={"type": "openrouter", "name": "OR", "api_key": "sk"})
    client.put("/api/settings", json={"research_model": "cheap/model"})
    IngestStore(db).append(uid, [_item("AAPL", "http://a"), _item("NVDA", "http://b")])

    run = client.post("/api/research/run")
    assert run.status_code == 200, run.text
    payload = run.json()
    assert payload["ran"] is True and payload["count"] == 2
    assert len(captured) == 1  # one batched paid call

    listed = client.get("/api/research").json()
    assert {b["ticker"] for b in listed} == {"AAPL", "NVDA"}
    # cockpit RESEARCH shape (aliases) + canonical Brief fields both present
    one = listed[0]
    for key in ("who", "topic", "text", "tags", "ticker", "summary", "sentiment", "sources", "ts"):
        assert key in one
    assert one["who"] == "research"
    assert one["topic"] == one["ticker"] and one["text"] == one["summary"]


def test_research_list_filtered_by_ticker(web: tuple[TestClient, Database, Any]) -> None:
    client, db, _cap = web
    _signup(client)
    uid = _uid(db)
    client.post("/api/endpoints", json={"type": "openrouter", "name": "OR", "api_key": "sk"})
    client.put("/api/settings", json={"research_model": "cheap/model"})
    IngestStore(db).append(uid, [_item("AAPL", "http://a"), _item("NVDA", "http://b")])
    client.post("/api/research/run")

    only = client.get("/api/research", params={"ticker": "AAPL"}).json()
    assert {b["ticker"] for b in only} == {"AAPL"}


def test_run_gated_returns_402_over_ceiling(web: tuple[TestClient, Database, Any]) -> None:
    client, db, captured = web
    _signup(client)
    uid = _uid(db)
    client.post("/api/endpoints", json={"type": "openrouter", "name": "OR", "api_key": "sk"})
    client.put("/api/settings", json={"research_model": "cheap/model", "daily_usd_ceiling": 0.0})
    IngestStore(db).append(uid, [_item("AAPL", "http://a")])

    r = client.post("/api/research/run")
    assert r.status_code == 402
    assert captured == []  # refused before spending


def test_run_without_endpoint_is_400(web: tuple[TestClient, Database, Any]) -> None:
    client, db, _cap = web
    _signup(client)
    uid = _uid(db)
    client.put("/api/settings", json={"research_model": "cheap/model"})
    IngestStore(db).append(uid, [_item("AAPL", "http://a")])
    r = client.post("/api/research/run")
    assert r.status_code == 400


def test_run_without_model_is_400(web: tuple[TestClient, Database, Any]) -> None:
    client, db, _cap = web
    _signup(client)
    client.post("/api/endpoints", json={"type": "openrouter", "name": "OR", "api_key": "sk"})
    IngestStore(db).append(_uid(db), [_item("AAPL", "http://a")])
    # research_model defaults to None → no model resolvable
    r = client.post("/api/research/run")
    assert r.status_code == 400
