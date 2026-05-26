"""WS-D Memory tests: vector stores (sqlite-vec + qdrant, same suite), namespaced
MemoryStore isolation, local embedder (mocked, no Ollama), gated reflection
(cap + dedup + cost gate), and Artoo-style hygiene (dedup + staleness).

No live services: vector backends run embedded/in-SQLite, the embedder uses a
deterministic FakeEmbedder, and the local-embedder HTTP path uses MockTransport.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from trading_agent.config.db import Database
from trading_agent.config.endpoints import EndpointRegistry, ModelRef
from trading_agent.config.settings_store import SettingsStore
from trading_agent.memory import (
    BM25,
    CostGate,
    CostGateError,
    EmbedError,
    FakeEmbedder,
    Hygiene,
    LocalEmbedder,
    MemoryStore,
    Reflector,
    collection_for,
    make_vector_store,
)
from trading_agent.memory.vector.base import Hit, cosine_similarity

# --- fixtures ----------------------------------------------------------------


@pytest.fixture(params=["sqlite-vec", "qdrant"])
def store(request: Any, tmp_path: Any) -> Any:
    """Both backends, run against the identical store/memory suite."""
    if request.param == "sqlite-vec":
        return make_vector_store("sqlite-vec", path=str(tmp_path / "mem.db"))
    return make_vector_store("qdrant", location=":memory:")


@pytest.fixture
def memory(store: Any) -> MemoryStore:
    return MemoryStore(store, FakeEmbedder())


# --- VectorStore contract (both backends) ------------------------------------


def test_vectorstore_upsert_search_delete(store: Any) -> None:
    emb = FakeEmbedder()
    store.upsert("c", "1", emb.embed("apple banana cherry"), {"trader_id": "a", "status": "active"})
    store.upsert("c", "2", emb.embed("zebra yak walrus"), {"trader_id": "a", "status": "active"})
    hits = store.search("c", emb.embed("apple banana"), k=5)
    assert hits and hits[0].id == "1"  # closest is the fruit doc
    assert isinstance(hits[0], Hit) and -1.001 <= hits[0].score <= 1.001
    store.delete("c", "1")
    assert store.get("c", "1") is None
    assert {h.id for h in store.search("c", emb.embed("apple"), k=5)} == {"2"}


def test_vectorstore_filter_and_upsert_overwrites(store: Any) -> None:
    emb = FakeEmbedder()
    store.upsert("c", "1", emb.embed("alpha lesson"), {"trader_id": "a", "status": "active"})
    store.upsert("c", "2", emb.embed("beta lesson"), {"trader_id": "b", "status": "active"})
    only_a = store.search("c", emb.embed("lesson"), k=5, flt={"trader_id": "a"})
    assert {h.id for h in only_a} == {"1"}
    # re-upsert id 1 with a new payload -> overwrite, not duplicate
    store.upsert("c", "1", emb.embed("alpha lesson v2"), {"trader_id": "a", "status": "archived"})
    assert store.count("c", {"trader_id": "a"}) == 1
    assert store.get("c", "1").payload["status"] == "archived"


def test_vectorstore_search_empty_and_k_zero(store: Any) -> None:
    emb = FakeEmbedder()
    assert store.search("missing", emb.embed("x"), k=5) == []
    store.upsert("c", "1", emb.embed("x"), {})
    assert store.search("c", emb.embed("x"), k=0) == []


def test_vectorstore_iter_and_set_payload(store: Any) -> None:
    emb = FakeEmbedder()
    store.upsert("c", "1", emb.embed("one"), {"trader_id": "a", "status": "active"})
    store.upsert("c", "2", emb.embed("two"), {"trader_id": "a", "status": "active"})
    assert len(store.iter_points("c")) == 2
    assert {p.id for p in store.iter_points("c", {"status": "active"})} == {"1", "2"}
    store.set_payload("c", "1", {"trader_id": "a", "status": "archived"})
    assert {p.id for p in store.iter_points("c", {"status": "active"})} == {"2"}


# --- MemoryStore namespacing (the whole point) -------------------------------


def test_two_traders_isolated(memory: MemoryStore) -> None:
    memory.remember("u1", "alpha", "keep single-name risk under 0.9 percent", tags=["risk"])
    memory.remember("u1", "beta", "buy breakouts backed by heavy volume", tags=["entry"])
    a = memory.recall("u1", "alpha", "how much should I risk per trade", k=10)
    b = memory.recall("u1", "beta", "how much should I risk per trade", k=10)
    assert a and all(lesson.trader_id == "alpha" for lesson in a)
    assert b and all(lesson.trader_id == "beta" for lesson in b)
    # trader A never sees trader B's lesson, even querying B's own words
    leak = memory.recall("u1", "alpha", "breakouts backed by heavy volume", k=10)
    assert all("breakout" not in lesson.text for lesson in leak)


def test_users_isolated(memory: MemoryStore, store: Any) -> None:
    memory.remember("u1", "alpha", "user one private lesson")
    memory.remember("u2", "alpha", "user two private lesson")
    # different collections -> no cross-user bleed even with same trader_id
    assert collection_for("u1") != collection_for("u2")
    u1 = memory.recall("u1", "alpha", "private lesson", k=10)
    assert all(lesson.user_id == "u1" for lesson in u1)
    assert all("user two" not in lesson.text for lesson in u1)


def test_remember_recall_roundtrip_fields(memory: MemoryStore) -> None:
    lesson = memory.remember("u1", "alpha", "wait for a pullback", tags=["entry", "patience"])
    assert lesson.status == "active" and lesson.created_at > 0
    got = memory.recall("u1", "alpha", "wait for a pullback", k=1)
    assert got[0].id == lesson.id
    assert got[0].tags == ["entry", "patience"]
    assert got[0].score is not None


def test_archive_hides_from_recall_but_recoverable(memory: MemoryStore) -> None:
    lesson = memory.remember("u1", "alpha", "trim into strength")
    assert memory.archive("u1", lesson.id) is True
    assert memory.recall("u1", "alpha", "trim into strength", k=5) == []
    assert memory.list("u1", "alpha") == []  # active only
    assert any(x.id == lesson.id for x in memory.list("u1", "alpha", include_archived=True))
    assert memory.restore("u1", lesson.id) is True
    assert memory.recall("u1", "alpha", "trim into strength", k=5)


def test_archive_missing_returns_false(memory: MemoryStore) -> None:
    assert memory.archive("u1", "nope") is False


# --- gated reflection --------------------------------------------------------


def test_reflection_caps_writes(memory: MemoryStore) -> None:
    r = Reflector(memory, max_writes=2)
    res = r.reflect(
        "u1",
        "alpha",
        ["alpha lesson one", "bravo lesson two", "charlie lesson three", "delta lesson four"],
    )
    assert len(res.written) == 2
    assert any("cap" in s.reason for s in res.skipped)


def test_reflection_dedups_within_batch(memory: MemoryStore) -> None:
    r = Reflector(memory, max_writes=5, dedup_threshold=0.9)
    res = r.reflect("u1", "alpha", ["cut losers fast", "cut losers fast", "let winners run"])
    assert len(res.written) == 2
    assert any("duplicate" in s.reason for s in res.skipped)


def test_reflection_dedups_against_existing(memory: MemoryStore) -> None:
    memory.remember("u1", "alpha", "size positions to about one percent risk")
    r = Reflector(memory, max_writes=5, dedup_threshold=0.9)
    res = r.reflect("u1", "alpha", ["size positions to about one percent risk", "scale out winners"])
    assert len(res.written) == 1 and res.written[0].text == "scale out winners"


def test_reflection_skips_empty(memory: MemoryStore) -> None:
    r = Reflector(memory, max_writes=5)
    res = r.reflect("u1", "alpha", ["  ", "real lesson here"])
    assert len(res.written) == 1
    assert any(s.reason == "empty" for s in res.skipped)


# --- cost gate ---------------------------------------------------------------


def test_cost_gate_blocks_over_ceiling(tmp_path: Any) -> None:
    db = Database(tmp_path / "c.db")
    settings = SettingsStore(db)
    settings.set("u1", "daily_usd_ceiling", 0.10)
    gate = CostGate(settings)
    gate.check("u1", 0.05)  # under ceiling: ok
    gate.record("u1", 0.05)
    gate.record("u1", 0.05)
    assert gate.spent_today("u1") == pytest.approx(0.10)
    with pytest.raises(CostGateError):
        gate.check("u1", 0.01)  # would exceed


def test_distill_is_cost_gated_and_records_spend(tmp_path: Any) -> None:
    db = Database(tmp_path / "c.db")
    settings = SettingsStore(db)
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": json.dumps({"lessons": ["a", "b"]})}}],
                "usage": {"cost": 0.03},
            },
        )

    reg = EndpointRegistry(db, transport=httpx.MockTransport(handler))
    ep = reg.add("u1", "openrouter", "OR", api_key="k")
    store = make_vector_store("sqlite-vec", path=str(tmp_path / "mem.db"))
    memory = MemoryStore(store, FakeEmbedder())
    r = Reflector(memory, settings=settings, registry=reg)

    lessons = r.distill("u1", "alpha", "round log: bought AMD, stopped out", ModelRef(ep.id, "m"))
    assert lessons == ["a", "b"]
    assert captured[-1]["response_format"] == {"type": "json_object"}  # json_mode honored
    assert r.cost_gate.spent_today("u1") == pytest.approx(0.03)

    # now exhaust the budget and confirm the paid path refuses
    settings.set("u1", "daily_usd_ceiling", 0.03)
    with pytest.raises(CostGateError):
        r.distill("u1", "alpha", "another round", ModelRef(ep.id, "m"))


def test_distill_requires_registry(memory: MemoryStore) -> None:
    r = Reflector(memory)  # no settings/registry
    with pytest.raises(RuntimeError):
        r.distill("u1", "alpha", "ctx", ModelRef("e", "m"))


# --- hygiene -----------------------------------------------------------------


def test_hygiene_dedup_semantic(memory: MemoryStore) -> None:
    keep = memory.remember("u1", "alpha", "cut losers quickly and let winners run")
    memory.remember("u1", "alpha", "cut losers quickly and let the winners run")  # near-dup
    memory.remember("u1", "alpha", "watch the fed calendar for volatility")
    hy = Hygiene(memory, semantic_threshold=0.99, bm25_threshold=0.5)
    report = hy.dedup("u1", "alpha")
    assert len(report.deduped) == 1
    archived_id, kept_id = report.deduped[0]
    assert kept_id == keep.id  # earliest kept
    remaining = {lesson.text for lesson in memory.list("u1", "alpha")}
    assert len(remaining) == 2


def test_hygiene_dedup_respects_trader_scope(memory: MemoryStore) -> None:
    memory.remember("u1", "alpha", "identical lesson text here")
    memory.remember("u1", "beta", "identical lesson text here")  # same text, other trader
    hy = Hygiene(memory, semantic_threshold=0.5, bm25_threshold=0.1)
    report = hy.dedup("u1")  # all traders
    assert report.deduped == []  # never dedup across traders
    assert len(memory.list("u1", "alpha")) == 1
    assert len(memory.list("u1", "beta")) == 1


def test_hygiene_sweep_stale_archives_cold(memory: MemoryStore) -> None:
    memory.remember("u1", "alpha", "an old cold lesson")
    memory.remember("u1", "alpha", "another old lesson")
    later = time.time() + 100 * 86400
    report = hygiene_sweep(memory, "u1", later)
    assert len(report.archived_stale) == 2
    assert memory.list("u1", "alpha") == []  # all archived (soft delete)
    assert len(memory.list("u1", "alpha", include_archived=True)) == 2  # not hard-deleted


def hygiene_sweep(memory: MemoryStore, user_id: str, now: float) -> Any:
    return Hygiene(memory).sweep_stale(user_id, max_age_days=90, now=now)


def test_hygiene_run_combines_passes(memory: MemoryStore) -> None:
    memory.remember("u1", "alpha", "fresh lesson about discipline")
    report = Hygiene(memory).run("u1", max_age_days=90)
    assert report.scanned >= 1


def test_bm25_similarity_self_high() -> None:
    docs = [["cut", "losers", "fast"], ["cut", "losers", "fast"], ["buy", "the", "dip"]]
    bm25 = BM25(docs)
    assert bm25.similarity(0, 1, docs) > bm25.similarity(0, 2, docs)


# --- local embedder (mocked HTTP, no Ollama) ---------------------------------


def test_local_embedder_calls_local_endpoint(tmp_path: Any) -> None:
    db = Database(tmp_path / "c.db")
    settings = SettingsStore(db)
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in range(n)]})

    reg = EndpointRegistry(db)
    reg.add("u1", "local", "Ollama", base_url="http://10.0.0.26:11434/v1")
    emb = LocalEmbedder(reg, settings, "u1", transport=httpx.MockTransport(handler))
    vec = emb.embed("hello world")
    assert vec == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert emb.dim == 3
    assert seen[-1]["url"].endswith("/v1/embeddings")
    assert seen[-1]["body"]["model"] == "bge-small-en-v1.5"


def test_local_embedder_refuses_non_local(tmp_path: Any) -> None:
    db = Database(tmp_path / "c.db")
    settings = SettingsStore(db)
    reg = EndpointRegistry(db)
    ep = reg.add("u1", "openrouter", "OR", api_key="k")
    settings.set("u1", "embed_endpoint_id", ep.id)
    emb = LocalEmbedder(reg, settings, "u1")
    with pytest.raises(EmbedError):  # embeddings must stay on-box (no WAN)
        emb.embed("nope")


def test_local_embedder_no_endpoint(tmp_path: Any) -> None:
    db = Database(tmp_path / "c.db")
    settings = SettingsStore(db)
    reg = EndpointRegistry(db)
    emb = LocalEmbedder(reg, settings, "u1")
    with pytest.raises(EmbedError):
        emb.embed("no endpoints at all")


# --- vstore switch is a setting ----------------------------------------------


@pytest.mark.parametrize("vstore", ["sqlite-vec", "qdrant", "unknown-falls-back"])
def test_make_vector_store_switch(vstore: str, tmp_path: Any) -> None:
    kwargs = {"path": str(tmp_path / "m.db"), "location": ":memory:"}
    store = make_vector_store(vstore, **kwargs)
    emb = FakeEmbedder()
    store.upsert("c", "1", emb.embed("hello"), {"k": "v"})
    assert store.search("c", emb.embed("hello"), k=1)[0].id == "1"


def test_cosine_similarity_edges() -> None:
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
