"""WS-A P3: the BenchController post-round reflection write-path.

No network: the reflection distill model is an ``httpx.MockTransport`` returning
per-trader lessons JSON, memory is a real ``MemoryStore`` over a tmp sqlite-vec
store with the deterministic ``FakeEmbedder``. Asserts reflection fires on the
cadence (not below it), a budget exhaustion is swallowed, and one book's lessons
never leak into another's namespace.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from trading_agent.bench.bench import Bench
from trading_agent.bench.controller import BenchController
from trading_agent.config.db import Database
from trading_agent.config.endpoints import EndpointRegistry
from trading_agent.config.settings_store import SettingsStore
from trading_agent.config.users import create_user
from trading_agent.llm.openrouter import OpenRouterClient
from trading_agent.llm.trader import DecisionResult
from trading_agent.memory import FakeEmbedder, MemoryStore, Reflector, make_vector_store

# --- fakes -------------------------------------------------------------------


class _Quiet:
    """A trader that observes and always holds — flat books, no trades."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.model = f"test/{name}"

    def observe(self, bar: dict[str, Any]) -> None:
        pass

    def decide(self, account: dict[str, Any]) -> DecisionResult:
        return DecisionResult(comment="hold")


def _lessons_transport() -> httpx.MockTransport:
    """Chat transport that distills a *trader-specific* lesson from the context,
    so cross-book leakage is detectable."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        context = body.get("messages", [{}])[-1].get("content", "")
        who = "alpha" if "Trader: alpha" in context else "beta"
        lessons = {"lessons": [f"{who}: keep single-name risk tight"]}
        return httpx.Response(
            200,
            json={
                "model": body.get("model", "x"),
                "choices": [{"message": {"content": json.dumps(lessons)}}],
                "usage": {"total_tokens": 40, "cost": 0.0},
            },
        )

    return httpx.MockTransport(handler)


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Any, monkeypatch: Any) -> dict[str, Any]:
    monkeypatch.delenv("TRADING_AGENT_OWNER_ID", raising=False)
    transport = _lessons_transport()
    db = Database(tmp_path / "config.db")
    user = create_user(db, "solo", "pw")  # single user → owner resolves lazily
    registry = EndpointRegistry(db, transport=transport)
    registry.add(user.id, "openrouter", "OR", api_key="k")
    settings = SettingsStore(db)
    memory = MemoryStore(make_vector_store("sqlite-vec", path=str(tmp_path / "mem.db")), FakeEmbedder())
    reflector = Reflector(memory, settings=settings, registry=registry)

    bench = Bench(["AAPL"], initial_balance=100_000.0)
    bench.add_competitor("alpha", _Quiet("alpha"))
    bench.add_competitor("beta", _Quiet("beta"))
    bench.observe_bar({"symbol": "AAPL", "close": 100.0})

    controller = BenchController(
        bench,
        OpenRouterClient(api_key="k", transport=transport),
        symbols=["AAPL"],
        reflector=reflector,
        owner_user_id=user.id,
        db=db,
    )
    return {
        "controller": controller,
        "memory": memory,
        "settings": settings,
        "user_id": user.id,
    }


# --- tests -------------------------------------------------------------------


def test_reflection_writes_at_cadence(env: dict[str, Any]) -> None:
    controller, memory, uid = env["controller"], env["memory"], env["user_id"]
    for _ in range(4):  # default cadence is 4
        controller.tick_now()
    assert memory.list(uid, "alpha"), "alpha should have a lesson after the cadence"
    assert memory.list(uid, "beta"), "beta should have a lesson after the cadence"


def test_no_reflection_below_cadence(env: dict[str, Any]) -> None:
    controller, memory, uid = env["controller"], env["memory"], env["user_id"]
    for _ in range(3):  # one short of the cadence
        controller.tick_now()
    assert memory.list(uid, "alpha") == []
    assert memory.list(uid, "beta") == []


def test_budget_exhaustion_is_swallowed(env: dict[str, Any]) -> None:
    controller, memory, settings, uid = (
        env["controller"], env["memory"], env["settings"], env["user_id"]
    )
    settings.set(uid, "daily_usd_ceiling", 0.0)  # nothing left to spend
    for _ in range(4):
        controller.tick_now()  # must not raise
    assert memory.list(uid, "alpha") == []  # gated before any write


def test_one_books_lessons_never_surface_for_another(env: dict[str, Any]) -> None:
    controller, memory, uid = env["controller"], env["memory"], env["user_id"]
    for _ in range(4):
        controller.tick_now()
    alpha = [lesson.text for lesson in memory.list(uid, "alpha")]
    beta = [lesson.text for lesson in memory.list(uid, "beta")]
    assert any(t.startswith("alpha:") for t in alpha)
    assert any(t.startswith("beta:") for t in beta)
    # strict namespacing: neither book carries the other's lesson
    assert all(not t.startswith("beta:") for t in alpha)
    assert all(not t.startswith("alpha:") for t in beta)
    # and recall (the trader's own read path) is filtered the same way
    recalled = memory.recall(uid, "alpha", "risk", k=5)
    assert recalled and all(r.trader_id == "alpha" for r in recalled)


def test_reflection_dark_without_owner(tmp_path: Any, monkeypatch: Any) -> None:
    """No resolvable owner → the hook is a no-op (no crash, no writes)."""
    monkeypatch.delenv("TRADING_AGENT_OWNER_ID", raising=False)
    transport = _lessons_transport()
    db = Database(tmp_path / "config.db")  # zero users → owner stays None
    settings = SettingsStore(db)
    memory = MemoryStore(make_vector_store("sqlite-vec", path=str(tmp_path / "m.db")), FakeEmbedder())
    reflector = Reflector(memory, settings=settings, registry=EndpointRegistry(db, transport=transport))
    bench = Bench(["AAPL"], initial_balance=100_000.0)
    bench.add_competitor("alpha", _Quiet("alpha"))
    controller = BenchController(
        bench, OpenRouterClient(api_key="k", transport=transport),
        symbols=["AAPL"], reflector=reflector, db=db,
    )
    for _ in range(8):
        controller.tick_now()
    assert memory.list("anything", "alpha") == []
    assert controller._round == 0  # never even counted a round (owner gate first)
