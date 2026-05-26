"""Tests for the bench FastAPI app + controller (no real network)."""

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from trading_agent.bench.bench import Bench
from trading_agent.bench.controller import BenchController
from trading_agent.llm.openrouter import OpenRouterClient
from trading_agent.web.bench_app import create_bench_app

_MODELS = {
    "data": [
        {"id": "anthropic/claude-opus-4.7", "name": "Opus 4.7",
         "pricing": {"prompt": "0.000005", "completion": "0.00002"}},
        {"id": "~anthropic/claude-opus-latest", "name": "alias (skipped)"},
        {"id": "some/other-model", "name": "Other"},
    ]
}
_BUY = json.dumps({"decisions": [{"symbol": "AAPL", "action": "BUY", "quantity": 2}],
                   "comment": "buying"})


def _transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=_MODELS)
        return httpx.Response(200, json={
            "model": "anthropic/claude-opus-4.7",
            "choices": [{"message": {"content": _BUY}}],
            "usage": {"total_tokens": 20},
        })
    return httpx.MockTransport(handler)


@pytest.fixture
def client() -> Any:
    oc = OpenRouterClient(api_key="sk-test", transport=_transport())
    bench = Bench(["AAPL"], initial_balance=10_000.0)
    ctl = BenchController(bench, oc, symbols=["AAPL"], cadence_seconds=300)
    c = TestClient(create_bench_app(ctl))
    c.ctl = ctl  # type: ignore[attr-defined]
    c.bench = bench  # type: ignore[attr-defined]
    return c


def test_health_and_index(client: Any) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    assert "Model Bench" in client.get("/").text


def test_models_menu_featured_and_filtered(client: Any) -> None:
    d = client.get("/api/bench/models").json()
    ids_all = {m["id"] for m in d["all"]}
    assert "anthropic/claude-opus-4.7" in ids_all
    assert "~anthropic/claude-opus-latest" not in ids_all  # alias skipped
    assert any(m["id"] == "anthropic/claude-opus-4.7" for m in d["featured"])


def test_add_competitor_then_appears(client: Any) -> None:
    r = client.post("/api/bench/competitors", json={"model": "anthropic/claude-opus-4.7"})
    assert r.status_code == 200 and r.json()["name"] == "anthropic/claude-opus-4.7"
    board = client.get("/api/bench").json()["leaderboard"]
    assert board[0]["name"] == "anthropic/claude-opus-4.7"
    assert board[0]["account_value"] == 10_000.0


def test_duplicate_competitor_409(client: Any) -> None:
    client.post("/api/bench/competitors", json={"model": "anthropic/claude-opus-4.7"})
    r = client.post("/api/bench/competitors", json={"model": "anthropic/claude-opus-4.7"})
    assert r.status_code == 409


def test_remove_competitor(client: Any) -> None:
    client.post("/api/bench/competitors", json={"model": "some/other-model"})
    client.delete("/api/bench/competitors/some%2Fother-model")
    assert client.get("/api/bench").json()["leaderboard"] == []


def test_set_cadence(client: Any) -> None:
    r = client.post("/api/bench/cadence", json={"seconds": 60})
    assert r.json()["cadence_seconds"] == 60
    assert client.get("/api/bench").json()["status"]["cadence_seconds"] == 60


def test_start_stop_status(client: Any) -> None:
    client.post("/api/bench/start")
    assert client.get("/api/bench").json()["status"]["running"] is True
    client.post("/api/bench/stop")
    assert client.get("/api/bench").json()["status"]["running"] is False


def test_tick_runs_a_decision_round(client: Any) -> None:
    client.post("/api/bench/competitors", json={"model": "anthropic/claude-opus-4.7"})
    client.bench.observe_bar({"symbol": "AAPL", "close": 100.0})  # set a price so it can fill
    r = client.post("/api/bench/tick")
    assert r.status_code == 200
    snap = client.get("/api/bench").json()
    assert snap["leaderboard"][0]["trades"] == 1  # the mocked BUY filled
    assert snap["recent_decisions"][0]["status"] == "filled"
