"""WS-I engine-wiring tests: the cockpit's core trading routers over a live
engine attached to ``app.state``.

Covers bench (accounts/leaderboard/positions/activity + add-trader create),
risk (limits + emergency stop), approvals (pending + approve/reject), the
graceful-empty fallback when no engine is attached, auth enforcement, and the
``build_cockpit`` serve factory. No network — the model client uses MockTransport
and is never actually called by these read/write routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from trading_agent.approval_queue import ApprovalQueue
from trading_agent.bench.bench import Bench, DecisionLogEntry
from trading_agent.bench.controller import BenchController
from trading_agent.config.db import Database
from trading_agent.llm.openrouter import OpenRouterClient
from trading_agent.risk_manager import RiskManager
from trading_agent.scripts.serve import build_cockpit
from trading_agent.web.app import create_cockpit_app

# --- fakes -------------------------------------------------------------------


class FakeTrader:
    """Minimal Trader stand-in: carries a ``model`` and absorbs observations.

    ``decide`` is never called by the routers under test, so it can stay trivial.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.name = model

    def observe(self, bar: dict[str, Any]) -> None:  # noqa: D401 - protocol shim
        pass


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "x", "choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    return httpx.MockTransport(handler)


def _live_bench() -> Bench:
    """A bench with two competitors, prices, a filled position, and a logged trade."""
    bench = Bench(["AAPL", "NVDA"], initial_balance=100_000.0, max_position_size=1_000.0)
    opus = bench.add_competitor("opus", FakeTrader("anthropic/claude-opus-4.7"))
    bench.add_competitor("gemini", FakeTrader("google/gemini-3.5-flash"))
    bench.observe_bar({"symbol": "AAPL", "close": 228.40})
    bench.observe_bar({"symbol": "NVDA", "close": 880.10})
    # opus opens a real AAPL position on its isolated paper book
    opus.broker.place_order(
        {"symbol": "AAPL", "side": "BUY", "amount": 10, "order_type": "market"}
    )
    # and logs a decision (as run_decisions would: log entry + decision_count)
    opus.decisions.appendleft(
        DecisionLogEntry(
            timestamp="2026-05-26T09:59:00",
            competitor="opus",
            symbol="AAPL",
            action="BUY",
            quantity=10,
            status="filled",
            reason="oversold bounce",
        )
    )
    opus.decision_count = 1
    return bench


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_kill_env(monkeypatch: Any) -> None:
    # Never let an ambient kill-switch env var leak into the risk tests.
    monkeypatch.delenv("TRADING_AGENT_KILL_SWITCH", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def bench() -> Bench:
    return _live_bench()


@pytest.fixture
def client(tmp_path: Path, bench: Bench) -> TestClient:
    """Cockpit app with a live bench/controller/risk/approvals on app.state."""
    app = create_cockpit_app(Database(tmp_path / "c.db"), transport=_mock_transport())
    app.state.bench = bench
    app.state.bench_controller = BenchController(
        bench, OpenRouterClient(api_key="k", transport=_mock_transport()), symbols=["AAPL", "NVDA"]
    )
    app.state.risk = RiskManager(kill_switch_file=tmp_path / ".kill")
    app.state.approvals = ApprovalQueue(
        db_path=tmp_path / "approvals.db", executor=lambda sig: {"filled": sig}
    )
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "lukas", "password": "pw"})
    return c


@pytest.fixture
def bare_client(tmp_path: Path) -> TestClient:
    """Authed cockpit with NO engine attached — every read route must fall back to []."""
    app = create_cockpit_app(Database(tmp_path / "c.db"), transport=_mock_transport())
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "nobench", "password": "pw"})
    return c


# --- bench: accounts / leaderboard -------------------------------------------


def test_accounts_returns_live_rows(client: TestClient) -> None:
    rows = client.get("/api/accounts").json()
    assert {r["name"] for r in rows} == {"opus", "gemini"}
    opus = next(r for r in rows if r["name"] == "opus")
    # cockpit ACCOUNTS card keys
    for key in ("prov", "value", "ret", "status", "cash", "pos", "trades", "win", "dec", "act", "sym"):
        assert key in opus
    assert opus["prov"] == "anthropic"
    assert opus["pos"] == 1  # the AAPL position
    assert opus["act"] == "BUY" and opus["sym"] == "AAPL"
    assert opus["status"] == "trading"  # has a logged decision


def test_leaderboard_same_shape_and_ranked(client: TestClient) -> None:
    rows = client.get("/api/leaderboard").json()
    assert [r["rank"] for r in rows] == [1, 2]  # ranked, contiguous
    assert all("value" in r and "ret" in r for r in rows)


def test_accounts_empty_without_bench(bare_client: TestClient) -> None:
    # No app.state.bench -> [] so the cockpit keeps its mock fallback.
    for path in ("/api/accounts", "/api/leaderboard", "/api/positions", "/api/activity"):
        assert bare_client.get(path).json() == []


# --- bench: positions --------------------------------------------------------


def test_positions_aggregate_holders(client: TestClient) -> None:
    cards = client.get("/api/positions").json()
    aapl = next(c for c in cards if c["sym"] == "AAPL")
    assert aapl["price"] == pytest.approx(228.40)
    assert aapl["holders"] == [{"acct": "opus", "qty": 10, "avg": pytest.approx(228.40)}]
    for key in ("name", "chg", "pct", "seed", "trend", "notes"):
        assert key in aapl


# --- bench: activity ---------------------------------------------------------


def test_activity_maps_decisions(client: TestClient) -> None:
    log = client.get("/api/activity").json()
    assert log, "expected the logged decision"
    row = log[0]
    assert row["lv"] == "trade"  # filled -> trade
    assert "opus" in row["text"] and "AAPL" in row["text"]
    assert row["ts"] == "09:59:00"  # iso timestamp formatted to a clock


# --- bench: add-trader create ------------------------------------------------


def test_create_trader_adds_competitor(client: TestClient) -> None:
    r = client.post(
        "/api/accounts",
        json={"name": "kimi", "model": "moonshotai/kimi-k2.6", "cash": 100000, "style": "Bold"},
    )
    assert r.status_code == 200
    assert r.json()["created"] == "kimi"
    assert "kimi" in {a["name"] for a in client.get("/api/accounts").json()}


def test_create_trader_duplicate_is_409(client: TestClient) -> None:
    body = {"model": "anthropic/claude-opus-4.7", "name": "opus"}
    assert client.post("/api/accounts", json=body).status_code == 409


def test_create_trader_503_without_controller(bare_client: TestClient) -> None:
    r = bare_client.post("/api/accounts", json={"model": "z-ai/glm-5.1"})
    assert r.status_code == 503


# --- risk --------------------------------------------------------------------


def test_risk_get_reflects_engine(client: TestClient) -> None:
    body = client.get("/api/risk").json()
    assert body["kill"] is False
    assert body["limits"] == {}


def test_risk_limits_persist_and_apply(client: TestClient, tmp_path: Path) -> None:
    limits = {"dailyLoss": 500, "maxPos": 25000, "tradesHour": 4, "openPos": 9, "perTrade": 0.9}
    put = client.put("/api/risk/limits", json=limits)
    assert put.status_code == 200 and put.json()["limits"]["dailyLoss"] == 500
    # persisted to user_settings (read back via the config router)
    assert client.get("/api/settings").json()["risk_limits"]["openPos"] == 9
    # applied to the live RiskManager
    rm = client.app.state.risk
    assert rm.limits.max_daily_loss == 500.0
    assert rm.limits.max_trades_per_hour == 4
    assert rm.limits.max_open_positions == 9
    assert rm.limits.max_position_size == 25000.0


def test_risk_kill_toggles_engine(client: TestClient) -> None:
    assert client.post("/api/risk/kill", json={"active": True}).json()["kill"] is True
    assert client.app.state.risk.kill_switch_active is True
    assert client.get("/api/risk").json()["kill"] is True
    assert client.post("/api/risk/kill", json={"active": False}).json()["kill"] is False
    assert client.app.state.risk.kill_switch_active is False


def test_risk_works_without_engine(bare_client: TestClient) -> None:
    # No app.state.risk: state round-trips through user_settings instead.
    assert bare_client.get("/api/risk").json() == {"kill": False, "limits": {}}
    bare_client.post("/api/risk/kill", json={"active": True})
    assert bare_client.get("/api/risk").json()["kill"] is True
    bare_client.put("/api/risk/limits", json={"openPos": 3})
    assert bare_client.get("/api/risk").json()["limits"]["openPos"] == 3


# --- approvals ---------------------------------------------------------------


def test_approvals_pending_and_approve(client: TestClient) -> None:
    queue: ApprovalQueue = client.app.state.approvals
    pid = queue.add({"symbol": "AAPL", "side": "BUY", "amount": 12, "reason": "dip", "model": "opus"})
    pending = client.get("/api/approvals").json()
    assert len(pending) == 1
    card = pending[0]
    assert card["id"] == pid and card["m"] == "opus"
    assert card["t"] == "Buy 12 shares of AAPL" and card["meta"] == "dip"
    # approve it -> executor runs, queue drains
    assert client.post(f"/api/approvals/{pid}/approve").json()["status"] == "approved"
    assert client.get("/api/approvals").json() == []


def test_approvals_reject_and_missing(client: TestClient) -> None:
    queue: ApprovalQueue = client.app.state.approvals
    pid = queue.add({"symbol": "NVDA", "side": "SELL", "quantity": 3})
    assert client.post(f"/api/approvals/{pid}/reject").json()["status"] == "rejected"
    # second decision on a non-pending proposal -> 409
    assert client.post(f"/api/approvals/{pid}/approve").status_code == 409
    # unknown proposal -> 404
    assert client.post("/api/approvals/nope/reject").status_code == 404


def test_approvals_empty_and_503_without_queue(bare_client: TestClient) -> None:
    assert bare_client.get("/api/approvals").json() == []
    assert bare_client.post("/api/approvals/x/approve").status_code == 503


# --- auth enforcement (formerly covered by foundation STUB_ROUTES) -----------


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("get", "/api/accounts", None),
        ("post", "/api/accounts", {"model": "m"}),
        ("get", "/api/leaderboard", None),
        ("get", "/api/positions", None),
        ("get", "/api/activity", None),
        ("get", "/api/risk", None),
        ("put", "/api/risk/limits", {"openPos": 3}),
        ("post", "/api/risk/kill", {"active": True}),
        ("get", "/api/approvals", None),
        ("post", "/api/approvals/abc/approve", None),
        ("post", "/api/approvals/abc/reject", None),
    ],
)
def test_routes_require_auth(tmp_path: Path, method: str, path: str, body: Any) -> None:
    app = create_cockpit_app(Database(tmp_path / "c.db"), transport=_mock_transport())
    unauthed = TestClient(app)
    resp = getattr(unauthed, method)(path, json=body) if body else getattr(unauthed, method)(path)
    assert resp.status_code == 401, f"{method.upper()} {path} -> {resp.status_code}"


# --- serve factory -----------------------------------------------------------


def test_build_cockpit_attaches_engine(tmp_path: Path) -> None:
    app = build_cockpit(
        db=Database(tmp_path / "c.db"), data_dir=tmp_path, transport=_mock_transport()
    )
    try:
        # CONTRACTS §Runtime wiring: the serve process attaches these.
        assert isinstance(app.state.bench, Bench)
        assert app.state.market_watch is not None
        assert isinstance(app.state.risk, RiskManager)
        assert isinstance(app.state.approvals, ApprovalQueue)
        # no OPENROUTER_API_KEY -> no controller (add-trader disabled, reads still work)
        assert getattr(app.state, "bench_controller", None) is None
    finally:
        app.state.approvals.close()


def test_build_cockpit_with_client_enables_controller(tmp_path: Path) -> None:
    app = build_cockpit(
        db=Database(tmp_path / "c.db"),
        data_dir=tmp_path,
        transport=_mock_transport(),
        openrouter_client=OpenRouterClient(api_key="k", transport=_mock_transport()),
    )
    try:
        assert isinstance(app.state.bench_controller, BenchController)
    finally:
        app.state.approvals.close()
