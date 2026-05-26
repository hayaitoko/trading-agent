"""Tests for the FastAPI notification-center app (TestClient = no live server)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_agent.approval_queue import ApprovalQueue
from trading_agent.audit import AuditLogger
from trading_agent.db import DatabaseManager
from trading_agent.web.app import create_app
from trading_agent.web.market_watch import MarketMoveWatcher
from trading_agent.web.notifications import NotificationCenter


@pytest.fixture
def client(tmp_path: Any) -> Any:
    db = DatabaseManager(str(tmp_path / "serve.db"))
    audit = AuditLogger(db, data_dir=tmp_path)
    executed: list[dict[str, Any]] = []

    def executor(signal: dict[str, Any]) -> dict[str, Any]:
        executed.append(signal)
        return {"status": "FILLED", "symbol": signal["asset"]}

    queue = ApprovalQueue(db_path=tmp_path / "approvals.db", executor=executor)
    watch = MarketMoveWatcher(threshold_pct=2.0)
    center = NotificationCenter(queue, db, watch, account_provider=lambda: {"cash": 100.0})
    app = create_app(center, queue)
    c = TestClient(app)
    c.audit = audit  # type: ignore[attr-defined]
    c.queue = queue  # type: ignore[attr-defined]
    c.watch = watch  # type: ignore[attr-defined]
    c.executed = executed  # type: ignore[attr-defined]
    return c


def test_health(client: Any) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_index_served(client: Any) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Trading Agent" in r.text


def test_notifications_payload(client: Any) -> None:
    client.queue.add({"asset": "AAPL", "side": "LONG", "amount": 2})
    client.audit.warn("kill_switch_active", module="serve")
    client.watch.observe("AAPL", 100.0)
    client.watch.observe("AAPL", 106.0)

    data = client.get("/api/notifications").json()
    assert data["counts"] == {"approvals": 1, "risk": 1, "market": 1}
    assert data["account"]["cash"] == 100.0


def test_approve_executes(client: Any) -> None:
    pid = client.queue.add({"asset": "AAPL", "side": "LONG", "amount": 2})
    r = client.post(f"/api/approvals/{pid}/approve", json={"note": "lgtm"})
    assert r.status_code == 200
    assert r.json()["status"] == "approved"
    assert client.executed == [{"asset": "AAPL", "side": "LONG", "amount": 2}]
    assert client.queue.get(pid).status == "approved"


def test_approve_empty_body_ok(client: Any) -> None:
    pid = client.queue.add({"asset": "X", "side": "LONG", "amount": 1})
    r = client.post(f"/api/approvals/{pid}/approve")  # no body
    assert r.status_code == 200


def test_reject(client: Any) -> None:
    pid = client.queue.add({"asset": "X", "side": "LONG", "amount": 1})
    r = client.post(f"/api/approvals/{pid}/reject", json={"note": "nope"})
    assert r.status_code == 200
    assert client.queue.get(pid).status == "rejected"
    assert client.executed == []


def test_approve_unknown_is_404(client: Any) -> None:
    r = client.post("/api/approvals/does-not-exist/approve")
    assert r.status_code == 404


def test_approve_twice_is_409(client: Any) -> None:
    pid = client.queue.add({"asset": "X", "side": "LONG", "amount": 1})
    client.post(f"/api/approvals/{pid}/approve")
    r = client.post(f"/api/approvals/{pid}/approve")
    assert r.status_code == 409
