"""Tests for the /ws/quotes WebSocket bridge over the MessageBus.

Drives the in-process bus from the test's own thread (the WS handler hops back
to the loop via ``call_soon_threadsafe`` either way), and uses
``TestClient.websocket_connect`` so the same session cookie the rest of the
test suite uses is reused on the WS handshake.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from trading_agent.config.db import Database
from trading_agent.data_feed import MessageBus
from trading_agent.web.app import create_cockpit_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    app.state.bus = MessageBus()
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "lukas", "password": "pw"})
    return c


def _publish_from_other_thread(bus: MessageBus, topic: str, payload: dict[str, object]) -> None:
    """Hit the bus from a non-loop thread — closer to how live_quote runs."""
    t = threading.Thread(target=bus.publish, args=(topic, payload), daemon=True)
    t.start()
    t.join(timeout=2.0)


def test_ws_subscribe_then_receive_quote(client: TestClient) -> None:
    with client.websocket_connect("/ws/quotes") as ws:
        ws.send_json({"action": "subscribe", "symbols": ["aapl", "MSFT"]})
        ack = ws.receive_json()
        assert ack == {"subscribed": ["AAPL", "MSFT"]}

        _publish_from_other_thread(
            client.app.state.bus,
            "quote.AAPL",
            {"symbol": "AAPL", "price": 199.5, "timestamp": "2026-05-28T15:00:00Z"},
        )
        msg = ws.receive_json()
        assert msg == {"symbol": "AAPL", "price": 199.5, "ts": "2026-05-28T15:00:00Z"}


def test_ws_unsubscribe_stops_pushes(client: TestClient) -> None:
    with client.websocket_connect("/ws/quotes") as ws:
        ws.send_json({"action": "subscribe", "symbols": ["AAPL"]})
        ws.receive_json()  # ack
        ws.send_json({"action": "unsubscribe", "symbols": ["AAPL"]})
        assert ws.receive_json() == {"unsubscribed": ["AAPL"]}

        # The bus should now have no subscribers for quote.AAPL.
        bus: MessageBus = client.app.state.bus
        assert "quote.AAPL" not in bus._subscribers  # type: ignore[attr-defined]


def test_ws_unknown_action_returns_error(client: TestClient) -> None:
    with client.websocket_connect("/ws/quotes") as ws:
        ws.send_json({"action": "wat", "symbols": ["AAPL"]})
        msg = ws.receive_json()
        assert "error" in msg


def test_ws_invalid_symbol_silently_dropped(client: TestClient) -> None:
    with client.websocket_connect("/ws/quotes") as ws:
        ws.send_json({"action": "subscribe", "symbols": ["AAPL", "@@@", ""]})
        assert ws.receive_json() == {"subscribed": ["AAPL"]}


def test_ws_disconnect_clears_subscriptions(client: TestClient) -> None:
    bus: MessageBus = client.app.state.bus
    with client.websocket_connect("/ws/quotes") as ws:
        ws.send_json({"action": "subscribe", "symbols": ["AAPL"]})
        ws.receive_json()
        assert "quote.AAPL" in bus._subscribers  # type: ignore[attr-defined]
    # Closing the with-block triggers WS shutdown; the handler must drop subs.
    # The cleanup runs in the server task; give it a brief moment.
    for _ in range(50):
        if "quote.AAPL" not in bus._subscribers:  # type: ignore[attr-defined]
            break
        time.sleep(0.01)
    assert "quote.AAPL" not in bus._subscribers  # type: ignore[attr-defined]


def test_ws_requires_auth(tmp_path: Path) -> None:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    app.state.bus = MessageBus()
    c = TestClient(app)  # not logged in
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws/quotes") as ws:
            ws.receive_json()


def test_ws_no_bus_attached_rejects(tmp_path: Path) -> None:
    """No bus on app.state means the realtime path is unavailable — close 1011."""
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    # explicitly no app.state.bus
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "u", "password": "pw"})
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws/quotes") as ws:
            ws.receive_json()


def test_ws_bearer_token_handshake(tmp_path: Path) -> None:
    """Auth also resolves from Authorization: Bearer (the non-cookie path)."""
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    app.state.bus = MessageBus()
    c = TestClient(app)
    signup = c.post("/api/auth/signup", json={"username": "u", "password": "pw"})
    assert signup.status_code == 200
    # Grab the session cookie issued on signup; we'll pass it as a Bearer token
    # instead of as a cookie to exercise that branch.
    token = signup.cookies.get("session") or c.cookies.get("session")
    assert token, "signup did not set a session cookie"
    # Use a fresh client (no cookies) and send the token via Authorization.
    c2 = TestClient(app)
    with c2.websocket_connect(
        "/ws/quotes", headers={"authorization": f"Bearer {token}"}
    ) as ws:
        ws.send_json({"action": "subscribe", "symbols": ["AAPL"]})
        assert ws.receive_json() == {"subscribed": ["AAPL"]}
