"""WS-H tests: advisor notes, per-trader universe, the stock-request flow, and
the merged notification feed (requests + alerts + fills + blocks).

Store-level tests run against an isolated ``Database(tmp)``; HTTP tests drive the
cockpit app via ``TestClient`` (no live server, no model calls).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.notes import NoteError, NotesStore
from trading_agent.requests import (
    STATUS_DECLINED,
    STATUS_FULFILLED,
    STATUS_PENDING,
    RequestError,
    RequestService,
    RequestStore,
    UniverseStore,
)
from trading_agent.web.app import create_cockpit_app
from trading_agent.web.market_watch import MarketMove, MarketMoveWatcher
from trading_agent.web.routers.notifications import (
    NotificationReadStore,
    build_items,
)

# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Any) -> Database:
    return Database(tmp_path / "config.db")


@pytest.fixture
def client(tmp_path: Any) -> TestClient:
    return TestClient(create_cockpit_app(Database(tmp_path / "config.db")))


def _signup(client: TestClient, username: str = "ada") -> str:
    r = client.post("/api/auth/signup", json={"username": username, "password": "pw"})
    assert r.status_code == 200
    return str(r.json()["user_id"])


# --- notes store -------------------------------------------------------------


def test_notes_put_get_roundtrip_and_upsert(db: Database) -> None:
    store = NotesStore(db)
    assert store.get("u1", "ticker", "NVDA") is None
    n = store.put("u1", "ticker", "NVDA", "looks oversold")
    assert n.text == "looks oversold"
    assert store.get("u1", "ticker", "NVDA").text == "looks oversold"
    # upsert overwrites in place, keeping the same id
    n2 = store.put("u1", "ticker", "NVDA", "changed my mind")
    assert n2.id == n.id
    assert store.get("u1", "ticker", "NVDA").text == "changed my mind"


def test_notes_isolated_per_user_and_scope(db: Database) -> None:
    store = NotesStore(db)
    store.put("u1", "trader", "glm-5.1", "watch its sizing")
    store.put("u2", "trader", "glm-5.1", "different user note")
    store.put("u1", "ticker", "glm-5.1", "ticker-scoped, same ref")
    assert store.get("u1", "trader", "glm-5.1").text == "watch its sizing"
    assert store.get("u2", "trader", "glm-5.1").text == "different user note"
    # same ref, different scope, same user → distinct rows
    assert store.get("u1", "ticker", "glm-5.1").text == "ticker-scoped, same ref"
    assert len(store.list("u1")) == 2
    assert len(store.list("u1", scope="trader")) == 1


def test_notes_invalid_scope_rejected(db: Database) -> None:
    store = NotesStore(db)
    with pytest.raises(NoteError):
        store.put("u1", "bogus", "X", "nope")
    with pytest.raises(NoteError):
        store.get("u1", "ticker", "")


def test_notes_delete(db: Database) -> None:
    store = NotesStore(db)
    store.put("u1", "ticker", "AAPL", "x")
    assert store.delete("u1", "ticker", "AAPL") is True
    assert store.delete("u1", "ticker", "AAPL") is False


# --- universe store ----------------------------------------------------------


def test_universe_add_contains_dedup_and_norm(db: Database) -> None:
    uni = UniverseStore(db)
    assert uni.add("u1", "glm-5.1", "nvda") is True  # lowercase normalised
    assert uni.add("u1", "glm-5.1", "NVDA") is False  # already present
    assert uni.contains("u1", "glm-5.1", "nvda") is True
    assert uni.get("u1", "glm-5.1") == ["NVDA"]


def test_universe_isolated_per_user_and_trader(db: Database) -> None:
    uni = UniverseStore(db)
    uni.add("u1", "glm-5.1", "NVDA")
    uni.add("u1", "opus", "AMD")
    uni.add("u2", "glm-5.1", "TSLA")
    assert uni.get("u1", "glm-5.1") == ["NVDA"]
    assert uni.get("u1", "opus") == ["AMD"]
    assert uni.get("u2", "glm-5.1") == ["TSLA"]


def test_universe_set_and_remove(db: Database) -> None:
    uni = UniverseStore(db)
    assert uni.set("u1", "t", ["AAPL", "MSFT", "aapl"]) == ["AAPL", "MSFT"]
    assert uni.remove("u1", "t", "AAPL") is True
    assert uni.get("u1", "t") == ["MSFT"]


# --- request service flow ----------------------------------------------------


def test_submit_creates_pending(db: Database) -> None:
    svc = RequestService(db)
    req = svc.submit("u1", "glm-5.1", "nvda", "breakout to new highs")
    assert req.status == STATUS_PENDING
    assert req.symbol == "NVDA"
    assert [r.id for r in svc.pending("u1")] == [req.id]


def test_submit_requires_trader(db: Database) -> None:
    with pytest.raises(RequestError):
        RequestService(db).submit("u1", "", "NVDA")


def test_allow_adds_to_universe_and_fulfills(db: Database) -> None:
    seen: list[tuple[str, str, str]] = []
    svc = RequestService(db, universe_listener=lambda u, t, s: seen.append((u, t, s)))
    req = svc.submit("u1", "glm-5.1", "NVDA", "wants it")
    assert svc.universe.contains("u1", "glm-5.1", "NVDA") is False
    allowed = svc.allow("u1", req.id)
    assert allowed.status == STATUS_FULFILLED
    assert svc.universe.contains("u1", "glm-5.1", "NVDA") is True
    assert seen == [("u1", "glm-5.1", "NVDA")]  # live-coordination hook fired
    assert svc.pending("u1") == []  # no longer pending


def test_decline_leaves_universe_unchanged(db: Database) -> None:
    svc = RequestService(db)
    req = svc.submit("u1", "glm-5.1", "NVDA", "wants it")
    declined = svc.decline("u1", req.id)
    assert declined.status == STATUS_DECLINED
    assert svc.universe.get("u1", "glm-5.1") == []
    assert svc.pending("u1") == []


def test_allow_unknown_and_non_pending(db: Database) -> None:
    svc = RequestService(db)
    with pytest.raises(KeyError):
        svc.allow("u1", "does-not-exist")
    req = svc.submit("u1", "glm-5.1", "NVDA")
    svc.allow("u1", req.id)
    with pytest.raises(RequestError):  # second allow → not pending
        svc.allow("u1", req.id)


def test_requests_isolated_per_user(db: Database) -> None:
    svc = RequestService(db)
    req = svc.submit("u1", "glm-5.1", "NVDA")
    assert svc.requests.get("u2", req.id) is None  # other user can't see it
    with pytest.raises(KeyError):
        svc.allow("u2", req.id)


# --- notification feed builder ----------------------------------------------


def test_build_items_merges_all_types_sorted(db: Database) -> None:
    store = RequestStore(db)
    store.create("u1", "glm-5.1", "NVDA", "breakout")
    move = MarketMove(
        symbol="TSLA",
        reference_price=200.0,
        current_price=193.0,
        pct_change=-0.035,
        direction="down",
        timestamp="2026-05-26T12:00:00",
    )
    decisions = [
        {
            "timestamp": "2026-05-26T12:30:00",
            "competitor": "opus",
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 12,
            "status": "filled",
            "reason": "oversold",
            "detail": "",
        },
        {
            "timestamp": "2026-05-26T12:10:00",
            "competitor": "gemini",
            "symbol": "MSFT",
            "action": "BUY",
            "quantity": 5,
            "status": "blocked",
            "reason": "",
            "detail": "exceeds max position size",
        },
        {
            "timestamp": "2026-05-26T12:05:00",
            "competitor": "kimi",
            "symbol": "AMD",
            "action": "SELL",
            "quantity": 3,
            "status": "hold",  # ignored
        },
    ]
    items = build_items(
        "u1", request_store=store, moves=[move], decisions=decisions, read_ids=set()
    )
    types = [it["type"] for it in items]
    assert types.count("request") == 1
    assert types.count("alert") == 1
    assert types.count("fill") == 1
    assert types.count("block") == 1
    assert "hold" not in [it.get("data", {}).get("status") for it in items if it["type"] == "fill"]
    # every item carries the cockpit NOTIFS fields
    for it in items:
        assert {"id", "type", "unread", "who", "t", "m", "ts"} <= set(it)
    fill = next(it for it in items if it["type"] == "fill")
    assert fill["t"] == "opus bought 12 shares of AAPL"
    assert next(it for it in items if it["type"] == "request")["unread"] is True


def test_build_items_respects_read_ids(db: Database) -> None:
    store = RequestStore(db)
    req = store.create("u1", "glm-5.1", "NVDA")
    items = build_items("u1", request_store=store, read_ids={req.id})
    assert items[0]["unread"] is False


def test_notification_read_store_roundtrip(db: Database) -> None:
    reads = NotificationReadStore(db)
    assert reads.read_ids("u1") == set()
    assert reads.mark("u1", ["a", "b", "a"]) == 2  # dedup on insert
    assert reads.read_ids("u1") == {"a", "b"}
    assert reads.read_ids("u2") == set()  # per-user


# --- HTTP: notes -------------------------------------------------------------


def test_notes_http_roundtrip_and_isolation(client: TestClient) -> None:
    _signup(client, "ada")
    # unset → empty text
    r = client.get("/api/notes", params={"scope": "ticker", "ref": "NVDA"})
    assert r.status_code == 200 and r.json()["text"] == ""
    put = client.put(
        "/api/notes", params={"scope": "ticker", "ref": "NVDA"}, json={"text": "buy the dip"}
    )
    assert put.status_code == 200
    assert client.get("/api/notes", params={"scope": "ticker", "ref": "NVDA"}).json()["text"] == (
        "buy the dip"
    )

    other = TestClient(client.app)
    _signup(other, "bob")
    assert other.get("/api/notes", params={"scope": "ticker", "ref": "NVDA"}).json()["text"] == ""


def test_notes_http_bad_scope_400_and_missing_param_422(client: TestClient) -> None:
    _signup(client)
    assert client.get("/api/notes", params={"scope": "bogus", "ref": "X"}).status_code == 400
    assert client.get("/api/notes").status_code == 422  # scope/ref required


def test_notes_http_requires_auth(client: TestClient) -> None:
    assert client.get("/api/notes", params={"scope": "ticker", "ref": "NVDA"}).status_code == 401


# --- HTTP: requests + universe ----------------------------------------------


def test_requests_http_allow_updates_universe(client: TestClient) -> None:
    user_id = _signup(client)
    db: Database = client.app.state.db
    req = RequestStore(db).create(user_id, "glm-5.1", "NVDA", "breakout")

    listed = client.get("/api/requests").json()
    assert [r["id"] for r in listed] == [req.id]

    allowed = client.post(f"/api/requests/{req.id}/allow")
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "allowed"
    assert "NVDA" in allowed.json()["universe"]
    assert UniverseStore(db).contains(user_id, "glm-5.1", "NVDA")
    # status flipped; second allow → 409
    assert client.post(f"/api/requests/{req.id}/allow").status_code == 409


def test_requests_http_decline_leaves_universe(client: TestClient) -> None:
    user_id = _signup(client)
    db: Database = client.app.state.db
    req = RequestStore(db).create(user_id, "glm-5.1", "NVDA")
    assert client.post(f"/api/requests/{req.id}/decline").status_code == 200
    assert UniverseStore(db).get(user_id, "glm-5.1") == []


def test_requests_http_unknown_404(client: TestClient) -> None:
    _signup(client)
    assert client.post("/api/requests/nope/allow").status_code == 404


# --- HTTP: notifications -----------------------------------------------------


def test_notifications_http_shows_request_then_clears_on_allow(client: TestClient) -> None:
    user_id = _signup(client)
    db: Database = client.app.state.db
    req = RequestStore(db).create(user_id, "glm-5.1", "NVDA", "breakout")

    snap = client.get("/api/notifications").json()
    assert snap["unread"] == 1
    item = snap["items"][0]
    assert item["type"] == "request" and item["id"] == req.id and item["who"] == "glm-5.1"

    client.post(f"/api/requests/{req.id}/allow")
    snap2 = client.get("/api/notifications").json()
    assert snap2["items"] == [] and snap2["unread"] == 0


def test_notifications_http_mark_read(client: TestClient) -> None:
    user_id = _signup(client)
    db: Database = client.app.state.db
    RequestStore(db).create(user_id, "glm-5.1", "NVDA")
    assert client.get("/api/notifications").json()["unread"] == 1
    read = client.post("/api/notifications/read")
    assert read.status_code == 200 and read.json()["unread"] == 0
    # request itself is still pending/visible, just no longer unread
    snap = client.get("/api/notifications").json()
    assert len(snap["items"]) == 1 and snap["items"][0]["unread"] is False


def test_notifications_http_merges_live_alerts_and_fills(client: TestClient) -> None:
    user_id = _signup(client)
    db: Database = client.app.state.db
    RequestStore(db).create(user_id, "glm-5.1", "NVDA")

    # attach live sources the way serve/WS-0 wiring will
    watch = MarketMoveWatcher(threshold_pct=2.0)
    watch.observe("TSLA", 200.0)  # reference
    assert watch.observe("TSLA", 193.0) is not None  # fires a -3.5% move

    class _FakeBench:
        def recent_decisions(self, limit: int = 30) -> list[dict[str, Any]]:
            return [
                {
                    "timestamp": "2026-05-26T12:30:00",
                    "competitor": "opus",
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 12,
                    "status": "filled",
                    "reason": "oversold",
                    "detail": "",
                }
            ]

    client.app.state.market_watch = watch
    client.app.state.bench = _FakeBench()

    items = client.get("/api/notifications").json()["items"]
    types = {it["type"] for it in items}
    assert {"request", "alert", "fill"} <= types
