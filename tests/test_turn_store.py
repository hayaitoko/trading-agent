"""Tests for WS-Agent A5: TurnStore + traces router + recent_turns() integration.

Coverage:
  - TurnRecord: to_trader_dict never includes book_type (MONEY IS REAL)
  - TurnRecord: to_operator_dict includes book_type + book_badge
  - TurnRecord: to_summary_dict includes book_type + book_badge
  - ToolCallRecord: round-trip to_dict / from_dict
  - TurnStore: record + recent (newest-first, capped at 50)
  - TurnStore: open_turn + close_turn lifecycle
  - TurnStore: get by turn_id
  - TurnStore: summaries (operator path, includes book_type)
  - TurnStore: cost_rollup (today / week / lifetime)
  - TurnStore: orphaned_turns detects NULL ended_at rows older than 5 min
  - TurnStore: graceful degradation when DB is absent / bad path
  - RecentTurnsTool: integrates with TurnStore, returns trader-dict (no book_type)
  - Traces router: GET /api/traces, GET /api/traces/{id}, GET /api/traces/cost,
    GET /api/traces/attention — all degrade gracefully when store absent
  - MONEY IS REAL grepping: no "paper" / "sim" in any trader-facing result
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from trading_agent.intel.tools.look.recent_turns import RecentTurnsTool
from trading_agent.intel.turn_store import ToolCallRecord, TurnRecord, TurnStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path):
    """An isolated TurnStore backed by a temp SQLite file."""
    return TurnStore(db_path=str(tmp_path / "turns.db"))


def _make_record(
    turn_id: str = "t-001",
    trader_id: str = "Alpha",
    book_type: str = "paper",
    final_action: str = "pass",
    tool_calls: list | None = None,
) -> TurnRecord:
    now = datetime.now(tz=UTC)
    return TurnRecord(
        turn_id=turn_id,
        trader_id=trader_id,
        started_at=now,
        ended_at=now,
        wake_reason="scheduled",
        turn_type="regular",
        first_look_snapshot={"account": {"cash": 100000}},
        tool_calls=tool_calls or [],
        final_action=final_action,
        final_action_args={"reason": "nothing interesting"},
        total_cost_usd=0.0042,
        total_tokens={"input": 500, "output": 80, "cached": 120},
        previous_attempt_turn_id=None,
        _book_type=book_type,
    )


def _make_tool_call(
    tool_name: str = "account_state",
    ok: bool = True,
    latency_ms: int = 45,
    cost_usd: float = 0.0,
) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=tool_name,
        args={"n": 3},
        result={"ok": ok, "data": {"cash": 100000}} if ok else {"ok": False, "error": {"kind": "unavailable", "message": "down"}},
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


# ---------------------------------------------------------------------------
# ToolCallRecord tests
# ---------------------------------------------------------------------------


def test_tool_call_record_round_trip():
    tc = _make_tool_call()
    d = tc.to_dict()
    tc2 = ToolCallRecord.from_dict(d)
    assert tc2.tool_name == tc.tool_name
    assert tc2.args == tc.args
    assert tc2.result == tc.result
    assert tc2.latency_ms == tc.latency_ms
    assert tc2.cost_usd == tc.cost_usd


def test_tool_call_record_bad_dict():
    """from_dict with missing keys returns defaults, no exception."""
    tc = ToolCallRecord.from_dict({})
    assert tc.tool_name == ""
    assert tc.latency_ms == 0
    assert tc.cost_usd == 0.0


# ---------------------------------------------------------------------------
# TurnRecord serialisation — MONEY IS REAL
# ---------------------------------------------------------------------------


def test_turn_record_trader_dict_no_book_type():
    """MONEY IS REAL: to_trader_dict must never contain book_type."""
    rec = _make_record(book_type="paper")
    d = rec.to_trader_dict()
    assert "book_type" not in d, "book_type must NOT appear in trader-facing dict"
    assert "book_badge" not in d, "book_badge must NOT appear in trader-facing dict"
    # Serialise to JSON string and check there's no paper-status leak.
    s = json.dumps(d)
    for forbidden in ("paper", "sim", "demo", "fake", "book_type", "book_badge"):
        assert forbidden not in s.lower(), (
            f"MONEY IS REAL violation: '{forbidden}' found in trader-dict JSON"
        )


def test_turn_record_operator_dict_has_book_type():
    """Operator dict must include book_type and book_badge."""
    rec = _make_record(book_type="paper")
    d = rec.to_operator_dict()
    assert d["book_type"] == "paper"
    assert d["book_badge"] == "[PAPER]"


def test_turn_record_operator_dict_live():
    rec = _make_record(book_type="live")
    d = rec.to_operator_dict()
    assert d["book_type"] == "live"
    assert d["book_badge"] == "[LIVE]"


def test_turn_record_summary_dict_has_book_type():
    rec = _make_record(book_type="paper")
    d = rec.to_summary_dict()
    assert d["book_type"] == "paper"
    assert d["book_badge"] == "[PAPER]"
    assert "tool_call_count" in d
    assert "first_look_snapshot" not in d  # summary omits full snapshot


def test_turn_record_trader_dict_with_tool_calls():
    tc = _make_tool_call()
    rec = _make_record(tool_calls=[tc])
    d = rec.to_trader_dict(include_tool_calls=True)
    assert len(d["tool_calls"]) == 1
    assert d["tool_calls"][0]["tool_name"] == "account_state"
    # Ensure no book_type sneaks in through tool calls.
    s = json.dumps(d)
    assert "book_type" not in s


def test_turn_record_trader_dict_no_tool_calls():
    tc = _make_tool_call()
    rec = _make_record(tool_calls=[tc])
    d = rec.to_trader_dict(include_tool_calls=False)
    assert "tool_calls" not in d
    assert d["tool_call_count"] == 1


# ---------------------------------------------------------------------------
# TurnStore write + read
# ---------------------------------------------------------------------------


def test_turn_store_record_and_recent(tmp_store):
    rec = _make_record(turn_id="t-001", trader_id="Alpha")
    tmp_store.record(rec)
    result = tmp_store.recent("Alpha", n=5)
    assert len(result) == 1
    assert result[0].turn_id == "t-001"
    assert result[0].trader_id == "Alpha"
    assert result[0].final_action == "pass"


def test_turn_store_recent_newest_first(tmp_store):
    """recent() should return turns newest-first."""
    import time as _time
    for i in range(3):
        r = _make_record(turn_id=f"t-{i:03}", trader_id="Beta")
        # Stagger start times by manually writing — use open+close pattern.
        tmp_store.record(r)
        _time.sleep(0.01)  # tiny gap to ensure ordering
    results = tmp_store.recent("Beta", n=10)
    assert len(results) == 3
    # All turn_ids present, order is by started_at DESC.
    ids = [r.turn_id for r in results]
    assert set(ids) == {"t-000", "t-001", "t-002"}


def test_turn_store_recent_capped_at_50(tmp_store):
    for i in range(60):
        tmp_store.record(_make_record(turn_id=f"t-{i:03}", trader_id="Cap"))
    results = tmp_store.recent("Cap", n=200)
    assert len(results) == 50  # hard cap in recent()


def test_turn_store_recent_empty_when_no_turns(tmp_store):
    result = tmp_store.recent("Nobody", n=5)
    assert result == []


def test_turn_store_get_by_id(tmp_store):
    rec = _make_record(turn_id="t-get", trader_id="Alpha")
    tmp_store.record(rec)
    fetched = tmp_store.get("t-get")
    assert fetched is not None
    assert fetched.turn_id == "t-get"
    assert fetched.total_cost_usd == pytest.approx(0.0042, rel=1e-4)


def test_turn_store_get_missing_returns_none(tmp_store):
    assert tmp_store.get("does-not-exist") is None


def test_turn_store_summaries_include_book_type(tmp_store):
    tmp_store.record(_make_record(turn_id="t-sum", trader_id="Alpha", book_type="live"))
    sums = tmp_store.summaries("Alpha", limit=10)
    assert len(sums) == 1
    assert sums[0]["book_type"] == "live"
    assert sums[0]["book_badge"] == "[LIVE]"
    assert sums[0]["tool_call_count"] == 0


def test_turn_store_open_and_close(tmp_store):
    """open_turn writes interrupted row; close_turn finalises it."""
    tid = "t-lifecycle"
    tmp_store.open_turn(
        turn_id=tid,
        trader_id="Gamma",
        wake_reason="scheduled",
        turn_type="regular",
        first_look_snapshot={"account": {}},
    )
    # Should be findable via get() even before close.
    mid = tmp_store.get(tid)
    assert mid is not None
    assert mid.final_action == "interrupted"
    assert mid.ended_at is None

    tmp_store.close_turn(
        turn_id=tid,
        tool_calls=[_make_tool_call()],
        final_action="hold",
        final_action_args={"reason": "nothing"},
        total_cost_usd=0.01,
        total_tokens={"input": 100, "output": 20, "cached": 0},
    )
    done = tmp_store.get(tid)
    assert done is not None
    assert done.final_action == "hold"
    assert done.ended_at is not None
    assert len(done.tool_calls) == 1


def test_turn_store_orphaned_turns(tmp_store):
    """open_turn rows older than 5min should appear as orphaned."""
    # Insert a turn with a synthetic old started_at via direct SQL.
    old_ts = time.time() - 400  # 400s ago > 5min threshold
    tmp_store._conn.execute(
        """INSERT INTO turn_records
           (turn_id, trader_id, started_at, ended_at, wake_reason, turn_type, book_type,
            first_look_json, tool_calls_json, final_action, final_action_args_json,
            total_cost_usd, tokens_input, tokens_output, tokens_cached)
           VALUES (?,?,?,NULL,?,?,?,?,?,?,?,?,?,?,?)
        """,
        ("t-orphan", "Delta", old_ts, "event", "regular", "paper",
         "{}", "[]", "interrupted", "{}", 0.0, 0, 0, 0),
    )
    tmp_store._conn.commit()
    orphans = tmp_store.orphaned_turns()
    assert len(orphans) == 1
    assert orphans[0]["turn_id"] == "t-orphan"
    assert orphans[0]["trader_id"] == "Delta"


def test_turn_store_cost_rollup(tmp_store):
    # Record 3 turns with different costs.
    for i in range(3):
        rec = _make_record(turn_id=f"tc-{i}", trader_id="Cost")
        rec = TurnRecord(
            turn_id=f"tc-{i}",
            trader_id="Cost",
            started_at=datetime.now(tz=UTC),
            ended_at=datetime.now(tz=UTC),
            wake_reason="scheduled",
            turn_type="regular",
            first_look_snapshot={},
            tool_calls=[],
            final_action="pass",
            final_action_args={},
            total_cost_usd=0.10 * (i + 1),
            total_tokens={"input": 0, "output": 0, "cached": 0},
            previous_attempt_turn_id=None,
        )
        tmp_store.record(rec)
    rollup = tmp_store.cost_rollup("Cost")
    assert rollup["today"] == pytest.approx(0.60, rel=1e-4)
    assert rollup["week"] == pytest.approx(0.60, rel=1e-4)
    assert rollup["lifetime"] == pytest.approx(0.60, rel=1e-4)


def test_turn_store_cost_rollup_absent():
    """cost_rollup on absent store returns zeros."""
    store = TurnStore(db_path="/nonexistent/path/turns.db")
    rollup = store.cost_rollup("X")
    assert rollup == {"today": 0.0, "week": 0.0, "lifetime": 0.0}


def test_turn_store_record_swallows_error():
    """record() on a broken store must not raise."""
    store = TurnStore(db_path="/nonexistent/path/turns.db")
    rec = _make_record()
    store.record(rec)  # must not raise


# ---------------------------------------------------------------------------
# RecentTurnsTool integration with TurnStore
# ---------------------------------------------------------------------------


def test_recent_turns_tool_no_store():
    """recent_turns() with no store returns ok=True, empty list."""
    tool = RecentTurnsTool(trader_id="Alpha", turn_store=None)
    result = tool(n=5)
    assert result.ok is True
    assert result.data["turns"] == []


def test_recent_turns_tool_with_store(tmp_store):
    rec = _make_record(turn_id="rt-001", trader_id="RTest")
    tmp_store.record(rec)
    tool = RecentTurnsTool(trader_id="RTest", turn_store=tmp_store)
    result = tool(n=5)
    assert result.ok is True
    assert len(result.data["turns"]) == 1
    turn = result.data["turns"][0]
    assert turn["turn_id"] == "rt-001"
    assert turn["final_action"] == "pass"
    # MONEY IS REAL: no book_type in the trader-facing payload.
    s = json.dumps(result.data)
    assert "book_type" not in s, "MONEY IS REAL: book_type must not appear in recent_turns output"
    assert "book_badge" not in s


def test_recent_turns_tool_include_tool_calls(tmp_store):
    tc = _make_tool_call(tool_name="history")
    rec = _make_record(turn_id="rt-002", trader_id="TCTest", tool_calls=[tc])
    tmp_store.record(rec)
    tool = RecentTurnsTool(trader_id="TCTest", turn_store=tmp_store)
    # with tool calls
    result = tool(n=5, include_tool_calls=True)
    assert result.ok is True
    t = result.data["turns"][0]
    assert len(t["tool_calls"]) == 1
    assert t["tool_calls"][0]["name"] == "history"
    # without tool calls
    result2 = tool(n=5, include_tool_calls=False)
    assert "tool_calls" not in result2.data["turns"][0]


def test_recent_turns_tool_money_is_real_grepped(tmp_store):
    """Red-team: serialise recent_turns result and grep for forbidden strings."""
    rec = _make_record(turn_id="rt-rr", trader_id="RedTeam", book_type="paper")
    tmp_store.record(rec)
    tool = RecentTurnsTool(trader_id="RedTeam", turn_store=tmp_store)
    result = tool(n=5)
    payload = json.dumps(result.data)
    for forbidden in ("paper", "sim", "demo", "fake", "book_type", "book_badge"):
        assert forbidden not in payload.lower(), (
            f"MONEY IS REAL violation in recent_turns output: '{forbidden}'"
        )


# ---------------------------------------------------------------------------
# Traces router tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def cockpit_app_with_store(tmp_path):
    """Minimal cockpit app with a real TurnStore wired into app.state."""
    from trading_agent.config.db import Database
    from trading_agent.web.app import create_cockpit_app

    db = Database(tmp_path / "config.db")
    app = create_cockpit_app(db=db)
    store = TurnStore(db_path=str(tmp_path / "turns.db"))
    app.state.turn_store = store
    return app, store


def _auth_client(app):
    """Return a TestClient with a session cookie."""
    client = TestClient(app, raise_server_exceptions=True)
    r = client.post("/api/auth/signup", json={"username": "op", "password": "pass1234"})
    assert r.status_code == 200
    return client


def test_traces_list_empty(cockpit_app_with_store):
    app, store = cockpit_app_with_store
    client = _auth_client(app)
    r = client.get("/api/traces?trader_id=Alpha")
    assert r.status_code == 200
    assert r.json() == []


def test_traces_list_returns_summaries(cockpit_app_with_store):
    app, store = cockpit_app_with_store
    store.record(_make_record(turn_id="tapi-001", trader_id="Alpha", book_type="paper"))
    client = _auth_client(app)
    r = client.get("/api/traces?trader_id=Alpha&limit=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["turn_id"] == "tapi-001"
    assert data[0]["book_type"] == "paper"      # operator path — allowed
    assert data[0]["book_badge"] == "[PAPER]"
    assert "first_look_snapshot" not in data[0]  # summary, not full record


def test_traces_get_full_record(cockpit_app_with_store):
    app, store = cockpit_app_with_store
    tc = _make_tool_call(tool_name="account_state")
    store.record(_make_record(turn_id="tapi-002", trader_id="Alpha", tool_calls=[tc]))
    client = _auth_client(app)
    r = client.get("/api/traces/tapi-002")
    assert r.status_code == 200
    d = r.json()
    assert d["turn_id"] == "tapi-002"
    assert d["book_type"] == "paper"         # operator path — allowed
    assert "first_look_snapshot" in d
    assert len(d["tool_calls"]) == 1
    assert d["tool_calls"][0]["tool_name"] == "account_state"


def test_traces_get_missing_returns_404(cockpit_app_with_store):
    app, _ = cockpit_app_with_store
    client = _auth_client(app)
    r = client.get("/api/traces/does-not-exist")
    assert r.status_code == 404


def test_traces_cost_endpoint(cockpit_app_with_store):
    app, store = cockpit_app_with_store
    store.record(_make_record(turn_id="tcost-1", trader_id="CostTr"))
    client = _auth_client(app)
    r = client.get("/api/traces/cost?trader_id=CostTr")
    assert r.status_code == 200
    d = r.json()
    assert "today" in d
    assert "week" in d
    assert "lifetime" in d
    assert d["today"] >= 0.0


def test_traces_attention_no_queue(cockpit_app_with_store):
    """Degrades gracefully when attention_queue not wired."""
    app, _ = cockpit_app_with_store
    client = _auth_client(app)
    r = client.get("/api/traces/attention?trader_id=Alpha")
    assert r.status_code == 200
    d = r.json()
    assert d["watchpoints"] == []
    assert d["reminders"] == []
    assert d["total"] == 0


def test_traces_no_store_list_degrades(tmp_path):
    """GET /api/traces degrades gracefully when turn_store not wired."""
    from trading_agent.config.db import Database
    from trading_agent.web.app import create_cockpit_app

    db = Database(tmp_path / "config.db")
    app = create_cockpit_app(db=db)
    # Deliberately do not set app.state.turn_store
    client = TestClient(app)
    r = client.post("/api/auth/signup", json={"username": "op2", "password": "pass5678"})
    r = client.get("/api/traces?trader_id=Alpha")
    assert r.status_code == 200
    assert r.json() == []


def test_traces_no_store_get_returns_503(tmp_path):
    """GET /api/traces/{id} returns 503 when store not wired."""
    from trading_agent.config.db import Database
    from trading_agent.web.app import create_cockpit_app

    db = Database(tmp_path / "config.db")
    app = create_cockpit_app(db=db)
    client = TestClient(app)
    client.post("/api/auth/signup", json={"username": "op3", "password": "pass9012"})
    r = client.get("/api/traces/some-id")
    assert r.status_code == 503


def test_traces_requires_auth(cockpit_app_with_store):
    """All /api/traces endpoints must reject unauthenticated requests."""
    app, _ = cockpit_app_with_store
    client = TestClient(app)  # no auth
    assert client.get("/api/traces?trader_id=Alpha").status_code == 401
    assert client.get("/api/traces/some-id").status_code == 401
    assert client.get("/api/traces/cost?trader_id=Alpha").status_code == 401
    assert client.get("/api/traces/attention?trader_id=Alpha").status_code == 401
