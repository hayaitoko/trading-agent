"""WS-Agent A2 — tests for the NOTE toolkit and pending-attention queue.

Coverage:
  - AttentionQueue: enqueue, count_active, can_add (soft/hard cap), poll_due,
    poll_all_due, mark_fired, expire_old — both in-memory (db=None) and with
    a real SQLite DB (tmp_path).
  - ReflectTool: writes lesson, provenance tags, empty note error, no-store path.
  - RemindMeTool: relative/ISO/tomorrow-ET parsing, past-time guard, cap guard.
  - WatchpointTool: condition forms, ttl validation, cap guard, interesting-move
    heuristic via evaluate_condition.
  - WatchSymbolTool / UnwatchSymbolTool: watchlist add/remove, idempotency.
  - AgentTrader integration: NOTE tools available via list_tools; reflect/remind/
    watchpoint/watch/unwatch dispatch correctly; first-look surfaces attention counts.
  - Scheduler scan: _scan_attention fires reminder + watchpoint rows.

MONEY IS REAL check: reflect / remind_me / watchpoint responses contain no
forbidden disclosure words.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from trading_agent.intel.attention_queue import (
    HARD_CAP_MULTIPLIER,
    AttentionQueue,
)
from trading_agent.intel.tools.note import (
    ReflectTool,
    RemindMeTool,
    UnwatchSymbolTool,
    WatchpointTool,
    WatchSymbolTool,
    evaluate_condition,
)
from trading_agent.intel.tools.note.remind_me import _parse_when
from trading_agent.intel.turn_context import build_first_look
from trading_agent.llm.openrouter import ToolCall, ToolCallChatResult
from trading_agent.llm.trader import AgentTrader

# ── Helpers ───────────────────────────────────────────────────────────────────

_FORBIDDEN = {"paper", "sim", "demo", "fake", "test mode"}


def _no_disclosure(text: str) -> bool:
    lower = text.lower()
    return not any(word in lower for word in _FORBIDDEN)


def _make_db(tmp_path: Path) -> Any:
    """Build a real config Database backed by a temp file."""
    from trading_agent.config.db import Database

    return Database(tmp_path / "test.db")


def _aq(db: Any = None, **kw: Any) -> AttentionQueue:
    return AttentionQueue(db, **kw)


# ── AttentionQueue (no-DB / in-memory degrade path) ──────────────────────────


def test_aq_no_db_enqueue_returns_sentinel() -> None:
    aq = _aq()
    row = aq.enqueue("trader-1", "reminder", {"about": "hi"}, ttl_seconds=60)
    assert row.id == -1
    assert row.kind == "reminder"


def test_aq_no_db_count_active_returns_zero() -> None:
    aq = _aq()
    assert aq.count_active("any", "reminder") == 0


def test_aq_no_db_can_add_always_ok() -> None:
    aq = _aq()
    ok, msg = aq.can_add("any", "watchpoint")
    assert ok is True


def test_aq_no_db_poll_due_empty() -> None:
    aq = _aq()
    assert aq.poll_due("any") == []


def test_aq_no_db_expire_old_zero() -> None:
    aq = _aq()
    assert aq.expire_old() == 0


# ── AttentionQueue (real SQLite) ──────────────────────────────────────────────


def test_aq_enqueue_and_count(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    aq = _aq(db)
    aq.enqueue("alice", "reminder", {"about": "check AAPL"}, ttl_seconds=300)
    aq.enqueue("alice", "watchpoint", {"symbol": "AAPL", "why": "watching"}, ttl_seconds=300)
    assert aq.count_active("alice", "reminder") == 1
    assert aq.count_active("alice", "watchpoint") == 1
    # Different trader → 0
    assert aq.count_active("bob", "reminder") == 0


def test_aq_mark_fired(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    aq = _aq(db)
    row = aq.enqueue("alice", "reminder", {"about": "hi"}, ttl_seconds=300)
    assert aq.count_active("alice", "reminder") == 1
    aq.mark_fired(row.id, "elapsed")
    assert aq.count_active("alice", "reminder") == 0


def test_aq_hard_cap(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    limit = 5
    aq = _aq(db, watchpoint_soft_limit=limit)
    hard_cap = limit * HARD_CAP_MULTIPLIER
    for i in range(hard_cap):
        ok, _ = aq.can_add("alice", "watchpoint")
        assert ok, f"Expected can_add=True at {i} entries"
        aq.enqueue("alice", "watchpoint", {"symbol": "X", "why": str(i)}, ttl_seconds=300)
    # Now at hard cap
    ok, msg = aq.can_add("alice", "watchpoint")
    assert ok is False
    assert "hard cap" in msg.lower()


def test_aq_soft_limit_not_blocking(tmp_path: Path) -> None:
    """Soft limit is informational (shown in first-look), never blocks adds."""
    db = _make_db(tmp_path)
    soft = 2
    aq = _aq(db, reminder_soft_limit=soft)
    # Add up to soft limit — all OK
    for i in range(soft + 1):
        ok, _ = aq.can_add("alice", "reminder")
        assert ok, f"should be ok at {i}"
        aq.enqueue("alice", "reminder", {"about": f"r{i}"}, ttl_seconds=300)


def test_aq_poll_due_reminder(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    aq = _aq(db)
    past = int(time.time()) - 10
    future = int(time.time()) + 300
    aq.enqueue("alice", "reminder", {"when_unix": past, "about": "old"}, ttl_seconds=3600)
    aq.enqueue("alice", "reminder", {"when_unix": future, "about": "future"}, ttl_seconds=3600)
    rows = aq.poll_due("alice")
    # Both rows are returned by poll_due (scheduler decides which to fire based on when_unix)
    assert len(rows) == 2


def test_aq_expire_old(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    aq = _aq(db)
    # Insert a row that "expires" in the past by manipulating expires_at directly.
    now = int(time.time())
    conn = db.connect()
    conn.execute(
        "INSERT INTO attention_queue (trader_id, kind, payload_json, created_at, expires_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("alice", "reminder", '{"about":"old"}', now - 200, now - 100),
    )
    assert aq.count_active("alice", "reminder") == 0  # already past expires_at filter
    fired = aq.expire_old()
    assert fired == 1


def test_aq_poll_all_due(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    aq = _aq(db)
    aq.enqueue("alice", "watchpoint", {"symbol": "AAPL", "why": "a"}, ttl_seconds=300)
    aq.enqueue("bob", "reminder", {"when_unix": int(time.time()) - 5, "about": "b"}, ttl_seconds=300)
    rows = aq.poll_all_due()
    traders = {r.trader_id for r in rows}
    assert "alice" in traders
    assert "bob" in traders


# ── evaluate_condition ────────────────────────────────────────────────────────


def test_evaluate_price_condition_trips() -> None:
    payload = {"symbol": "AAPL", "why": "test", "condition": "price > 580"}
    tripped, reason = evaluate_condition(payload, last_prices={"AAPL": 585.0})
    assert tripped is True
    assert "585" in reason


def test_evaluate_price_condition_does_not_trip() -> None:
    payload = {"symbol": "AAPL", "why": "test", "condition": "price > 580"}
    tripped, _ = evaluate_condition(payload, last_prices={"AAPL": 575.0})
    assert tripped is False


def test_evaluate_news_rate_condition() -> None:
    payload = {"symbol": "TSLA", "why": "spike", "condition": "news_rate > 2x"}
    tripped, reason = evaluate_condition(payload, news_rate_ratio={"TSLA": 3.5})
    assert tripped is True
    assert "3.5" in reason or "news_rate" in reason


def test_evaluate_realized_vol_condition() -> None:
    payload = {"symbol": "SPY", "why": "vol spike", "condition": "realized_vol > 1.5x"}
    tripped, reason = evaluate_condition(payload, realized_vol_ratio={"SPY": 2.0})
    assert tripped is True


def test_evaluate_heuristic_price_sigma() -> None:
    """Heuristic: price move > 1σ."""
    payload = {"symbol": "NVDA", "why": "move", "condition": None}
    tripped, reason = evaluate_condition(
        payload,
        price_sigma={"NVDA": 5.0},
        price_change_1h={"NVDA": 6.5},  # > 1σ
    )
    assert tripped is True
    assert "sigma" in reason.lower() or "1σ" in reason


def test_evaluate_heuristic_news_rate() -> None:
    payload = {"symbol": "NVDA", "why": "news", "condition": None}
    tripped, reason = evaluate_condition(payload, news_rate_ratio={"NVDA": 2.5})
    assert tripped is True


def test_evaluate_heuristic_approval_queue() -> None:
    payload = {"symbol": "MSFT", "why": "approval", "condition": None}
    tripped, reason = evaluate_condition(payload, approval_symbols={"MSFT"})
    assert tripped is True
    assert "approval" in reason.lower()


def test_evaluate_heuristic_no_trigger() -> None:
    payload = {"symbol": "MSFT", "why": "watching", "condition": None}
    tripped, _ = evaluate_condition(payload)
    assert tripped is False


# ── ReflectTool ───────────────────────────────────────────────────────────────


@pytest.fixture
def stub_memory() -> Any:
    """Simple stub MemoryStore for tests."""

    @dataclass
    class _Lesson:
        id: str
        user_id: str
        trader_id: str
        text: str
        tags: list[str] = field(default_factory=list)

    class _MemoryStore:
        def __init__(self) -> None:
            self.lessons: list[_Lesson] = []

        def remember(
            self, user_id: str, trader_id: str, text: str, tags: list[str] | None = None
        ) -> _Lesson:
            lesson = _Lesson(
                id=f"l-{len(self.lessons)}",
                user_id=user_id,
                trader_id=trader_id,
                text=text,
                tags=list(tags or []),
            )
            self.lessons.append(lesson)
            return lesson

    return _MemoryStore()


def test_reflect_stores_lesson(stub_memory: Any) -> None:
    tool = ReflectTool(
        memory=stub_memory,
        owner_user_id="user-1",
        trader_id="Alpha",
    )
    result = tool.run("AAPL tends to gap up on earnings in low vol environments.")
    assert result.ok is True
    assert result.data["stored"] is True
    assert result.data["lesson_id"] is not None
    assert len(stub_memory.lessons) == 1


def test_reflect_provenance_in_tags(stub_memory: Any) -> None:
    tool = ReflectTool(
        memory=stub_memory,
        owner_user_id="user-1",
        trader_id="Beta",
    )
    tool.run("lesson text", tags=["AAPL"], tool_call_names=["history", "news"])
    tags = stub_memory.lessons[0].tags
    prov_tags = [t for t in tags if t.startswith("provenance:")]
    assert prov_tags, "Expected provenance tag"
    assert "history" in prov_tags[0]


def test_reflect_empty_note_error() -> None:
    tool = ReflectTool(trader_id="T")
    result = tool.run("")
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind == "invalid_input"


def test_reflect_no_store_graceful() -> None:
    """No memory store → ok=True but stored=False (lesson is lost gracefully)."""
    tool = ReflectTool(memory=None, owner_user_id=None, trader_id="T")
    result = tool.run("a lesson")
    assert result.ok is True
    assert result.data["stored"] is False


def test_reflect_no_disclosure(stub_memory: Any) -> None:
    tool = ReflectTool(
        memory=stub_memory,
        owner_user_id="user-1",
        trader_id="T",
    )
    result = tool.run("momentum play worked well on AAPL today")
    assert _no_disclosure(json.dumps(result.to_dict()))


# ── RemindMeTool ─────────────────────────────────────────────────────────────


def test_parse_when_relative_seconds() -> None:
    now = int(time.time())
    ts = _parse_when("in 5s")
    assert abs(ts - (now + 5)) <= 2


def test_parse_when_relative_minutes() -> None:
    now = int(time.time())
    ts = _parse_when("in 15min")
    assert abs(ts - (now + 15 * 60)) <= 2


def test_parse_when_relative_hours() -> None:
    now = int(time.time())
    ts = _parse_when("in 2h")
    assert abs(ts - (now + 7200)) <= 2


def test_parse_when_relative_days() -> None:
    now = int(time.time())
    ts = _parse_when("in 1d")
    assert abs(ts - (now + 86400)) <= 5


def test_parse_when_iso() -> None:
    ts = _parse_when("2030-01-01T12:00:00Z")
    expected = int(datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp())
    assert ts == expected


def test_parse_when_iso_no_tz() -> None:
    """No-TZ ISO string treated as UTC."""
    ts = _parse_when("2030-06-15T09:30:00")
    expected = int(datetime(2030, 6, 15, 9, 30, 0, tzinfo=UTC).timestamp())
    assert ts == expected


def test_parse_when_invalid_raises() -> None:
    with pytest.raises(ValueError):
        _parse_when("next friday at noon")


def test_remind_me_future(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    aq = _aq(db)
    tool = RemindMeTool(attention_queue=aq, trader_id="Alpha")
    result = tool.run("in 1d", "check AAPL after earnings")
    assert result.ok is True
    assert result.data["stored"] is True
    assert aq.count_active("Alpha", "reminder") == 1


def test_remind_me_past_time_error(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    aq = _aq(db)
    tool = RemindMeTool(attention_queue=aq, trader_id="Alpha")
    # A time clearly in the past.
    result = tool.run("2020-01-01T00:00:00Z", "old reminder")
    assert result.ok is False
    assert result.error is not None
    assert result.error.kind == "invalid_input"


def test_remind_me_empty_about_error() -> None:
    tool = RemindMeTool(trader_id="T")
    result = tool.run("in 1h", "")
    assert result.ok is False
    assert result.error.kind == "invalid_input"


def test_remind_me_hard_cap(tmp_path: Path) -> None:
    """Can't exceed hard cap × soft_limit reminders."""
    db = _make_db(tmp_path)
    aq = _aq(db, reminder_soft_limit=2)
    hard_cap = 2 * HARD_CAP_MULTIPLIER
    for i in range(hard_cap):
        tool = RemindMeTool(attention_queue=aq, trader_id="T")
        result = tool.run(f"in {i + 1}d", f"reminder {i}")
        assert result.ok, f"Expected ok at {i}"
    # At cap → error.
    tool = RemindMeTool(attention_queue=aq, trader_id="T")
    result = tool.run("in 99d", "one too many")
    assert result.ok is False
    assert result.error.kind == "unavailable"


def test_remind_me_no_disclosure() -> None:
    tool = RemindMeTool(trader_id="T")
    result = tool.run("in 1h", "follow up on my open positions")
    assert _no_disclosure(json.dumps(result.to_dict()))


# ── WatchpointTool ────────────────────────────────────────────────────────────


def test_watchpoint_stores(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    aq = _aq(db)
    tool = WatchpointTool(attention_queue=aq, trader_id="Alpha")
    result = tool.run("AAPL", "earnings breakout setup")
    assert result.ok is True
    assert result.data["stored"] is True
    assert result.data["symbol"] == "AAPL"
    assert aq.count_active("Alpha", "watchpoint") == 1


def test_watchpoint_with_condition(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    aq = _aq(db)
    tool = WatchpointTool(attention_queue=aq, trader_id="Alpha")
    result = tool.run("TSLA", "price target", condition="price > 300")
    assert result.ok is True
    assert "price > 300" in result.data["condition"]


def test_watchpoint_default_ttl_label() -> None:
    tool = WatchpointTool(trader_id="T")
    result = tool.run("MSFT", "watching")
    assert result.ok is True
    assert result.data["condition"] == "interesting-move heuristic"


def test_watchpoint_ttl_too_large_error() -> None:
    tool = WatchpointTool(trader_id="T")
    result = tool.run("X", "why", ttl_hours=999)
    assert result.ok is False
    assert result.error.kind == "invalid_input"


def test_watchpoint_empty_symbol_error() -> None:
    tool = WatchpointTool(trader_id="T")
    result = tool.run("", "watching")
    assert result.ok is False


def test_watchpoint_hard_cap(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    aq = _aq(db, watchpoint_soft_limit=2)
    hard_cap = 2 * HARD_CAP_MULTIPLIER
    for i in range(hard_cap):
        tool = WatchpointTool(attention_queue=aq, trader_id="T")
        result = tool.run(f"SYM{i}", f"watching {i}")
        assert result.ok, f"Failed at {i}"
    tool = WatchpointTool(attention_queue=aq, trader_id="T")
    result = tool.run("OVERFLOW", "one too many")
    assert result.ok is False
    assert result.error.kind == "unavailable"


def test_watchpoint_no_disclosure() -> None:
    tool = WatchpointTool(trader_id="T")
    result = tool.run("AAPL", "momentum setup on breakout")
    assert _no_disclosure(json.dumps(result.to_dict()))


# ── WatchSymbolTool / UnwatchSymbolTool ───────────────────────────────────────


@pytest.fixture
def stub_settings() -> Any:
    """Simple in-memory settings store stub."""

    class _Settings:
        def __init__(self) -> None:
            self._data: dict[str, dict[str, str]] = {}

        def get(self, user_id: str, key: str, default: Any = None) -> Any:
            import json as _json

            raw = self._data.get(user_id, {}).get(key)
            if raw is None:
                return default
            return _json.loads(raw)

        def set(self, user_id: str, key: str, value: Any) -> None:
            import json as _json

            self._data.setdefault(user_id, {})[key] = _json.dumps(value)

    return _Settings()


def test_watch_symbol_adds(stub_settings: Any) -> None:
    tool = WatchSymbolTool(
        settings_store=stub_settings, owner_user_id="u1", trader_id="T"
    )
    result = tool.run("AAPL")
    assert result.ok is True
    assert result.data["added"] is True
    assert "AAPL" in result.data["watchlist"]


def test_watch_symbol_idempotent(stub_settings: Any) -> None:
    tool = WatchSymbolTool(
        settings_store=stub_settings, owner_user_id="u1", trader_id="T"
    )
    tool.run("AAPL")
    result = tool.run("AAPL")
    assert result.data["added"] is False
    assert result.data["watchlist"].count("AAPL") == 1


def test_unwatch_symbol_removes(stub_settings: Any) -> None:
    w_tool = WatchSymbolTool(
        settings_store=stub_settings, owner_user_id="u1", trader_id="T"
    )
    uw_tool = UnwatchSymbolTool(
        settings_store=stub_settings, owner_user_id="u1", trader_id="T"
    )
    w_tool.run("MSFT")
    result = uw_tool.run("MSFT")
    assert result.ok is True
    assert result.data["removed"] is True
    assert "MSFT" not in result.data["watchlist"]


def test_unwatch_symbol_idempotent(stub_settings: Any) -> None:
    tool = UnwatchSymbolTool(
        settings_store=None, owner_user_id=None, trader_id="T"
    )
    result = tool.run("NOTHERE")
    assert result.ok is True
    assert result.data["removed"] is False


def test_watch_empty_symbol_error() -> None:
    tool = WatchSymbolTool(trader_id="T")
    result = tool.run("")
    assert result.ok is False
    assert result.error.kind == "invalid_input"


# ── AgentTrader integration (NOTE tools wired) ───────────────────────────────

# Stub LLM client that emits a scripted sequence of ToolCallChatResult.


@dataclass
class _FakeTCResult:
    tool_calls: list[ToolCall]
    content: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0


class _ScriptedClient:
    def __init__(self, script: list[_FakeTCResult]) -> None:
        self._script = list(script)
        self._idx = 0

    def chat_with_tools(self, model: str, messages: list, **kw: Any) -> ToolCallChatResult:
        result = self._script[self._idx % len(self._script)]
        self._idx += 1
        return ToolCallChatResult(
            content=result.content,
            tool_calls=result.tool_calls,
            model=model,
            usage=result.usage,
            cost=result.cost,
        )


def _tc(name: str, **args: Any) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=args)


def test_agent_note_list_tools_includes_note_catalog() -> None:
    """list_tools() must include all five A2 NOTE tools."""
    client = _ScriptedClient(
        [
            _FakeTCResult(tool_calls=[_tc("list_tools")]),
            _FakeTCResult(tool_calls=[_tc("pass")]),
        ]
    )
    trader = AgentTrader(
        "test/model", client, symbols=["AAPL"], name="NoteTrader"
    )
    result = trader.decide({"cash": 10_000.0, "positions": []})
    assert result.error is None
    # Inspect what list_tools returned by checking the trader directly.
    catalog_result = trader._tool_list_tools()
    names = [e["name"] for e in catalog_result.data["tools"]]
    for expected in ["reflect", "remind_me", "watchpoint", "watch_symbol", "unwatch_symbol"]:
        assert expected in names, f"'{expected}' missing from list_tools catalog"


def test_agent_reflect_dispatches(stub_memory: Any, tmp_path: Path) -> None:
    """reflect tool stores a lesson in the memory store."""
    client = _ScriptedClient(
        [
            _FakeTCResult(
                tool_calls=[
                    _tc("reflect", note="AAPL gaps up on earnings in calm vol"),
                    _tc("hold", reason="no trades today"),
                ]
            ),
        ]
    )
    trader = AgentTrader(
        "test/model",
        client,
        symbols=["AAPL"],
        name="ReflectTrader",
        memory=stub_memory,
        owner_user_id="u1",
    )
    result = trader.decide({"cash": 10_000.0, "positions": []})
    assert result.error is None
    assert len(stub_memory.lessons) == 1
    assert "AAPL" in stub_memory.lessons[0].text


def test_agent_remind_me_dispatches(tmp_path: Path) -> None:
    """remind_me tool inserts a row in the attention queue."""
    db = _make_db(tmp_path)
    aq = _aq(db)
    client = _ScriptedClient(
        [
            _FakeTCResult(
                tool_calls=[
                    _tc("remind_me", when="in 1h", about="check AAPL momentum"),
                    _tc("pass"),
                ]
            ),
        ]
    )
    trader = AgentTrader(
        "test/model",
        client,
        symbols=["AAPL"],
        name="RemindTrader",
        attention_queue=aq,
    )
    trader.decide({"cash": 10_000.0, "positions": []})
    assert aq.count_active("RemindTrader", "reminder") == 1


def test_agent_watchpoint_dispatches(tmp_path: Path) -> None:
    """watchpoint tool inserts a row in the attention queue."""
    db = _make_db(tmp_path)
    aq = _aq(db)
    client = _ScriptedClient(
        [
            _FakeTCResult(
                tool_calls=[
                    _tc("watchpoint", symbol="AAPL", why="earnings setup", condition="price > 200"),
                    _tc("pass"),
                ]
            ),
        ]
    )
    trader = AgentTrader(
        "test/model",
        client,
        symbols=["AAPL"],
        name="WPTrader",
        attention_queue=aq,
    )
    trader.decide({"cash": 10_000.0, "positions": []})
    assert aq.count_active("WPTrader", "watchpoint") == 1


def test_agent_watch_unwatch_dispatches(tmp_path: Path) -> None:
    """watch_symbol / unwatch_symbol round-trip via the trader."""
    from trading_agent.config.db import Database
    from trading_agent.config.settings_store import SettingsStore

    db_real = Database(tmp_path / "cfg.db")
    settings = SettingsStore(db_real)

    # Need at least one user so we have an owner_user_id.
    owner = "u-test"

    watch_client = _ScriptedClient(
        [
            _FakeTCResult(tool_calls=[_tc("watch_symbol", symbol="NVDA"), _tc("pass")]),
        ]
    )
    trader = AgentTrader(
        "test/model",
        watch_client,
        symbols=["NVDA"],
        name="WatchTrader",
        owner_user_id=owner,
        settings_store=settings,
    )
    trader.decide({"cash": 10_000.0, "positions": []})

    from trading_agent.intel.tools.note.watch_symbol import _load_watchlist

    wl = _load_watchlist(settings, owner, "WatchTrader")
    assert "NVDA" in wl

    uw_client = _ScriptedClient(
        [
            _FakeTCResult(tool_calls=[_tc("unwatch_symbol", symbol="NVDA"), _tc("pass")]),
        ]
    )
    trader2 = AgentTrader(
        "test/model",
        uw_client,
        symbols=["NVDA"],
        name="WatchTrader",
        owner_user_id=owner,
        settings_store=settings,
    )
    trader2.decide({"cash": 10_000.0, "positions": []})

    wl2 = _load_watchlist(settings, owner, "WatchTrader")
    assert "NVDA" not in wl2


def test_agent_first_look_attention_counts(tmp_path: Path) -> None:
    """Active watchpoints + reminders must surface in first-look."""
    db = _make_db(tmp_path)
    aq = _aq(db)
    aq.enqueue("CountTrader", "watchpoint", {"symbol": "AAPL", "why": "x"}, ttl_seconds=3600)
    aq.enqueue("CountTrader", "reminder", {"about": "y", "when_unix": int(time.time()) + 3600}, ttl_seconds=86400)

    client = _ScriptedClient([_FakeTCResult(tool_calls=[_tc("pass")])])
    trader = AgentTrader(
        "test/model",
        client,
        symbols=["AAPL"],
        name="CountTrader",
        attention_queue=aq,
    )
    # Peek at the first-look string directly.
    ctx = trader._build_turn_context({"cash": 50_000.0, "positions": []})
    fl = build_first_look(ctx)
    assert "1 active watchpoints" in fl
    assert "1 active reminders" in fl


# ── Scheduler scan ────────────────────────────────────────────────────────────


def test_scan_attention_fires_elapsed_reminder(tmp_path: Path) -> None:
    """_scan_attention marks reminders whose when_unix has elapsed."""
    db = _make_db(tmp_path)
    aq = _aq(db)
    # Insert a reminder that's already due.
    past = int(time.time()) - 5
    aq.enqueue(
        "ScanTrader", "reminder",
        {"when_unix": past, "about": "fire me"},
        ttl_seconds=3600,
    )

    # Build a minimal BenchController with our trader + queue.
    from trading_agent.bench.bench import Bench
    from trading_agent.bench.controller import BenchController

    # Stub client — should NOT be called during _scan_attention itself.
    no_call_client = _ScriptedClient([_FakeTCResult(tool_calls=[_tc("pass")])])

    bench = Bench(["AAPL"])
    trader = AgentTrader(
        "test/model",
        no_call_client,
        symbols=["AAPL"],
        name="ScanTrader",
        attention_queue=aq,
    )
    # Use a duck-typed stub for the OpenRouterClient parameter.
    ctrl = BenchController(bench, no_call_client, symbols=["AAPL"])
    # Add trader manually (bypassing LLMTrader creation).
    bench.add_competitor("ScanTrader", trader)

    # Run the scan.
    ctrl._do_scan_attention()

    # Row should now be fired.
    remaining = aq.count_active("ScanTrader", "reminder")
    assert remaining == 0


def test_scan_attention_fires_watchpoint_condition(tmp_path: Path) -> None:
    """_scan_attention fires a watchpoint when the price condition trips."""
    db = _make_db(tmp_path)
    aq = _aq(db)
    aq.enqueue(
        "ScanTrader2", "watchpoint",
        {"symbol": "AAPL", "why": "price target", "condition": "price > 200"},
        ttl_seconds=3600,
    )

    from trading_agent.bench.bench import Bench
    from trading_agent.bench.controller import BenchController

    no_call_client = _ScriptedClient([_FakeTCResult(tool_calls=[_tc("pass")])])
    bench = Bench(["AAPL"])
    bench._last_prices["AAPL"] = 210.0  # above 200

    trader = AgentTrader(
        "test/model",
        no_call_client,
        symbols=["AAPL"],
        name="ScanTrader2",
        attention_queue=aq,
    )
    bench.add_competitor("ScanTrader2", trader)
    ctrl = BenchController(bench, no_call_client, symbols=["AAPL"])

    ctrl._do_scan_attention()

    remaining = aq.count_active("ScanTrader2", "watchpoint")
    assert remaining == 0
