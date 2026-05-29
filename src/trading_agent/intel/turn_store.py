"""Persistent SQLite store for agent turn traces — the backbone of A5 observability.

**Design role:** every AgentTrader turn is written here — wake reason, first-look
snapshot, the ordered list of tool calls (name, args, result, latency, cost), the
final action, total cost, and total tokens.  The cockpit's ``/api/traces`` endpoints
read from this store; the trader-side ``recent_turns()`` LOOK tool (A1) also reads
from this store so operator and trader share the same ground truth.

**MONEY IS REAL — critical invariant:**
``book_type`` is stored at the row level for operator forensics and audit.
It is **never** included in the dict returned by :meth:`TurnRecord.to_trader_dict`
(the trader-facing path).  The operator-facing :meth:`TurnRecord.to_operator_dict`
includes it with a prominent ``[PAPER]`` / ``[LIVE]`` badge.

**MANAGER FRUGALITY — no LLM calls from this module.**
This module is pure SQLite read/write.  Zero LLM calls, zero HTTP calls.
The cockpit tiles read from ``/api/traces``; the router calls this module;
no polling path invokes any model.

**Thread safety:** a threading.Lock guards all mutations.  Reads are lock-free
(SQLite WAL mode allows concurrent reads).

**Storage:** same ``data/config.db`` as the rest of the cockpit by default.
Pass a different path for test isolation.  Schema is bootstrapped idempotently
on construction via :meth:`TurnStore._bootstrap`.

**Failure mode:** if the DB is absent, :meth:`record` silently returns ``None``
and :meth:`recent` returns an empty list — the agent keeps running without traces.
Errors inside :meth:`record` are swallowed so a trace write never kills a turn.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turn_records (
    turn_id          TEXT PRIMARY KEY,
    trader_id        TEXT NOT NULL,
    started_at       REAL NOT NULL,   -- Unix seconds UTC
    ended_at         REAL,            -- NULL until turn completes
    wake_reason      TEXT NOT NULL,
    turn_type        TEXT NOT NULL,
    book_type        TEXT NOT NULL DEFAULT 'paper',  -- 'paper' | 'live'; OPERATOR ONLY
    first_look_json  TEXT NOT NULL,   -- JSON dict
    tool_calls_json  TEXT NOT NULL,   -- JSON list of ToolCallRecord dicts
    final_action     TEXT NOT NULL DEFAULT 'interrupted',
    final_action_args_json TEXT NOT NULL DEFAULT '{}',
    total_cost_usd   REAL NOT NULL DEFAULT 0.0,
    tokens_input     INTEGER NOT NULL DEFAULT 0,
    tokens_output    INTEGER NOT NULL DEFAULT 0,
    tokens_cached    INTEGER NOT NULL DEFAULT 0,
    previous_attempt_turn_id TEXT      -- NULL unless crash-recovery turn
);

CREATE INDEX IF NOT EXISTS idx_turns_trader
    ON turn_records(trader_id, started_at DESC);
"""

_DEFAULT_DB = os.environ.get("TRADING_AGENT_DB", "data/config.db")

# ---------------------------------------------------------------------------
# Dataclasses (frozen contracts — §Contracts in the plan)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool invocation within a turn.

    ``cost_usd`` is non-zero only for ``model_call`` / ``queued`` tools
    (e.g. ``ask_manager``, ``request_research``).  All LOOK tools that call no
    model have ``cost_usd=0.0``.

    **MONEY IS REAL:** no ``book_type`` here — this record is safe to return
    to the trader via ``recent_turns(include_tool_calls=True)``.
    """

    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any]  # serialised ToolResult.to_dict()
    latency_ms: int
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "result": self.result,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCallRecord:
        return cls(
            tool_name=d.get("tool_name", ""),
            args=d.get("args", {}),
            result=d.get("result", {}),
            latency_ms=int(d.get("latency_ms", 0)),
            cost_usd=float(d.get("cost_usd", 0.0)),
        )


@dataclass
class TurnRecord:
    """Full trace of one agent turn.

    Fields match §Contracts/TurnRecord in the ws-agent plan.

    **MONEY IS REAL:**
    - :meth:`to_trader_dict` never includes ``book_type`` — safe to surface via
      ``recent_turns()`` tool.
    - :meth:`to_operator_dict` includes ``book_type`` with a clear badge for the
      cockpit / audit path.

    ``book_type`` MUST NOT appear in any trader-facing serialisation path.  This
    is enforced by having only one serialiser that omits it (``to_trader_dict``)
    and a separate one for operators.
    """

    turn_id: str
    trader_id: str
    started_at: datetime
    ended_at: datetime | None
    wake_reason: str
    turn_type: Literal["SoD", "regular", "event", "reminder", "callback", "EoD", "tutorial"]
    first_look_snapshot: dict[str, Any]
    tool_calls: list[ToolCallRecord]
    final_action: str
    final_action_args: dict[str, Any]
    total_cost_usd: float
    total_tokens: dict[str, int]  # {input, output, cached}
    previous_attempt_turn_id: str | None
    # OPERATOR ONLY — never in trader-facing path
    _book_type: str = field(default="paper", repr=False)

    # -- Trader-facing serialiser (NO book_type) --

    def to_trader_dict(
        self, *, include_tool_calls: bool = True
    ) -> dict[str, Any]:
        """Serialise for the trader's ``recent_turns()`` tool.

        ``book_type`` is deliberately excluded here.  Adding it would violate
        the MONEY IS REAL cross-cutting rule.
        """
        d: dict[str, Any] = {
            "turn_id": self.turn_id,
            "trader_id": self.trader_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "wake_reason": self.wake_reason,
            "turn_type": self.turn_type,
            "final_action": self.final_action,
            "final_action_args": self.final_action_args,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "previous_attempt_turn_id": self.previous_attempt_turn_id,
        }
        if include_tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        else:
            d["tool_call_count"] = len(self.tool_calls)
        return d

    # -- Operator-facing serialiser (includes book_type badge) --

    def to_operator_dict(
        self, *, include_tool_calls: bool = True
    ) -> dict[str, Any]:
        """Serialise for the cockpit operator UI.

        Includes ``book_type`` with a prominent ``[PAPER]`` / ``[LIVE]`` label
        so operators can clearly see the account mode.  Also includes
        ``first_look_snapshot`` — the full structured context the trader saw
        at the start of the turn (used by the ``turnReplay`` modal).

        This path MUST NOT be used for any trader-facing response.
        """
        d = self.to_trader_dict(include_tool_calls=include_tool_calls)
        d["book_type"] = self._book_type
        d["book_badge"] = "[PAPER]" if self._book_type == "paper" else "[LIVE]"
        d["first_look_snapshot"] = self.first_look_snapshot
        return d

    def to_summary_dict(self) -> dict[str, Any]:
        """Compact summary for the ``GET /api/traces`` list endpoint (operator)."""
        return {
            "turn_id": self.turn_id,
            "trader_id": self.trader_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "wake_reason": self.wake_reason,
            "turn_type": self.turn_type,
            "final_action": self.final_action,
            "total_cost_usd": self.total_cost_usd,
            "tool_call_count": len(self.tool_calls),
            "book_type": self._book_type,
            "book_badge": "[PAPER]" if self._book_type == "paper" else "[LIVE]",
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TurnStore:
    """SQLite-backed store of agent turn traces.

    Usage in the agent loop (AgentTrader)::

        store = TurnStore()          # uses data/config.db by default
        rec = TurnRecord(...)
        store.record(rec)            # write; silently swallows failures

    Usage in A1's ``recent_turns()`` tool::

        records = store.recent("Alpha", n=5)   # list[TurnRecord], newest first

    Usage in the ``/api/traces`` router::

        summaries = store.summaries("Alpha", limit=20)
        full      = store.get("turn-abc-123")

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Defaults to ``TRADING_AGENT_DB`` env var or
        ``data/config.db``.  Pass a ``str(tmp_path)`` in tests for isolation.
    """

    def __init__(self, db_path: str = _DEFAULT_DB) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        try:
            self._conn = self._open()
            self._bootstrap()
        except Exception:
            self._conn = None

    # ------------------------------------------------------------------ internal

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _bootstrap(self) -> None:
        if self._conn is None:
            return
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ write

    def record(self, rec: TurnRecord) -> None:
        """Persist a TurnRecord.  Silently swallows all exceptions.

        The agent turn MUST NOT fail because the trace write failed.
        Callers should not catch any exception from this method — it is
        fully swallowed here.
        """
        if self._conn is None:
            return
        try:
            started_unix = rec.started_at.timestamp()
            ended_unix = rec.ended_at.timestamp() if rec.ended_at else None
            tool_calls_json = json.dumps([tc.to_dict() for tc in rec.tool_calls])
            first_look_json = json.dumps(rec.first_look_snapshot)
            final_args_json = json.dumps(rec.final_action_args)
            tokens = rec.total_tokens or {}
            with self._lock:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO turn_records (
                        turn_id, trader_id, started_at, ended_at,
                        wake_reason, turn_type, book_type,
                        first_look_json, tool_calls_json,
                        final_action, final_action_args_json,
                        total_cost_usd,
                        tokens_input, tokens_output, tokens_cached,
                        previous_attempt_turn_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        rec.turn_id,
                        rec.trader_id,
                        started_unix,
                        ended_unix,
                        rec.wake_reason,
                        rec.turn_type,
                        rec._book_type,
                        first_look_json,
                        tool_calls_json,
                        rec.final_action,
                        final_args_json,
                        rec.total_cost_usd,
                        tokens.get("input", 0),
                        tokens.get("output", 0),
                        tokens.get("cached", 0),
                        rec.previous_attempt_turn_id,
                    ),
                )
                self._conn.commit()
        except Exception:
            # Silent swallow — trace write must never kill a turn.
            pass

    def open_turn(
        self,
        turn_id: str,
        trader_id: str,
        wake_reason: str,
        turn_type: str,
        first_look_snapshot: dict[str, Any],
        book_type: str = "paper",
        previous_attempt_turn_id: str | None = None,
    ) -> None:
        """Insert a minimal in-progress row so crashes are detectable.

        The row will have ``ended_at=NULL`` and ``final_action="interrupted"``
        until :meth:`close_turn` is called.  A4's crash-recovery scan can
        identify these orphaned rows.
        """
        now = time.time()
        if self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO turn_records (
                        turn_id, trader_id, started_at, ended_at,
                        wake_reason, turn_type, book_type,
                        first_look_json, tool_calls_json,
                        final_action, final_action_args_json,
                        total_cost_usd,
                        tokens_input, tokens_output, tokens_cached,
                        previous_attempt_turn_id
                    ) VALUES (?,?,?,NULL,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        turn_id,
                        trader_id,
                        now,
                        wake_reason,
                        turn_type,
                        book_type,
                        json.dumps(first_look_snapshot),
                        "[]",
                        "interrupted",
                        "{}",
                        0.0,
                        0,
                        0,
                        0,
                        previous_attempt_turn_id,
                    ),
                )
                self._conn.commit()
        except Exception:
            pass

    def close_turn(
        self,
        turn_id: str,
        *,
        tool_calls: list[ToolCallRecord],
        final_action: str,
        final_action_args: dict[str, Any],
        total_cost_usd: float,
        total_tokens: dict[str, int],
    ) -> None:
        """Finalise a turn opened with :meth:`open_turn`."""
        if self._conn is None:
            return
        try:
            now = time.time()
            tool_calls_json = json.dumps([tc.to_dict() for tc in tool_calls])
            tokens = total_tokens or {}
            with self._lock:
                self._conn.execute(
                    """
                    UPDATE turn_records
                    SET ended_at=?, tool_calls_json=?, final_action=?,
                        final_action_args_json=?, total_cost_usd=?,
                        tokens_input=?, tokens_output=?, tokens_cached=?
                    WHERE turn_id=?
                    """,
                    (
                        now,
                        tool_calls_json,
                        final_action,
                        json.dumps(final_action_args),
                        total_cost_usd,
                        tokens.get("input", 0),
                        tokens.get("output", 0),
                        tokens.get("cached", 0),
                        turn_id,
                    ),
                )
                self._conn.commit()
        except Exception:
            pass

    # ------------------------------------------------------------------ read

    def recent(self, trader_id: str, n: int = 5) -> list[TurnRecord]:
        """Return the N most recent completed turns for ``trader_id``, newest first.

        Used by A1's ``recent_turns()`` LOOK tool.  Returns completed turns
        only (``ended_at IS NOT NULL``).  The trader-side caller should use
        :meth:`TurnRecord.to_trader_dict` — NOT ``to_operator_dict`` — to
        ensure ``book_type`` is never serialised into the agent-facing response.
        """
        if self._conn is None:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT * FROM turn_records
                WHERE trader_id=? AND ended_at IS NOT NULL
                ORDER BY started_at DESC LIMIT ?
                """,
                (trader_id, max(1, min(int(n), 50))),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]
        except Exception:
            return []

    def recent_all(self, limit: int = 30) -> list[TurnRecord]:
        """Return the N most recent completed turns across ALL traders, newest first.

        Operator-facing: backs the cockpit ``/api/activity`` feed (Gap B —
        WS-LOOKTOOL-WIRING).  Under the agent model, trades settle via ACT tools so
        the legacy ``Bench.recent_decisions()`` log stays empty; agent turn activity
        lives here instead.  Returns completed turns only (``ended_at IS NOT NULL``).
        """
        if self._conn is None:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT * FROM turn_records
                WHERE ended_at IS NOT NULL
                ORDER BY started_at DESC LIMIT ?
                """,
                (max(1, min(int(limit), 200)),),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]
        except Exception:
            return []

    def summaries(
        self,
        trader_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return compact summary dicts for the operator ``GET /api/traces`` list.

        Includes ``book_type`` / ``book_badge`` — OPERATOR path only.
        """
        if self._conn is None:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT * FROM turn_records
                WHERE trader_id=?
                ORDER BY started_at DESC LIMIT ?
                """,
                (trader_id, max(1, min(int(limit), 200))),
            ).fetchall()
            return [self._row_to_record(r).to_summary_dict() for r in rows]
        except Exception:
            return []

    def get(self, turn_id: str) -> TurnRecord | None:
        """Return a full TurnRecord by ID, or None if not found."""
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT * FROM turn_records WHERE turn_id=?", (turn_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        except Exception:
            return None

    def cost_rollup(self, trader_id: str) -> dict[str, float]:
        """Return rolling spend totals for ``trader_id`` (today / week / lifetime).

        Used by the ``costPerTrader`` cockpit tile via ``GET /api/traces``.
        OPERATOR path — callers should not surface these numbers to the trader.
        """
        if self._conn is None:
            return {"today": 0.0, "week": 0.0, "lifetime": 0.0}
        try:
            now = time.time()
            day_start = now - 86400
            week_start = now - 7 * 86400
            row = self._conn.execute(
                """
                SELECT
                    SUM(CASE WHEN started_at >= ? THEN total_cost_usd ELSE 0 END) AS today,
                    SUM(CASE WHEN started_at >= ? THEN total_cost_usd ELSE 0 END) AS week,
                    SUM(total_cost_usd) AS lifetime
                FROM turn_records
                WHERE trader_id=?
                """,
                (day_start, week_start, trader_id),
            ).fetchone()
            if row is None:
                return {"today": 0.0, "week": 0.0, "lifetime": 0.0}
            return {
                "today": float(row["today"] or 0.0),
                "week": float(row["week"] or 0.0),
                "lifetime": float(row["lifetime"] or 0.0),
            }
        except Exception:
            return {"today": 0.0, "week": 0.0, "lifetime": 0.0}

    def orphaned_turns(self) -> list[dict[str, Any]]:
        """Return in-progress (ended_at IS NULL) turns older than 5 minutes.

        Used by A4's crash-recovery scanner to detect stale turns and fire
        new recovery turns.
        """
        if self._conn is None:
            return []
        cutoff = time.time() - 300  # 5 minutes
        try:
            rows = self._conn.execute(
                """
                SELECT turn_id, trader_id, started_at, wake_reason, turn_type,
                       tool_calls_json, book_type
                FROM turn_records
                WHERE ended_at IS NULL AND started_at < ?
                ORDER BY started_at ASC
                """,
                (cutoff,),
            ).fetchall()
            return [
                {
                    "turn_id": r["turn_id"],
                    "trader_id": r["trader_id"],
                    "started_at": r["started_at"],
                    "wake_reason": r["wake_reason"],
                    "turn_type": r["turn_type"],
                    "tool_names": [
                        tc.get("tool_name", "?")
                        for tc in json.loads(r["tool_calls_json"] or "[]")
                    ],
                    "book_type": r["book_type"],
                }
                for r in rows
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TurnRecord:
        tool_calls = [
            ToolCallRecord.from_dict(d)
            for d in json.loads(row["tool_calls_json"] or "[]")
        ]
        started = datetime.fromtimestamp(row["started_at"], tz=UTC)
        ended = (
            datetime.fromtimestamp(row["ended_at"], tz=UTC)
            if row["ended_at"] is not None
            else None
        )
        return TurnRecord(
            turn_id=row["turn_id"],
            trader_id=row["trader_id"],
            started_at=started,
            ended_at=ended,
            wake_reason=row["wake_reason"],
            turn_type=row["turn_type"],  # type: ignore[arg-type]
            first_look_snapshot=json.loads(row["first_look_json"] or "{}"),
            tool_calls=tool_calls,
            final_action=row["final_action"],
            final_action_args=json.loads(row["final_action_args_json"] or "{}"),
            total_cost_usd=float(row["total_cost_usd"] or 0.0),
            total_tokens={
                "input": int(row["tokens_input"] or 0),
                "output": int(row["tokens_output"] or 0),
                "cached": int(row["tokens_cached"] or 0),
            },
            previous_attempt_turn_id=row["previous_attempt_turn_id"],
            _book_type=row["book_type"] or "paper",
        )
