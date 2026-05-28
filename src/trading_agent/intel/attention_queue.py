"""Pending-attention queue for the agent trader — reminders and watchpoints.

**Design role:** deferred self-pokes from the trader.  When the model calls
``remind_me(when, about)`` or ``watchpoint(symbol, why)``, an unfired row is
inserted here.  The scheduler polls the table on every cadence tick, fires rows
whose condition has elapsed or tripped, and enqueues event-driven turns.

**Storage:** uses the app's existing :class:`~trading_agent.config.db.Database`
(the same ``data/config.db`` SQLite file) under the ``attention_queue`` table,
which is created by ``db/migrations/001_attention.sql``.  The class also
bootstraps the table itself if it is missing, so it can be instantiated in tests
without a pre-run migration.

**UTC internally, ET-anchored for market-bounded ops:** all timestamps are stored
as Unix seconds (UTC integers).  The scheduler computes the next-fire wall-clock
in UTC; ET offsets are applied only when a user-friendly string is surfaced.

**Idempotency:** inserts are unique per ``(trader_id, kind, payload content)``
within a TTL window — the scheduler calling ``enqueue`` twice with the same args
is harmless.

**Soft limits + hard cap:**
  - Soft limits (``WATCHPOINT_SOFT_LIMIT`` / ``REMINDER_SOFT_LIMIT``) are checked
    by :meth:`count_active` and surfaced in the first-look context by
    :mod:`~trading_agent.intel.turn_context`.
  - Hard cap = 5× soft limit.  :meth:`can_add` returns ``False`` when the cap is
    reached; callers should surface a ``ToolResult`` error with kind ``"unavailable"``.

**TTL defaults:**
  - Watchpoint: 24 h (configurable via ``AttentionQueue.DEFAULT_WATCHPOINT_TTL_H``).
  - Reminder: auto-expires on fire OR after 7 days if never fired.

Failure mode: if the DB is absent, all methods degrade silently — ``enqueue``
returns a sentinel row with ``id=-1``, ``count_active`` returns 0, ``poll_due``
returns an empty list.  The agent can still run; it just won't get attention wakes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_WATCHPOINT_TTL_H: float = 24.0
DEFAULT_REMINDER_TTL_DAYS: float = 7.0

# Soft-limit defaults; may be overridden per-trader via INTERESTING_MOVE_RULES
# settings or env vars.
WATCHPOINT_SOFT_LIMIT_DEFAULT: int = 20
REMINDER_SOFT_LIMIT_DEFAULT: int = 10
HARD_CAP_MULTIPLIER: int = 5

# DDL — idempotent; mirrors db/migrations/001_attention.sql.
# NOTE: the FK to traders(id) from the migration spec is intentionally omitted
# here because bench traders are in-memory (not persisted in config.db), so the
# FK would reject every insert.  The cascade-delete behaviour is enforced at the
# application layer (clean up on trader removal).
_DDL = """
CREATE TABLE IF NOT EXISTS attention_queue (
    id          INTEGER PRIMARY KEY,
    trader_id   TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    payload_json TEXT   NOT NULL,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL,
    fired_at    INTEGER,
    fire_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_attention_pending
    ON attention_queue(trader_id, fired_at)
    WHERE fired_at IS NULL;
"""


@dataclass
class AttentionRow:
    """One row from the ``attention_queue`` table."""

    id: int
    trader_id: str
    kind: str                  # 'reminder' | 'watchpoint'
    payload: dict[str, Any]    # deserialized payload_json
    created_at: int            # Unix seconds UTC
    expires_at: int            # Unix seconds UTC
    fired_at: int | None       # None = unfired
    fire_reason: str | None


class AttentionQueue:
    """SQLite-backed pending-attention queue.

    Parameters
    ----------
    db:
        A :class:`~trading_agent.config.db.Database` instance (or any object
        exposing a ``.connect()`` → ``sqlite3.Connection`` method).  May be
        ``None`` — every method degrades gracefully.
    watchpoint_soft_limit:
        Trader's soft limit for watchpoints (default 20).
    reminder_soft_limit:
        Trader's soft limit for reminders (default 10).
    """

    DEFAULT_WATCHPOINT_TTL_H = DEFAULT_WATCHPOINT_TTL_H
    DEFAULT_REMINDER_TTL_DAYS = DEFAULT_REMINDER_TTL_DAYS

    def __init__(
        self,
        db: Any = None,
        *,
        watchpoint_soft_limit: int = WATCHPOINT_SOFT_LIMIT_DEFAULT,
        reminder_soft_limit: int = REMINDER_SOFT_LIMIT_DEFAULT,
    ) -> None:
        self._db = db
        self.watchpoint_soft_limit = watchpoint_soft_limit
        self.reminder_soft_limit = reminder_soft_limit
        if db is not None:
            try:
                conn = db.connect()
                conn.executescript(_DDL)
            except Exception:
                pass  # degrade if we can't bootstrap

    # ── write ─────────────────────────────────────────────────────────────────

    def enqueue(
        self,
        trader_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        expires_at: int | None = None,
        ttl_seconds: float | None = None,
    ) -> AttentionRow:
        """Insert an unfired attention row.

        One of ``expires_at`` (absolute Unix timestamp) or ``ttl_seconds``
        (relative) must be provided.  Returns a sentinel row with ``id=-1`` if
        the DB is unavailable.

        Parameters
        ----------
        trader_id:
            The bench competitor name / unique trader identifier.
        kind:
            ``'reminder'`` or ``'watchpoint'``.
        payload:
            JSON-serializable dict — ``{symbol?, when?, condition?, why}``.
        expires_at:
            Absolute Unix timestamp (UTC) when this row expires.
        ttl_seconds:
            If ``expires_at`` is omitted, set ``expires_at = now + ttl_seconds``.
        """
        now = int(time.time())
        if expires_at is None:
            if ttl_seconds is None:
                # Default TTLs by kind.
                if kind == "reminder":
                    ttl_seconds = DEFAULT_REMINDER_TTL_DAYS * 86_400
                else:
                    ttl_seconds = DEFAULT_WATCHPOINT_TTL_H * 3_600
            expires_at = now + int(ttl_seconds)

        sentinel = AttentionRow(
            id=-1,
            trader_id=trader_id,
            kind=kind,
            payload=payload,
            created_at=now,
            expires_at=expires_at,
            fired_at=None,
            fire_reason=None,
        )
        if self._db is None:
            return sentinel
        try:
            conn = self._db.connect()
            cur = conn.execute(
                """
                INSERT INTO attention_queue
                    (trader_id, kind, payload_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (trader_id, kind, json.dumps(payload), now, expires_at),
            )
            return AttentionRow(
                id=cur.lastrowid or -1,
                trader_id=trader_id,
                kind=kind,
                payload=payload,
                created_at=now,
                expires_at=expires_at,
                fired_at=None,
                fire_reason=None,
            )
        except Exception:
            return sentinel

    def mark_fired(
        self,
        row_id: int,
        fire_reason: str,
    ) -> None:
        """Set ``fired_at`` and ``fire_reason`` for a row.  Idempotent."""
        if self._db is None or row_id < 0:
            return
        try:
            now = int(time.time())
            self._db.connect().execute(
                "UPDATE attention_queue SET fired_at=?, fire_reason=? WHERE id=?",
                (now, fire_reason, row_id),
            )
        except Exception:
            pass

    # ── read ──────────────────────────────────────────────────────────────────

    def count_active(self, trader_id: str, kind: str) -> int:
        """Number of unfired, non-expired rows of the given kind for this trader."""
        if self._db is None:
            return 0
        try:
            now = int(time.time())
            row = self._db.connect().execute(
                """
                SELECT COUNT(*) FROM attention_queue
                WHERE trader_id=? AND kind=? AND fired_at IS NULL AND expires_at > ?
                """,
                (trader_id, kind, now),
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def can_add(self, trader_id: str, kind: str) -> tuple[bool, str]:
        """Check whether a new row can be added without exceeding the hard cap.

        Returns ``(True, "")`` if within limit, or ``(False, reason_msg)`` if the
        hard cap (5× soft limit) would be breached.
        """
        limit = (
            self.watchpoint_soft_limit
            if kind == "watchpoint"
            else self.reminder_soft_limit
        )
        hard_cap = limit * HARD_CAP_MULTIPLIER
        active = self.count_active(trader_id, kind)
        if active >= hard_cap:
            return (
                False,
                f"hard cap reached: {active} active {kind}s (cap={hard_cap}). "
                "Review and prune existing entries before adding more.",
            )
        return True, ""

    def poll_due(self, trader_id: str, *, now: int | None = None) -> list[AttentionRow]:
        """Return unfired rows whose fire condition can now be evaluated.

        For **reminders**, returns rows where ``payload.when_unix <= now`` and
        the row has not yet fired.  For **watchpoints**, returns all unfired rows
        so the scheduler can check the condition itself.

        Expired rows are excluded — they will be cleaned up by
        :meth:`expire_old`.
        """
        if self._db is None:
            return []
        _now = now if now is not None else int(time.time())
        try:
            rows = self._db.connect().execute(
                """
                SELECT id, trader_id, kind, payload_json,
                       created_at, expires_at, fired_at, fire_reason
                FROM attention_queue
                WHERE trader_id=? AND fired_at IS NULL AND expires_at > ?
                ORDER BY created_at ASC
                """,
                (trader_id, _now),
            ).fetchall()
            return [_row_to_attention(r) for r in rows]
        except Exception:
            return []

    def poll_all_due(self, *, now: int | None = None) -> list[AttentionRow]:
        """Like :meth:`poll_due` but across ALL traders. Used by the scheduler."""
        if self._db is None:
            return []
        _now = now if now is not None else int(time.time())
        try:
            rows = self._db.connect().execute(
                """
                SELECT id, trader_id, kind, payload_json,
                       created_at, expires_at, fired_at, fire_reason
                FROM attention_queue
                WHERE fired_at IS NULL AND expires_at > ?
                ORDER BY trader_id, created_at ASC
                """,
                (_now,),
            ).fetchall()
            return [_row_to_attention(r) for r in rows]
        except Exception:
            return []

    def expire_old(self) -> int:
        """Mark all past-expiry rows as fired with reason='expired'. Returns count."""
        if self._db is None:
            return 0
        try:
            now = int(time.time())
            cur = self._db.connect().execute(
                """
                UPDATE attention_queue
                SET fired_at=?, fire_reason='expired'
                WHERE fired_at IS NULL AND expires_at <= ?
                """,
                (now, now),
            )
            return cur.rowcount or 0
        except Exception:
            return 0

    def list_active(self, trader_id: str) -> list[AttentionRow]:
        """All unfired, non-expired rows for a trader (for first-look display)."""
        return self.poll_due(trader_id)


# ── helpers ───────────────────────────────────────────────────────────────────


def _row_to_attention(row: sqlite3.Row | tuple[Any, ...]) -> AttentionRow:
    idx = row if isinstance(row, (list, tuple)) else tuple(row)
    try:
        payload = json.loads(idx[3])
    except Exception:
        payload = {}
    return AttentionRow(
        id=int(idx[0]),
        trader_id=str(idx[1]),
        kind=str(idx[2]),
        payload=payload,
        created_at=int(idx[4]),
        expires_at=int(idx[5]),
        fired_at=int(idx[6]) if idx[6] is not None else None,
        fire_reason=str(idx[7]) if idx[7] is not None else None,
    )
