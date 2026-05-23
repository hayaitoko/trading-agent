"""SQLite-backed approval queue for human-in-the-loop trade dispatch.

When SignalRouter is in APPROVAL mode it hands proposals here. A consumer
(Telegram bot, CLI, web UI) iterates pending proposals, calls
:meth:`approve` or :meth:`reject`, and the actual execution callback wired
on ``ApprovalQueue`` is invoked on approve.

Pending proposals carry a deadline (now + timeout_seconds). On every
:meth:`pending` or :meth:`process_expirations` call we sweep expired ones
to status ``expired``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Union


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


PathLike = Union[str, Path]


@dataclass
class ApprovalRecord:
    proposal_id: str
    signal: dict[str, Any]
    status: str          # 'pending' | 'approved' | 'rejected' | 'expired'
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None = None
    note: str | None = None
    execution_result: dict[str, Any] | None = None


class ApprovalQueue:
    """Persistent approval queue.

    ``executor`` is the callable invoked on approval — typically a wrapped
    ``broker.place_order``. If unset, approve() raises.
    """

    DEFAULT_TIMEOUT_SECONDS = 300

    def __init__(
        self,
        db_path: PathLike = "data/approvals.db",
        executor: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        default_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.executor = executor
        self.default_timeout_seconds = default_timeout_seconds
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS approvals (
                proposal_id      TEXT PRIMARY KEY,
                signal_json      TEXT NOT NULL,
                status           TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                expires_at       TEXT NOT NULL,
                decided_at       TEXT,
                note             TEXT,
                execution_result TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status)"
        )

    # --- Public API ---------------------------------------------------------

    def add(self, signal: dict[str, Any], timeout_seconds: int | None = None) -> str:
        """Enqueue a proposal. Returns the proposal_id."""
        proposal_id = str(uuid.uuid4())
        now = _utcnow()
        ttl = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        expires = now + timedelta(seconds=ttl)
        self._conn.execute(
            "INSERT INTO approvals(proposal_id, signal_json, status, created_at, expires_at) "
            "VALUES (?, ?, 'pending', ?, ?)",
            (proposal_id, json.dumps(signal, default=str), now.isoformat(), expires.isoformat()),
        )
        return proposal_id

    def approve(self, proposal_id: str, note: str | None = None) -> dict[str, Any] | None:
        """Approve and execute. Returns the executor's result."""
        if self.executor is None:
            raise RuntimeError("Cannot approve: ApprovalQueue has no executor configured")

        self.process_expirations()
        record = self.get(proposal_id)
        if record is None:
            raise KeyError(f"Proposal {proposal_id} not found")
        if record.status != "pending":
            raise ValueError(f"Proposal {proposal_id} is {record.status!r}, not pending")

        result = self.executor(record.signal)
        now = _utcnow().isoformat()
        self._conn.execute(
            "UPDATE approvals SET status='approved', decided_at=?, note=?, execution_result=? "
            "WHERE proposal_id=?",
            (now, note, json.dumps(result, default=str), proposal_id),
        )
        return result

    def reject(self, proposal_id: str, note: str | None = None) -> None:
        record = self.get(proposal_id)
        if record is None:
            raise KeyError(f"Proposal {proposal_id} not found")
        if record.status != "pending":
            raise ValueError(f"Proposal {proposal_id} is {record.status!r}, not pending")
        now = _utcnow().isoformat()
        self._conn.execute(
            "UPDATE approvals SET status='rejected', decided_at=?, note=? WHERE proposal_id=?",
            (now, note, proposal_id),
        )

    def pending(self) -> list[ApprovalRecord]:
        """Return all currently pending (and not-yet-expired) proposals."""
        self.process_expirations()
        rows = self._conn.execute(
            "SELECT * FROM approvals WHERE status='pending' ORDER BY created_at ASC"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get(self, proposal_id: str) -> ApprovalRecord | None:
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE proposal_id=?", (proposal_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def process_expirations(self) -> int:
        """Mark any pending proposals past their expires_at as ``expired``.
        Returns the count moved.
        """
        now_iso = _utcnow().isoformat()
        cursor = self._conn.execute(
            "UPDATE approvals SET status='expired', decided_at=? "
            "WHERE status='pending' AND expires_at <= ?",
            (now_iso, now_iso),
        )
        return cursor.rowcount or 0

    def close(self) -> None:
        self._conn.close()

    # --- Helpers ------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ApprovalRecord:
        return ApprovalRecord(
            proposal_id=row["proposal_id"],
            signal=json.loads(row["signal_json"]),
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            decided_at=datetime.fromisoformat(row["decided_at"]) if row["decided_at"] else None,
            note=row["note"],
            execution_result=json.loads(row["execution_result"]) if row["execution_result"] else None,
        )
