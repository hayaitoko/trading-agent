"""Advisor notes: free-text the operator writes about a trader or a ticker.

A note is keyed by ``(user_id, scope, ref)`` where ``scope ∈ {trader, ticker}``
and ``ref`` is the trader name or the ticker symbol. This backs the cockpit's
account-window notes box (``scope="trader"``) and the ticker-card notes
(``scope="ticker"``) — see ``CONTRACTS.md §Per-user model`` (``notes`` table) and
``design/handoff/workstreams/H-requests-notes.md``.

There is at most one note per ``(user_id, scope, ref)``: :meth:`NotesStore.put`
upserts. Everything is isolated per ``user_id``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from .config.db import Database

# The only valid scopes (CONTRACTS.md: scope ∈ {trader, ticker}). The cockpit
# mock labels them "acct"/"ticker"; WS-G maps its label → "trader"/"ticker".
VALID_SCOPES = ("trader", "ticker")


class NoteError(ValueError):
    """Invalid scope or missing reference."""


@dataclass
class Note:
    id: str
    user_id: str
    scope: str
    ref: str
    text: str
    updated_at: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check(scope: str, ref: str) -> None:
    if scope not in VALID_SCOPES:
        raise NoteError(f"scope must be one of {VALID_SCOPES}, got {scope!r}")
    if not ref or not ref.strip():
        raise NoteError("ref is required")


class NotesStore:
    """CRUD over the ``notes`` table, scoped to one user.

    The table is created by WS-0's bootstrap (``config/db.py``); this store only
    reads/writes it.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, user_id: str, scope: str, ref: str) -> Note | None:
        _check(scope, ref)
        row = self._db.query_one(
            "SELECT id, user_id, scope, ref, text, updated_at FROM notes "
            "WHERE user_id = ? AND scope = ? AND ref = ?",
            (user_id, scope, ref),
        )
        return self._row_to_note(row) if row else None

    def put(self, user_id: str, scope: str, ref: str, text: str) -> Note:
        """Upsert the note for ``(user_id, scope, ref)`` and return it."""
        _check(scope, ref)
        now = time.time()
        existing = self.get(user_id, scope, ref)
        if existing is not None:
            self._db.execute(
                "UPDATE notes SET text = ?, updated_at = ? WHERE id = ?",
                (text, now, existing.id),
            )
            return Note(existing.id, user_id, scope, ref, text, now)
        note = Note(uuid.uuid4().hex, user_id, scope, ref, text, now)
        self._db.execute(
            "INSERT INTO notes (id, user_id, scope, ref, text, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (note.id, user_id, scope, ref, text, now),
        )
        return note

    def list(self, user_id: str, scope: str | None = None) -> list[Note]:
        """All of a user's notes, newest first; optionally filtered by scope."""
        if scope is not None and scope not in VALID_SCOPES:
            raise NoteError(f"scope must be one of {VALID_SCOPES}, got {scope!r}")
        if scope is None:
            rows = self._db.query(
                "SELECT id, user_id, scope, ref, text, updated_at FROM notes "
                "WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            )
        else:
            rows = self._db.query(
                "SELECT id, user_id, scope, ref, text, updated_at FROM notes "
                "WHERE user_id = ? AND scope = ? ORDER BY updated_at DESC",
                (user_id, scope),
            )
        return [self._row_to_note(r) for r in rows]

    def delete(self, user_id: str, scope: str, ref: str) -> bool:
        _check(scope, ref)
        cur = self._db.execute(
            "DELETE FROM notes WHERE user_id = ? AND scope = ? AND ref = ?",
            (user_id, scope, ref),
        )
        return (cur.rowcount or 0) > 0

    @staticmethod
    def _row_to_note(row: Any) -> Note:
        return Note(
            id=row["id"],
            user_id=row["user_id"],
            scope=row["scope"],
            ref=row["ref"],
            text=row["text"],
            updated_at=row["updated_at"],
        )
