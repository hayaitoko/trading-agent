"""Conversation persistence for the Manager chat (WS-E).

Backs the cockpit's left-rail chat + "Previous chats" using the ``conversations``
and ``turns`` tables WS-0 created. Everything keys on ``user_id`` for per-user
isolation, mirroring the rest of the cockpit spine.

The lifecycle mirrors the cockpit's mental model:

- Every ``POST /api/chat`` turn is **persisted** to a conversation row (so chats
  survive a refresh). A fresh conversation starts **untitled** (``title IS NULL``).
- **Saving** a conversation sets its title; :meth:`ConversationStore.list_saved`
  returns only titled ones — exactly the cockpit's "Previous chats" list.

There is deliberately no per-id GET route (it isn't in the CONTRACTS route
table): :meth:`list_saved` returns each saved conversation **with its turns**, so
the client can both list and load without an extra round-trip.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.db import Database

# Roles persisted in ``turns``. "system" is never stored — it is rebuilt per
# call from live state by the agent — so history replays cleanly.
USER = "user"
ASSISTANT = "assistant"

_TITLE_MAX = 60
_MAX_HISTORY_TURNS = 40  # cap replayed history so context stays bounded


@dataclass
class Turn:
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: float

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }


@dataclass
class Conversation:
    id: str
    user_id: str
    title: str | None
    started_at: float
    turns: list[Turn] = field(default_factory=list)

    @property
    def saved(self) -> bool:
        """A conversation is "saved" (shown in Previous chats) once titled."""
        return bool(self.title)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "started_at": self.started_at,
            "turns": [t.as_dict() for t in self.turns],
        }


class ConversationStore:
    """CRUD over ``conversations``/``turns``, scoped to a ``user_id``."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # --- create / fetch ------------------------------------------------------

    def create(self, user_id: str, title: str | None = None) -> Conversation:
        conv = Conversation(
            id=uuid.uuid4().hex,
            user_id=user_id,
            title=title,
            started_at=time.time(),
        )
        self._db.execute(
            "INSERT INTO conversations (id, user_id, title, started_at) VALUES (?, ?, ?, ?)",
            (conv.id, conv.user_id, conv.title, conv.started_at),
        )
        return conv

    def get(self, user_id: str, conversation_id: str) -> Conversation | None:
        row = self._db.query_one(
            "SELECT id, user_id, title, started_at FROM conversations"
            " WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        if row is None:
            return None
        conv = Conversation(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            started_at=row["started_at"],
        )
        conv.turns = self.turns(conversation_id)
        return conv

    def get_or_create(self, user_id: str, conversation_id: str | None) -> Conversation:
        """Resolve an existing conversation or start a fresh, untitled one.

        An unknown/foreign ``conversation_id`` yields a brand-new conversation
        (the client adopts the returned id) — never a cross-user leak.
        """
        if conversation_id:
            existing = self.get(user_id, conversation_id)
            if existing is not None:
                return existing
        return self.create(user_id)

    # --- turns ---------------------------------------------------------------

    def add_turn(self, conversation_id: str, role: str, content: str) -> Turn:
        now = time.time()
        cur = self._db.execute(
            "INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        return Turn(
            id=int(cur.lastrowid or 0),
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
        )

    def turns(self, conversation_id: str, *, limit: int | None = None) -> list[Turn]:
        sql = (
            "SELECT id, conversation_id, role, content, created_at FROM turns"
            " WHERE conversation_id = ? ORDER BY id"
        )
        rows = self._db.query(sql, (conversation_id,))
        turns = [
            Turn(
                id=r["id"],
                conversation_id=r["conversation_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
        if limit is not None and len(turns) > limit:
            return turns[-limit:]
        return turns

    def history_messages(self, conversation_id: str) -> list[dict[str, str]]:
        """Prior user/assistant turns as chat messages, newest-bounded.

        Excludes any persisted system rows (there are none today) so the agent
        owns the system prompt and rebuilds live state every call.
        """
        recent = self.turns(conversation_id, limit=_MAX_HISTORY_TURNS)
        return [
            {"role": t.role, "content": t.content}
            for t in recent
            if t.role in (USER, ASSISTANT)
        ]

    # --- save / list / delete ------------------------------------------------

    def save(self, user_id: str, conversation_id: str, title: str | None = None) -> Conversation:
        """Title (and thereby "save") a conversation. Raises ``KeyError`` if absent."""
        conv = self.get(user_id, conversation_id)
        if conv is None:
            raise KeyError(conversation_id)
        new_title = (title or "").strip() or self._derive_title(conv)
        self._db.execute(
            "UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
            (new_title, conversation_id, user_id),
        )
        conv.title = new_title
        return conv

    def list_saved(self, user_id: str) -> list[Conversation]:
        """Saved (titled) conversations, newest first, each with its turns."""
        rows = self._db.query(
            "SELECT id, user_id, title, started_at FROM conversations"
            " WHERE user_id = ? AND title IS NOT NULL ORDER BY started_at DESC",
            (user_id,),
        )
        out: list[Conversation] = []
        for r in rows:
            conv = Conversation(
                id=r["id"],
                user_id=r["user_id"],
                title=r["title"],
                started_at=r["started_at"],
            )
            conv.turns = self.turns(conv.id)
            out.append(conv)
        return out

    def delete(self, user_id: str, conversation_id: str) -> bool:
        """Delete a conversation and its turns. False if not owned/found."""
        cur = self._db.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        if cur.rowcount == 0:
            return False
        # turns has no FK cascade in the WS-0 schema — drop them explicitly.
        self._db.execute("DELETE FROM turns WHERE conversation_id = ?", (conversation_id,))
        return True

    # --- internals -----------------------------------------------------------

    @staticmethod
    def _derive_title(conv: Conversation) -> str:
        """First user line (trimmed) makes the chat list legible, like the mock."""
        for turn in conv.turns:
            if turn.role == USER and turn.content.strip():
                return turn.content.strip()[:_TITLE_MAX]
        stamp = time.strftime("%H:%M", time.localtime(conv.started_at))
        return f"Chat · {stamp}"
