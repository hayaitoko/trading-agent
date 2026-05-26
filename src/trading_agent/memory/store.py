"""MemoryStore: each trader's private, namespaced long-term memory.

**Namespacing is the whole point** (and what Artoo lacks). Every lesson lives in
a per-user collection and carries both ``user_id`` and ``trader_id`` in its
payload; ``recall`` filters to that exact pair, so trader A can never surface
trader B's lessons — not even within the same user account.

Lessons are soft-deleted, never silently dropped: ``status`` flips
``active → archived`` (see :mod:`trading_agent.memory.hygiene`), and ``recall``
only returns ``active`` ones.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .embed import Embedder
from .vector.base import StoredPoint, VectorStore

KIND = "lessons"
STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"


@dataclass
class Lesson:
    """A durable, decision-changing lesson a trader has learned."""

    id: str
    user_id: str
    trader_id: str
    text: str
    tags: list[str] = field(default_factory=list)
    status: str = STATUS_ACTIVE
    created_at: float = 0.0
    updated_at: float = 0.0
    score: float | None = None  # similarity, set on recall only

    def to_payload(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "trader_id": self.trader_id,
            "text": self.text,
            "tags": list(self.tags),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_point(cls, point: StoredPoint, *, score: float | None = None) -> Lesson:
        p = point.payload
        return cls(
            id=point.id,
            user_id=p.get("user_id", ""),
            trader_id=p.get("trader_id", ""),
            text=p.get("text", ""),
            tags=list(p.get("tags", [])),
            status=p.get("status", STATUS_ACTIVE),
            created_at=float(p.get("created_at", 0.0)),
            updated_at=float(p.get("updated_at", 0.0)),
            score=score,
        )


def collection_for(user_id: str) -> str:
    """The per-(user, kind) collection name. Trader is a payload filter."""
    return f"u:{user_id}:{KIND}"


class MemoryStore:
    """Private per-trader lesson memory over any :class:`VectorStore`."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        """The embedder, exposed so reflection/hygiene can score candidates."""
        return self._embedder

    # --- write ---------------------------------------------------------------

    def remember(
        self,
        user_id: str,
        trader_id: str,
        lesson: str,
        tags: list[str] | None = None,
        *,
        lesson_id: str | None = None,
    ) -> Lesson:
        """Store a lesson for ``(user_id, trader_id)`` and return it."""
        now = time.time()
        obj = Lesson(
            id=lesson_id or uuid.uuid4().hex,
            user_id=user_id,
            trader_id=trader_id,
            text=lesson.strip(),
            tags=tags or [],
            status=STATUS_ACTIVE,
            created_at=now,
            updated_at=now,
        )
        vector = self._embedder.embed(obj.text)
        self._store.upsert(collection_for(user_id), obj.id, vector, obj.to_payload())
        return obj

    # --- read ----------------------------------------------------------------

    def recall(self, user_id: str, trader_id: str, query: str, k: int = 5) -> list[Lesson]:
        """Most-similar **active** lessons for this exact (user, trader) pair."""
        vector = self._embedder.embed(query)
        hits = self._store.search(
            collection_for(user_id),
            vector,
            k,
            flt={"trader_id": trader_id, "status": STATUS_ACTIVE},
        )
        return [
            Lesson.from_point(StoredPoint(id=h.id, vector=[], payload=h.payload), score=h.score)
            for h in hits
        ]

    def get(self, user_id: str, lesson_id: str) -> Lesson | None:
        point = self._store.get(collection_for(user_id), lesson_id)
        return Lesson.from_point(point) if point else None

    def list(
        self,
        user_id: str,
        trader_id: str | None = None,
        *,
        include_archived: bool = False,
    ) -> list[Lesson]:
        """All lessons for the user, optionally narrowed to one trader."""
        flt: dict[str, Any] = {}
        if trader_id is not None:
            flt["trader_id"] = trader_id
        if not include_archived:
            flt["status"] = STATUS_ACTIVE
        points = self._store.iter_points(collection_for(user_id), flt or None)
        lessons = [Lesson.from_point(p) for p in points]
        lessons.sort(key=lambda lesson: lesson.created_at, reverse=True)
        return lessons

    # --- lifecycle -----------------------------------------------------------

    def archive(self, user_id: str, lesson_id: str) -> bool:
        """Soft-delete: flip ``status`` to archived. Returns False if missing."""
        return self._set_status(user_id, lesson_id, STATUS_ARCHIVED)

    def restore(self, user_id: str, lesson_id: str) -> bool:
        return self._set_status(user_id, lesson_id, STATUS_ACTIVE)

    def _set_status(self, user_id: str, lesson_id: str, status: str) -> bool:
        col = collection_for(user_id)
        point = self._store.get(col, lesson_id)
        if point is None:
            return False
        payload = dict(point.payload)
        payload["status"] = status
        payload["updated_at"] = time.time()
        self._store.set_payload(col, lesson_id, payload)
        return True

    def forget(self, user_id: str, lesson_id: str) -> None:
        """Hard-delete. Prefer :meth:`archive`; hygiene never calls this."""
        self._store.delete(collection_for(user_id), lesson_id)
