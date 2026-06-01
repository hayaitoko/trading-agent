"""Memory router: read trader memories from MemoryStore for the customer UI.

``GET /api/memory`` lists all active lessons for the authenticated user.
An optional ``?trader_id=`` narrows to a single trader; ``?q=`` does a
semantic recall search when a MemoryStore is attached to ``app.state``.

Degrades gracefully: when ``app.state.memory`` is absent (no embedder
configured, plain cockpit app) it falls back to an empty list so the
customer UI just shows "no memories yet".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request

from ...config.users import current_user

if TYPE_CHECKING:
    from ...memory.store import MemoryStore

router = APIRouter(tags=["memory"])


def _memory(request: Request) -> MemoryStore | None:
    return getattr(request.app.state, "memory", None)


def _lesson_public(lesson: Any) -> dict[str, Any]:
    return {
        "id": lesson.id,
        "trader_id": lesson.trader_id,
        "text": lesson.text,
        "tags": list(lesson.tags),
        "status": lesson.status,
        "created_at": lesson.created_at,
        "updated_at": lesson.updated_at,
        "score": lesson.score,
    }


@router.get("/api/memory")
def list_memories(
    request: Request,
    trader_id: str | None = None,
    q: str | None = None,
    k: int = 20,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Active trader lessons for *user_id*.

    Query params:
    * ``trader_id`` — optional, filter to one trader.
    * ``q`` — optional semantic search query (requires ``app.state.memory``).
    * ``k`` — max results for semantic search (default 20).

    Returns ``{"lessons": [...], "total": int, "source": "memory"|"empty"}``.
    """
    store = _memory(request)
    if store is None:
        return {"lessons": [], "total": 0, "source": "empty"}

    if q and q.strip():
        # Semantic recall — needs a trader_id to filter namespace properly.
        # When no trader_id given, fall back to listing (can't recall without trader scope).
        if trader_id:
            lessons = store.recall(user_id, trader_id, q.strip(), k=k)
        else:
            lessons = store.list(user_id, trader_id=None)
            # Sort by updated_at as approximation when no query scope.
    else:
        lessons = store.list(user_id, trader_id=trader_id)

    # Cap list results to k as well.
    if not (q and q.strip()):
        lessons = lessons[:k]

    return {
        "lessons": [_lesson_public(lesson) for lesson in lessons],
        "total": len(lessons),
        "source": "memory",
    }
