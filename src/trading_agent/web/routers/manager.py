"""Manager-chat router (WS-E): operator chat + saved conversations.

Wires the cockpit's left-rail chat to the :class:`ManagerAgent` (overseer that
reads the live bench snapshot + research + memory and **only advises**) and the
:class:`ConversationStore` (persistence over the WS-0 ``conversations``/``turns``
tables).

Optional context sources are read off ``app.state`` when present and degrade
gracefully when they are not:
- ``app.state.bench``    — anything exposing ``snapshot()`` (the live ``Bench``)
- ``app.state.research`` — WS-C research store (``recent(user_id, n)``)
- ``app.state.memory``   — WS-D ``MemoryStore`` (``recall(...)``)

Every model call resolves through the endpoint registry (no hardcoded provider)
and is cost-gated to one call per message against the per-user daily ceiling.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config.endpoints import EndpointError, EndpointRegistry
from ...config.settings_store import SettingsStore
from ...config.users import current_user, get_db
from ...manager.agent import ManagerAgent, ManagerConfigError, resolve_manager_ref
from ...manager.chat import ConversationStore
from ...memory.reflect import CostGateError

router = APIRouter(tags=["manager"])


# --- request bodies ----------------------------------------------------------


class ChatIn(BaseModel):
    message: str
    conversation_id: str | None = None


class SaveIn(BaseModel):
    conversation_id: str
    title: str | None = None


# --- wiring helpers ----------------------------------------------------------


def _settings(request: Request) -> SettingsStore:
    return request.app.state.settings  # type: ignore[no-any-return]


def _registry(request: Request) -> EndpointRegistry:
    return request.app.state.endpoints  # type: ignore[no-any-return]


def _conversations(request: Request) -> ConversationStore:
    return ConversationStore(get_db(request))


def _agent(request: Request) -> ManagerAgent:
    state = request.app.state
    return ManagerAgent(
        _registry(request),
        _settings(request),
        _conversations(request),
        bench=getattr(state, "bench", None),
        research=getattr(state, "research", None),
        memory=getattr(state, "memory", None),
    )


# --- routes ------------------------------------------------------------------


@router.get("/api/chats")
def list_chats(request: Request, user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    """Saved (titled) conversations, newest first, each with its turns."""
    return [conv.as_dict() for conv in _conversations(request).list_saved(user_id)]


@router.post("/api/chat")
def chat(
    body: ChatIn, request: Request, user_id: str = Depends(current_user)
) -> dict[str, Any]:
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message is required")

    try:
        ref = resolve_manager_ref(_settings(request), _registry(request), user_id)
    except ManagerConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    store = _conversations(request)
    conv = store.get_or_create(user_id, body.conversation_id)

    try:
        reply = _agent(request).chat(user_id, conv.id, message, ref)
    except CostGateError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except EndpointError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Persist only after a successful reply, so a failed call leaves no orphans.
    store.add_turn(conv.id, "user", message)
    assistant_turn = store.add_turn(conv.id, "assistant", reply)
    return {
        "conversation_id": conv.id,
        "reply": reply,
        "model": ref.model,
        "created_at": assistant_turn.created_at,
    }


@router.post("/api/chats")
def save_chat(
    body: SaveIn, request: Request, user_id: str = Depends(current_user)
) -> dict[str, Any]:
    try:
        conv = _conversations(request).save(user_id, body.conversation_id, body.title)
    except KeyError:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"id": conv.id, "title": conv.title, "started_at": conv.started_at}


@router.delete("/api/chats/{chat_id}")
def delete_chat(
    chat_id: str, request: Request, user_id: str = Depends(current_user)
) -> dict[str, bool]:
    if not _conversations(request).delete(user_id, chat_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"ok": True}
