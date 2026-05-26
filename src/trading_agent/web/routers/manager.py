"""Manager-chat router (WS-E fills in; reads bench + research + memory). Stubbed 501."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...config.users import current_user

router = APIRouter(tags=["manager"])

_NOT_IMPLEMENTED = "not implemented yet (WS-E)"


@router.get("/api/chats")
def list_chats(user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post("/api/chat")
def chat(user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post("/api/chats")
def save_chat(user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)
