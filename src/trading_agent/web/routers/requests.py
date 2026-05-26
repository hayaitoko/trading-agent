"""Stock-requests router (WS-H fills in; over stock_requests table). Stubbed 501."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...config.users import current_user

router = APIRouter(tags=["requests"])

_NOT_IMPLEMENTED = "not implemented yet (WS-H)"


@router.get("/api/requests")
def requests(user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post("/api/requests/{request_id}/allow")
def allow(request_id: str, user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post("/api/requests/{request_id}/decline")
def decline(request_id: str, user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)
