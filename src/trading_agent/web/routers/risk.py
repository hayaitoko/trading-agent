"""Risk router (WS-H/G fill in; over risk_manager + limits). Stubbed 501."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...config.users import current_user

router = APIRouter(tags=["risk"])

_NOT_IMPLEMENTED = "not implemented yet"


@router.get("/api/risk")
def risk(user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.put("/api/risk/limits")
def put_limits(user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post("/api/risk/kill")
def kill(user_id: str = Depends(current_user)) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)
