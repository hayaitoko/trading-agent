"""Bench router (WS-G1 fills in; reads bench/risk/approvals state). Stubbed 501."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...config.users import current_user

router = APIRouter(tags=["bench"])

_NOT_IMPLEMENTED = "not implemented yet (WS-G1)"


@router.get("/api/accounts")
def accounts(user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.get("/api/leaderboard")
def leaderboard(user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.get("/api/positions")
def positions(user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.get("/api/activity")
def activity(user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)
