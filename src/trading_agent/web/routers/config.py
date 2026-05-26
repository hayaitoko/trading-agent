"""Config router (WS-0, fully implemented): auth, settings, endpoints, sources.

This is the only router WS-0 fills in; the rest are 501 stubs for their owners.
All per-user state keys on the ``current_user`` ``user_id``. Endpoint API keys are
never returned in full — :meth:`Endpoint.public` masks them.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from ...config import users as users_mod
from ...config.endpoints import EndpointError, EndpointRegistry
from ...config.settings_store import SettingsStore
from ...config.users import SESSION_COOKIE, AuthError, current_user, get_db

router = APIRouter(tags=["config"])


# --- request bodies ----------------------------------------------------------


class Credentials(BaseModel):
    username: str
    password: str


class EndpointIn(BaseModel):
    id: str | None = None
    type: str
    name: str
    base_url: str | None = None
    api_key: str | None = None
    enabled: bool = True


class SourceIn(BaseModel):
    id: str | None = None
    kind: str
    name: str
    config: dict[str, Any] = {}
    enabled: bool = True


# --- helpers -----------------------------------------------------------------


def _registry(request: Request) -> EndpointRegistry:
    return request.app.state.endpoints  # type: ignore[no-any-return]


def _settings(request: Request) -> SettingsStore:
    return request.app.state.settings  # type: ignore[no-any-return]


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE, token, httponly=True, samesite="lax", path="/", max_age=users_mod.SESSION_TTL_SECONDS
    )


# --- auth --------------------------------------------------------------------


@router.post("/api/auth/signup")
def signup(creds: Credentials, request: Request, response: Response) -> dict[str, str]:
    db = get_db(request)
    try:
        user = users_mod.create_user(db, creds.username, creds.password)
    except AuthError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    _registry(request).seed_defaults(user.id)  # default OpenRouter endpoint from env
    token = users_mod.create_session(db, user.id)
    _set_session_cookie(response, token)
    return {"user_id": user.id, "username": user.username, "token": token}


@router.post("/api/auth/login")
def login(creds: Credentials, request: Request, response: Response) -> dict[str, str]:
    db = get_db(request)
    try:
        user = users_mod.authenticate(db, creds.username, creds.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    token = users_mod.create_session(db, user.id)
    _set_session_cookie(response, token)
    return {"user_id": user.id, "username": user.username, "token": token}


@router.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    db = get_db(request)
    token = request.cookies.get(SESSION_COOKIE) or users_mod._token_from_request(request)
    users_mod.delete_session(db, token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/api/me")
def me(request: Request, user_id: str = Depends(current_user)) -> dict[str, Any]:
    user = users_mod.get_user(get_db(request), user_id)
    if user is None:  # session resolved but user vanished
        raise HTTPException(status_code=401, detail="not authenticated")
    return {"user_id": user.id, "username": user.username, "created_at": user.created_at}


# --- settings ----------------------------------------------------------------


@router.get("/api/settings")
def get_settings(request: Request, user_id: str = Depends(current_user)) -> dict[str, Any]:
    return _settings(request).all(user_id)


@router.put("/api/settings")
def put_settings(
    request: Request,
    values: dict[str, Any],
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    # A bare ``dict`` param is read as the JSON request body by FastAPI.
    return _settings(request).update(user_id, values)


# --- endpoints ---------------------------------------------------------------


@router.get("/api/endpoints")
def list_endpoints(request: Request, user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    return [ep.public() for ep in _registry(request).list(user_id)]


@router.post("/api/endpoints")
def upsert_endpoint(
    body: EndpointIn, request: Request, user_id: str = Depends(current_user)
) -> dict[str, Any]:
    reg = _registry(request)
    try:
        if body.id:
            ep = reg.update(
                user_id,
                body.id,
                type=body.type,
                name=body.name,
                base_url=body.base_url,
                api_key=body.api_key,
                enabled=body.enabled,
            )
        else:
            ep = reg.add(
                user_id,
                body.type,
                body.name,
                base_url=body.base_url,
                api_key=body.api_key or "",
                enabled=body.enabled,
            )
    except EndpointError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ep.public()


@router.delete("/api/endpoints/{endpoint_id}")
def delete_endpoint(
    endpoint_id: str, request: Request, user_id: str = Depends(current_user)
) -> dict[str, bool]:
    if not _registry(request).remove(user_id, endpoint_id):
        raise HTTPException(status_code=404, detail="endpoint not found")
    return {"ok": True}


# --- sources (WS-B consumes; WS-0 provides CRUD) -----------------------------


def _source_public(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "name": row["name"],
        "config": json.loads(row["config_json"]),
        "enabled": bool(row["enabled"]),
    }


@router.get("/api/sources")
def list_sources(request: Request, user_id: str = Depends(current_user)) -> list[dict[str, Any]]:
    rows = get_db(request).query(
        "SELECT * FROM sources WHERE user_id = ? ORDER BY name", (user_id,)
    )
    return [_source_public(r) for r in rows]


@router.post("/api/sources")
def upsert_source(
    body: SourceIn, request: Request, user_id: str = Depends(current_user)
) -> dict[str, Any]:
    db = get_db(request)
    config_json = json.dumps(body.config)
    if body.id:
        existing = db.query_one(
            "SELECT id FROM sources WHERE id = ? AND user_id = ?", (body.id, user_id)
        )
        if existing is None:
            raise HTTPException(status_code=404, detail="source not found")
        db.execute(
            "UPDATE sources SET kind = ?, name = ?, config_json = ?, enabled = ?"
            " WHERE id = ? AND user_id = ?",
            (body.kind, body.name, config_json, int(body.enabled), body.id, user_id),
        )
        source_id = body.id
    else:
        source_id = uuid.uuid4().hex
        db.execute(
            "INSERT INTO sources (id, user_id, kind, name, config_json, enabled)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, user_id, body.kind, body.name, config_json, int(body.enabled)),
        )
    row = db.query_one("SELECT * FROM sources WHERE id = ?", (source_id,))
    return _source_public(row)


@router.delete("/api/sources/{source_id}")
def delete_source(
    source_id: str, request: Request, user_id: str = Depends(current_user)
) -> dict[str, bool]:
    cur = get_db(request).execute(
        "DELETE FROM sources WHERE id = ? AND user_id = ?", (source_id, user_id)
    )
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="source not found")
    return {"ok": True}
