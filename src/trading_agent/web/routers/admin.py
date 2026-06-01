"""Admin router: sysadmin back-office for the trading-agent operator.

Provides operator-level endpoints (prefixed ``/api/admin/*``) to:
  * List, create, and reset passwords for local user accounts.
  * Assign / update API keys and provider endpoints per user.
  * Read/write model defaults and arbitrary system settings.
  * Inspect storage locations and system-level config.

Auth is still the same session cookie / Bearer token, but callers must use the
``admin_user`` dependency instead of ``current_user``. Currently any
authenticated user is treated as an admin (single-operator deployment). The
seam is here so a future role column can gate it without changing callers.

Routes owned by this router (all under /api/admin):
  GET    /api/admin/users
  POST   /api/admin/users
  POST   /api/admin/users/{user_id}/reset-password
  GET    /api/admin/users/{user_id}/endpoints
  POST   /api/admin/users/{user_id}/endpoints
  PUT    /api/admin/users/{user_id}/endpoints/{endpoint_id}
  DELETE /api/admin/users/{user_id}/endpoints/{endpoint_id}
  GET    /api/admin/users/{user_id}/settings
  PUT    /api/admin/users/{user_id}/settings
  GET    /api/admin/system
  GET    /api/admin/storage
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config import users as users_mod
from ...config.db import Database
from ...config.endpoints import EndpointError, EndpointRegistry
from ...config.settings_store import SettingsStore
from ...config.users import AuthError, current_user

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Helpers / dependencies
# ---------------------------------------------------------------------------


def _db(request: Request) -> Database:
    return request.app.state.db  # type: ignore[no-any-return]


def _registry(request: Request) -> EndpointRegistry:
    return request.app.state.endpoints  # type: ignore[no-any-return]


def _settings(request: Request) -> SettingsStore:
    return request.app.state.settings  # type: ignore[no-any-return]


# Any authenticated session is treated as admin in a single-operator deploy.
# Rename this dependency to `require_role("admin")` when roles land.
admin_user = current_user


def _get_user_or_404(db: Database, uid: str) -> dict[str, Any]:
    row = db.query_one(
        "SELECT id, username, created_at FROM users WHERE id = ?", (uid,)
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"user not found: {uid}")
    return {"id": row["id"], "username": row["username"], "created_at": row["created_at"]}


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class CreateUserBody(BaseModel):
    username: str
    password: str


class ResetPasswordBody(BaseModel):
    new_password: str


class EndpointBody(BaseModel):
    type: str
    name: str
    base_url: str | None = None
    api_key: str | None = None
    enabled: bool = True


class SettingsPatch(BaseModel):
    values: dict[str, Any]


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------


@router.get("/users")
def list_users(
    request: Request,
    _: str = Depends(admin_user),
) -> list[dict[str, Any]]:
    """List all user accounts (id, username, created_at, endpoint_count)."""
    db = _db(request)
    rows = db.query("SELECT id, username, created_at FROM users ORDER BY created_at")
    out: list[dict[str, Any]] = []
    for row in rows:
        ep_count = db.query_one(
            "SELECT COUNT(*) AS n FROM endpoints WHERE user_id = ?", (row["id"],)
        )
        out.append(
            {
                "id": row["id"],
                "username": row["username"],
                "created_at": row["created_at"],
                "endpoint_count": int(ep_count["n"]) if ep_count else 0,
            }
        )
    return out


@router.post("/users", status_code=201)
def create_user(
    body: CreateUserBody,
    request: Request,
    _: str = Depends(admin_user),
) -> dict[str, Any]:
    """Provision a new user account and seed default endpoints from env."""
    db = _db(request)
    try:
        user = users_mod.create_user(db, body.username, body.password)
    except AuthError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    # Seed default endpoint from env so the new user can work immediately.
    _registry(request).seed_defaults(user.id)
    return {"id": user.id, "username": user.username, "created_at": user.created_at}


@router.post("/users/{uid}/reset-password")
def reset_password(
    uid: str,
    body: ResetPasswordBody,
    request: Request,
    _: str = Depends(admin_user),
) -> dict[str, str]:
    """Reset a user's password (admin action — no old-password verification)."""
    db = _db(request)
    _get_user_or_404(db, uid)  # raises 404 if absent
    if not body.new_password:
        raise HTTPException(status_code=400, detail="new_password is required")
    new_hash = users_mod.hash_password(body.new_password)
    db.execute("UPDATE users SET pw_hash = ? WHERE id = ?", (new_hash, uid))
    # Invalidate all existing sessions for this user so old tokens stop working.
    db.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
    return {"status": "ok", "user_id": uid}


# ---------------------------------------------------------------------------
# Per-user endpoint (API key) management
# ---------------------------------------------------------------------------


@router.get("/users/{uid}/endpoints")
def list_user_endpoints(
    uid: str,
    request: Request,
    _: str = Depends(admin_user),
) -> list[dict[str, Any]]:
    """List all provider endpoints for a user (keys masked)."""
    db = _db(request)
    _get_user_or_404(db, uid)
    registry = _registry(request)
    return [ep.public() for ep in registry.list(uid)]


@router.post("/users/{uid}/endpoints", status_code=201)
def create_user_endpoint(
    uid: str,
    body: EndpointBody,
    request: Request,
    _: str = Depends(admin_user),
) -> dict[str, Any]:
    """Add a provider endpoint for a user."""
    db = _db(request)
    _get_user_or_404(db, uid)
    registry = _registry(request)
    try:
        ep = registry.add(
            uid,
            body.type,
            body.name,
            base_url=body.base_url,
            api_key=body.api_key or "",
            enabled=body.enabled,
        )
    except EndpointError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ep.public()


@router.put("/users/{uid}/endpoints/{eid}")
def update_user_endpoint(
    uid: str,
    eid: str,
    body: EndpointBody,
    request: Request,
    _: str = Depends(admin_user),
) -> dict[str, Any]:
    """Update an existing endpoint for a user (partial-update semantics)."""
    db = _db(request)
    _get_user_or_404(db, uid)
    registry = _registry(request)
    try:
        ep = registry.update(
            uid,
            eid,
            type=body.type,
            name=body.name,
            base_url=body.base_url,
            api_key=body.api_key,
            enabled=body.enabled,
        )
    except EndpointError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ep.public()


@router.delete("/users/{uid}/endpoints/{eid}")
def delete_user_endpoint(
    uid: str,
    eid: str,
    request: Request,
    _: str = Depends(admin_user),
) -> dict[str, str]:
    """Remove a provider endpoint from a user."""
    db = _db(request)
    _get_user_or_404(db, uid)
    registry = _registry(request)
    removed = registry.remove(uid, eid)
    if not removed:
        raise HTTPException(status_code=404, detail=f"endpoint not found: {eid}")
    return {"status": "deleted", "id": eid}


# ---------------------------------------------------------------------------
# Per-user model defaults / settings
# ---------------------------------------------------------------------------


@router.get("/users/{uid}/settings")
def get_user_settings(
    uid: str,
    request: Request,
    _: str = Depends(admin_user),
) -> dict[str, Any]:
    """Return all settings for a user (merged over defaults)."""
    db = _db(request)
    _get_user_or_404(db, uid)
    store = _settings(request)
    return store.all(uid)


@router.put("/users/{uid}/settings")
def update_user_settings(
    uid: str,
    body: SettingsPatch,
    request: Request,
    _: str = Depends(admin_user),
) -> dict[str, Any]:
    """Bulk-update settings for a user. Returns the full merged settings."""
    db = _db(request)
    _get_user_or_404(db, uid)
    store = _settings(request)
    return store.update(uid, body.values)


# ---------------------------------------------------------------------------
# System info / storage
# ---------------------------------------------------------------------------


@router.get("/system")
def system_info(
    request: Request,
    _: str = Depends(admin_user),
) -> dict[str, Any]:
    """Return system-level configuration and runtime state."""
    db = _db(request)

    # DB stats
    user_count_row = db.query_one("SELECT COUNT(*) AS n FROM users")
    user_count = int(user_count_row["n"]) if user_count_row else 0

    session_count_row = db.query_one(
        "SELECT COUNT(*) AS n FROM sessions WHERE expires_at > ?", (time.time(),)
    )
    session_count = int(session_count_row["n"]) if session_count_row else 0

    endpoint_count_row = db.query_one("SELECT COUNT(*) AS n FROM endpoints")
    endpoint_count = int(endpoint_count_row["n"]) if endpoint_count_row else 0

    # Runtime engine objects attached by the serve process (may be None in dev)
    bench = getattr(request.app.state, "bench", None)
    risk = getattr(request.app.state, "risk", None)
    bus = getattr(request.app.state, "bus", None)

    return {
        "db_path": str(db.path) if hasattr(db, "path") else "unknown",
        "users": user_count,
        "active_sessions": session_count,
        "endpoints": endpoint_count,
        "engine": {
            "bench": bench is not None,
            "risk": risk is not None,
            "bus": bus is not None,
        },
        "env": {
            "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
            "ALPACA_API_KEY": bool(os.environ.get("ALPACA_API_KEY")),
            "FINNHUB_API_KEY": bool(os.environ.get("FINNHUB_API_KEY")),
        },
    }


@router.get("/storage")
def storage_info(
    request: Request,
    _: str = Depends(admin_user),
) -> dict[str, Any]:
    """Return storage locations and sizes for operator inspection."""
    db = _db(request)
    db_path = Path(db.path) if hasattr(db, "path") else None

    def _file_info(p: Path | None) -> dict[str, Any]:
        if p is None or not p.exists():
            return {"path": str(p) if p else None, "exists": False, "size_bytes": None}
        return {
            "path": str(p.resolve()),
            "exists": True,
            "size_bytes": p.stat().st_size,
        }

    # Walk the data directory for additional DB files (vector store, etc.)
    data_dir = db_path.parent if db_path else Path("data")
    extra: list[dict[str, Any]] = []
    if data_dir.is_dir():
        for f in sorted(data_dir.glob("*.db")):
            if db_path and f.resolve() == db_path.resolve():
                continue  # already reported as config_db
            extra.append({"name": f.name, "path": str(f.resolve()), "size_bytes": f.stat().st_size})

    return {
        "config_db": _file_info(db_path),
        "data_dir": str(data_dir.resolve()) if data_dir.is_dir() else str(data_dir),
        "other_dbs": extra,
    }
