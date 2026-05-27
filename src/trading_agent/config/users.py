"""Local accounts + session auth for the cockpit backend.

Real per-user accounts keyed by ``user_id``. Passwords are hashed with
``hashlib.scrypt`` (stdlib, no extra deps); sessions are opaque random tokens
stored server-side with an expiry. The :func:`current_user` FastAPI dependency
resolves the session cookie (or ``Authorization: Bearer``) to a ``user_id`` and
is the single seam every router authenticates through.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
import uuid
from dataclasses import dataclass

from fastapi import HTTPException, Request

from .db import Database

# scrypt cost params. n=2**14/r=8/p=1 ≈ 16 MiB working set — comfortable on a Pi
# 4B and well under OpenSSL's default 32 MiB maxmem ceiling.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 32

SESSION_TTL_SECONDS = 30 * 24 * 3600  # 30 days
SESSION_COOKIE = "session"

# Env fallback for the bench owner (WS-A). The trader-intelligence layer binds
# one owner_user_id so research/memory/settings/cost-gate (all user-namespaced)
# have an identity off the request path. See resolve_owner_user_id.
OWNER_ENV = "TRADING_AGENT_OWNER_ID"


@dataclass
class User:
    id: str
    username: str
    created_at: float


class AuthError(Exception):
    """Username already taken, unknown user, or bad credentials."""


# --- password hashing --------------------------------------------------------


def hash_password(password: str) -> str:
    """Return a self-describing ``scrypt$n$r$p$salt$hash`` string."""
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_DKLEN
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_hash(password: str, encoded: str) -> bool:
    """Constant-time check of ``password`` against a stored hash string."""
    try:
        scheme, n, r, p, salt_hex, hash_hex = encoded.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(bytes.fromhex(hash_hex)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


# --- users -------------------------------------------------------------------


def _row_to_user(row: object) -> User:
    return User(id=row["id"], username=row["username"], created_at=row["created_at"])  # type: ignore[index]


def create_user(db: Database, username: str, password: str) -> User:
    """Create a user. Raises :class:`AuthError` if the username is taken."""
    username = username.strip()
    if not username or not password:
        raise AuthError("username and password are required")
    if db.query_one("SELECT 1 FROM users WHERE username = ?", (username,)) is not None:
        raise AuthError(f"username already taken: {username}")
    user = User(id=uuid.uuid4().hex, username=username, created_at=time.time())
    db.execute(
        "INSERT INTO users (id, username, pw_hash, created_at) VALUES (?, ?, ?, ?)",
        (user.id, user.username, hash_password(password), user.created_at),
    )
    return user


def get_user(db: Database, user_id: str) -> User | None:
    row = db.query_one("SELECT id, username, created_at FROM users WHERE id = ?", (user_id,))
    return _row_to_user(row) if row else None


def authenticate(db: Database, username: str, password: str) -> User:
    """Verify credentials; raise :class:`AuthError` on any mismatch."""
    row = db.query_one(
        "SELECT id, username, pw_hash, created_at FROM users WHERE username = ?",
        (username.strip(),),
    )
    if row is None or not verify_hash(password, row["pw_hash"]):
        raise AuthError("invalid username or password")
    return _row_to_user(row)


# --- owner resolution (WS-A trader intelligence) -----------------------------


def list_user_ids(db: Database) -> list[str]:
    """Every user id, oldest first. Used to detect the single-operator case."""
    rows = db.query("SELECT id FROM users ORDER BY created_at, id")
    return [str(row["id"]) for row in rows]


def resolve_owner_user_id(db: Database, *, explicit: str | None = None) -> str | None:
    """Resolve the one ``user_id`` the bench's intelligence layer binds to.

    Priority: ``explicit`` (the ``--owner`` CLI flag) → the :data:`OWNER_ENV`
    env var → a **lazy single-user fallback** (if exactly one user exists, it is
    the owner) → ``None`` (intelligence features stay dark: history-only, like
    the manager's None-guards).

    A requested value (explicit or env) may be a ``user_id`` *or* a ``username``
    — the operator's convenience. A *stale* request (names no existing user)
    resolves to ``None`` rather than silently binding a different account; this
    is also what lets callers re-resolve each round on a fresh box, picking up
    the owner once they sign up. The single-user fallback only applies when no
    request was made.
    """
    requested = (explicit or os.environ.get(OWNER_ENV) or "").strip()
    if requested:
        if get_user(db, requested) is not None:
            return requested  # matched a real user id
        row = db.query_one("SELECT id FROM users WHERE username = ?", (requested,))
        return str(row["id"]) if row is not None else None  # username, else stale → None

    ids = list_user_ids(db)
    return ids[0] if len(ids) == 1 else None  # lone operator, else zero/many → None


# --- sessions ----------------------------------------------------------------


def create_session(db: Database, user_id: str, ttl: int = SESSION_TTL_SECONDS) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    db.execute(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, now, now + ttl),
    )
    return token


def resolve_session(db: Database, token: str | None) -> str | None:
    """Return the ``user_id`` for a live token, or None if missing/expired."""
    if not token:
        return None
    row = db.query_one(
        "SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)
    )
    if row is None:
        return None
    if row["expires_at"] < time.time():
        delete_session(db, token)
        return None
    return str(row["user_id"])


def delete_session(db: Database, token: str | None) -> None:
    if token:
        db.execute("DELETE FROM sessions WHERE token = ?", (token,))


# --- FastAPI dependencies ----------------------------------------------------


def get_db(request: Request) -> Database:
    """Resolve the app-wide :class:`Database` from app state."""
    return request.app.state.db  # type: ignore[no-any-return]


def _token_from_request(request: Request) -> str | None:
    cookie = request.cookies.get(SESSION_COOKIE)
    if cookie:
        return cookie
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer ") :].strip()
    return None


def current_user(request: Request) -> str:
    """FastAPI dependency → authenticated ``user_id`` (401 if not logged in).

    Every router depends on this; per-user state keys on the returned id.
    """
    db = get_db(request)
    user_id = resolve_session(db, _token_from_request(request))
    if user_id is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user_id
