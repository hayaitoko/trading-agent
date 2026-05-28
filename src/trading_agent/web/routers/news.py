"""News router: surface ingested ``raw_items`` as headlines for the cockpit.

The Research router (``/api/research``) returns **distilled briefs** — the
LLM's post-processed output. The news tile wants the *unprocessed* feed:
the original ticker-tagged items the WS-B ingest pipeline pulled from RSS,
Reddit, Bluesky, etc. This router exposes that backlog directly off the
``raw_items`` table the worker writes to.

Always per-user (``raw_items`` is keyed by ``user_id`` — same scoping rule as
research). Newest first. Optional ``?symbol=`` narrows to one ticker — the
common case for the in-depth quote window's news tab.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ...config.db import Database
from ...config.users import current_user
from ...ingest.store import IngestStore

router = APIRouter(tags=["news"])

_DEFAULT_LIMIT = 25
_MAX_LIMIT = 200


def _db(request: Request) -> Database:
    return request.app.state.db  # type: ignore[no-any-return]


def _clean_symbol(raw: str) -> str:
    sym = (raw or "").strip().upper()
    if not sym:
        return ""
    if not sym.replace(".", "").replace("-", "").isalnum() or len(sym) > 12:
        raise HTTPException(status_code=400, detail=f"invalid symbol: {raw!r}")
    return sym


@router.get("/api/news")
def news(
    request: Request,
    symbol: str = "",
    limit: int = _DEFAULT_LIMIT,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Recent ingested headlines for ``user_id`` (newest first).

    Params:
      * ``symbol`` — optional, narrows to items whose ``ticker`` column matches.
      * ``limit`` — 1..200, default 25.

    Return: ``{"items": [{title, source, url, ts, ticker}, ...]}``. ``title``
    is the item's text (RSS headline, post body excerpt, etc.); ``source`` is
    the ``source_id`` the worker stamped on the item.
    """
    if limit < 1 or limit > _MAX_LIMIT:
        raise HTTPException(
            status_code=400, detail=f"limit must be 1..{_MAX_LIMIT} (got {limit})"
        )
    db = _db(request)
    # IngestStore() ensures the raw_items schema exists; constructor is idempotent
    # and safe to call from a request — the existing research router does the same.
    IngestStore(db)

    ticker = _clean_symbol(symbol)
    if ticker:
        rows = db.query(
            "SELECT source_id, ticker, text, url, ts FROM raw_items "
            "WHERE user_id = ? AND ticker = ? "
            "ORDER BY fetched_at DESC, id DESC LIMIT ?",
            (user_id, ticker, limit),
        )
    else:
        rows = db.query(
            "SELECT source_id, ticker, text, url, ts FROM raw_items "
            "WHERE user_id = ? "
            "ORDER BY fetched_at DESC, id DESC LIMIT ?",
            (user_id, limit),
        )
    items = [
        {
            "title": str(r["text"]),
            "source": str(r["source_id"]),
            "url": str(r["url"]),
            "ts": str(r["ts"]),
            "ticker": r["ticker"],
        }
        for r in rows
    ]
    return {"symbol": ticker or None, "items": items}
