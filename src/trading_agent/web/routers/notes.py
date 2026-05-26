"""Advisor-notes router (WS-H): get/put notes keyed (user_id, scope, ref).

Backs the cockpit's account-window notes box (``scope=trader``) and the
ticker-card notes (``scope=ticker``). One note per (user, scope, ref); PUT
upserts. See ``CONTRACTS.md §HTTP route table`` (``GET/PUT /api/notes?scope=&ref=``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config.users import current_user, get_db
from ...notes import NoteError, NotesStore

router = APIRouter(tags=["notes"])


class NoteBody(BaseModel):
    text: str = ""


def _store(request: Request) -> NotesStore:
    return NotesStore(get_db(request))


@router.get("/api/notes")
def get_note(
    request: Request,
    scope: str,
    ref: str,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Return the note for ``(user, scope, ref)`` — ``text`` is "" if unset."""
    try:
        note = _store(request).get(user_id, scope, ref)
    except NoteError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if note is None:
        return {"scope": scope, "ref": ref, "text": "", "updated_at": None}
    return note.as_dict()


@router.put("/api/notes")
def put_note(
    request: Request,
    body: NoteBody,
    scope: str,
    ref: str,
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Upsert the note text for ``(user, scope, ref)``."""
    try:
        note = _store(request).put(user_id, scope, ref, body.text)
    except NoteError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return note.as_dict()
