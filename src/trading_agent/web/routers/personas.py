"""Personas router — expose the trader persona catalog over HTTP.

``GET /api/personas`` returns the full list of named style flavours that the
add-trader wizard (both the admin cockpit and the customer product UI) can
surface to end-users.  Each entry carries the stable ``id`` (pass as ``style``
to ``POST /api/accounts``), a human-readable ``name``, a one-line ``tagline``,
and the full ``mandate`` text that gets injected into the trader system prompt.

No auth required: the persona list is not sensitive — it is a static catalog
baked into the server binary, not user data.
"""

from __future__ import annotations

from fastapi import APIRouter

from ...prompts.personas import PERSONAS, PersonaEntry

router = APIRouter(tags=["personas"])


@router.get("/api/personas")
def list_personas() -> list[PersonaEntry]:
    """Return the full trader persona catalog.

    Each entry: ``{id, name, tagline, mandate}``.

    Pass the ``id`` as ``style`` when creating a trader via ``POST /api/accounts``
    to give the trader that persona's mandate.  The mandate is folded into the
    system prompt as: "Your mandate: <mandate>."

    The "animal_spirits" persona (Keynes' term for the instinct and crowd
    psychology that drives markets) trades on feel — sentiment, momentum,
    narrative, and social energy — explicitly downweighting hard fundamentals.
    """
    return list(PERSONAS)
