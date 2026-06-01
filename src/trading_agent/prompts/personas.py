"""Trader persona registry.

A *persona* is a named style flavour that shapes an AgentTrader's system-prompt
mandate (the ``style`` constructor parameter).  Each persona entry exposes a
human-readable name, a short tagline, and the full mandate text that gets
injected into the system prompt as:

    "Your mandate: <style>."

Design rules
------------
- The mandate is trader-facing text.  MONEY IS REAL invariant applies: the
  words "paper", "sim", "demo", "fake", and "test mode" must never appear here.
- Personas downweight fundamentals or overweight them as appropriate for their
  character — they do NOT make the fundamentals unavailable; the LOOK toolkit
  stays fully accessible.
- ``PERSONAS`` is the canonical source of truth; the ``/api/personas`` endpoint
  serialises it.  Both the admin cockpit and the customer product UI read from
  that endpoint to populate their add-trader wizards.

Adding a new persona
--------------------
Append an entry to ``PERSONAS``.  The ``id`` is the stable machine key used in
API payloads; ``name`` is the display string; ``mandate`` is the style string
threaded into ``AgentTrader(style=...)``.
"""

from __future__ import annotations

from typing import TypedDict


class PersonaEntry(TypedDict):
    id: str
    name: str
    tagline: str
    mandate: str


# ---------------------------------------------------------------------------
# Canonical persona registry
# ---------------------------------------------------------------------------

PERSONAS: list[PersonaEntry] = [
    {
        "id": "balanced",
        "name": "Balanced",
        "tagline": "Weighs fundamentals, technicals, and sentiment equally.",
        "mandate": (
            "Approach each decision with equal weight on fundamentals, technical "
            "signals, and market sentiment. Seek opportunities with strong confirmation "
            "across all three dimensions before committing capital."
        ),
    },
    {
        "id": "momentum",
        "name": "Momentum",
        "tagline": "Rides trends and price breakouts; fades on weakness.",
        "mandate": (
            "Follow price momentum: buy strength, sell weakness. Enter on confirmed "
            "breakouts above recent highs or below recent lows, ride the trend, and "
            "exit when momentum stalls. Ignore valuation — price action is truth."
        ),
    },
    {
        "id": "value",
        "name": "Value",
        "tagline": "Seeks undervalued assets with margin of safety.",
        "mandate": (
            "Hunt for assets priced below intrinsic value. Prioritise low P/E, "
            "strong balance sheets, and durable earnings. Be patient: only enter "
            "when there is a meaningful margin of safety. Ignore short-term noise."
        ),
    },
    {
        "id": "animal_spirits",
        "name": "Animal Spirits",
        "tagline": (
            "Trades on feel — sentiment, crowd psychology, narrative, and momentum. "
            "Explicitly downweights hard fundamentals."
        ),
        "mandate": (
            "You are driven by Keynes' 'animal spirits' — the raw instinct and "
            "crowd psychology that actually moves markets in the short run. "
            "Your edge is reading the room: what is the narrative right now? "
            "What will retail traders do when they wake up and see this headline? "
            "Where is the crowd's attention flowing? "
            "Prioritise: sentiment scores, social momentum, options flow, "
            "short-squeeze setups, meme-stock energy, earnings-reaction patterns, "
            "and narrative velocity. "
            "Explicitly downweight hard fundamentals and valuation metrics — "
            "a stock can be wildly overvalued and still run 40% on a good story. "
            "Your mantra: 'markets are not a weighing machine in the short run; "
            "they are a voting machine — and you are here to vote with the crowd "
            "before the crowd votes.' "
            "Use news(), research_brief(), and prediction_market_odds() as your "
            "primary lenses. Trust your gut when the data is ambiguous."
        ),
    },
    {
        "id": "contrarian",
        "name": "Contrarian",
        "tagline": "Fades consensus; buys fear, sells greed.",
        "mandate": (
            "Bet against the crowd. When sentiment is euphoric, look to fade the "
            "move; when the crowd is panicking, look for entry points. Seek "
            "situations where the narrative has overshot the reality. Your edge "
            "is keeping a cool head when everyone else is emotional."
        ),
    },
    {
        "id": "macro",
        "name": "Macro",
        "tagline": "Trades on economic regime, rates, and cross-asset flows.",
        "mandate": (
            "Think top-down: start from the macro regime (risk-on vs. risk-off, "
            "rate environment, credit spreads, USD trend) and work down to sector "
            "and individual names. World events, central bank policy, and "
            "geopolitical shifts are your primary signals."
        ),
    },
]

# Stable lookup by id for O(1) access by the trader constructor or API layer.
PERSONAS_BY_ID: dict[str, PersonaEntry] = {p["id"]: p for p in PERSONAS}


def get_persona_mandate(persona_id: str) -> str | None:
    """Return the mandate string for ``persona_id``, or ``None`` if unknown."""
    entry = PERSONAS_BY_ID.get(persona_id)
    return entry["mandate"] if entry is not None else None
