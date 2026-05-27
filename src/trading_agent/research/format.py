"""Render research briefs into a compact prompt block.

Shared by the manager (overseer chat) and the per-trader decision context so the
two never drift in how a brief reads. Duck-typed over anything brief-shaped
(``ticker`` / ``summary`` / ``sentiment`` / ``catalysts``) — no import of the
:class:`~trading_agent.research.store.Brief` type, so callers stay decoupled.
"""

from __future__ import annotations

from typing import Any

DEFAULT_HEADER = "## Research briefs"
_MAX_CATALYSTS = 3


def format_briefs(
    briefs: list[Any],
    *,
    header: str = DEFAULT_HEADER,
    max_catalysts: int = _MAX_CATALYSTS,
) -> str:
    """One line per brief: ``TICKER (sentiment ±x): summary | catalysts: …``.

    Returns ``""`` for an empty/falsy list so callers can drop the block.
    """
    if not briefs:
        return ""
    lines = [header]
    for brief in briefs:
        ticker = getattr(brief, "ticker", None) or "?"
        summary = (getattr(brief, "summary", "") or "").strip()
        sentiment = getattr(brief, "sentiment", None)
        head = str(ticker)
        if isinstance(sentiment, (int, float)):
            head += f" (sentiment {sentiment:+.2f})"
        line = f"{head}: {summary}".rstrip(": ").rstrip()
        catalysts = getattr(brief, "catalysts", None) or []
        if catalysts:
            line += f" | catalysts: {', '.join(str(c) for c in catalysts[:max_catalysts])}"
        lines.append(line)
    return "\n".join(lines)
