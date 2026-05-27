"""Render trader lessons into a compact prompt block.

Shared by the manager (which gathers lessons across every book and tags each
with its owning trader) and the per-trader decision context (which shows only
its own lessons, untagged). Duck-typed over anything lesson-shaped (``text``,
optionally ``trader_id``) so callers don't import the
:class:`~trading_agent.memory.store.Lesson` type.
"""

from __future__ import annotations

from typing import Any

DEFAULT_HEADER = "## Lessons"


def format_lessons(
    lessons: list[Any],
    *,
    header: str = DEFAULT_HEADER,
    show_trader: bool = False,
    limit: int | None = None,
) -> str:
    """One line per lesson, deduped by text, oldest-kept-first up to ``limit``.

    With ``show_trader`` each line is prefixed ``[trader_id]`` (the manager's
    cross-book view); otherwise it's a plain ``- bullet`` (a trader's own
    memory). Returns ``""`` when nothing survives so callers can drop the block.
    """
    rows: list[str] = []
    seen: set[str] = set()
    for lesson in lessons:
        text = (getattr(lesson, "text", "") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if show_trader:
            trader = getattr(lesson, "trader_id", "") or "?"
            rows.append(f"[{trader}] {text}")
        else:
            rows.append(f"- {text}")
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        return ""
    return "\n".join([header, *rows])
