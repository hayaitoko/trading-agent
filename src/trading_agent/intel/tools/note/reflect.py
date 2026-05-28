"""NOTE tool: ``reflect`` — write a durable lesson to the trader's memory.

**Design role:** P5 calibrated learning write path for the agent.  Every lesson
stored here is namespaced to ``(owner_user_id, trader_id)`` via the WS-D
:class:`~trading_agent.memory.store.MemoryStore`, so no two traders ever share
private notes.

**MONEY IS REAL invariant:** the caller (the LLM) writes the note text.  This
layer never adds or strips "paper" / "sim" strings — it stores exactly what the
agent wrote.  The agent's system prompt already enforces the invariant on its
output.

**Provenance:** the ``tool_call_names`` argument carries the list of tool names
already called in this turn before ``reflect``.  This enables the P5 calibration
router to correlate *what data the agent looked at* before crystallising a lesson
(e.g. "I called research_brief then reflect → lesson was research-informed").

**Latency tier:** fast
**Cost class:** free (local store write, no model call)
**Gating flag:** (none — reflect is always enabled)
"""

from __future__ import annotations

from typing import Any

from ...tool_envelope import ToolResult
from ._base import NoteToolBase

# OpenAI-compatible tool definition for injection into tool lists.
DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "reflect",
        "description": (
            "Write a durable lesson to your private memory. Use this to crystallise "
            "insight that should influence future decisions — e.g. 'AAPL tends to "
            "reverse after earnings gaps in low-volume environments'. The note is "
            "private to you; it surfaces via memory_search in future turns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The lesson or insight to store.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of short tags (e.g. ['AAPL', 'earnings', 'momentum']). "
                        "Tags improve future recall precision."
                    ),
                },
            },
            "required": ["note"],
        },
    },
}

# Catalog entry returned by list_tools().
CATALOG_ENTRY: dict[str, Any] = {
    "name": "reflect",
    "description": "Write a durable lesson to your private memory for future recall.",
    "args": {"note": "str", "tags": "list[str] (optional)"},
    "latency": "fast",
    "cost_class": "free",
    "enabled": True,
    "disabled_reason": None,
}


class ReflectTool(NoteToolBase):
    """Executes the ``reflect(note, *, tags)`` tool call.

    Parameters
    ----------
    tool_call_names:
        Tool names already executed in this turn before the reflect call.
        Stored as provenance metadata alongside the lesson.
    """

    def run(
        self,
        note: str,
        *,
        tags: list[str] | None = None,
        tool_call_names: list[str] | None = None,
    ) -> ToolResult:
        """Store a lesson in the trader's private memory namespace.

        Returns ``{ok: true, data: {lesson_id: ..., note: ..., tags: [...]}}`` on
        success.  Returns a ``not_found`` error if the memory store is absent.
        """
        note = note.strip() if note else ""
        if not note:
            return self._err("invalid_input", "note must not be empty")

        if self.memory is None or self.owner_user_id is None:
            # Memory store not wired (e.g. tests without memory); return ok with
            # a note rather than erroring — the lesson is lost but the loop
            # doesn't break.
            return self._ok(
                {
                    "lesson_id": None,
                    "note": note,
                    "tags": list(tags or []),
                    "stored": False,
                    "note_detail": "memory store unavailable — lesson not persisted",
                }
            )

        effective_tags = list(tags or [])
        if tool_call_names:
            # Encode provenance as a synthetic tag so P5 can filter.
            effective_tags.append(f"provenance:{','.join(tool_call_names)}")

        try:
            lesson = self.memory.remember(
                self.owner_user_id,
                self.trader_id,
                note,
                effective_tags,
            )
            return self._ok(
                {
                    "lesson_id": lesson.id,
                    "note": lesson.text,
                    "tags": lesson.tags,
                    "stored": True,
                }
            )
        except Exception as exc:
            return self._err("internal", f"memory write failed: {exc}")
