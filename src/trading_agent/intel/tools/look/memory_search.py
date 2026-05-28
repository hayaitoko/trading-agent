"""``memory_search`` — search this trader's private memory for relevant lessons.

Tool name:      memory_search
Args:           query (str), k=5 (int)
ToolResult:     ok=True, data={"memories": [MemoryEntry, …]}
Latency tier:   fast (requires local embedder; medium if remote)
Cost class:     free
Gating flag:    always enabled (empty list when memory store absent)
Example use:    memory_search("AAPL momentum breakout") to recall past AAPL lessons.

**Isolation guarantee:** only returns lessons for (owner_user_id, trader_id).
Never surfaces another trader's or another user's memory.

**recent_reflections slot integration:** when a query is made during first-look
assembly, the top-3 results tagged with today's symbols/themes are placed into
the TurnContext ``recent_reflections`` slot.  The rest of the matches stay behind
the tool.  This is governed by ``reflections_for_slot(query, symbols)``.

Each MemoryEntry: {text: str, score: float|None, tags: list[str]}

Wraps :class:`~trading_agent.memory.store.MemoryStore` (WS-D).
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase

_TOP_N_FOR_SLOT = 3


class MemorySearchTool(LookToolBase):
    """Search this trader's private lesson memory.

    Parameters
    ----------
    memory_store:
        Duck-typed: must expose
        ``recall(user_id, trader_id, query, k) -> list[Lesson]`` where each Lesson
        has .text/.tags/.score attributes.  ``None`` → empty result.
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "memory_search",
        "description": (
            "Search your private memory for lessons and reflections relevant to a query. "
            "Returns up to k items; empty result means no prior context — you may be new here."
        ),
        "args": {"query": "str", "k": "int (default 5)"},
        "latency": "fast",
        "cost_class": "free",
        "enabled": True,
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        memory_store: Any = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._memory = memory_store

    def __call__(self, query: str, k: int = 5) -> Any:
        """Search memory for lessons matching ``query``.

        Returns
        -------
        ToolResult
            ok=True, data={"memories": [{text, score, tags}, …]}
            ok=False when query is empty (invalid_input).

        Example
        -------
        >>> tool = MemorySearchTool(trader_id="Alpha")
        >>> result = tool("AAPL momentum")
        >>> result.ok
        True
        >>> result.data["memories"]
        []
        """
        query = (query or "").strip()
        if not query:
            return self._err("invalid_input", "query must not be empty")

        k = max(1, min(int(k), 50))

        if self._memory is None or self.owner_user_id is None:
            return self._ok(
                {
                    "memories": [],
                    "note": "memory store not yet available",
                }
            )

        try:
            lessons = self._memory.recall(
                self.owner_user_id, self.trader_id, query, k
            )
        except Exception as exc:
            return self._err("internal", f"memory_search failed: {exc}")

        memories = [
            {
                "text": str(getattr(lesson, "text", lesson)),
                "score": _maybe_float(getattr(lesson, "score", None)),
                "tags": list(getattr(lesson, "tags", [])),
            }
            for lesson in lessons
        ]
        return self._ok({"memories": memories})

    def reflections_for_slot(
        self,
        query: str,
        symbols: list[str] | None = None,
    ) -> list[str]:
        """Return top-N reflection texts for the TurnContext ``recent_reflections`` slot.

        Called during first-look assembly before the first model call.  Runs a
        memory recall against ``query`` (typically the trader's universe string),
        then selects up to ``_TOP_N_FOR_SLOT`` entries whose tags intersect with
        today's symbols or the query terms.  Returns plain text strings.

        Returns
        -------
        list[str]
            Up to _TOP_N_FOR_SLOT reflection text snippets, or empty list when
            memory unavailable or empty.
        """
        if not query or self._memory is None or self.owner_user_id is None:
            return []

        target_tags = {s.upper() for s in (symbols or [])}
        # Add lowercased words from the query as tag matches too.
        for word in query.lower().split():
            if len(word) >= 3:
                target_tags.add(word)

        try:
            # Fetch a bit more than _TOP_N_FOR_SLOT to allow tag filtering.
            lessons = self._memory.recall(
                self.owner_user_id,
                self.trader_id,
                query,
                max(_TOP_N_FOR_SLOT * 3, 10),
            )
        except Exception:
            return []

        # Prefer lessons with matching tags; fall back to top-by-score order.
        scored: list[tuple[int, Any]] = []
        for lesson in lessons:
            tags = {t.upper() for t in (getattr(lesson, "tags", []) or [])}
            overlap = len(tags & target_tags)
            scored.append((overlap, lesson))
        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        for _, lesson in scored:
            text = str(getattr(lesson, "text", "")).strip()
            if text:
                # Truncate long reflections for the first-look block.
                if len(text) > 200:
                    text = text[:197] + "…"
                result.append(text)
            if len(result) >= _TOP_N_FOR_SLOT:
                break
        return result


# --------------------------------------------------------------------------- helpers


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
