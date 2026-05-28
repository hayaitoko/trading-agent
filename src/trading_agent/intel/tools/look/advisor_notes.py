"""``advisor_notes`` — read operator advisor notes for this trader.

Tool name:      advisor_notes
Args:           symbol=None (str|None), scope="trader" ("ticker"|"trader"|"global")
ToolResult:     ok=True, data={"notes": [NoteEntry, …], "scope": str}
Latency tier:   fast
Cost class:     free
Gating flag:    always enabled (empty list when notes store absent)
Example use:    advisor_notes(scope="trader") for account-level operator notes.
                advisor_notes("AAPL", scope="ticker") for AAPL-specific notes.
                advisor_notes(scope="global") for operator-wide notes.

**Isolation guarantee:** this tool ONLY returns notes for:
  (this trader's owner_user_id, this trader_id)
Never returns notes for other traders or other users.

**Directed-notes slot integration:** when called during first-look assembly,
``directed_notes`` returns the text of every note that has NOT yet been surfaced
to this trader this session.  After being surfaced once (in the TurnContext
``directed_notes`` slot), those notes are marked read via the ``mark_read``
callback — they will NOT reappear in the slot on subsequent turns (though the
full text is still accessible by calling advisor_notes() as a tool).

Wraps :class:`~trading_agent.notes.NotesStore` (WS-H).
Scope mapping:
  "trader"  → scope="trader", ref=trader_id   (the bench competitor name)
  "ticker"  → scope="ticker", ref=symbol       (requires symbol arg)
  "global"  → scope="trader", ref="__global__" (operator-wide notes)
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase

_GLOBAL_REF = "__global__"


class AdvisorNotesTool(LookToolBase):
    """Read-only access to operator-authored advisor notes.

    Parameters
    ----------
    notes_store:
        Duck-typed: must expose ``get(user_id, scope, ref) -> Note|None``
        and ``list(user_id, scope=None) -> list[Note]``.  ``None`` → empty result.
    mark_read_fn:
        ``Callable[[str], None]`` — (note_id,) → None.  Called to mark a directed
        note as read after it surfaces in the TurnContext slot.  ``None`` → notes
        are never marked read (they reappear on every turn).
    session_read_ids:
        Set of note IDs already surfaced this session (mutated in-place).  If None,
        an empty set is used.  This prevents double-surfacing within a session if
        advisor_notes() is called multiple times in one turn.
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "advisor_notes",
        "description": (
            "Operator-written advisor notes scoped to your account (scope='trader'), "
            "a specific ticker (scope='ticker'), or global notes (scope='global'). "
            "Returns notes visible to this trader only — never other traders' notes."
        ),
        "args": {
            "symbol": "str|None (default None, required when scope='ticker')",
            "scope": "'ticker' | 'trader' | 'global' (default 'trader')",
        },
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
        notes_store: Any = None,
        mark_read_fn: Any = None,
        session_read_ids: set[str] | None = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._notes = notes_store
        self._mark_read = mark_read_fn
        self._session_read_ids: set[str] = session_read_ids if session_read_ids is not None else set()

    def __call__(
        self,
        symbol: str | None = None,
        scope: str = "trader",
    ) -> Any:
        """Return operator notes for the requested scope.

        Isolation: results are strictly scoped to (owner_user_id, trader_id).
        Never returns notes from other traders or users.

        Returns
        -------
        ToolResult
            ok=True, data={"notes": [{id, text, updated_at}], "scope": …}

        Example
        -------
        >>> tool = AdvisorNotesTool(trader_id="Alpha")
        >>> result = tool(scope="trader")
        >>> result.ok
        True
        >>> result.data["notes"]
        []
        """
        # Validate scope
        valid_scopes = ("ticker", "trader", "global")
        if scope not in valid_scopes:
            return self._err(
                "invalid_input",
                f"scope must be one of {valid_scopes!r}, got {scope!r}",
            )

        if scope == "ticker":
            sym = (symbol or "").strip().upper()
            if not sym:
                return self._err(
                    "invalid_input",
                    "symbol is required when scope='ticker'",
                )
        else:
            sym = ""

        if self._notes is None or self.owner_user_id is None:
            return self._ok({"notes": [], "scope": scope})

        try:
            note = self._fetch_note(scope, sym)
        except Exception as exc:
            return self._err("internal", f"advisor_notes fetch failed: {exc}")

        note_list = []
        if note is not None:
            note_list = [
                {
                    "id": str(getattr(note, "id", "")),
                    "text": str(getattr(note, "text", "")),
                    "updated_at": float(getattr(note, "updated_at", 0.0)),
                }
            ]

        return self._ok({"notes": note_list, "scope": scope})

    def directed_notes_for_slot(self) -> list[str]:
        """Return unread directed trader-scoped notes for the TurnContext slot.

        Called during first-look assembly (before the first model call in a turn).
        Returns the note text(s) for any unread directed notes.  After returning,
        marks each note as read so they will not reappear in the slot on the next
        turn.

        Returns
        -------
        list[str]
            Texts of unread directed notes, or an empty list.
        """
        if self._notes is None or self.owner_user_id is None:
            return []
        try:
            note = self._fetch_note("trader", "")
        except Exception:
            return []

        if note is None:
            return []

        note_id = str(getattr(note, "id", ""))
        if not note_id or note_id in self._session_read_ids:
            return []

        text = str(getattr(note, "text", "")).strip()
        if not text:
            return []

        # Mark as read for this session.
        self._session_read_ids.add(note_id)
        if self._mark_read is not None:
            try:
                self._mark_read(note_id)
            except Exception:
                pass  # best-effort; surfacing is more important than marking

        return [text]

    def _fetch_note(self, scope: str, symbol: str) -> Any:
        """Fetch a single note; enforces (user, trader) isolation.

        scope=="trader"  → ref = trader_id
        scope=="ticker"  → ref = symbol
        scope=="global"  → ref = _GLOBAL_REF
        """
        if scope == "trader":
            ref = self.trader_id
            db_scope = "trader"
        elif scope == "ticker":
            ref = symbol
            db_scope = "ticker"
        else:  # global
            ref = _GLOBAL_REF
            db_scope = "trader"

        return self._notes.get(self.owner_user_id, db_scope, ref)
