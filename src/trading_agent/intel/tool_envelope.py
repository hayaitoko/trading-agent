"""Universal tool result envelope for the agent catalog.

Every tool in the LOOK / NOTE / ACT catalog wraps its response in a
:class:`ToolResult`.  The LLM sees a uniform shape regardless of which tool
was called, enabling it to reason structurally about errors and absence.

Contract (frozen, never subclass):

    ToolResult(ok=True,  data=<payload>)   — success, ``data`` carries result
    ToolResult(ok=False, error=ToolError)  — failure, ``error`` carries cause

Never raise bare exceptions from tool handlers; always wrap in ToolResult.error.
This module has no intra-package imports to keep the dependency graph clean.

Failure modes:
- Malformed calls from the LLM → ``invalid_input``
- Upstream service down → ``unavailable`` (with optional ``retry_after``)
- Feature not yet enabled → ``disabled``
- Object not found → ``not_found``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ToolError:
    """Structured error payload inside a failed :class:`ToolResult`.

    ``retry_after`` (seconds) is set only for ``"rate_limit"`` or
    ``"unavailable"`` — it gives the agent a concrete wait hint so it can
    decide whether to retry immediately or continue with other tools.
    """

    kind: Literal[
        "network",
        "rate_limit",
        "unavailable",
        "invalid_input",
        "disabled",
        "not_found",
        "internal",
    ]
    message: str
    retry_after: int | None = None


@dataclass(frozen=True)
class ToolResult:
    """Universal return envelope for every agent tool.

    Callers must check ``ok`` before accessing ``data`` or ``error``.
    Both ``data`` and ``error`` are ``None`` on their unused side.

    Design choice: frozen so result objects are safe to cache or log without
    mutation risk; ``to_dict`` serialises to a JSON-safe dict for injection
    into the OpenAI-compatible message list.
    """

    ok: bool
    data: Any | None = None
    error: ToolError | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict for injection into message content."""
        if self.ok:
            return {"ok": True, "data": self.data}
        assert self.error is not None
        err: dict[str, Any] = {
            "kind": self.error.kind,
            "message": self.error.message,
        }
        if self.error.retry_after is not None:
            err["retry_after"] = self.error.retry_after
        return {"ok": False, "error": err}
