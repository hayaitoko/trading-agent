"""ManagerAgent — the overseer that reads the books and talks to the operator.

It assembles read-only context from three optional, independently-guarded
sources, then makes **exactly one** cost-gated model call per message:

1. the **live bench snapshot** (every book's P&L, recent decisions, risk state),
2. recent **research briefs** (WS-C), and
3. relevant **trader memories** (WS-D).

Each source is optional: if it isn't wired (or errors), its section is simply
omitted — the manager still answers from whatever it has. The model is resolved
through the :class:`EndpointRegistry` (never a hardcoded provider/key), defaults
to a cheap model, and every call is metered against the per-user daily ceiling
via the shared :class:`CostGate`.

**No trading capability by construction.** This class holds no broker and no
order path — it can read state and raise :class:`Notification` flags, nothing
more. Tests assert the broker is never touched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from ..config.endpoints import ModelRef
from ..memory.format import format_lessons
from ..memory.reflect import CostGate, CostGateError
from ..research.format import format_briefs
from ..web.notifications import Notification, _utcnow_iso

if TYPE_CHECKING:
    from ..config.endpoints import EndpointRegistry
    from ..config.settings_store import SettingsStore
    from .chat import ConversationStore

__all__ = [
    "ChatTurn",
    "CostGateError",
    "DEFAULT_MANAGER_MODEL",
    "DEFAULT_REFLECTION_MODEL",
    "ManagerAgent",
    "ManagerConfigError",
    "resolve_manager_ref",
    "resolve_reflection_ref",
]

# Cheap default overseer model (cockpit's featured cheap pick). Overridable via
# the ``manager_model`` user setting.
DEFAULT_MANAGER_MODEL = "google/gemini-3.5-flash"

# Cheap default for the post-round reflection distill (WS-A). Overridable via the
# ``reflection_model`` user setting.
DEFAULT_REFLECTION_MODEL = "google/gemini-3.5-flash"

# Per-message guards. The structural cap is one call/message; CostGate adds the
# daily $ ceiling. The estimate is conservative and only used for the pre-check;
# the actual cost from the response is what gets recorded.
_CALL_ESTIMATE_USD = 0.01
_TEMPERATURE = 0.4
_MAX_TOKENS = 700

# Defaults for what the manager proactively flags.
_DEFAULT_FLAG_DRAWDOWN_PCT = 5.0
_MAX_FLAGGED_BLOCKS = 5
_RESEARCH_BRIEFS = 5
_MEMORIES_PER_BOOK = 2
_MAX_MEMORIES = 8

# Endpoint-type preference when the user hasn't pinned one for the manager.
_TYPE_PRIORITY = {"openrouter": 0, "openai": 1, "anthropic": 2, "local": 3}

SYSTEM_PROMPT = (
    "You are the Manager — the overseer of a stable of autonomous paper-trading "
    "agents (each a 'book'). You watch every book's P&L, their recent decisions, "
    "the risk state, research briefs, and trader memories, and you talk to the "
    "operator (a human). Summarize, explain, and flag — clearly and concisely.\n"
    "You ADVISE ONLY. You cannot place, modify, or cancel trades and you have no "
    "broker access; only the per-book agents trade, and the operator approves. If "
    "asked to trade, say so and offer to flag it instead.\n"
    "Ground every answer in the state below. If the state doesn't cover something, "
    "say you don't have that rather than guessing."
)

# Tool-call envelope. When you want to *also* surface something on the cockpit
# dashboard alongside your reply, return a JSON object instead of plain text.
# The frontend executes the listed actions; the ``reply`` text is shown in the
# chat panel. Plain-text replies still work — only emit the JSON form when you
# actually have an action to take.
TOOL_PROMPT = (
    "\n\n# Surfacing things on the dashboard (optional)\n"
    "When the operator asks to *see*, *open*, *chart*, *pull up*, or *show* a "
    "ticker, account, or tab, you can surface it for them by returning a "
    "single JSON object instead of plain text:\n"
    '{"reply": "<your short answer>", "actions": [<one or more actions>]}\n'
    "Supported action types:\n"
    '  · {"type": "open_quote",   "symbol": "AAPL"}   — open the in-depth quote/chart window\n'
    '  · {"type": "open_chart",   "symbol": "AAPL"}   — same as open_quote (alias)\n'
    '  · {"type": "open_account", "name":   "opus"}   — open the named trader\'s account window\n'
    '  · {"type": "open_tab",     "tab":    "research"} — switch to a tab by id\n'
    "Rules: only use these action types; emit at most 3 actions per turn; "
    "if no surface action is needed, reply with plain text (no JSON envelope). "
    "Never invent action types — unknown types are dropped."
)


@dataclass
class ChatTurn:
    """One overseer turn: the human-facing reply plus optional UI actions.

    ``actions`` is a list of validated ``{type, ...args}`` dicts the cockpit
    frontend already knows how to execute (open_quote / open_chart /
    open_account / open_tab). Empty when the model just chats — the
    backward-compatible ``ManagerAgent.chat()`` returns just ``reply``.
    """

    reply: str
    actions: list[dict[str, Any]] = field(default_factory=list)


# Known action types, each with its required key. Unknown types or actions
# missing the required key are dropped silently so a hallucinating model
# can't push junk through to the UI.
_ACTION_REQUIRED_KEY: dict[str, str] = {
    "open_quote": "symbol",
    "open_chart": "symbol",
    "open_account": "name",
    "open_tab": "tab",
}
_MAX_ACTIONS_PER_TURN = 3

# Optional code-fence wrapper the model may emit (```json ... ```). We strip it
# before json.loads so the envelope still parses.
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def _normalize_action(raw: Any) -> dict[str, Any] | None:
    """Validate a single action dict; return the cleaned form or ``None``."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("type")
    if not isinstance(kind, str):
        return None
    required = _ACTION_REQUIRED_KEY.get(kind)
    if required is None:
        return None  # unknown action type — drop silently
    value = raw.get(required)
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = {"type": kind, required: value.strip()}
    # ``open_tab`` historically accepts an ``id`` alias; the cockpit reads
    # ``a.tab||a.id``. Pass an ``id`` through too if the model used it.
    if kind == "open_tab" and isinstance(raw.get("id"), str) and raw.get("id"):
        cleaned["id"] = raw["id"]
    return cleaned


def parse_tool_response(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Split a model response into ``(reply, actions)``.

    Accepts either a plain string (treated as the whole reply, no actions) or
    a JSON object ``{"reply": "...", "actions": [...]}`` optionally wrapped in
    a ```json ... ``` fence. Anything malformed degrades to ``(content, [])``
    so a model that drifts from the envelope still produces a usable reply.
    """
    text = (content or "").strip()
    if not text:
        return "", []
    candidate = _strip_fence(text)
    # Only attempt JSON parse if it looks like a JSON object — avoids matching
    # incidental "{" inside a normal sentence.
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return text, []
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return text, []
    if not isinstance(parsed, dict) or "reply" not in parsed:
        return text, []
    reply = parsed.get("reply")
    if not isinstance(reply, str):
        return text, []
    raw_actions = parsed.get("actions") or []
    if not isinstance(raw_actions, list):
        raw_actions = []
    actions: list[dict[str, Any]] = []
    for item in raw_actions:
        cleaned = _normalize_action(item)
        if cleaned is not None:
            actions.append(cleaned)
        if len(actions) >= _MAX_ACTIONS_PER_TURN:
            break
    return reply.strip(), actions


class ManagerConfigError(RuntimeError):
    """No usable model endpoint is configured for the manager."""


# --- optional, duck-typed context providers ---------------------------------
# Declared as Protocols so the manager composes with WS-C/WS-D (and the live
# Bench) without importing them; any object with these methods works, and tests
# can pass fakes.


class BenchSnapshotProvider(Protocol):
    def snapshot(self) -> dict[str, Any]: ...


class ResearchProvider(Protocol):
    def recent(self, user_id: str, n: int) -> list[Any]: ...


class MemoryProvider(Protocol):
    def recall(self, user_id: str, trader_id: str, query: str, k: int) -> list[Any]: ...


def _prefer_endpoint(endpoints: list[Any]) -> Any:
    return sorted(endpoints, key=lambda e: (_TYPE_PRIORITY.get(e.type, 9), e.name))[0]


def _resolve_ref(
    settings: SettingsStore,
    registry: EndpointRegistry,
    user_id: str,
    *,
    key: str,
    default: str,
) -> ModelRef:
    """Turn a model setting (``key``) into a concrete :class:`ModelRef`.

    The setting is either ``{"endpoint_id", "model"}`` (a pinned endpoint) or a
    bare model slug string (then the user's first enabled endpoint is chosen,
    preferring OpenRouter). Falls back to ``default`` for the model. Raises
    :class:`ManagerConfigError` if the user has no enabled endpoint.
    """
    raw = settings.get(user_id, key, None)
    endpoint_id: str | None = None
    model: str | None = None
    if isinstance(raw, dict):
        endpoint_id = raw.get("endpoint_id") or None
        model = raw.get("model") or None
    elif isinstance(raw, str):
        model = raw.strip() or None
    model = model or default

    if endpoint_id is not None:
        pinned = registry.get(user_id, endpoint_id)
        if pinned is None or not pinned.enabled:
            endpoint_id = None  # stale/disabled → fall back to an enabled one

    if endpoint_id is None:
        enabled = [e for e in registry.list(user_id) if e.enabled]
        if not enabled:
            raise ManagerConfigError(
                "no enabled model endpoint — add one in Settings before chatting"
            )
        endpoint_id = _prefer_endpoint(enabled).id

    return ModelRef(endpoint_id=endpoint_id, model=model)


def resolve_manager_ref(
    settings: SettingsStore, registry: EndpointRegistry, user_id: str
) -> ModelRef:
    """Resolve the ``manager_model`` setting to a :class:`ModelRef`."""
    return _resolve_ref(
        settings, registry, user_id, key="manager_model", default=DEFAULT_MANAGER_MODEL
    )


def resolve_reflection_ref(
    settings: SettingsStore, registry: EndpointRegistry, user_id: str
) -> ModelRef:
    """Resolve the ``reflection_model`` setting to a :class:`ModelRef` (WS-A)."""
    return _resolve_ref(
        settings, registry, user_id, key="reflection_model", default=DEFAULT_REFLECTION_MODEL
    )


class ManagerAgent:
    """Overseer chat backed by the endpoint registry. Reads; never trades."""

    def __init__(
        self,
        registry: EndpointRegistry,
        settings: SettingsStore,
        conversations: ConversationStore,
        *,
        bench: BenchSnapshotProvider | None = None,
        research: ResearchProvider | None = None,
        memory: MemoryProvider | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._conversations = conversations
        self._bench = bench
        self._research = research
        self._memory = memory
        self._cost_gate = CostGate(settings)

    # --- chat ----------------------------------------------------------------

    def chat(self, user_id: str, conversation_id: str, message: str, ref: ModelRef) -> str:
        """Backwards-compatible string reply. See :meth:`chat_with_actions`."""
        return self.chat_with_actions(user_id, conversation_id, message, ref).reply

    def chat_with_actions(
        self, user_id: str, conversation_id: str, message: str, ref: ModelRef
    ) -> ChatTurn:
        """One overseer turn: assemble context, make one cost-gated call, reply.

        Reads prior turns of ``conversation_id`` for continuity; the caller
        persists the new user/assistant turns. Raises :class:`CostGateError` if
        the daily ceiling would be exceeded and :class:`EndpointError` on a model
        failure — in both cases nothing is written.

        The returned :class:`ChatTurn` carries the human-facing ``reply`` plus
        any UI ``actions`` the model surfaced (open_quote / open_account / ...);
        ``actions`` is empty when the model replied as plain text.
        """
        context = self._build_context(user_id, message)
        system = SYSTEM_PROMPT + TOOL_PROMPT
        if context:
            system = f"{system}\n\n# Live state\n{context}"

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(self._conversations.history_messages(conversation_id))
        messages.append({"role": "user", "content": message})

        self._cost_gate.check(user_id, _CALL_ESTIMATE_USD)
        result = self._registry.chat(
            user_id, ref, messages, temperature=_TEMPERATURE, max_tokens=_MAX_TOKENS
        )
        spent = result.cost if result.cost is not None else _CALL_ESTIMATE_USD
        self._cost_gate.record(user_id, spent)
        reply, actions = parse_tool_response(result.content)
        return ChatTurn(reply=reply, actions=actions)

    # --- flags ---------------------------------------------------------------

    def flags(self, user_id: str) -> list[Notification]:
        """Read-only: things worth raising to the operator (no model call).

        Surfaces erroring books, books past the soft-drawdown limit, and recent
        risk-blocked orders, drawn straight from the live bench snapshot.
        """
        if self._bench is None:
            return []
        try:
            snap = self._bench.snapshot()
        except Exception:  # a flaky bench must never break the feed
            return []

        out: list[Notification] = []
        ts = _utcnow_iso()
        drawdown = float(
            self._settings.get(user_id, "manager_flag_drawdown_pct", _DEFAULT_FLAG_DRAWDOWN_PCT)
        )

        for row in snap.get("leaderboard", []):
            name = str(row.get("name", "?"))
            if row.get("error"):
                out.append(
                    Notification(
                        id=f"manager:error:{name}",
                        kind="risk",
                        severity="critical",
                        title=f"{name} is erroring",
                        body=str(row["error"]),
                        timestamp=ts,
                        data={"book": name},
                    )
                )
                continue
            ret = row.get("return_pct")
            if isinstance(ret, (int, float)) and ret <= -drawdown:
                pnl = row.get("pnl", 0.0)
                out.append(
                    Notification(
                        id=f"manager:drawdown:{name}",
                        kind="risk",
                        severity="warning",
                        title=f"{name} down {ret:.1f}%",
                        body=(
                            f"{name} is below the −{drawdown:.0f}% soft limit "
                            f"(P&L {pnl:,.0f}). Advisory only — still inside hard limits."
                        ),
                        timestamp=ts,
                        data={"book": name, "return_pct": float(ret)},
                    )
                )

        blocked = [d for d in snap.get("recent_decisions", []) if d.get("status") == "blocked"]
        for dec in blocked[:_MAX_FLAGGED_BLOCKS]:
            who = dec.get("competitor", "?")
            out.append(
                Notification(
                    id=f"manager:blocked:{who}:{dec.get('timestamp', '')}",
                    kind="risk",
                    severity="warning",
                    title=f"{who} order blocked by risk",
                    body=str(dec.get("detail") or dec.get("reason") or "blocked"),
                    timestamp=str(dec.get("timestamp") or ts),
                    data=dict(dec),
                )
            )
        return out

    # --- context assembly ----------------------------------------------------

    def _build_context(self, user_id: str, message: str) -> str:
        snap = self._safe_snapshot()
        sections = [
            self._bench_block(snap),
            self._research_block(user_id),
            self._memory_block(user_id, message, snap),
        ]
        return "\n\n".join(s for s in sections if s)

    def _safe_snapshot(self) -> dict[str, Any]:
        if self._bench is None:
            return {}
        try:
            return self._bench.snapshot()
        except Exception:
            return {}

    def _bench_block(self, snap: dict[str, Any]) -> str:
        leaderboard = snap.get("leaderboard") or []
        if not leaderboard:
            return ""
        lines = ["## Books (paper, live ranking)"]
        for row in leaderboard:
            bits = [
                f"#{row.get('rank', '?')} {row.get('name', '?')}",
                f"({row.get('model', '?')})",
                f"value ${row.get('account_value', 0):,.0f}",
                f"P&L {row.get('pnl', 0):+,.0f} ({row.get('return_pct', 0):+.2f}%)",
                f"{row.get('trades', 0)} trades / {row.get('decisions', 0)} decisions",
            ]
            if row.get("error"):
                bits.append(f"ERROR: {row['error']}")
            elif row.get("last_comment"):
                bits.append(f"— {row['last_comment']}")
            lines.append(" · ".join(bits))

        decisions = snap.get("recent_decisions") or []
        if decisions:
            lines.append("\n## Recent decisions (newest first)")
            for dec in decisions[:10]:
                line = (
                    f"{dec.get('timestamp', '')} {dec.get('competitor', '?')}: "
                    f"{dec.get('action', '?')} {dec.get('quantity', '')} "
                    f"{dec.get('symbol', '')} [{dec.get('status', '?')}]"
                )
                if dec.get("reason"):
                    line += f" — {dec['reason']}"
                lines.append(line.strip())

        blocked = sum(1 for d in decisions if d.get("status") == "blocked")
        errored = sum(1 for d in decisions if d.get("status") == "error")
        if blocked or errored:
            lines.append(
                f"\n## Risk: {blocked} blocked, {errored} errored in recent decisions."
            )
        return "\n".join(lines)

    def _research_block(self, user_id: str) -> str:
        if self._research is None:
            return ""
        try:
            briefs = self._research.recent(user_id, _RESEARCH_BRIEFS)
        except Exception:
            return ""
        return format_briefs(briefs, header="## Recent research briefs")

    def _memory_block(self, user_id: str, query: str, snap: dict[str, Any]) -> str:
        if self._memory is None:
            return ""
        names = [str(r.get("name")) for r in (snap.get("leaderboard") or []) if r.get("name")]
        if not names:
            return ""
        gathered: list[Any] = []  # lesson objects, tagged by their own trader_id
        for name in names:
            try:
                gathered.extend(self._memory.recall(user_id, name, query, _MEMORIES_PER_BOOK))
            except Exception:
                continue
        return format_lessons(
            gathered, header="## Relevant trader memories", show_trader=True, limit=_MAX_MEMORIES
        )
