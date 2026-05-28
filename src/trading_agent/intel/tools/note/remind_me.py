"""NOTE tool: ``remind_me`` — time-based deferred self-poke.

**Design role:** the trader schedules a future attention turn by name and
reason.  On the scheduled tick the scheduler fires the row and enqueues an
event-driven turn for this trader with the original ``about`` text in context.

**When format:** ISO-8601 datetime string OR a relative expression accepted by
:func:`_parse_when`:
  - ``"2026-05-28T10:30:00"`` / ``"2026-05-28T10:30:00-04:00"``
  - ``"in 5s"`` / ``"in 15min"`` / ``"in 2h"`` / ``"in 1d"``
  - ``"tomorrow 10am ET"`` → parsed as next-calendar-day 10:00 America/New_York

**Time anchoring:** all times stored as UTC Unix seconds internally; the
"tomorrow 10am ET" form is ET-anchored per the multi-tenant-time rule —
"ET" is the universal US-market frame, never Lukas's local PT.

**Auto-expiry:**
  - On fire: ``fired_at`` is set → no longer shows in active count.
  - After 7 days without firing (dormancy / shutdown): the scheduler's
    :meth:`expire_old` call soft-expires the row.

**Soft limits:** ``REMINDER_SOFT_LIMIT`` (default 10) is checked via
:class:`~trading_agent.intel.attention_queue.AttentionQueue.can_add`.  A
nudge appears in first-look context when the active count exceeds it.

**Latency tier:** fast (local write, no model call)
**Cost class:** free
**Gating flag:** (none — always enabled)
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from ...attention_queue import DEFAULT_REMINDER_TTL_DAYS
from ...tool_envelope import ToolResult
from ._base import NoteToolBase

# OpenAI-compatible tool definition.
DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "remind_me",
        "description": (
            "Schedule a deferred self-poke: you will be woken at the given time "
            "with the 'about' text in your context.  Use when you want to check "
            "back on something later without setting a persistent watchpoint. "
            "Accepts ISO datetime (e.g. '2026-05-28T14:00:00') or a relative "
            "expression ('in 15min', 'in 2h', 'in 1d', 'tomorrow 10am ET')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "when": {
                    "type": "string",
                    "description": (
                        "When to remind you. ISO-8601 string or relative: "
                        "'in 5s', 'in 15min', 'in 2h', 'in 1d', 'tomorrow 10am ET'."
                    ),
                },
                "about": {
                    "type": "string",
                    "description": "What to remind you about.",
                },
            },
            "required": ["when", "about"],
        },
    },
}

# Catalog entry for list_tools().
CATALOG_ENTRY: dict[str, Any] = {
    "name": "remind_me",
    "description": (
        "Schedule a deferred self-poke at a future time. "
        "You will wake with the 'about' text in your context."
    ),
    "args": {"when": "str (ISO datetime or relative)", "about": "str"},
    "latency": "fast",
    "cost_class": "free",
    "enabled": True,
    "disabled_reason": None,
}


class RemindMeTool(NoteToolBase):
    """Executes the ``remind_me(when, about)`` tool call."""

    def run(self, when: str, about: str) -> ToolResult:
        """Schedule a reminder.

        Returns ``{ok: true, data: {reminder_id, when_unix, when_iso, about}}``
        on success.  Returns an error result for unparseable ``when`` strings or
        if the reminder soft-cap is reached.
        """
        when = (when or "").strip()
        about = (about or "").strip()
        if not when:
            return self._err("invalid_input", "'when' must not be empty")
        if not about:
            return self._err("invalid_input", "'about' must not be empty")

        try:
            when_unix = _parse_when(when)
        except ValueError as exc:
            return self._err(
                "invalid_input",
                f"Could not parse 'when' = {when!r}: {exc}. "
                "Use ISO datetime or relative like 'in 15min', 'in 2h', 'in 1d', "
                "'tomorrow 10am ET'.",
            )

        now = int(time.time())
        if when_unix <= now:
            return self._err(
                "invalid_input",
                f"Reminder time is in the past (when_unix={when_unix}, now={now}). "
                "Provide a future time.",
            )

        # Hard-cap check via queue.
        if self.attention_queue is not None:
            ok, cap_msg = self.attention_queue.can_add(self.trader_id, "reminder")
            if not ok:
                return self._err("unavailable", cap_msg)

        expires_at = when_unix + int(DEFAULT_REMINDER_TTL_DAYS * 86_400)
        payload: dict[str, Any] = {
            "when": when,
            "when_unix": when_unix,
            "about": about,
        }

        if self.attention_queue is not None:
            row = self.attention_queue.enqueue(
                self.trader_id,
                "reminder",
                payload,
                expires_at=expires_at,
            )
            reminder_id: int | None = row.id if row.id >= 0 else None
        else:
            reminder_id = None

        when_iso = datetime.fromtimestamp(when_unix, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return self._ok(
            {
                "reminder_id": reminder_id,
                "when_unix": when_unix,
                "when_iso": when_iso,
                "about": about,
                "stored": reminder_id is not None,
            }
        )


# ── time parser ───────────────────────────────────────────────────────────────

# Relative patterns: "in 5s" / "in 15min" / "in 2h" / "in 1d"
_REL_PATTERN = re.compile(
    r"^\s*in\s+(\d+(?:\.\d+)?)\s*(s|sec|second|seconds|m|min|minute|minutes|h|hour|hours|d|day|days)\s*$",
    re.IGNORECASE,
)

# "tomorrow 10am ET" — simple ET-anchored next-day form.
_TOMORROW_ET = re.compile(
    r"^\s*tomorrow\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)?\s*(et|est|edt)?\s*$",
    re.IGNORECASE,
)


def _parse_when(when: str) -> int:
    """Return a UTC Unix timestamp from a when-expression.

    Raises ``ValueError`` if the string cannot be parsed.

    NEVER uses local TZ (Lukas's PT or any server TZ).  All anchoring is either
    UTC explicit or America/New_York (ET) for the market-bounded ``tomorrow`` form.
    """
    w = when.strip()

    # 1. Relative: "in N unit"
    m = _REL_PATTERN.match(w)
    if m:
        amount = float(m.group(1))
        unit = m.group(2).lower()
        if unit in ("s", "sec", "second", "seconds"):
            delta = timedelta(seconds=amount)
        elif unit in ("m", "min", "minute", "minutes"):
            delta = timedelta(minutes=amount)
        elif unit in ("h", "hour", "hours"):
            delta = timedelta(hours=amount)
        elif unit in ("d", "day", "days"):
            delta = timedelta(days=amount)
        else:
            raise ValueError(f"Unknown unit: {unit!r}")
        return int((datetime.now(UTC) + delta).timestamp())

    # 2. "tomorrow Nh[am/pm] [ET]"
    m2 = _TOMORROW_ET.match(w)
    if m2:
        return _parse_tomorrow_et(m2)

    # 3. ISO-8601 / date-string fallback — try stdlib.
    return _parse_iso(w)


def _parse_tomorrow_et(m: re.Match[str]) -> int:
    """Parse 'tomorrow HH[:MM][am|pm] ET' → UTC Unix seconds.

    "Tomorrow" = next calendar day in ET, never in local/server TZ.
    """
    try:
        from zoneinfo import ZoneInfo

        et_tz = ZoneInfo("America/New_York")
    except Exception as exc:
        raise ValueError(f"zoneinfo not available: {exc}") from exc

    now_et = datetime.now(et_tz)
    tomorrow_et = now_et.date() + timedelta(days=1)

    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    ampm = (m.group(3) or "").lower()

    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    dt_et = datetime(
        tomorrow_et.year,
        tomorrow_et.month,
        tomorrow_et.day,
        hour,
        minute,
        tzinfo=et_tz,
    )
    return int(dt_et.timestamp())


def _parse_iso(s: str) -> int:
    """Parse an ISO-8601 string → UTC Unix seconds.

    Handles:
    - "2026-05-28T10:30:00"         (treated as UTC)
    - "2026-05-28T10:30:00Z"        (UTC explicit)
    - "2026-05-28T10:30:00-04:00"   (offset-aware)
    - "2026-05-28 10:30:00"         (space separator, UTC)
    """
    s = s.strip()
    # Normalise space-separator to T.
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    # Append Z if no TZ info present so datetime.fromisoformat treats it as UTC.
    if not s.endswith("Z") and "+" not in s[10:] and "-" not in s[10:]:
        s += "Z"
    try:
        # Python 3.11+: datetime.fromisoformat handles full RFC 3339.
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp())
    except ValueError:
        raise ValueError(
            f"Cannot parse {s!r} as ISO-8601. "
            "Use e.g. '2026-05-28T10:30:00' or 'in 15min'."
        )
