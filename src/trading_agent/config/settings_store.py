"""Per-user settings store: JSON-valued key/value rows keyed on ``user_id``.

This is the server-side replacement for the cockpit mock's ``localStorage``
(see ``design/handoff/README.md`` — settings are per-user, not per-browser).
Holds theme, risk limits, embed_model, vstore, research_model,
research_cadence, the daily $ ceiling, etc. Values are arbitrary JSON.
"""

from __future__ import annotations

import json
from typing import Any

from .db import Database

# Defaults mirror the cockpit mock's initial state so a fresh user sees the
# same UI the design specifies. Streams may read more keys than appear here.
DEFAULTS: dict[str, Any] = {
    "theme": "teal",
    "embed_model": "bge-small-en-v1.5",
    "vstore": "sqlite-vec",
    "research_model": None,
    "research_cadence": "off",
    "daily_usd_ceiling": 5.0,  # cost-gating ceiling (CONTRACTS §Cost-gating)
    "risk_limits": {},
}


class SettingsStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, user_id: str, key: str, default: Any = None) -> Any:
        row = self._db.query_one(
            "SELECT value FROM user_settings WHERE user_id = ? AND key = ?", (user_id, key)
        )
        if row is not None:
            return json.loads(row["value"])
        if default is not None:
            return default
        return DEFAULTS.get(key)

    def set(self, user_id: str, key: str, value: Any) -> None:
        self._db.execute(
            """
            INSERT INTO user_settings (user_id, key, value) VALUES (?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value
            """,
            (user_id, key, json.dumps(value)),
        )

    def all(self, user_id: str) -> dict[str, Any]:
        """Every stored key for the user, merged over :data:`DEFAULTS`."""
        merged: dict[str, Any] = dict(DEFAULTS)
        for row in self._db.query(
            "SELECT key, value FROM user_settings WHERE user_id = ?", (user_id,)
        ):
            merged[row["key"]] = json.loads(row["value"])
        return merged

    def update(self, user_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """Set many keys at once; return the full merged settings."""
        for key, value in values.items():
            self.set(user_id, key, value)
        return self.all(user_id)
