"""Per-user settings store: JSON-valued key/value rows keyed on ``user_id``.

This is the server-side replacement for the cockpit mock's ``localStorage``
(see ``design/handoff/README.md`` — settings are per-user, not per-browser).
Holds theme, risk limits, embed_model, vstore, research_model,
research_cadence, the daily $ ceiling, etc. Values are arbitrary JSON.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .db import Database

# A few settings are NOT per-user: the embedding model, its dimension, and the
# vector-store backend are shared infrastructure (one embedder is pulled at the
# deploy, one collection holds everyone's vectors). They live under a reserved
# pseudo-user so the whole box reads/writes one value. ``get``/``set``/``all``
# transparently redirect these keys to that scope, so existing call sites
# (embed.py, serve.py, research.py) read the system value with no change. Only
# the admin console may write them (see web/routers/admin.py).
SYSTEM_USER_ID = "__system__"
SYSTEM_KEYS = frozenset({"embed_model", "vstore", "embed_dim"})

# Defaults mirror the cockpit mock's initial state so a fresh user sees the
# same UI the design specifies. Streams may read more keys than appear here.
DEFAULTS: dict[str, Any] = {
    "theme": "teal",
    "embed_model": "bge-small-en-v1.5",
    "vstore": "sqlite-vec",
    "research_model": None,
    "research_cadence": "1h",  # off | 15m | 1h | 4h — honored per-user by ResearchDaemon
    "daily_usd_ceiling": 5.0,  # cost-gating ceiling (CONTRACTS §Cost-gating)
    "risk_limits": {},
    # WS-A trader intelligence: how much research/memory each trader pulls per
    # decision, and the gated post-round reflection cadence/cost. daily_usd_ceiling
    # above is the shared hard gate; these only shape volume and frequency.
    "trader_research_read": True,  # let traders read shared research briefs
    "trader_research_k": 5,  # briefs pulled into a decision
    "trader_memory_recall_k": 5,  # own lessons recalled into a decision
    "reflection_cadence_rounds": 4,  # reflect every N rounds (0 = never on cadence)
    "reflection_model": None,  # None → cheap DEFAULT_REFLECTION_MODEL
    "reflection_estimated_usd": 0.01,  # per-book distill pre-check estimate
    # P0 risk mechanics (all None = disabled so existing behaviour is unchanged).
    "hard_floor_pct": None,          # catastrophic max-loss % before auto-flatten
    # P1 stale-decision guard defaults.
    "stale_ttl_seconds": 30,          # max decision age before discard
    "stale_drift_pct": 1.0,           # price-drift % before discard
    # P3 situation layer (all off by default so existing bench is unchanged).
    "situation_enabled": False,       # inject regime + social block into trader context
    "regime_vol_calm": 0.15,          # annualized vol below which regime is calm
    "regime_vol_elevated": 0.25,      # vol below which regime is elevated
    "regime_vol_risk_off": 0.40,      # vol at/above which regime is risk-off
    "regime_event_horizon_days": 3,   # calendar look-ahead window (days)
    # P4 pattern KB.
    "pattern_kb_enabled": False,      # inject pattern KB recall into trader context
    "pattern_k": 3,                   # similar patterns to recall per decision
    "pattern_prune_hit_rate": 0.52,   # archive labels that decay toward coin-flip
    # P6 calibration.
    "calibration_horizon_days": 5,    # forward price window for outcome scoring
    # WS-Situation Track A — new data providers (all default off).
    # Turn on to enable the corresponding LOOK tools for a user.
    "SITUATION_GDELT": False,               # world_events tool (GDELTProvider)
    "SITUATION_PREDICTION_MARKETS": False,  # prediction_market_odds tool (Polymarket + Kalshi)
    "SITUATION_OPTIONS_IV": False,          # options_iv tool (Alpaca IV passthrough)
    "SITUATION_FORECAST": False,            # forecast tool (price-cone; needs history)
    # Per-user approval mode (overridable per-trader via add_model(requires_approval=...)).
    # False = autonomous execution; True = every trade routed through PendingTradeQueue.
    "requires_approval": False,
}


class SettingsStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, user_id: str, key: str, default: Any = None) -> Any:
        if key in SYSTEM_KEYS:  # system-wide, not per-user
            user_id = SYSTEM_USER_ID
        row = self._db.query_one(
            "SELECT value FROM user_settings WHERE user_id = ? AND key = ?", (user_id, key)
        )
        if row is not None:
            return json.loads(row["value"])
        if default is not None:
            return default
        return DEFAULTS.get(key)

    def set(self, user_id: str, key: str, value: Any) -> None:
        if key in SYSTEM_KEYS:  # system-wide, not per-user
            user_id = SYSTEM_USER_ID
        self._db.execute(
            """
            INSERT INTO user_settings (user_id, key, value) VALUES (?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value
            """,
            (user_id, key, json.dumps(value)),
        )

    def all(self, user_id: str) -> dict[str, Any]:
        """Every stored key for the user, merged over :data:`DEFAULTS`.

        System-scoped keys (:data:`SYSTEM_KEYS`) always reflect the shared
        system value, never a stray per-user row, so any user's view shows the
        one true embed model / vector store.
        """
        merged: dict[str, Any] = dict(DEFAULTS)
        is_system = user_id == SYSTEM_USER_ID
        for row in self._db.query(
            "SELECT key, value FROM user_settings WHERE user_id = ?", (user_id,)
        ):
            if row["key"] in SYSTEM_KEYS and not is_system:
                continue  # system keys never come from a per-user row
            merged[row["key"]] = json.loads(row["value"])
        if not is_system:  # overlay the shared system values for system keys
            for row in self._db.query(
                "SELECT key, value FROM user_settings WHERE user_id = ?", (SYSTEM_USER_ID,)
            ):
                if row["key"] in SYSTEM_KEYS:
                    merged[row["key"]] = json.loads(row["value"])
        return merged

    def update(self, user_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """Set many keys at once; return the full merged settings."""
        for key, value in values.items():
            self.set(user_id, key, value)
        return self.all(user_id)

    def seed_system_from_env(self) -> None:
        """Seed system-scoped settings from deploy env when not already set.

        ``EMBED_MODEL`` / ``VSTORE`` / ``EMBED_DIM`` are the deploy's source of
        truth (they dictate what the embedder pulled). We only fill a key that
        has no system row yet, so an admin-UI override persists and wins on
        later boots. This also fixes the prior bug where the app ignored
        ``EMBED_MODEL`` and silently defaulted to ``bge-small-en-v1.5``.
        """
        env_map = {
            "embed_model": os.environ.get("EMBED_MODEL"),
            "vstore": os.environ.get("VSTORE"),
            "embed_dim": os.environ.get("EMBED_DIM"),
        }
        for key, raw in env_map.items():
            if not raw:
                continue
            existing = self._db.query_one(
                "SELECT 1 FROM user_settings WHERE user_id = ? AND key = ?",
                (SYSTEM_USER_ID, key),
            )
            if existing is not None:
                continue  # admin override already persisted — don't clobber
            value: Any = int(raw) if key == "embed_dim" and raw.isdigit() else raw
            self.set(SYSTEM_USER_ID, key, value)
