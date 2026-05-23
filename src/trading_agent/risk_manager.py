"""Risk gates: kill switch + per-strategy circuit breakers.

Sits between SignalRouter and broker. All ``check_*`` methods return ``True``
when the trade should be **blocked**, ``False`` when it may proceed.

Kill switch sources (any one trips it):
    1. ``activate_kill_switch()`` instance flag
    2. environment variable ``TRADING_AGENT_KILL_SWITCH`` set to a truthy value
    3. file flag at ``self.kill_switch_file`` (existence trips it)

When the kill switch is tripped, every ``check_*`` method returns ``True``
regardless of underlying counters — there is no path to ``False``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

KILL_SWITCH_ENV = "TRADING_AGENT_KILL_SWITCH"
DEFAULT_KILL_SWITCH_FILE = Path("data/.kill_switch")


@dataclass(frozen=True)
class RiskLimits:
    """Per-strategy risk thresholds. All limits inclusive."""

    max_daily_loss: float = 1000.0
    max_position_size: float = 100.0
    max_trades_per_hour: int = 10
    max_open_positions: int = 5


class RiskManager:
    """Kill switch + circuit breakers.

    ``scope_id`` identifies the strategy or user the counters belong to. The
    same instance can multiplex across multiple scopes.
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        kill_switch_file: Path | None = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.kill_switch_file = (
            Path(kill_switch_file) if kill_switch_file is not None else DEFAULT_KILL_SWITCH_FILE
        )
        self._kill_switch_flag = False

        self.daily_losses: dict[str, dict[str, Any]] = {}
        self.hourly_trades: dict[str, dict[str, Any]] = {}
        self.open_positions: dict[str, int] = {}
        self.position_sizes: dict[str, float] = {}

    # Backwards-compat shims for the dataclass thresholds. Older callers and
    # tests referenced these as class attributes.
    @property
    def MAX_DAILY_LOSS(self) -> float:  # noqa: N802
        return self.limits.max_daily_loss

    @property
    def MAX_POSITION_SIZE(self) -> float:  # noqa: N802
        return self.limits.max_position_size

    @property
    def MAX_TRADES_PER_HOUR(self) -> int:  # noqa: N802
        return self.limits.max_trades_per_hour

    @property
    def MAX_OPEN_POSITIONS(self) -> int:  # noqa: N802
        return self.limits.max_open_positions

    # --- Kill switch --------------------------------------------------------

    @property
    def kill_switch_active(self) -> bool:
        """True if any kill-switch source is tripped (checked dynamically)."""
        if self._kill_switch_flag:
            return True
        env_val = os.environ.get(KILL_SWITCH_ENV, "").strip().lower()
        if env_val and env_val not in {"0", "false", "no", "off", ""}:
            return True
        if self.kill_switch_file.exists():
            return True
        return False

    def activate_kill_switch(self) -> None:
        self._kill_switch_flag = True

    def deactivate_kill_switch(self) -> None:
        self._kill_switch_flag = False

    def check_kill_switch(self) -> bool:
        return self.kill_switch_active

    # --- Circuit breakers ---------------------------------------------------

    def check_daily_loss_limit(self, scope_id: str, loss_amount: float) -> bool:
        """Accumulate ``loss_amount`` (positive number) and return True if exceeded.

        Kill switch active → True. Otherwise compare cumulative daily loss
        against ``max_daily_loss``.
        """
        if self.kill_switch_active:
            return True

        today = self._today()
        bucket = self.daily_losses.setdefault(scope_id, {"date": today, "loss": 0.0})
        if bucket["date"] != today:
            bucket["date"] = today
            bucket["loss"] = 0.0
        bucket["loss"] += loss_amount
        return bucket["loss"] > self.limits.max_daily_loss

    def check_position_size(self, scope_id: str, size: float) -> bool:
        """True if ``|size|`` exceeds ``max_position_size``. Kill switch → True."""
        if self.kill_switch_active:
            return True
        return abs(size) > self.limits.max_position_size

    def increment_hourly_trades(self, scope_id: str) -> bool:
        """Increment counter, return True if limit exceeded. Kill switch → True."""
        if self.kill_switch_active:
            return True

        now_hour = self._current_hour()
        bucket = self.hourly_trades.setdefault(scope_id, {"hour": now_hour, "count": 0})
        if bucket["hour"] != now_hour:
            bucket["hour"] = now_hour
            bucket["count"] = 0
        bucket["count"] += 1
        return bucket["count"] > self.limits.max_trades_per_hour

    def update_open_positions(self, scope_id: str, delta: int) -> bool:
        """Apply ``delta`` to open-position counter. True if limit exceeded.

        Kill switch active → True (the count is still updated so accounting stays
        consistent across kill-switch toggles).
        """
        count = self.open_positions.get(scope_id, 0) + delta
        if count < 0:
            count = 0
        self.open_positions[scope_id] = count

        if self.kill_switch_active:
            return True
        return count > self.limits.max_open_positions

    # --- Position bookkeeping ----------------------------------------------

    def open_position(self, scope_id: str, size: float) -> None:
        self.update_open_positions(scope_id, 1)
        self.position_sizes[scope_id] = size

    def close_position(self, scope_id: str) -> None:
        self.update_open_positions(scope_id, -1)
        self.position_sizes.pop(scope_id, None)

    def partial_close(self, scope_id: str, amount: int) -> None:
        if scope_id in self.open_positions:
            self.open_positions[scope_id] = max(0, self.open_positions[scope_id] - amount)

    def flip_position(self, scope_id: str, new_position: int) -> None:
        self.open_positions[scope_id] = new_position

    def get_position(self, scope_id: str) -> int:
        return self.open_positions.get(scope_id, 0)

    def get_position_count(self, scope_id: str) -> int:
        return self.open_positions.get(scope_id, 0)

    def get_total_exposure(self) -> float:
        return sum(self.position_sizes.values())

    # --- Position sizing ----------------------------------------------------

    def calculate_position_size(
        self, account_value: float, risk_percent: float, stop_loss: float
    ) -> float:
        """Kelly-style sizing: (account_value * risk_percent) / stop_loss.

        ``risk_percent`` is a decimal (0.01 = 1%). ``stop_loss`` is the per-unit
        loss the strategy is willing to take. Non-positive inputs return 0.
        """
        if account_value <= 0 or risk_percent <= 0 or stop_loss <= 0:
            return 0.0
        return (account_value * risk_percent) / stop_loss

    # --- Helpers ------------------------------------------------------------

    @staticmethod
    def _today() -> str:
        return datetime.now().date().isoformat()

    @staticmethod
    def _current_hour() -> str:
        return datetime.now().replace(minute=0, second=0, microsecond=0).isoformat()
