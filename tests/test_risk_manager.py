"""RiskManager unit tests — kill switch + circuit breakers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from trading_agent.risk_manager import (
    KILL_SWITCH_ENV,
    RiskLimits,
    RiskManager,
)


@pytest.fixture
def rm(tmp_path: Path) -> RiskManager:
    """RiskManager with low limits and an isolated kill-switch file."""
    return RiskManager(
        limits=RiskLimits(
            max_daily_loss=100.0,
            max_position_size=10.0,
            max_trades_per_hour=3,
            max_open_positions=2,
        ),
        kill_switch_file=tmp_path / ".kill_switch",
    )


# --- check_daily_loss_limit ----------------------------------------------------

def test_daily_loss_below_limit_passes(rm: RiskManager) -> None:
    assert rm.check_daily_loss_limit("strat-a", 50.0) is False


def test_daily_loss_at_exact_limit_passes(rm: RiskManager) -> None:
    # threshold is "> max_daily_loss" so equal is allowed
    assert rm.check_daily_loss_limit("strat-a", 100.0) is False


def test_daily_loss_above_limit_blocks(rm: RiskManager) -> None:
    assert rm.check_daily_loss_limit("strat-a", 101.0) is True


def test_daily_loss_accumulates(rm: RiskManager) -> None:
    assert rm.check_daily_loss_limit("strat-a", 60.0) is False
    assert rm.check_daily_loss_limit("strat-a", 50.0) is True


def test_daily_loss_per_scope(rm: RiskManager) -> None:
    rm.check_daily_loss_limit("strat-a", 80.0)
    # different scope keeps its own counter
    assert rm.check_daily_loss_limit("strat-b", 80.0) is False


def test_daily_loss_resets_on_new_day(rm: RiskManager) -> None:
    with freeze_time("2026-05-21 12:00:00"):
        rm.check_daily_loss_limit("strat-a", 90.0)
    with freeze_time("2026-05-22 09:00:00"):
        # new day → counter reset, so 90 alone < 100 = not blocked
        assert rm.check_daily_loss_limit("strat-a", 90.0) is False


# --- check_position_size ------------------------------------------------------

def test_position_size_below_limit_passes(rm: RiskManager) -> None:
    assert rm.check_position_size("strat-a", 5.0) is False


def test_position_size_above_limit_blocks(rm: RiskManager) -> None:
    assert rm.check_position_size("strat-a", 11.0) is True


def test_position_size_uses_absolute_value(rm: RiskManager) -> None:
    # short positions also count toward the size cap
    assert rm.check_position_size("strat-a", -11.0) is True
    assert rm.check_position_size("strat-a", -5.0) is False


# --- increment_hourly_trades --------------------------------------------------

def test_hourly_trades_below_limit_passes(rm: RiskManager) -> None:
    for _ in range(3):
        assert rm.increment_hourly_trades("strat-a") is False


def test_hourly_trades_above_limit_blocks(rm: RiskManager) -> None:
    for _ in range(3):
        rm.increment_hourly_trades("strat-a")
    assert rm.increment_hourly_trades("strat-a") is True


def test_hourly_trades_resets_on_new_hour(rm: RiskManager) -> None:
    with freeze_time("2026-05-22 09:30:00"):
        for _ in range(3):
            rm.increment_hourly_trades("strat-a")
    with freeze_time("2026-05-22 10:00:00"):
        assert rm.increment_hourly_trades("strat-a") is False


# --- update_open_positions ----------------------------------------------------

def test_open_positions_below_limit_passes(rm: RiskManager) -> None:
    assert rm.update_open_positions("strat-a", 1) is False
    assert rm.update_open_positions("strat-a", 1) is False


def test_open_positions_above_limit_blocks(rm: RiskManager) -> None:
    rm.update_open_positions("strat-a", 2)
    assert rm.update_open_positions("strat-a", 1) is True


def test_open_positions_cannot_go_negative(rm: RiskManager) -> None:
    rm.update_open_positions("strat-a", -5)
    assert rm.get_position("strat-a") == 0


def test_open_close_position_accounting(rm: RiskManager) -> None:
    rm.open_position("strat-a", 5.0)
    assert rm.get_position("strat-a") == 1
    assert rm.position_sizes["strat-a"] == 5.0
    rm.close_position("strat-a")
    assert rm.get_position("strat-a") == 0
    assert "strat-a" not in rm.position_sizes


def test_partial_close_decrements(rm: RiskManager) -> None:
    rm.update_open_positions("strat-a", 2)
    rm.partial_close("strat-a", 1)
    assert rm.get_position("strat-a") == 1


def test_flip_position_overrides_count(rm: RiskManager) -> None:
    rm.update_open_positions("strat-a", 2)
    rm.flip_position("strat-a", -1)
    assert rm.get_position("strat-a") == -1


def test_total_exposure_sums_across_scopes(rm: RiskManager) -> None:
    rm.open_position("strat-a", 30.0)
    rm.open_position("strat-b", 70.0)
    assert rm.get_total_exposure() == 100.0


# --- Kill switch --------------------------------------------------------------

def test_kill_switch_flag_blocks_all_checks(rm: RiskManager) -> None:
    rm.activate_kill_switch()
    assert rm.check_kill_switch() is True
    assert rm.check_daily_loss_limit("strat-a", 1.0) is True
    assert rm.check_position_size("strat-a", 0.1) is True
    assert rm.increment_hourly_trades("strat-a") is True
    assert rm.update_open_positions("strat-a", 0) is True


def test_kill_switch_deactivate(rm: RiskManager) -> None:
    rm.activate_kill_switch()
    rm.deactivate_kill_switch()
    assert rm.kill_switch_active is False
    assert rm.check_position_size("strat-a", 1.0) is False


def test_kill_switch_via_env_var(rm: RiskManager) -> None:
    with patch.dict("os.environ", {KILL_SWITCH_ENV: "1"}):
        assert rm.kill_switch_active is True
        assert rm.check_position_size("strat-a", 1.0) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_kill_switch_env_falsey_values_dont_trip(rm: RiskManager, value: str) -> None:
    with patch.dict("os.environ", {KILL_SWITCH_ENV: value}):
        assert rm.kill_switch_active is False


def test_kill_switch_via_file_flag(tmp_path: Path) -> None:
    flag = tmp_path / ".kill_switch"
    rm = RiskManager(kill_switch_file=flag)
    assert rm.kill_switch_active is False
    flag.touch()
    assert rm.kill_switch_active is True


# --- Position sizing ---------------------------------------------------------

def test_calculate_position_size_normal(rm: RiskManager) -> None:
    # 10000 * 0.01 / 2 = 50
    assert rm.calculate_position_size(10_000, 0.01, 2.0) == 50.0


@pytest.mark.parametrize(
    "account,risk,stop",
    [(0, 0.01, 1.0), (10_000, 0, 1.0), (10_000, 0.01, 0), (-100, 0.01, 1.0)],
)
def test_calculate_position_size_invalid_inputs_return_zero(
    rm: RiskManager, account: float, risk: float, stop: float
) -> None:
    assert rm.calculate_position_size(account, risk, stop) == 0.0


# --- Defaults & properties ----------------------------------------------------

def test_default_limits() -> None:
    rm = RiskManager()
    assert rm.MAX_DAILY_LOSS == 1000.0
    assert rm.MAX_POSITION_SIZE == 100.0
    assert rm.MAX_TRADES_PER_HOUR == 10
    assert rm.MAX_OPEN_POSITIONS == 5


def test_today_helper_returns_iso_date() -> None:
    today = RiskManager._today()
    datetime.fromisoformat(today)
