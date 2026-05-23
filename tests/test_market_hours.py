"""Tests for the US equity market-hours helper."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from trading_agent.market_hours import is_us_equity_market_open

_ET = ZoneInfo("America/New_York")


def test_open_midday_weekday():
    # Friday 2026-05-22, 12:00 ET
    assert is_us_equity_market_open(datetime(2026, 5, 22, 12, 0, tzinfo=_ET)) is True


def test_closed_before_open():
    assert is_us_equity_market_open(datetime(2026, 5, 22, 9, 0, tzinfo=_ET)) is False


def test_closed_at_and_after_close():
    assert is_us_equity_market_open(datetime(2026, 5, 22, 16, 0, tzinfo=_ET)) is False
    assert is_us_equity_market_open(datetime(2026, 5, 22, 16, 30, tzinfo=_ET)) is False


def test_open_boundary_at_930():
    assert is_us_equity_market_open(datetime(2026, 5, 22, 9, 30, tzinfo=_ET)) is True


def test_closed_on_weekend():
    # Saturday / Sunday
    assert is_us_equity_market_open(datetime(2026, 5, 23, 12, 0, tzinfo=_ET)) is False
    assert is_us_equity_market_open(datetime(2026, 5, 24, 12, 0, tzinfo=_ET)) is False


def test_converts_from_other_timezone():
    # 2026-05-22 16:00 UTC == 12:00 ET (EDT) -> open
    assert is_us_equity_market_open(datetime(2026, 5, 22, 16, 0, tzinfo=ZoneInfo("UTC"))) is True
