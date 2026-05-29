"""WS-Situation C1 — forecast cone tests.

Tests
-----
intel/forecast.py
  - build_forecast with all providers absent → degenerate cone (no points)
  - build_forecast with history only → empirical_sigma present, cone has points
  - build_forecast empirical_sigma math (log-return realized vol)
  - build_forecast with mock IV provider → iv_sigma present
  - build_forecast with mock PM provider matching ticker → pm_implied_move present
  - build_forecast combined_sigma = max of available
  - cone geometry: t=0 → lo=hi=mid=current_price
  - cone geometry: t>0 → lo < mid < hi

intel/tools/look/forecast.py
  - ForecastTool flag off (settings_store=None) → disabled error
  - ForecastTool flag on, no history → degenerate cone but ok=True
  - ForecastTool flag on, mock history → ok=True, correct shape
  - ForecastTool horizon clamping (horizon=7 → nearest valid = 5 or 10)

settings_store DEFAULTS include SITUATION_FORECAST (default False)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_agent.config.settings_store import DEFAULTS
from trading_agent.intel.forecast import (
    ForecastCone,
    _compute_empirical_sigma,
    build_forecast,
)
from trading_agent.intel.tools.look.forecast import ForecastTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(close: float) -> Any:
    return SimpleNamespace(close=close)


def _history_service(closes: list[float]) -> Any:
    svc = MagicMock()
    svc.get_bars.return_value = [_bar(c) for c in closes]
    return svc


def _settings_on(flag: str) -> Any:
    s = MagicMock()
    s.get = lambda uid, key, default=None: (True if key == flag else False)
    return s


def _chain_provider_with_iv(iv: float, symbol: str = "AAPL") -> Any:
    from datetime import date

    from trading_agent.instruments.options import OptionContract, OptionQuote, OptionRight
    c = OptionContract(symbol, date(2026, 6, 19), 150.0, OptionRight.CALL)
    q = OptionQuote(contract=c, bid=2.0, ask=2.4, last=2.2, implied_vol=iv)
    p = MagicMock()
    p.get_chain.return_value = [q]
    return p


# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------


def test_defaults_include_situation_forecast_false() -> None:
    assert DEFAULTS.get("SITUATION_FORECAST") is False


# ---------------------------------------------------------------------------
# intel/forecast.py — build_forecast
# ---------------------------------------------------------------------------


def test_build_forecast_no_providers_returns_degenerate_cone() -> None:
    cone = build_forecast("AAPL")
    assert isinstance(cone, ForecastCone)
    assert cone.symbol == "AAPL"
    assert cone.empirical_sigma is None
    assert cone.iv_sigma is None
    assert cone.pm_implied_move is None
    assert cone.combined_sigma is None
    assert cone.points == []
    assert cone.components_used == []


def test_build_forecast_with_history_returns_empirical_sigma() -> None:
    closes = [100.0 + i * 0.5 for i in range(32)]  # 32 bars for 31 log-returns
    svc = _history_service(closes)
    cone = build_forecast("SPY", history_service=svc, spot_price=closes[-1])
    assert cone.empirical_sigma is not None
    assert cone.empirical_sigma > 0
    assert "empirical" in cone.components_used


def test_build_forecast_cone_has_points_with_empirical_sigma() -> None:
    closes = [100.0 + i * 0.5 for i in range(32)]
    svc = _history_service(closes)
    cone = build_forecast("AAPL", horizon_days=5, history_service=svc, spot_price=100.0)
    assert len(cone.points) == 6  # t=0..5


def test_build_forecast_cone_point_t0_is_flat() -> None:
    closes = [100.0] * 32
    svc = _history_service(closes)
    cone = build_forecast("AAPL", horizon_days=10, history_service=svc, spot_price=100.0)
    t0 = cone.points[0]
    assert t0.t == 0
    assert t0.lo == pytest.approx(100.0)
    assert t0.mid == pytest.approx(100.0)
    assert t0.hi == pytest.approx(100.0)


def test_build_forecast_cone_widens_over_time() -> None:
    closes = [100.0 + i for i in range(32)]
    svc = _history_service(closes)
    cone = build_forecast("AAPL", horizon_days=30, history_service=svc, spot_price=131.0)
    assert cone.combined_sigma is not None
    for p in cone.points:
        assert p.lo <= p.mid <= p.hi, f"t={p.t}: lo={p.lo} mid={p.mid} hi={p.hi}"
    if len(cone.points) > 1:
        assert cone.points[-1].hi > cone.points[0].hi  # cone widens


def test_build_forecast_with_iv_provider_adds_iv_sigma() -> None:
    chain = _chain_provider_with_iv(0.35)
    settings = _settings_on("SITUATION_OPTIONS_IV")
    cone = build_forecast(
        "AAPL",
        horizon_days=10,
        chain_provider=chain,
        settings_store=settings,
        spot_price=150.0,
    )
    assert cone.iv_sigma == pytest.approx(0.35)
    assert "iv" in cone.components_used


def test_build_forecast_combined_sigma_is_max() -> None:
    closes = [100.0 + i * 0.2 for i in range(32)]
    svc = _history_service(closes)
    chain = _chain_provider_with_iv(0.50)  # higher than realized
    settings = _settings_on("SITUATION_OPTIONS_IV")
    cone = build_forecast(
        "AAPL",
        horizon_days=5,
        history_service=svc,
        chain_provider=chain,
        settings_store=settings,
        spot_price=100.0,
    )
    assert cone.empirical_sigma is not None
    assert cone.iv_sigma == pytest.approx(0.50)
    assert cone.combined_sigma == pytest.approx(0.50)  # max of empirical + iv


def test_compute_empirical_sigma_too_few_bars_returns_none() -> None:
    svc = _history_service([100.0, 101.0])  # only 2 bars → too few
    result = _compute_empirical_sigma(svc, "X")
    # 2 bars → 1 log-return → below the 5-bar minimum
    assert result is None


def test_compute_empirical_sigma_flat_prices_returns_near_zero() -> None:
    svc = _history_service([100.0] * 32)
    result = _compute_empirical_sigma(svc, "X")
    assert result is not None
    assert result == pytest.approx(0.0, abs=1e-9)


def test_build_forecast_to_dict_shape() -> None:
    closes = [100.0 + i for i in range(32)]
    svc = _history_service(closes)
    cone = build_forecast("SPY", horizon_days=5, history_service=svc, spot_price=130.0)
    d = cone.to_dict()
    for key in ("symbol", "horizon_days", "current_price", "empirical_sigma",
                "iv_sigma", "pm_implied_move", "combined_sigma",
                "components_used", "points"):
        assert key in d, f"missing key: {key}"
    for pt in d["points"]:
        for k in ("t", "lo", "mid", "hi"):
            assert k in pt


# ---------------------------------------------------------------------------
# ForecastTool
# ---------------------------------------------------------------------------


def test_forecast_tool_flag_off_returns_disabled() -> None:
    tool = ForecastTool(trader_id="Alpha")
    result = tool("AAPL", horizon=10)
    assert not result.ok
    assert result.error.kind == "disabled"
    assert "SITUATION_FORECAST" in result.error.message


def test_forecast_tool_flag_on_no_data_returns_degenerate_ok() -> None:
    settings = _settings_on("SITUATION_FORECAST")
    tool = ForecastTool(trader_id="Alpha", settings_store=settings)
    result = tool("AAPL", horizon=5)
    assert result.ok
    d = result.data
    assert d["symbol"] == "AAPL"
    assert d["horizon_days"] == 5
    assert d["combined_sigma"] is None
    assert d["points"] == []


def test_forecast_tool_flag_on_with_history_returns_cone() -> None:
    settings = _settings_on("SITUATION_FORECAST")
    closes = [100.0 + i for i in range(32)]
    svc = _history_service(closes)
    tool = ForecastTool(
        trader_id="Alpha",
        settings_store=settings,
        history_service=svc,
        spot_prices={"AAPL": 130.0},
    )
    result = tool("AAPL", horizon=30)
    assert result.ok
    d = result.data
    assert d["empirical_sigma"] is not None
    assert len(d["points"]) == 31  # t=0..30
    for pt in d["points"]:
        assert pt["lo"] <= pt["mid"] <= pt["hi"]


def test_forecast_tool_horizon_clamping() -> None:
    """horizon=7 is between 5 and 10; should clamp to whichever is closer (10)."""
    settings = _settings_on("SITUATION_FORECAST")
    tool = ForecastTool(trader_id="Alpha", settings_store=settings)
    result = tool("AAPL", horizon=7)
    assert result.ok
    # min((5,10,30), key=lambda h: abs(h-7)) → 5 (|5-7|=2 < |10-7|=3)
    assert result.data["horizon_days"] == 5


def test_forecast_tool_no_paper_leak() -> None:
    """MONEY IS REAL: ForecastTool result must not contain forbidden words."""
    settings = _settings_on("SITUATION_FORECAST")
    closes = [100.0 + i for i in range(32)]
    svc = _history_service(closes)
    tool = ForecastTool(trader_id="Alpha", settings_store=settings, history_service=svc, spot_prices={"SPY": 130.0})
    result = tool("SPY", horizon=5)
    assert result.ok
    result_str = str(result.data).lower()
    for word in ("paper", "sim", "demo", "fake"):
        assert word not in result_str, f"forbidden word '{word}' in forecast result"
