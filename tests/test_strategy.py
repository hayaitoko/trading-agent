"""Tests for Strategy ABC and MeanReversionStrategy."""

from __future__ import annotations

from pathlib import Path

import pytest

from trading_agent.strategies.mean_reversion import MeanReversionStrategy
from trading_agent.strategy import Strategy

# Path to the real TOML config shipped with the repo.
STRATEGY_TOML = Path(__file__).resolve().parents[1] / "strategies" / "config" / "mean_reversion.toml"


# --- Strategy ABC -------------------------------------------------------------


def test_strategy_is_abstract_and_cannot_instantiate():
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_strategy_subclass_missing_abstract_methods_fails():
    class Incomplete(Strategy):
        def get_symbols(self):  # type: ignore[override]
            return []

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_strategy_full_subclass_works():
    class Tiny(Strategy):
        def get_symbols(self):  # type: ignore[override]
            return ["AAPL"]

        def get_timeframe(self):  # type: ignore[override]
            return "1m"

        def get_params(self):  # type: ignore[override]
            return {"k": 1}

        def on_data(self, bar_or_tick):  # type: ignore[override]
            return {"side": "NEUTRAL"}

    s = Tiny()
    assert s.get_symbols() == ["AAPL"]
    assert s.get_timeframe() == "1m"
    assert s.get_params() == {"k": 1}
    assert s.on_data({"close": 1.0})["side"] == "NEUTRAL"


# --- MeanReversionStrategy ---------------------------------------------------


def _fast_params(sma_period=5, std_multiplier=2.0):
    return {
        "timeframe": "1m",
        "symbols": ["TEST"],
        "strategy": {
            "sma_period": sma_period,
            "std_multiplier": std_multiplier,
            "position_size": 1.0,
        },
    }


def test_mean_reversion_loads_from_toml_file():
    strat = MeanReversionStrategy(config_path=STRATEGY_TOML)
    assert strat.get_timeframe() == "1m"
    assert strat.get_symbols() == ["SYNTH-USD"]
    params = strat.get_params()
    assert params["sma_period"] == 20
    assert params["std_multiplier"] == 2.0
    assert params["position_size"] == 1.0


def test_mean_reversion_requires_params_or_config():
    with pytest.raises(ValueError):
        MeanReversionStrategy()


def test_mean_reversion_neutral_until_window_fills():
    strat = MeanReversionStrategy(params=_fast_params(sma_period=5))
    # First sma_period-1 bars must all return NEUTRAL with sma/lower_band = None.
    for i in range(4):
        out = strat.on_data({"symbol": "TEST", "close": 100.0 + i})
        assert out["side"] == "NEUTRAL"
        assert out["sma"] is None
        assert out["lower_band"] is None


def test_mean_reversion_emits_long_on_lower_band_then_short_on_sma():
    strat = MeanReversionStrategy(params=_fast_params(sma_period=5, std_multiplier=2.0))
    # Feed a flat window so std=0; with std_multiplier=2 -> lower_band == sma == 100.
    # We need a drop below sma to trigger LONG, then rise back to sma to trigger SHORT.
    bars = [
        {"symbol": "TEST", "close": 100.0},
        {"symbol": "TEST", "close": 100.0},
        {"symbol": "TEST", "close": 100.0},
        {"symbol": "TEST", "close": 100.0},
    ]
    for b in bars:
        out = strat.on_data(b)
        assert out["side"] == "NEUTRAL"

    # 5th bar fills window. With perfectly flat window and a lower close
    # the sma drops just under previous level; force the window to be flat at 100
    # then submit a deviating close to trip the band.
    # Use a fresh strategy with a window that already has 4 100s, then one big drop.
    out = strat.on_data({"symbol": "TEST", "close": 90.0})
    # Window is now [100,100,100,100,90]; sma=98; std>0; lower_band < 98.
    # 90 may or may not be below lower_band depending on std; assert it triggered
    # OR keep pushing low values. Push another sharp drop to guarantee a LONG.
    if out["side"] != "LONG":
        out = strat.on_data({"symbol": "TEST", "close": 80.0})
    assert out["side"] == "LONG"
    assert out["sma"] is not None
    assert out["lower_band"] is not None
    assert out["price"] <= out["lower_band"]

    # Now push closes back up to / above the SMA to trigger SHORT (exit).
    for close in (100.0, 110.0, 120.0, 130.0, 140.0):
        out = strat.on_data({"symbol": "TEST", "close": close})
        if out["side"] == "SHORT":
            break
    assert out["side"] == "SHORT"


def test_mean_reversion_accepts_dataframe_input():
    import pandas as pd

    strat = MeanReversionStrategy(params=_fast_params(sma_period=3))
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    out = strat.on_data(df)
    # Only the last close is consumed; need 3 calls to fill window.
    # First call only adds 102; sma not yet ready.
    assert out["side"] == "NEUTRAL"


def test_mean_reversion_signal_payload_shape():
    strat = MeanReversionStrategy(params=_fast_params(sma_period=2))
    strat.on_data({"symbol": "TEST", "close": 100.0})
    out = strat.on_data({"symbol": "TEST", "close": 101.0})
    assert set(out.keys()) == {"asset", "side", "amount", "price", "sma", "lower_band"}
    assert out["asset"] == "TEST"
    assert out["amount"] == 1.0


def test_mean_reversion_uses_default_symbol_when_bar_lacks_one():
    strat = MeanReversionStrategy(params=_fast_params(sma_period=2))
    out = strat.on_data({"close": 100.0})
    assert out["asset"] == "TEST"
