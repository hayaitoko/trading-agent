"""Deterministic feature → canonical pattern label taxonomy (P4).

Feature extraction from price bars (gap %, volume z-score, trend break) maps
to a small canonical vocabulary. This vocabulary is objective market description —
no LLM involved. The LLM can enrich a label with a narrative, but the label
itself comes from deterministic computation so it's reproducible and comparable.

Canonical labels:
  gap-up-no-news          large gap up with normal social signal
  gap-down-no-news        large gap down with normal social signal
  gap-up-catalyst         gap up with elevated social velocity
  gap-down-catalyst       gap down with elevated social velocity
  capitulation-volume-spike   vol z-score extreme on a down move
  volume-spike-up             vol z-score extreme on an up move
  failed-breakout             prior high approach then close-below
  bull-flag                   tight range after strong up move
  bear-flag                   tight range after strong down move
  earnings-drift              gap at earnings, drift in same direction
  mean-reversion-setup        extended trend, reversal signal
  unknown                     inputs insufficient for classification
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any


class PatternLabel(str):
    """Canonical pattern label string constants."""

    GAP_UP_NO_NEWS = "gap-up-no-news"
    GAP_DOWN_NO_NEWS = "gap-down-no-news"
    GAP_UP_CATALYST = "gap-up-catalyst"
    GAP_DOWN_CATALYST = "gap-down-catalyst"
    CAPITULATION_VOLUME_SPIKE = "capitulation-volume-spike"
    VOLUME_SPIKE_UP = "volume-spike-up"
    FAILED_BREAKOUT = "failed-breakout"
    BULL_FLAG = "bull-flag"
    BEAR_FLAG = "bear-flag"
    EARNINGS_DRIFT = "earnings-drift"
    MEAN_REVERSION_SETUP = "mean-reversion-setup"
    UNKNOWN = "unknown"


# Thresholds for feature extraction.
_GAP_PCT_THRESHOLD = 0.02       # 2% gap is notable
_LARGE_GAP_PCT = 0.04           # 4% is large
_VOL_SPIKE_Z = 2.0              # z-score above this = spike
_TREND_LOOKBACK = 10            # bars for trend measurement
_FLAG_RANGE_PCT = 0.03          # price range ≤ 3% of midpoint = flag
_EARNINGS_GAP = 0.03            # 3% gap near earnings


@dataclass
class BarFeatures:
    """Extracted features from a price bar sequence."""

    gap_pct: float          # (open_today - close_yesterday) / close_yesterday
    volume_zscore: float    # standard scores vs recent volume history
    trend_pct: float        # close[-1] vs close[-N] / close[-N]
    range_pct: float        # (max_close - min_close) / mean_close over window
    has_earnings: bool      # caller signals this is near an earnings event
    social_velocity: float  # normalized social velocity (from situation layer)


def extract_features(
    bars: list[dict[str, Any]],
    *,
    has_earnings: bool = False,
    social_velocity: float = 0.0,
) -> BarFeatures | None:
    """Extract features from a list of OHLCV bar dicts.

    Bars are oldest-first. Returns ``None`` if there aren't enough bars.
    Each bar must have at least ``close`` and ``volume`` keys.
    """
    if len(bars) < 3:
        return None

    closes = [float(b["close"]) for b in bars if b.get("close") is not None]
    volumes = [float(b.get("volume", 0) or 0) for b in bars]
    opens = [float(b.get("open", b["close"])) for b in bars if b.get("close") is not None]

    if len(closes) < 3 or len(volumes) < 3:
        return None

    # Gap: today's open vs yesterday's close.
    gap_pct = (opens[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0.0

    # Volume z-score: last bar vs prior bars.
    prior_vols = volumes[:-1]
    if len(prior_vols) >= 2:
        v_mean = statistics.mean(prior_vols)
        v_std = statistics.stdev(prior_vols)
        vol_z = (volumes[-1] - v_mean) / v_std if v_std > 0 else 0.0
    else:
        vol_z = 0.0

    # Trend: compare last close to N bars ago.
    n = min(_TREND_LOOKBACK, len(closes) - 1)
    trend_pct = (closes[-1] - closes[-1 - n]) / closes[-1 - n] if closes[-1 - n] > 0 else 0.0

    # Range: max-min / mean over the window.
    window_closes = closes[-min(10, len(closes)):]
    c_mean = statistics.mean(window_closes)
    range_pct = (max(window_closes) - min(window_closes)) / c_mean if c_mean > 0 else 0.0

    return BarFeatures(
        gap_pct=gap_pct,
        volume_zscore=vol_z,
        trend_pct=trend_pct,
        range_pct=range_pct,
        has_earnings=has_earnings,
        social_velocity=float(social_velocity) if not math.isnan(social_velocity) else 0.0,
    )


def compute_label(
    bars: list[dict[str, Any]],
    *,
    has_earnings: bool = False,
    social_velocity: float = 0.0,
) -> str:
    """Deterministically compute a canonical label from bars.

    Priority order (first match wins):
    1. Earnings-related patterns
    2. Large gap patterns
    3. Capitulation / volume spike
    4. Flag patterns (tight range)
    5. Breakout / mean-reversion
    6. Unknown (insufficient features)
    """
    features = extract_features(bars, has_earnings=has_earnings, social_velocity=social_velocity)
    if features is None:
        return PatternLabel.UNKNOWN

    f = features
    gap = f.gap_pct
    vol_z = f.volume_zscore
    trend = f.trend_pct
    high_social = abs(f.social_velocity) > 0.5

    # 1. Earnings drift: gap near earnings
    if f.has_earnings and abs(gap) >= _EARNINGS_GAP:
        return PatternLabel.EARNINGS_DRIFT

    # 2. Large gaps with/without catalyst
    if gap >= _LARGE_GAP_PCT:
        return PatternLabel.GAP_UP_CATALYST if high_social else PatternLabel.GAP_UP_NO_NEWS
    if gap <= -_LARGE_GAP_PCT:
        return PatternLabel.GAP_DOWN_CATALYST if high_social else PatternLabel.GAP_DOWN_NO_NEWS

    # 3. Volume spikes
    if vol_z >= _VOL_SPIKE_Z:
        if trend < 0:
            return PatternLabel.CAPITULATION_VOLUME_SPIKE
        return PatternLabel.VOLUME_SPIKE_UP

    # 4. Flag patterns: tight range after a directional move
    if f.range_pct <= _FLAG_RANGE_PCT:
        if trend > 0.05:
            return PatternLabel.BULL_FLAG
        if trend < -0.05:
            return PatternLabel.BEAR_FLAG

    # 5. Mean reversion setup: extended trend
    if trend > 0.15:
        return PatternLabel.MEAN_REVERSION_SETUP
    if trend < -0.15:
        return PatternLabel.MEAN_REVERSION_SETUP

    # 6. Fallback
    return PatternLabel.UNKNOWN
