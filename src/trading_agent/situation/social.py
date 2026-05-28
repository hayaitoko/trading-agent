"""Social aggregator (P3): compact metrics from raw social items.

Social is an adversarial input — actively gamed by pump rings, bots, and
coordinated shilling. We NEVER feed raw posts into prompts. Instead we
distill into a small metrics block and sanitize all text before any LLM
consumption.

Aggregation produces :class:`SocialMetrics` per ticker (or overall):
  mention_volume   — raw count of mentions in the window
  velocity         — Δvolume / prior_volume (positive = acceleration)
  sentiment_mean   — mean sentiment score [-1, 1]
  sentiment_std    — spread (high std = conflicted crowd)
  bullish_pct      — fraction of positive-sentiment items
  credibility_mean — mean source credibility weight [0, 1]

Sanitization strips URLs, escapes prompt-injection patterns, and truncates.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

# ---- sanitization -----------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# Patterns that could break out of a prompt boundary or inject instructions.
_INJECTION_RE = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?"
    r"|<\|im_start\|>"
    r"|<\|im_end\|>"
    r"|system\s*:)",
    re.IGNORECASE,
)
_MAX_SANITIZED_LEN = 500


def sanitize_social_text(raw: str) -> str:
    """Strip URLs, escape injection patterns, truncate.

    The sanitized output is still "social text" — never a directive. The
    caller decides whether to use it at all (aggregate stats are preferred
    over individual post text).
    """
    text = _URL_RE.sub("[url]", raw)
    text = _INJECTION_RE.sub("[filtered]", text)
    # Remove backtick triplets that could close a code-block in a prompt.
    text = text.replace("```", "'''")
    # Flatten excessive whitespace.
    text = " ".join(text.split())
    return text[:_MAX_SANITIZED_LEN]


# ---- social item shape ------------------------------------------------------

@dataclass
class SocialItem:
    """One normalized social mention."""

    source: str              # reddit | stocktwits | bluesky | rss | ...
    ticker: str | None       # None → general market chatter
    text: str                # raw text (sanitize before LLM use)
    sentiment: float = 0.0  # [-1, 1] negative=bearish
    credibility: float = 0.5  # source credibility weight [0, 1]
    ts: str = ""             # ISO-8601 UTC


# ---- credibility weights ----------------------------------------------------

_SOURCE_CREDIBILITY: dict[str, float] = {
    "reuters": 0.9,
    "ap": 0.9,
    "bloomberg": 0.85,
    "wsj": 0.85,
    "rss": 0.7,
    "reddit": 0.45,
    "stocktwits": 0.40,
    "bluesky": 0.35,
    "browser": 0.5,
}


def source_credibility(source: str) -> float:
    return _SOURCE_CREDIBILITY.get(source.lower(), 0.4)


# ---- metrics ----------------------------------------------------------------

@dataclass
class SocialMetrics:
    """Compact social signal for one ticker (or market-wide)."""

    ticker: str | None
    mention_volume: int
    velocity: float           # Δ vs prior window, NaN if no prior
    sentiment_mean: float
    sentiment_std: float
    bullish_pct: float
    credibility_mean: float
    sources: list[str] = field(default_factory=list)  # distinct sources seen

    def is_empty(self) -> bool:
        return self.mention_volume == 0

    def to_context_lines(self) -> list[str]:
        if self.is_empty():
            return [f"  Social ({self.ticker or 'market'}): no mentions"]
        vel_str = (
            f"{self.velocity:+.0%}" if not math.isnan(self.velocity) else "n/a"
        )
        return [
            f"  Social ({self.ticker or 'market'}): "
            f"{self.mention_volume} mentions, velocity {vel_str}, "
            f"sentiment {self.sentiment_mean:+.2f}±{self.sentiment_std:.2f}, "
            f"bullish {self.bullish_pct:.0%}, "
            f"credibility {self.credibility_mean:.2f} "
            f"(sources: {', '.join(self.sources) if self.sources else 'none'})"
        ]


# ---- aggregator -------------------------------------------------------------

class SocialAggregator:
    """Aggregate raw :class:`SocialItem` lists into compact :class:`SocialMetrics`.

    Parameters
    ----------
    prior_volumes : historical mention counts keyed by ticker (for velocity).
    """

    def __init__(self, prior_volumes: dict[str, int] | None = None) -> None:
        self._prior: dict[str, int] = dict(prior_volumes or {})

    def aggregate(
        self,
        items: list[SocialItem],
        ticker: str | None = None,
    ) -> SocialMetrics:
        """Aggregate ``items`` into metrics.

        If ``ticker`` is given, only items for that ticker are counted;
        otherwise all items are included (market-wide).
        """
        if ticker:
            subset = [it for it in items if it.ticker and it.ticker.upper() == ticker.upper()]
        else:
            subset = list(items)

        volume = len(subset)
        prior = self._prior.get(ticker or "", 0)
        velocity = (
            (volume - prior) / max(prior, 1)
            if prior > 0 or volume > 0
            else math.nan
        )
        if prior == 0 and volume == 0:
            velocity = math.nan

        sentiments = [it.sentiment for it in subset]
        credibilities = [it.credibility for it in subset]
        sources = sorted({it.source for it in subset})

        s_mean = statistics.mean(sentiments) if sentiments else 0.0
        s_std = statistics.stdev(sentiments) if len(sentiments) > 1 else 0.0
        bullish = (
            sum(1 for s in sentiments if s > 0.0) / len(sentiments)
            if sentiments else 0.0
        )
        c_mean = statistics.mean(credibilities) if credibilities else 0.0

        # Update prior for next call.
        self._prior[ticker or ""] = volume

        return SocialMetrics(
            ticker=ticker,
            mention_volume=volume,
            velocity=velocity,
            sentiment_mean=s_mean,
            sentiment_std=s_std,
            bullish_pct=bullish,
            credibility_mean=c_mean,
            sources=sources,
        )

    def aggregate_all(
        self,
        items: list[SocialItem],
        tickers: list[str],
    ) -> dict[str, SocialMetrics]:
        """Aggregate for each ticker in ``tickers`` + a market-wide entry."""
        result: dict[str, SocialMetrics] = {}
        for ticker in tickers:
            result[ticker.upper()] = self.aggregate(items, ticker=ticker)
        result["__market__"] = self.aggregate(items, ticker=None)
        return result


def social_items_from_raw(
    raw_items: list[Any],
    source: str,
    ticker: str | None = None,
) -> list[SocialItem]:
    """Convert ingest :class:`~trading_agent.ingest.fetchers.base.RawItem`
    objects into :class:`SocialItem` with credibility and zero sentiment
    (caller enriches sentiment if needed).
    """
    cred = source_credibility(source)
    out: list[SocialItem] = []
    for item in raw_items:
        text = str(getattr(item, "text", "") or "")
        tick = ticker or getattr(item, "ticker", None)
        ts = str(getattr(item, "ts", "") or "")
        out.append(SocialItem(
            source=source,
            ticker=tick,
            text=text,
            sentiment=0.0,
            credibility=cred,
            ts=ts,
        ))
    return out
