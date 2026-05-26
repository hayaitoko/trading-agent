"""Historical bars + fundamentals service and the richer trader context block.

WS-A. Today each bench trader sees only the last 30 *close* prices
(``llm/trader.py`` ``_build_context``). :class:`HistoryService` gives agents real
historical depth — a downsampled long view (e.g. ~1y daily) plus a dense recent
window (intraday last N bars, full OHLCV) — and optional fundamentals, assembled
into a single context block that drops in for the 30-close path.

Design notes:

* **No hardcoded provider.** Bars come from a :class:`BarProvider`, fundamentals
  from an optional :class:`FundamentalsProvider`. The default Alpaca bar provider
  reads the *same* Alpaca data key that feeds live books; provider choice resolves
  from ``user_settings`` / env via :func:`build_history_service`, never hardcoded.
* **Caching.** History rarely changes intraday, so bars and fundamentals are cached
  with a TTL to avoid refetching every cadence tick.
* **Token budget.** ``context_block`` downsamples the long view and caps the recent
  window so the prompt stays small at the default depth; depth is configurable
  (``history_depth`` in ``user_settings``).
* **No model spend here.** This module calls market-data providers only; nothing is
  cost-gated because no paid *model* is invoked (CONTRACTS §Cost-gating concerns
  model calls).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..config.settings_store import SettingsStore


class HistoryProviderError(RuntimeError):
    """A data provider could not be reached or is missing credentials."""


@dataclass
class Bar:
    """One OHLCV candle. ``timestamp`` is an ISO-8601 string (provider-native tz)."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@runtime_checkable
class BarProvider(Protocol):
    """Source of historical OHLCV bars (Alpaca, a recorded fixture, ...)."""

    def get_bars(self, symbol: str, timeframe: str, lookback: int) -> list[Bar]: ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    """Source of company fundamentals; returns a canonical dict or ``None``."""

    def get_fundamentals(self, symbol: str) -> dict[str, Any] | None: ...


# --- depth configuration -----------------------------------------------------


@dataclass(frozen=True)
class HistoryDepth:
    """How much history to feed the model.

    The long view is fetched at ``long_timeframe`` × ``long_lookback`` bars then
    *downsampled* to ``downsample_long_to`` points; the recent window is
    ``recent_timeframe`` × ``recent_lookback`` bars shown as full OHLCV.
    """

    long_timeframe: str = "1D"
    long_lookback: int = 365
    recent_timeframe: str = "1H"
    recent_lookback: int = 24
    downsample_long_to: int = 40


# Named presets the UI's history-depth control selects between (WS-G/Settings
# just stores the string; we read it). "off" still yields a small block when a
# service is injected — whether to inject at all is the caller's decision.
PRESETS: dict[str, HistoryDepth] = {
    "off": HistoryDepth("1D", 90, "1D", 5, 15),
    "shallow": HistoryDepth("1D", 120, "1D", 10, 20),
    "standard": HistoryDepth("1D", 365, "1H", 24, 40),
    "deep": HistoryDepth("1D", 730, "1H", 48, 60),
}
DEFAULT_DEPTH = "standard"


def resolve_depth(value: Any) -> HistoryDepth:
    """Coerce a setting value into a :class:`HistoryDepth`.

    Accepts a :class:`HistoryDepth`, a preset name, or a dict of overrides on top
    of the ``standard`` preset. Anything else falls back to ``standard``.
    """
    if isinstance(value, HistoryDepth):
        return value
    if isinstance(value, dict):
        valid = {f.name for f in fields(HistoryDepth)}
        overrides = {k: v for k, v in value.items() if k in valid}
        return replace(PRESETS[DEFAULT_DEPTH], **overrides)
    if isinstance(value, str):
        return PRESETS.get(value.strip().lower(), PRESETS[DEFAULT_DEPTH])
    return PRESETS[DEFAULT_DEPTH]


# --- helpers -----------------------------------------------------------------


def _downsample(bars: Sequence[Bar], n: int) -> list[Bar]:
    """Evenly sample ``bars`` down to at most ``n`` points, keeping the last bar."""
    count = len(bars)
    if n <= 0 or count <= n:
        return list(bars)
    step = count / n
    out = [bars[min(count - 1, int(i * step))] for i in range(n)]
    if out[-1] is not bars[-1]:
        out[-1] = bars[-1]
    return out


def _short_ts(ts: str) -> str:
    """Trim an ISO timestamp to ``YYYY-MM-DD HH:MM`` for compact prompt rows."""
    if not ts:
        return "?"
    return ts.replace("T", " ")[:16]


def _fmt_fundamentals(f: dict[str, Any]) -> str:
    """One-line summary of the populated canonical fundamentals fields."""
    parts: list[str] = []
    label = {
        "name": "name",
        "sector": "sector",
        "market_cap": "mktcap",
        "pe": "P/E",
        "eps": "EPS",
        "dividend_yield": "div%",
        "beta": "beta",
        "week52_high": "52wH",
        "week52_low": "52wL",
    }
    for key, tag in label.items():
        val = f.get(key)
        if val in (None, ""):
            continue
        if isinstance(val, float):
            parts.append(f"{tag}={val:,.2f}")
        else:
            parts.append(f"{tag}={val}")
    return "; ".join(parts) if parts else "(none)"


# --- service -----------------------------------------------------------------


class HistoryService:
    """Bars + fundamentals with TTL caching and a token-bounded context block.

    Construct with a :class:`BarProvider` (required) and an optional
    :class:`FundamentalsProvider`. ``depth`` may be a :class:`HistoryDepth`, a
    preset name, or an overrides dict. ``time_fn`` is a test seam for the cache
    clock.
    """

    def __init__(
        self,
        bar_provider: BarProvider,
        *,
        fundamentals_provider: FundamentalsProvider | None = None,
        depth: Any = DEFAULT_DEPTH,
        bar_ttl: float = 900.0,
        fundamentals_ttl: float = 6 * 3600.0,
        max_chars: int = 14_000,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.bar_provider = bar_provider
        self.fundamentals_provider = fundamentals_provider
        self.depth = resolve_depth(depth)
        self.bar_ttl = bar_ttl
        self.fundamentals_ttl = fundamentals_ttl
        self.max_chars = max_chars
        self._now = time_fn
        self._bar_cache: dict[tuple[str, str, int], tuple[float, list[Bar]]] = {}
        self._fund_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}

    # --- data access (CONTRACTS surface) ------------------------------------

    def bars(self, symbol: str, timeframe: str, lookback: int) -> list[Bar]:
        """Return up to ``lookback`` bars for ``symbol``, cached for ``bar_ttl``."""
        key = (symbol, timeframe, lookback)
        now = self._now()
        hit = self._bar_cache.get(key)
        if hit is not None and now - hit[0] < self.bar_ttl:
            return list(hit[1])
        fetched = list(self.bar_provider.get_bars(symbol, timeframe, lookback))
        self._bar_cache[key] = (now, fetched)
        return list(fetched)

    def fundamentals(self, symbol: str) -> dict[str, Any] | None:
        """Canonical fundamentals for ``symbol`` (or ``None``), cached longer."""
        if self.fundamentals_provider is None:
            return None
        now = self._now()
        hit = self._fund_cache.get(symbol)
        if hit is not None and now - hit[0] < self.fundamentals_ttl:
            return hit[1]
        data = self.fundamentals_provider.get_fundamentals(symbol)
        self._fund_cache[symbol] = (now, data)
        return data

    def context_block(self, symbols: Any, account: dict[str, Any]) -> str:
        """Assemble the richer prompt context: long view + recent window + fundamentals.

        Fetches once per symbol (cached), then renders. If the rendered block
        exceeds ``max_chars`` it is re-rendered with fewer recent rows / long
        samples until it fits — never refetching.
        """
        syms = [str(s) for s in symbols]
        data: dict[str, tuple[list[Bar], list[Bar], dict[str, Any] | None]] = {}
        for symbol in syms:
            long_bars = self.bars(symbol, self.depth.long_timeframe, self.depth.long_lookback)
            recent_bars = self.bars(symbol, self.depth.recent_timeframe, self.depth.recent_lookback)
            data[symbol] = (long_bars, recent_bars, self._maybe_fundamentals(symbol))

        recent_n = self.depth.recent_lookback
        long_n = self.depth.downsample_long_to
        block = self._render(syms, account, data, recent_n, long_n)
        while len(block) > self.max_chars and not (recent_n <= 3 and long_n <= 5):
            recent_n = max(3, recent_n // 2)
            long_n = max(5, long_n // 2)
            block = self._render(syms, account, data, recent_n, long_n)
        return block

    # --- internals ----------------------------------------------------------

    def _maybe_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        try:
            return self.fundamentals(symbol)
        except Exception:
            # Fundamentals are a nice-to-have; never let them break a decision.
            return None

    def _render(
        self,
        symbols: list[str],
        account: dict[str, Any],
        data: dict[str, tuple[list[Bar], list[Bar], dict[str, Any] | None]],
        recent_n: int,
        long_n: int,
    ) -> str:
        cash = float(account.get("cash", 0) or 0)
        lines = [
            f"Cash available: {cash:,.2f}",
            f"Positions: {account.get('positions', [])}",
            f"Tradable symbols: {', '.join(symbols)}",
        ]
        for symbol in symbols:
            long_bars, recent_bars, fund = data[symbol]
            lines.append("")
            lines.extend(self._render_symbol(symbol, long_bars, recent_bars, fund, recent_n, long_n))
        lines.extend(["", "Return your JSON decision now."])
        return "\n".join(lines)

    def _render_symbol(
        self,
        symbol: str,
        long_bars: list[Bar],
        recent_bars: list[Bar],
        fund: dict[str, Any] | None,
        recent_n: int,
        long_n: int,
    ) -> list[str]:
        out = [f"=== {symbol} ==="]
        if not long_bars and not recent_bars:
            out.append("  (no history available)")
            if fund:
                out.append("  Fundamentals: " + _fmt_fundamentals(fund))
            return out

        if long_bars:
            sampled = _downsample(long_bars, long_n)
            closes = [round(b.close, 2) for b in sampled]
            first, last = long_bars[0], long_bars[-1]
            ret = ((last.close - first.close) / first.close * 100.0) if first.close else 0.0
            hi = max(b.high for b in long_bars)
            lo = min(b.low for b in long_bars)
            out.append(
                f"  Long view ({self.depth.long_timeframe} ×{len(long_bars)}): "
                f"start {first.close:.2f} -> last {last.close:.2f} ({ret:+.1f}%), "
                f"range {lo:.2f}-{hi:.2f}"
            )
            out.append(f"  Downsampled closes: {closes}")

        if recent_bars:
            window = recent_bars[-recent_n:]
            out.append(
                f"  Recent ({self.depth.recent_timeframe} ×{len(window)}, time O/H/L/C/V):"
            )
            for b in window:
                out.append(
                    f"    {_short_ts(b.timestamp)}  "
                    f"{b.open:.2f} {b.high:.2f} {b.low:.2f} {b.close:.2f} {int(b.volume)}"
                )

        if fund:
            out.append("  Fundamentals: " + _fmt_fundamentals(fund))
        return out


# --- factory: resolve provider + depth from settings/env (never hardcoded) ---


def build_history_service(
    *,
    settings: SettingsStore | None = None,
    user_id: str | None = None,
    bar_provider: BarProvider | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
    transport: Any = None,
) -> HistoryService:
    """Build a :class:`HistoryService`, resolving provider and depth choices.

    Depth comes from the user's ``history_depth`` setting; the fundamentals
    provider from ``fundamentals_provider`` (``none`` | ``finnhub`` | ``polygon``).
    The default bar provider is Alpaca (same data key as the live books). Pass an
    explicit ``bar_provider`` / ``fundamentals_provider`` in tests to stay offline.
    """
    depth: Any = DEFAULT_DEPTH
    fund_name = "none"
    if settings is not None and user_id is not None:
        depth = settings.get(user_id, "history_depth", DEFAULT_DEPTH)
        fund_name = settings.get(user_id, "fundamentals_provider", "none")

    if bar_provider is None:
        from .providers.alpaca import AlpacaBarProvider

        bar_provider = AlpacaBarProvider()

    if fundamentals_provider is None and fund_name and fund_name.lower() != "none":
        fundamentals_provider = _resolve_fundamentals(fund_name, transport=transport)

    return HistoryService(
        bar_provider, fundamentals_provider=fundamentals_provider, depth=depth
    )


def _resolve_fundamentals(name: str, *, transport: Any = None) -> FundamentalsProvider | None:
    key = name.strip().lower()
    if key == "finnhub":
        from .providers.finnhub import FinnhubProvider

        return FinnhubProvider(transport=transport)
    if key == "polygon":
        from .providers.polygon import PolygonProvider

        return PolygonProvider(transport=transport)
    return None
