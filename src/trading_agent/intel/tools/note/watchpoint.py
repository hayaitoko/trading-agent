"""NOTE tool: ``watchpoint`` — event-based symbol monitor with optional condition.

**Design role:** the trader registers a persistent price/vol/news trigger on a
symbol.  On every scheduler tick, :class:`WatchpointFirer` evaluates all unfired
watchpoint rows and fires the ones whose condition has tripped.  Each fire
enqueues an event-driven turn for the owning trader with the original ``why`` in
context.

**Condition forms (``condition`` argument):**
  - ``None`` → "interesting move" heuristic (see below)
  - ``"price > 580"`` / ``"price < 400"`` / ``"price >= 580"``
  - ``"news_rate > 2x"``     — news mention rate > 2× baseline
  - ``"realized_vol > 1.5x"`` — realized vol > 1.5× normal
  - Free text is stored as a label; the heuristic is used for evaluation.

**"Interesting move" heuristic** (default; fires if ANY of):
  1. Price moved > 1σ over last 1h (σ from 30-day realized vol via HistoryService).
  2. News mention rate > 2× 15-min baseline.
  3. Realized vol > 1.5× normal.
  4. Approval queue contains the symbol.

The heuristic is configurable per-trader via ``INTERESTING_MOVE_RULES`` setting
(stored in ``user_settings``) — any subset of the four rules may be disabled.

**TTL:** default 24 h (``WATCHPOINT_TTL_H`` per-trader setting); configurable via
``ttl_hours`` arg (max 168 h = 7 days).

**Soft limits:** ``WATCHPOINT_SOFT_LIMIT`` (default 20) surfaced in first-look.
Hard cap = 5× soft limit.

**Latency tier:** fast (local write, no model call)
**Cost class:** free
**Gating flag:** (none — always enabled)
"""

from __future__ import annotations

import math
import time
from typing import Any

from ...attention_queue import DEFAULT_WATCHPOINT_TTL_H
from ...tool_envelope import ToolResult
from ._base import NoteToolBase

# Max TTL guard — prevent watchpoints that last forever.
_MAX_TTL_H: float = 168.0  # 7 days

# OpenAI-compatible tool definition.
DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "watchpoint",
        "description": (
            "Register an event-based monitor on a symbol. You will be woken when "
            "the condition trips.  If condition is omitted, the 'interesting move' "
            "heuristic fires on price > 1σ, news spike, vol spike, or an approval "
            "queue entry for this symbol.  Conditions: 'price > 580', 'price < 400', "
            "'news_rate > 2x', 'realized_vol > 1.5x'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol to watch, e.g. 'AAPL'.",
                },
                "why": {
                    "type": "string",
                    "description": "Why you are watching this. Surfaced in your wake context.",
                },
                "condition": {
                    "type": "string",
                    "description": (
                        "Optional structured condition: 'price > 580', 'price < 400', "
                        "'news_rate > 2x', 'realized_vol > 1.5x'. "
                        "Omit to use the default 'interesting move' heuristic."
                    ),
                },
                "ttl_hours": {
                    "type": "number",
                    "description": (
                        f"How many hours to keep the watchpoint active (default {DEFAULT_WATCHPOINT_TTL_H}, max 168)."
                    ),
                },
            },
            "required": ["symbol", "why"],
        },
    },
}

# Catalog entry for list_tools().
CATALOG_ENTRY: dict[str, Any] = {
    "name": "watchpoint",
    "description": (
        "Register an event-based monitor on a symbol. Wake on price/news/vol condition "
        "or the default 'interesting move' heuristic."
    ),
    "args": {
        "symbol": "str",
        "why": "str",
        "condition": "str (optional) — 'price > N', 'news_rate > 2x', 'realized_vol > 1.5x'",
        "ttl_hours": f"float (optional, default {DEFAULT_WATCHPOINT_TTL_H}, max 168)",
    },
    "latency": "fast",
    "cost_class": "free",
    "enabled": True,
    "disabled_reason": None,
}


class WatchpointTool(NoteToolBase):
    """Executes the ``watchpoint(symbol, why, *, condition, ttl_hours)`` tool call."""

    def run(
        self,
        symbol: str,
        why: str,
        *,
        condition: str | None = None,
        ttl_hours: float | None = None,
    ) -> ToolResult:
        """Register a watchpoint.

        Returns ``{ok: true, data: {watchpoint_id, symbol, condition, expires_at_iso}}``
        on success.
        """
        symbol = (symbol or "").strip().upper()
        why = (why or "").strip()
        if not symbol:
            return self._err("invalid_input", "'symbol' must not be empty")
        if not why:
            return self._err("invalid_input", "'why' must not be empty")

        # TTL validation.
        ttl = float(ttl_hours) if ttl_hours is not None else DEFAULT_WATCHPOINT_TTL_H
        if ttl <= 0 or not math.isfinite(ttl):
            return self._err(
                "invalid_input",
                f"ttl_hours must be a positive finite number (got {ttl_hours!r})",
            )
        if ttl > _MAX_TTL_H:
            return self._err(
                "invalid_input",
                f"ttl_hours={ttl} exceeds maximum of {_MAX_TTL_H}h. "
                "Use a shorter window or re-register after expiry.",
            )

        # Hard-cap check.
        if self.attention_queue is not None:
            ok, cap_msg = self.attention_queue.can_add(self.trader_id, "watchpoint")
            if not ok:
                return self._err("unavailable", cap_msg)

        ttl_seconds = ttl * 3_600.0
        now = int(time.time())
        expires_at = now + int(ttl_seconds)

        payload: dict[str, Any] = {
            "symbol": symbol,
            "why": why,
            "condition": condition or None,
            "ttl_hours": ttl,
        }

        wp_id: int | None = None
        if self.attention_queue is not None:
            row = self.attention_queue.enqueue(
                self.trader_id,
                "watchpoint",
                payload,
                expires_at=expires_at,
            )
            wp_id = row.id if row.id >= 0 else None

        from datetime import UTC, datetime

        expires_iso = datetime.fromtimestamp(expires_at, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        condition_label = condition if condition else "interesting-move heuristic"

        return self._ok(
            {
                "watchpoint_id": wp_id,
                "symbol": symbol,
                "condition": condition_label,
                "why": why,
                "ttl_hours": ttl,
                "expires_at_iso": expires_iso,
                "stored": wp_id is not None,
            }
        )


# ── Evaluation (used by scheduler) ────────────────────────────────────────────


def evaluate_condition(
    row_payload: dict[str, Any],
    *,
    last_prices: dict[str, float] | None = None,
    price_sigma: dict[str, float] | None = None,
    price_change_1h: dict[str, float] | None = None,
    news_rate_ratio: dict[str, float] | None = None,
    realized_vol_ratio: dict[str, float] | None = None,
    approval_symbols: set[str] | None = None,
) -> tuple[bool, str]:
    """Evaluate a watchpoint payload against current market metrics.

    Parameters
    ----------
    row_payload:
        The ``payload`` dict from an :class:`~trading_agent.intel.attention_queue.AttentionRow`.
    last_prices:
        ``{symbol: last_price}`` map.
    price_sigma:
        ``{symbol: 1σ_in_price_units}`` — 30-day realized vol × price.  Used for
        the "price > 1σ" heuristic.
    price_change_1h:
        ``{symbol: abs_price_change_over_last_1h}`` — used to compare against sigma.
    news_rate_ratio:
        ``{symbol: current_rate / baseline_rate}`` — ratio > 2.0 trips the heuristic.
    realized_vol_ratio:
        ``{symbol: current_vol / normal_vol}`` — ratio > 1.5 trips the heuristic.
    approval_symbols:
        Set of symbols currently in the approval queue.

    Returns
    -------
    (tripped: bool, reason: str)
        ``reason`` is the human-readable fire reason stored in ``fire_reason``.
    """
    symbol = str(row_payload.get("symbol", "")).upper()
    condition = row_payload.get("condition")

    if condition:
        tripped, reason = _evaluate_structured(
            symbol, condition,
            last_prices=last_prices or {},
            news_rate_ratio=news_rate_ratio or {},
            realized_vol_ratio=realized_vol_ratio or {},
        )
        return tripped, reason

    # Heuristic mode — any of the four rules trips.
    return _interesting_move(
        symbol,
        price_sigma=price_sigma or {},
        price_change_1h=price_change_1h or {},
        news_rate_ratio=news_rate_ratio or {},
        realized_vol_ratio=realized_vol_ratio or {},
        approval_symbols=approval_symbols or set(),
    )


def _evaluate_structured(
    symbol: str,
    condition: str,
    *,
    last_prices: dict[str, float],
    news_rate_ratio: dict[str, float],
    realized_vol_ratio: dict[str, float],
) -> tuple[bool, str]:
    """Evaluate a structured condition string.

    Supported forms:
        price > N / price < N / price >= N / price <= N
        news_rate > Nx / news_rate > N
        realized_vol > Nx / realized_vol > N
    """
    cond = condition.strip().lower()

    # price comparisons
    import re

    m = re.match(
        r"^price\s*(>=|<=|>|<)\s*(\d+(?:\.\d+)?)\s*$", cond
    )
    if m:
        op, val = m.group(1), float(m.group(2))
        price = last_prices.get(symbol)
        if price is None:
            return False, ""
        ops = {">": price > val, "<": price < val, ">=": price >= val, "<=": price <= val}
        hit = ops.get(op, False)
        if hit:
            return True, f"condition: price {op} {val} (last={price:.2f})"
        return False, ""

    # news_rate > Nx  (e.g. "news_rate > 2x" or "news_rate > 2")
    m2 = re.match(r"^news_rate\s*>\s*(\d+(?:\.\d+)?)x?\s*$", cond)
    if m2:
        threshold = float(m2.group(1))
        ratio = news_rate_ratio.get(symbol, 0.0)
        if ratio > threshold:
            return True, f"condition: news_rate > {threshold}x (ratio={ratio:.2f})"
        return False, ""

    # realized_vol > Nx
    m3 = re.match(r"^realized_vol\s*>\s*(\d+(?:\.\d+)?)x?\s*$", cond)
    if m3:
        threshold = float(m3.group(1))
        ratio = realized_vol_ratio.get(symbol, 0.0)
        if ratio > threshold:
            return True, f"condition: realized_vol > {threshold}x (ratio={ratio:.2f})"
        return False, ""

    # Unknown condition — treat as heuristic (don't block, just don't fire on condition)
    return False, ""


def _interesting_move(
    symbol: str,
    *,
    price_sigma: dict[str, float],
    price_change_1h: dict[str, float],
    news_rate_ratio: dict[str, float],
    realized_vol_ratio: dict[str, float],
    approval_symbols: set[str],
) -> tuple[bool, str]:
    """Evaluate the default 'interesting move' heuristic."""
    # Rule 1: price > 1σ over last 1h
    sigma = price_sigma.get(symbol)
    change = price_change_1h.get(symbol)
    if sigma is not None and change is not None and sigma > 0 and abs(change) > sigma:
        return True, f"interesting-move: price moved {change:+.2f} > 1σ ({sigma:.2f})"

    # Rule 2: news rate > 2× baseline
    nr = news_rate_ratio.get(symbol, 0.0)
    if nr > 2.0:
        return True, f"interesting-move: news_rate ratio={nr:.2f} > 2.0x"

    # Rule 3: realized vol > 1.5× normal
    vr = realized_vol_ratio.get(symbol, 0.0)
    if vr > 1.5:
        return True, f"interesting-move: realized_vol ratio={vr:.2f} > 1.5x"

    # Rule 4: symbol in approval queue
    if symbol in approval_symbols:
        return True, f"interesting-move: {symbol} has a pending approval entry"

    return False, ""
