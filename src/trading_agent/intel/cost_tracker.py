"""Per-turn rolling cost tracker for the agent decision loop.

:class:`CostTracker` accumulates model-call costs and nested LLM costs (e.g.
``ask_manager``, ``request_research``) across all calls in one agent turn.
When the running total exceeds the soft-warning threshold, :meth:`check_warn`
returns a message string that the loop injects as a system message before the
next model call.

This is a **soft warning only** — it never caps or interrupts the loop.
The hard cap in the loop is the runaway-guard call count (100 calls per turn),
not cost.

Configuration:
    ``COST_WARN_PER_TURN`` env var (default ``"1.00"``): threshold in USD.
    Deliberately a soft nudge so one expensive ``ask_manager`` call doesn't
    silence a turn that genuinely needs more work.

Failure mode: malformed env var falls back to the $1.00 default silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_DEFAULT_WARN_USD = 1.00


def _warn_threshold() -> float:
    try:
        return float(os.environ.get("COST_WARN_PER_TURN", _DEFAULT_WARN_USD))
    except (ValueError, TypeError):
        return _DEFAULT_WARN_USD


@dataclass
class CostTracker:
    """Accumulates per-turn spending across model calls and nested LLM tools.

    Usage::

        tracker = CostTracker()
        tracker.add_model_call(cost_usd=0.002, input_tokens=800, output_tokens=200)
        tracker.add_nested_llm("ask_manager", cost_usd=0.010)
        warn = tracker.check_warn()   # returns warning string or None
        summary = tracker.rollup()    # {"total_usd": ..., "model_calls": ..., ...}
    """

    _total_usd: float = field(default=0.0, init=False)
    _model_calls: list[dict[str, object]] = field(default_factory=list, init=False)
    _nested_calls: list[dict[str, object]] = field(default_factory=list, init=False)
    _warned: bool = field(default=False, init=False)

    def add_model_call(
        self,
        cost_usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        """Record cost for one main-loop LLM call."""
        self._total_usd += max(0.0, cost_usd)
        self._model_calls.append(
            {
                "cost_usd": cost_usd,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
            }
        )

    def add_nested_llm(self, tool_name: str, cost_usd: float) -> None:
        """Record cost for a nested LLM call made inside a tool (e.g. ask_manager)."""
        self._total_usd += max(0.0, cost_usd)
        self._nested_calls.append({"tool": tool_name, "cost_usd": cost_usd})

    @property
    def total_usd(self) -> float:
        return self._total_usd

    @property
    def call_count(self) -> int:
        """Number of main-loop model calls this turn."""
        return len(self._model_calls)

    def check_warn(self) -> str | None:
        """Return a soft-warning string if the spend threshold is exceeded (once).

        The loop injects the returned string as a system message before the
        next model call.  Returns ``None`` if the threshold hasn't been crossed
        or the warning has already been issued this turn.
        """
        if self._warned:
            return None
        threshold = _warn_threshold()
        if self._total_usd > threshold:
            self._warned = True
            return (
                f"[Cost notice] This turn has spent ${self._total_usd:.4f} so far "
                f"(soft threshold: ${threshold:.2f}). Consider wrapping up — "
                "call pass() or hold() unless a decision genuinely requires more tool calls."
            )
        return None

    def rollup(self) -> dict[str, object]:
        """Return a summary dict suitable for the turn record."""
        return {
            "total_usd": round(self._total_usd, 6),
            "model_calls": len(self._model_calls),
            "nested_llm_calls": len(self._nested_calls),
        }
