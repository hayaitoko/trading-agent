"""Bench competitors: a uniform ``Trader`` interface with two implementations.

* :class:`LLMTrader` — prompts an OpenRouter model with recent price context +
  the current portfolio and parses a structured BUY/SELL/HOLD decision.
* :class:`StrategyTrader` — wraps any deterministic :class:`Strategy` (e.g.
  mean-reversion) so it can compete in the same bench as a baseline.

Both expose ``observe(bar)`` (cheap, every bar) and ``decide(account)`` (called
by the bench at the chosen cadence). ``decide`` returns a :class:`DecisionResult`
the bench turns into broker orders via :func:`decision_to_signal`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .openrouter import OpenRouterError, parse_json_object

if TYPE_CHECKING:
    from ..data.history import HistoryService
    from ..strategy import Strategy
    from .openrouter import OpenRouterClient

_VALID_ACTIONS = {"BUY", "SELL", "HOLD"}


@dataclass
class TradeDecision:
    symbol: str
    action: str  # BUY | SELL | HOLD
    quantity: float
    reason: str = ""


@dataclass
class DecisionResult:
    decisions: list[TradeDecision] = field(default_factory=list)
    comment: str = ""
    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class Trader(Protocol):
    name: str

    def observe(self, bar: dict[str, Any]) -> None: ...
    def decide(self, account: dict[str, Any]) -> DecisionResult: ...


def decision_to_signal(d: TradeDecision) -> dict[str, Any] | None:
    """Map a decision to the canonical strategy-signal dict, or None for HOLD."""
    action = d.action.upper()
    if action == "HOLD" or d.quantity <= 0:
        return None
    side = "BUY" if action == "BUY" else "SELL"
    return {
        "asset": d.symbol,
        "side": side,
        "type": "market",
        "amount": float(d.quantity),
        "reason": d.reason,
    }


# --- LLM trader -------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an autonomous trading agent managing a paper account of US equities. "
    "Given recent price bars and your current portfolio, decide what to trade to "
    "maximize risk-adjusted return. You may only trade the listed symbols, in whole "
    "shares, and must not spend more than your available cash. Respond with ONLY a "
    "JSON object of this exact shape:\n"
    '{"decisions": [{"symbol": "AAPL", "action": "BUY|SELL|HOLD", '
    '"quantity": <int>, "reason": "<short>"}], "comment": "<one line>"}\n'
    "Use HOLD with quantity 0 when you want no change for a symbol. Be decisive but "
    "manage risk. Do not include any text outside the JSON object."
)


class LLMTrader:
    """A model that trades by reasoning over recent bars + its portfolio."""

    def __init__(
        self,
        model: str,
        client: OpenRouterClient,
        *,
        symbols: list[str],
        name: str | None = None,
        lookback: int = 30,
        temperature: float = 0.3,
        max_tokens: int = 800,
        history: HistoryService | None = None,
    ) -> None:
        self.model = model
        self.client = client
        self.symbols = list(symbols)
        self.name = name or model
        self.lookback = lookback
        self.temperature = temperature
        self.max_tokens = max_tokens
        # WS-A: when injected, the trader sees a richer historical + fundamentals
        # context block instead of just the last `lookback` closes. Optional so
        # the bench/back-compat path (no history) is unchanged.
        self.history = history
        self._bars: dict[str, deque[dict[str, Any]]] = {
            s: deque(maxlen=lookback) for s in self.symbols
        }

    def observe(self, bar: dict[str, Any]) -> None:
        symbol = bar.get("symbol")
        if symbol in self._bars:
            self._bars[symbol].append(bar)

    def decide(self, account: dict[str, Any]) -> DecisionResult:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._build_context(account)},
        ]
        try:
            res = self.client.chat(
                self.model,
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                json_mode=True,
            )
        except OpenRouterError as exc:
            return DecisionResult(error=str(exc))

        try:
            payload = parse_json_object(res.content)
        except ValueError as exc:
            return DecisionResult(raw=res.content, usage=res.usage, error=f"parse: {exc}")

        return DecisionResult(
            decisions=self._coerce_decisions(payload.get("decisions", [])),
            comment=str(payload.get("comment", ""))[:200],
            raw=res.content,
            usage=res.usage,
        )

    # --- internals ----------------------------------------------------------

    def _build_context(self, account: dict[str, Any]) -> str:
        # When a HistoryService is injected, delegate to its richer context block
        # (downsampled long view + dense recent OHLCV window + fundamentals).
        if self.history is not None:
            return self.history.context_block(self.symbols, account)
        lines = [
            f"Cash available: {account.get('cash', 0):,.2f}",
            f"Positions: {account.get('positions', [])}",
            f"Tradable symbols: {', '.join(self.symbols)}",
            "",
            "Recent bars (oldest first) — close prices:",
        ]
        for symbol, bars in self._bars.items():
            closes = [round(float(b["close"]), 2) for b in bars if b.get("close") is not None]
            lines.append(f"  {symbol}: {closes[-self.lookback:]}")
        lines.append("\nReturn your JSON decision now.")
        return "\n".join(lines)

    def _coerce_decisions(self, raw: Any) -> list[TradeDecision]:
        out: list[TradeDecision] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).strip()
            action = str(item.get("action", "HOLD")).strip().upper()
            if symbol not in self.symbols or action not in _VALID_ACTIONS:
                continue
            try:
                qty = max(0.0, float(item.get("quantity", 0) or 0))
            except (TypeError, ValueError):
                qty = 0.0
            out.append(
                TradeDecision(
                    symbol=symbol,
                    action=action,
                    quantity=qty,
                    reason=str(item.get("reason", ""))[:200],
                )
            )
        return out


# --- Strategy baseline trader ----------------------------------------------


class StrategyTrader:
    """Adapt a deterministic :class:`Strategy` into a bench competitor.

    The wrapped strategy emits a signal per bar (LONG/SHORT/NEUTRAL); we hold the
    most recent non-NEUTRAL signal and surface it on the next ``decide`` tick so
    every competitor trades on the same cadence.
    """

    def __init__(self, strategy: Strategy, *, name: str, default_quantity: float = 1.0) -> None:
        self.strategy = strategy
        self.name = name
        self.default_quantity = default_quantity
        self._pending: TradeDecision | None = None

    def observe(self, bar: dict[str, Any]) -> None:
        signal = self.strategy.on_data(bar)
        side = str(signal.get("side", "NEUTRAL")).upper()
        if side in ("LONG", "BUY"):
            action = "BUY"
        elif side in ("SHORT", "SELL"):
            action = "SELL"
        else:
            return
        self._pending = TradeDecision(
            symbol=str(signal.get("asset", bar.get("symbol", ""))),
            action=action,
            quantity=float(signal.get("amount", self.default_quantity) or self.default_quantity),
            reason=str(signal.get("reason", "strategy signal")),
        )

    def decide(self, account: dict[str, Any]) -> DecisionResult:
        if self._pending is None:
            return DecisionResult(comment="no signal")
        decision, self._pending = self._pending, None
        return DecisionResult(decisions=[decision], comment="strategy signal")
