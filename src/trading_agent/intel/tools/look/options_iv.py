"""``options_iv`` — implied volatility and Greeks for a symbol (gated by SITUATION_OPTIONS_IV).

Tool name:      options_iv
Args:           symbol (str), expiry=None (str|None)
ToolResult:     ok=True,  data={"symbol": str, "contracts": [...]}
                ok=False, error=ToolError(kind="disabled", …) when flag off
                ok=False, error=ToolError(kind="network_error", …) on provider failure
Latency tier:   fast (Alpaca snapshot is near-real-time; indicative on paper feed)
Cost class:     free (uses existing ALPACA_API_KEY)
Gating flag:    SITUATION_OPTIONS_IV (user_settings, default False)

Purpose
-------
Wraps :class:`~trading_agent.instruments.options_chain.AlpacaOptionChainProvider`
to surface per-contract implied volatility and Greeks as a LOOK tool.  Uses
the WS-Situation A2 additive fields on :class:`~trading_agent.instruments.options.OptionQuote`
(``implied_vol`` and ``greeks``) that Alpaca already computes server-side.

The trader calls this tool when:
  - Evaluating the options market's forward vol estimate before a trade.
  - Checking gamma exposure near a key strike level.
  - Comparing IV vs realised vol to form a vol-regime opinion.

Returns the near-the-money contracts (within ±20% of spot) for the specified
expiry (or all contracts if no expiry given).  When no contracts have IV data,
all contracts are returned (no silent drop).

Money-is-real note: this tool never discloses paper/live account status.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase

_DEFAULT_MONEYNESS_BAND = 0.20  # ±20% from spot


class OptionsIVTool(LookToolBase):
    """Near-the-money options implied volatility + Greeks for a symbol.

    Parameters
    ----------
    owner_user_id, trader_id:
        Namespace identifiers.
    settings_store:
        Duck-typed: ``get(user_id, key, default)`` — checks
        ``SITUATION_OPTIONS_IV`` flag.
    chain_provider:
        Pre-constructed :class:`~trading_agent.instruments.options_chain.AlpacaOptionChainProvider`.
        ``None`` → returns disabled error even when flag is on.
    spot_prices:
        Dict mapping symbol → current spot price.  Used to filter
        near-the-money contracts.  ``None`` → include all contracts.
    """

    TOOL_META: dict[str, Any] = {
        "name": "options_iv",
        "description": (
            "Implied volatility and Greeks for near-the-money options on a symbol. "
            "Returns per-contract IV (annualised), delta, gamma, theta, vega, rho. "
            "Use to gauge the options market's forward vol expectation or check "
            "gamma exposure near a key strike level. "
            "Enable via SITUATION_OPTIONS_IV in trader settings. "
            "Requires ALPACA_API_KEY."
        ),
        "args": {
            "symbol": "str (equity ticker, e.g. 'AAPL', 'SPY')",
            "expiry": "str|None (default None — nearest expiry; ISO 'YYYY-MM-DD')",
        },
        "latency": "fast",
        "cost_class": "free",
        "enabled": True,  # wired — returns disabled error when flag off
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        settings_store: Any = None,
        chain_provider: Any = None,
        spot_prices: dict[str, float] | None = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._settings = settings_store
        self._chain = chain_provider
        self._spots: dict[str, float] = dict(spot_prices or {})

    def __call__(
        self,
        symbol: str,
        expiry: str | None = None,
    ) -> Any:
        """Fetch near-the-money IV and Greeks for ``symbol``.

        Parameters
        ----------
        symbol:
            The underlying equity ticker.
        expiry:
            ISO date string ``YYYY-MM-DD``.  ``None`` → nearest listed expiry.

        Returns
        -------
        ToolResult
            ok=True, data={"symbol": str, "expiry_filter": str|None,
              "contracts": [
                {"occ": str, "underlying": str, "strike": float, "right": "C"|"P",
                 "expiry": str, "bid": float|None, "ask": float|None,
                 "mark": float|None, "implied_vol": float|None,
                 "greeks": {"delta":..., "gamma":..., "theta":..., "vega":..., "rho":...}|None},
              ...]}

            ok=False, error=ToolError(kind="disabled")
                When SITUATION_OPTIONS_IV flag is off.

            ok=False, error=ToolError(kind="network_error", message=...)
                On provider network/HTTP failure.

        Example
        -------
        >>> tool = OptionsIVTool(trader_id="Alpha")
        >>> result = tool("AAPL")
        >>> result.ok
        False
        >>> result.error.kind
        'disabled'
        """
        if not self._flag_enabled():
            return self._err(
                "disabled",
                "options_iv: enable SITUATION_OPTIONS_IV in trader settings to use this tool.",
            )

        if self._chain is None:
            return self._err(
                "disabled",
                "options_iv: options chain provider not initialised.",
            )

        symbol = symbol.upper()
        try:
            quotes = self._chain.get_chain(symbol, expiry)
        except Exception as exc:  # noqa: BLE001
            return self._err("network_error", f"options chain error for {symbol}: {exc}")

        # Filter near-the-money if we have spot
        spot = self._spots.get(symbol)
        if spot is not None and spot > 0:
            lo = spot * (1 - _DEFAULT_MONEYNESS_BAND)
            hi = spot * (1 + _DEFAULT_MONEYNESS_BAND)
            quotes = [q for q in quotes if lo <= q.contract.strike <= hi]

        # Prefer contracts that have IV; fall back to all when none have it
        quotes_with_iv = [q for q in quotes if q.implied_vol is not None]
        display = quotes_with_iv if quotes_with_iv else quotes

        return self._ok(
            {
                "symbol": symbol,
                "expiry_filter": expiry,
                "contracts": [_quote_to_dict(q) for q in display],
            }
        )

    # ------------------------------------------------------------------

    def _flag_enabled(self) -> bool:
        if self._settings is None:
            return False
        try:
            return bool(
                self._settings.get(
                    self.owner_user_id or "", "SITUATION_OPTIONS_IV", False
                )
            )
        except Exception:  # noqa: BLE001
            return False


def _quote_to_dict(q: Any) -> dict[str, Any]:
    return {
        "occ": q.contract.occ_symbol,
        "underlying": q.contract.underlying,
        "strike": q.contract.strike,
        "right": q.contract.right.value,
        "expiry": q.contract.expiry.isoformat(),
        "bid": q.bid,
        "ask": q.ask,
        "mark": q.mark,
        "implied_vol": q.implied_vol,
        "greeks": dict(q.greeks) if q.greeks is not None else None,
    }
