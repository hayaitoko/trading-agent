"""Single-leg equity options: contract identity, quotes/marks, pricing, and a
×100-multiplier position + P&L book with expiry/assignment settlement.

Scope (single-leg first; multi-leg spreads are a follow-up):

* :class:`OptionContract` — the (underlying, expiry, strike, right) identity, with
  OCC-symbol round-tripping and intrinsic value.
* :class:`OptionQuote` + :func:`mark_price` — the mark mechanism (mid of bid/ask,
  falling back to last).
* :func:`black_scholes_price` — a theoretical pricer (no market data needed),
  useful for valuation and tests.
* :class:`OptionsBook` — cash + positions in *contracts*, every dollar figure
  scaled by ``multiplier`` (100 shares per contract), realized-P&L ledger, and
  expiry settlement (long → intrinsic; short ITM → assignment).

Margin for *written* (short) options is out of scope here — writing is gated
behind ``allow_short`` and assumed covered/cash-secured; naked-option margin and
multi-leg spreads come later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

from ..enums import OrderSide

# 21-char OCC: 6-char root (space-padded right) + YYMMDD + C/P + strike×1000 (8).
_OCC_TAIL = 15  # YYMMDD(6) + right(1) + strike(8)


class OptionRight(Enum):
    CALL = "C"
    PUT = "P"


@dataclass(frozen=True)
class OptionContract:
    """A single option's identity. ``strike`` is in dollars; ``expiry`` a date."""

    underlying: str
    expiry: date
    strike: float
    right: OptionRight

    @property
    def occ_symbol(self) -> str:
        """The OCC option symbol, e.g. ``AAPL  240119C00150000``."""
        root = f"{self.underlying.upper():<6}"
        ymd = self.expiry.strftime("%y%m%d")
        strike_mils = int(round(self.strike * 1000))
        return f"{root}{ymd}{self.right.value}{strike_mils:08d}"

    @classmethod
    def from_occ(cls, symbol: str) -> OptionContract:
        """Parse an OCC symbol (padded or compact) back into a contract."""
        if len(symbol) <= _OCC_TAIL:
            raise ValueError(f"not an OCC option symbol: {symbol!r}")
        tail = symbol[-_OCC_TAIL:]
        root = symbol[:-_OCC_TAIL].strip()
        if not root:
            raise ValueError(f"missing underlying in OCC symbol: {symbol!r}")
        try:
            expiry = date(2000 + int(tail[0:2]), int(tail[2:4]), int(tail[4:6]))
            right = OptionRight(tail[6])
            strike = int(tail[7:15]) / 1000.0
        except (ValueError, KeyError) as exc:
            raise ValueError(f"malformed OCC symbol: {symbol!r}") from exc
        return cls(underlying=root, expiry=expiry, strike=strike, right=right)

    def intrinsic(self, underlying_price: float) -> float:
        """Intrinsic value per share at ``underlying_price`` (never negative)."""
        if self.right is OptionRight.CALL:
            return max(0.0, underlying_price - self.strike)
        return max(0.0, self.strike - underlying_price)

    def is_expired(self, on: date) -> bool:
        """True once ``on`` reaches the expiry date (expiry settles at the close)."""
        return on >= self.expiry


@dataclass
class OptionQuote:
    """A market quote for one contract. Any field may be ``None``."""

    contract: OptionContract
    bid: float | None = None
    ask: float | None = None
    last: float | None = None

    @property
    def mark(self) -> float | None:
        """Mark price: mid of bid/ask, else last, else whichever side exists."""
        return mark_price(self.bid, self.ask, self.last)


def mark_price(
    bid: float | None, ask: float | None, last: float | None
) -> float | None:
    """The conventional option mark: mid of a two-sided quote, else last, else a
    single-sided quote. ``None`` when nothing is quoted."""
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if last is not None:
        return last
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return None


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    right: OptionRight,
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    vol: float,
) -> float:
    """Black–Scholes price of a European option (per share).

    Degenerate inputs fall back to discounted intrinsic: at/under zero time or
    zero vol the option is worth its intrinsic value (discounting the strike at
    ``rate``). Used as a theoretical mark when no live quote is available.
    """
    if spot <= 0 or strike <= 0:
        return 0.0
    if t_years <= 0 or vol <= 0:
        disc_k = strike * math.exp(-rate * max(0.0, t_years))
        if right is OptionRight.CALL:
            return max(0.0, spot - disc_k)
        return max(0.0, disc_k - spot)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * math.sqrt(t_years))
    d2 = d1 - vol * math.sqrt(t_years)
    disc_k = strike * math.exp(-rate * t_years)
    if right is OptionRight.CALL:
        return spot * _norm_cdf(d1) - disc_k * _norm_cdf(d2)
    return disc_k * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


@dataclass
class OptionPosition:
    """An open option position. ``quantity`` is in contracts (negative = written);
    ``avg_price`` is the per-share premium basis (not yet ×multiplier)."""

    contract: OptionContract
    quantity: float
    avg_price: float


class OptionsBook:
    """Cash + option positions with ×``multiplier`` accounting and expiry settlement.

    Long-only by default; set ``allow_short`` to write (sell-to-open) contracts.
    Cash moves by ``quantity × price × multiplier`` on every fill. Realized P&L is
    booked when a fill reduces existing exposure, gross of any commission.
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        *,
        multiplier: int = 100,
        allow_short: bool = False,
    ) -> None:
        self._initial_balance = initial_balance
        self._cash = initial_balance
        self.multiplier = multiplier
        self.allow_short = allow_short
        self._positions: dict[str, OptionPosition] = {}
        self._realized_pnl = 0.0
        self._closed_pnls: list[float] = []

    # --- accessors ----------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    def position(self, contract: OptionContract) -> OptionPosition | None:
        pos = self._positions.get(contract.occ_symbol)
        if pos is None:
            return None
        return OptionPosition(pos.contract, pos.quantity, pos.avg_price)

    def positions(self) -> list[OptionPosition]:
        return [OptionPosition(p.contract, p.quantity, p.avg_price) for p in self._positions.values()]

    def get_realized_pnl(self) -> float:
        return self._realized_pnl

    def get_win_loss(self) -> tuple[int, int]:
        wins = sum(1 for p in self._closed_pnls if p > 0)
        losses = sum(1 for p in self._closed_pnls if p < 0)
        return wins, losses

    def account_value(self, marks: dict[str, float]) -> float:
        """Cash + mark-to-market of open positions. ``marks`` is keyed by OCC symbol
        (per-share price); positions without a mark contribute only via cash."""
        total = self._cash
        for occ, pos in self._positions.items():
            if occ in marks:
                total += pos.quantity * marks[occ] * self.multiplier
        return total

    # --- trading ------------------------------------------------------------

    def buy(self, contract: OptionContract, quantity: float, price: float) -> dict[str, Any]:
        """Buy ``quantity`` contracts at ``price`` per share. Rejects if unaffordable."""
        cost = quantity * price * self.multiplier
        # Buying that opens/extends a long must be affordable; covering a short
        # spends cash too but the premium received at open already sits in cash.
        if cost > self._cash + 1e-9:
            return self._result(contract, OrderSide.BUY, quantity, price, "REJECTED")
        self._fill(contract, OrderSide.BUY, quantity, price)
        return self._result(contract, OrderSide.BUY, quantity, price, "FILLED")

    def sell(self, contract: OptionContract, quantity: float, price: float) -> dict[str, Any]:
        """Sell ``quantity`` contracts at ``price``. Selling beyond an existing long
        writes a short, which requires ``allow_short``."""
        existing = self._positions.get(contract.occ_symbol)
        long_qty = existing.quantity if existing is not None and existing.quantity > 0 else 0.0
        if quantity > long_qty and not self.allow_short:
            return self._result(contract, OrderSide.SELL, quantity, price, "REJECTED")
        self._fill(contract, OrderSide.SELL, quantity, price)
        return self._result(contract, OrderSide.SELL, quantity, price, "FILLED")

    # --- expiry / assignment ------------------------------------------------

    def settle_expiry(self, contract: OptionContract, underlying_price: float) -> float:
        """Settle one contract at expiry against ``underlying_price``.

        A long is closed at its intrinsic value (OTM → worthless, lose premium); a
        short is assigned at intrinsic (OTM → keep full premium, ITM → pay
        intrinsic). Returns the per-share intrinsic value settled at. No-op (0.0)
        if the position isn't held.
        """
        pos = self._positions.get(contract.occ_symbol)
        if pos is None:
            return 0.0
        intrinsic = contract.intrinsic(underlying_price)
        if pos.quantity > 0:
            self._fill(contract, OrderSide.SELL, pos.quantity, intrinsic)
        else:
            self._fill(contract, OrderSide.BUY, -pos.quantity, intrinsic)
        return intrinsic

    def expire(self, on_date: date, underlying_prices: dict[str, float]) -> list[str]:
        """Settle every held contract whose expiry has arrived by ``on_date``.

        ``underlying_prices`` maps underlying symbol → spot. A contract whose
        underlying has no price is left untouched. Returns the OCC symbols settled.
        """
        settled: list[str] = []
        for occ, pos in list(self._positions.items()):
            if not pos.contract.is_expired(on_date):
                continue
            spot = underlying_prices.get(pos.contract.underlying)
            if spot is None:
                continue
            self.settle_expiry(pos.contract, spot)
            settled.append(occ)
        return settled

    # --- internals ----------------------------------------------------------

    def _fill(self, contract: OptionContract, side: OrderSide, quantity: float, price: float) -> None:
        occ = contract.occ_symbol
        existing = self._positions.get(occ)
        mult = self.multiplier

        # Realized P&L on the portion that reduces existing exposure (captured
        # before the position mutates): selling a long, or buying back a short.
        closed_qty = 0.0
        realized = 0.0
        if existing is not None:
            if side is OrderSide.SELL and existing.quantity > 0:
                closed_qty = min(quantity, existing.quantity)
                realized = closed_qty * (price - existing.avg_price) * mult
            elif side is OrderSide.BUY and existing.quantity < 0:
                closed_qty = min(quantity, -existing.quantity)
                realized = closed_qty * (existing.avg_price - price) * mult

        notional = quantity * price * mult
        self._cash += -notional if side is OrderSide.BUY else notional

        old = existing.quantity if existing is not None else 0.0
        new = old + (quantity if side is OrderSide.BUY else -quantity)

        if abs(new) < 1e-12:
            self._positions.pop(occ, None)
        elif existing is None:
            self._positions[occ] = OptionPosition(contract, new, price)
        else:
            same_direction = (old > 0 and new > 0) or (old < 0 and new < 0)
            if same_direction and abs(new) > abs(old):
                # adding to the position → weighted-average the new lot in
                existing.avg_price = (abs(old) * existing.avg_price + quantity * price) / abs(new)
            elif not same_direction:
                existing.avg_price = price  # crossed zero (flip): basis resets
            existing.quantity = new

        if closed_qty > 0:
            self._realized_pnl += realized
            self._closed_pnls.append(realized)

    def _result(
        self, contract: OptionContract, side: OrderSide, quantity: float, price: float, status: str
    ) -> dict:
        return {
            "symbol": contract.occ_symbol,
            "side": side.value,
            "quantity": quantity,
            "price": price,
            "status": status,
        }
