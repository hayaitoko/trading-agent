"""In-memory paper broker for backtesting and dry-run.

Implements the canonical BrokerAdapter interface. Broker-internal order /
position records are named PaperOrder / PaperPosition to avoid shadowing the
models.Order / models.Position dataclasses (which carry different fields).

Realism knobs (all default to off, so behaviour is unchanged unless enabled):

* **Quotes** — feed real bid/ask via :meth:`update_quote` and market orders
  fill at the ask (buy) / bid (sell) instead of a single mid price.
* **slippage_bps** — adverse slippage applied to *market* fills.
* **commission_bps** — per-trade commission deducted from cash.
* **is_market_open** — optional ``Callable[[str], bool]``; when it returns
  False, market orders are rejected and limit orders stay queued.
* **allow_short** — when False (default) the book is long-only: a SELL is only
  fillable up to the quantity you already hold. When True, a SELL may drive the
  position negative (open/extend a short) subject to a margin check: the total
  short market value across the book, multiplied by ``short_margin_ratio``, must
  not exceed account equity, and may be hard-capped by ``max_short_notional``.
  Covering (a BUY against a short) reduces it and books realized P&L.

Limit orders now match against the prevailing quote: a buy limit fills when the
market trades at or below the limit, a sell limit when at or above it. Matching
runs on placement and on every :meth:`update_market_prices` / :meth:`update_quote`.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .broker_adapter import BrokerAdapter
from .enums import OrderSide, OrderType


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class PaperOrder:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float | None = None
    # Protective-order fields (P0). stop_price is the current trigger for STOP
    # and TRAILING_STOP orders; trail_offset is the dollar gap maintained between
    # the running high/low and the trigger for trailing stops.
    stop_price: float | None = None
    trail_offset: float | None = None


@dataclass
class PaperPosition:
    symbol: str
    quantity: float
    avg_price: float


class PaperBroker(BrokerAdapter):
    """In-memory paper broker."""

    def __init__(
        self,
        initial_balance: float = 100000.0,
        *,
        slippage_bps: float = 0.0,
        commission_bps: float = 0.0,
        is_market_open: Callable[[str], bool] | None = None,
        allow_short: bool = False,
        short_margin_ratio: float = 1.5,
        max_short_notional: float | None = None,
    ):
        self._initial_balance: float = initial_balance
        self._balance: float = initial_balance
        self._positions: dict[str, PaperPosition] = {}
        self._orders: dict[str, PaperOrder] = {}
        self._trade_history: list[tuple[str, float, float, str]] = []
        # Realized-P&L ledger: total realized + the per-close P&L of every trade
        # that reduced or closed a position (used for win/loss counts). Gross of
        # commission; commission is a separate cash drag, not a position outcome.
        self._realized_pnl: float = 0.0
        self._closed_pnls: list[float] = []
        # Last-price view, kept for valuation + get_quote backward compatibility.
        self.market_prices: dict[str, float] = {}
        # Richer per-symbol quote: {'bid': float|None, 'ask': float|None, 'last': float|None}.
        self._quotes: dict[str, dict[str, float | None]] = {}
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.is_market_open = is_market_open
        # Short-selling config (see class docstring). Off by default → long-only.
        self.allow_short = allow_short
        self.short_margin_ratio = short_margin_ratio
        self.max_short_notional = max_short_notional
        self._connected: bool = False
        # Protective-order state (P0). Keyed by order_id.
        # trail_peak tracks the running high (long) or low (short) for each
        # trailing-stop order so we can ratchet the trigger on every price update.
        self._trail_peak: dict[str, float] = {}
        # hard_floor_pct: if not None and equity / initial_balance - 1 < -pct,
        # flatten_all() is auto-called with no model in the loop.
        self.hard_floor_pct: float | None = None

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def balance(self) -> float:
        return self._balance

    def get_balance(self) -> dict[str, Any]:
        return {"cash": self._balance}

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get current quote for symbol.

        Returns a dict with a ``price`` key (last/mid). Includes ``bid``/``ask``
        when a richer quote has been supplied via :meth:`update_quote`.

        Raises:
            ValueError: If no price is known for the symbol.
        """
        price = self.market_prices.get(symbol)
        if price is None:
            raise ValueError(f"Symbol {symbol} not found in market_prices")
        q = self._quotes.get(symbol, {})
        out: dict[str, Any] = {"price": price}
        if q.get("bid") is not None:
            out["bid"] = q["bid"]
        if q.get("ask") is not None:
            out["ask"] = q["ask"]
        return out

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        if not self._connected:
            raise RuntimeError("Broker is not connected")
        order = self._orders.get(order_id)
        if not order:
            return {}
        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "filled_price": order.filled_price,
        }

    @property
    def positions(self) -> dict[str, PaperPosition]:
        return {k: PaperPosition(v.symbol, v.quantity, v.avg_price) for k, v in self._positions.items()}

    @property
    def orders(self) -> dict[str, PaperOrder]:
        return {
            order_id: PaperOrder(
                id=order.id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                status=order.status,
                filled_quantity=order.filled_quantity,
                filled_price=order.filled_price,
            )
            for order_id, order in self._orders.items()
        }

    def place_order(self, order_details: dict[str, Any]) -> dict[str, Any] | None:
        if not self._connected:
            raise RuntimeError("Broker is not connected")

        try:
            symbol = order_details["symbol"]
            quantity_raw = order_details.get("amount", order_details.get("quantity"))
            if quantity_raw is None:
                return None
            quantity = float(quantity_raw)

            # Handle OrderSide conversion (enum values are uppercase: BUY/SELL)
            side_input = order_details["side"]
            if isinstance(side_input, str):
                side = OrderSide(side_input.upper())
            elif isinstance(side_input, OrderSide):
                side = side_input
            else:
                return None

            # Handle OrderType conversion (enum values are lowercase: market/limit/...)
            order_type_input = order_details["order_type"]
            if isinstance(order_type_input, str):
                order_type = OrderType(order_type_input.lower())
            elif isinstance(order_type_input, OrderType):
                order_type = order_type_input
            else:
                return None

            price = order_details.get("price")
        except (KeyError, ValueError, TypeError):
            return None

        order_id = str(uuid.uuid4())
        order_price = None if order_type == OrderType.MARKET else price

        # --- Protective-order fields (P0) ------------------------------------
        stop_price: float | None = None
        trail_offset: float | None = None
        if order_type in (OrderType.STOP, OrderType.TRAILING_STOP, OrderType.TAKE_PROFIT):
            raw_sp = order_details.get("stop_price")
            raw_ta = order_details.get("trail_amount")
            if raw_sp is not None:
                stop_price = float(raw_sp)
            elif order_type == OrderType.TAKE_PROFIT:
                # take_profit uses 'price' as the trigger
                stop_price = float(price) if price is not None else None
            if raw_ta is not None:
                trail_offset = float(raw_ta)
                # Compute initial stop_price from trail_amount if not explicitly set.
                if stop_price is None:
                    mkt = self.market_prices.get(symbol)
                    if mkt is not None:
                        if side == OrderSide.SELL:
                            stop_price = mkt - trail_offset
                        else:
                            stop_price = mkt + trail_offset
            elif order_type == OrderType.TRAILING_STOP and stop_price is not None:
                # Infer trail_offset from current market vs supplied stop_price.
                mkt = self.market_prices.get(symbol)
                if mkt is not None:
                    trail_offset = abs(mkt - stop_price)

        order = PaperOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=order_price,
            stop_price=stop_price,
            trail_offset=trail_offset,
        )
        self._orders[order_id] = order

        if order_type == OrderType.MARKET:
            self._fill_market_order(order)
        elif order_type in (OrderType.STOP, OrderType.TRAILING_STOP, OrderType.TAKE_PROFIT):
            # Protective orders rest until a price update fires them; initialize
            # the trailing peak to the current market price so the first update
            # can ratchet immediately if the price has already moved.
            mkt = self.market_prices.get(symbol)
            if mkt is not None and order_type == OrderType.TRAILING_STOP:
                self._trail_peak[order_id] = mkt
            # Evaluate immediately in case the market is already past the trigger.
            self._evaluate_protective_order(order)
        else:
            # Limit (and other resting types): try to fill immediately if the
            # current quote already crosses; otherwise leave it PENDING.
            self._try_fill_limit_order(order)

        return self._order_result(order)

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected:
            raise RuntimeError("Broker is not connected")

        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def get_order(self, order_id: str) -> PaperOrder | None:
        return self._orders.get(order_id)

    # --- Market data --------------------------------------------------------

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """Set the last price for one or more symbols, then match resting orders."""
        self.market_prices.update(prices)
        for symbol, px in prices.items():
            q = self._quotes.setdefault(symbol, {"bid": None, "ask": None, "last": None})
            q["last"] = px
        self._match_pending_limit_orders(list(prices.keys()))
        self._evaluate_protective_orders(list(prices.keys()))
        self._check_hard_floor()

    def update_quote(
        self,
        symbol: str,
        *,
        bid: float | None = None,
        ask: float | None = None,
        last: float | None = None,
    ) -> None:
        """Supply a richer bid/ask/last quote (e.g. from a live feed).

        Market orders fill at ask (buy) / bid (sell) when those are present.
        ``market_prices`` is kept in sync (last, else mid) for valuation.
        """
        q = self._quotes.setdefault(symbol, {"bid": None, "ask": None, "last": None})
        if bid is not None:
            q["bid"] = bid
        if ask is not None:
            q["ask"] = ask
        if last is not None:
            q["last"] = last

        reference = q["last"]
        if reference is None and q["bid"] is not None and q["ask"] is not None:
            reference = (q["bid"] + q["ask"]) / 2.0
        if reference is not None:
            self.market_prices[symbol] = reference

        self._match_pending_limit_orders([symbol])
        self._evaluate_protective_orders([symbol])
        self._check_hard_floor()

    def update_quotes(self, quotes: dict[str, dict[str, float | None]]) -> None:
        """Bulk variant of :meth:`update_quote`."""
        for symbol, q in quotes.items():
            self.update_quote(
                symbol, bid=q.get("bid"), ask=q.get("ask"), last=q.get("last")
            )

    # --- Fill engine --------------------------------------------------------

    def _market_fill_price(self, symbol: str, side: OrderSide) -> float | None:
        """Marketable price for a market order: ask for buys, bid for sells,
        falling back to the last price. Adverse slippage is then applied."""
        q = self._quotes.get(symbol, {})
        ref = self.market_prices.get(symbol)
        quoted = q.get("ask") if side == OrderSide.BUY else q.get("bid")
        base = quoted if quoted is not None else ref
        if base is None:
            return None
        slip = base * self.slippage_bps / 10_000.0
        return base + slip if side == OrderSide.BUY else base - slip

    def _limit_marketable_price(
        self, symbol: str, side: OrderSide, limit_price: float
    ) -> float | None:
        """Fill price for a limit order if it currently crosses, else None.

        Buy fills when the market (ask, else last) is at or below the limit;
        sell when the market (bid, else last) is at or above it. Filled at the
        better of the limit and the prevailing quote — no slippage on limits.
        """
        q = self._quotes.get(symbol, {})
        ref = self.market_prices.get(symbol)
        if side == OrderSide.BUY:
            mkt = q.get("ask")
            mkt = mkt if mkt is not None else ref
            if mkt is None or mkt > limit_price:
                return None
            return min(mkt, limit_price)
        mkt = q.get("bid")
        mkt = mkt if mkt is not None else ref
        if mkt is None or mkt < limit_price:
            return None
        return max(mkt, limit_price)

    def _fill_market_order(self, order: PaperOrder) -> None:
        if self.is_market_open is not None and not self.is_market_open(order.symbol):
            order.status = OrderStatus.REJECTED
            return

        fill_price = self._market_fill_price(order.symbol, order.side)
        if fill_price is None:
            order.status = OrderStatus.REJECTED
            return

        if not self._can_fill(order, fill_price):
            order.status = OrderStatus.REJECTED
            return

        self._execute_trade(order, fill_price)

    def _try_fill_limit_order(self, order: PaperOrder) -> None:
        """Attempt to fill a resting limit order against the current quote."""
        if order.status != OrderStatus.PENDING or order.price is None:
            return
        # Closed market: leave the order queued rather than rejecting it.
        if self.is_market_open is not None and not self.is_market_open(order.symbol):
            return

        fill_price = self._limit_marketable_price(order.symbol, order.side, order.price)
        if fill_price is None:
            return  # not marketable yet — stays PENDING
        if not self._can_fill(order, fill_price):
            return  # can't afford / no position — stays PENDING
        self._execute_trade(order, fill_price)

    def _match_pending_limit_orders(self, symbols: list[str]) -> None:
        affected = set(symbols)
        _protective = {OrderType.STOP, OrderType.TRAILING_STOP, OrderType.TAKE_PROFIT}
        for order in list(self._orders.values()):
            if (
                order.status == OrderStatus.PENDING
                and order.order_type not in (OrderType.MARKET, *_protective)
                and order.symbol in affected
            ):
                self._try_fill_limit_order(order)

    # --- Protective-order engine (P0) ---------------------------------------

    def _evaluate_protective_orders(self, symbols: list[str]) -> None:
        """Evaluate all resting protective orders for the updated symbols."""
        affected = set(symbols)
        for order in list(self._orders.values()):
            if order.status == OrderStatus.PENDING and order.symbol in affected:
                if order.order_type in (OrderType.STOP, OrderType.TRAILING_STOP, OrderType.TAKE_PROFIT):
                    self._evaluate_protective_order(order)

    def _evaluate_protective_order(self, order: PaperOrder) -> None:
        """Fire a single protective order if its trigger is met."""
        if order.status != OrderStatus.PENDING or order.stop_price is None:
            return
        px = self.market_prices.get(order.symbol)
        if px is None:
            return

        if order.order_type == OrderType.TRAILING_STOP:
            self._ratchet_trailing_stop(order, px)
            if order.stop_price is None:
                return

        triggered = False
        if order.order_type == OrderType.TAKE_PROFIT:
            # For a SELL take-profit: fire when price rises to or above the target.
            # For a BUY take-profit (covering a short): fire when price falls to or below.
            triggered = (
                (order.side == OrderSide.SELL and px >= order.stop_price)
                or (order.side == OrderSide.BUY and px <= order.stop_price)
            )
        else:
            # STOP and TRAILING_STOP: sell when price falls to/below trigger,
            # buy (stop-to-cover) when price rises to/above trigger.
            triggered = (
                (order.side == OrderSide.SELL and px <= order.stop_price)
                or (order.side == OrderSide.BUY and px >= order.stop_price)
            )

        if triggered:
            # Execute as a market fill at the current price (override open-market
            # check — protective orders must fire even in after-hours).
            fill_price = self._market_fill_price(order.symbol, order.side)
            if fill_price is None:
                fill_price = px
            if self._can_fill(order, fill_price):
                self._execute_trade(order, fill_price)
            else:
                order.status = OrderStatus.REJECTED
            self._trail_peak.pop(order.id, None)

    def _ratchet_trailing_stop(self, order: PaperOrder, px: float) -> None:
        """Update the trailing-stop trigger if the price has moved in our favour."""
        if order.trail_offset is None or order.stop_price is None:
            return
        peak = self._trail_peak.get(order.id, px)
        if order.side == OrderSide.SELL:
            # Long protection: ratchet up with the new high.
            if px > peak:
                self._trail_peak[order.id] = px
                order.stop_price = px - order.trail_offset
        else:
            # Short protection (buy to cover): ratchet down with the new low.
            if px < peak:
                self._trail_peak[order.id] = px
                order.stop_price = px + order.trail_offset

    def _check_hard_floor(self) -> None:
        """Auto-flatten if account equity has fallen below the hard-floor threshold.

        No model in the loop — fires deterministically on every price update if
        ``hard_floor_pct`` is set and the loss threshold is breached.
        """
        if self.hard_floor_pct is None:
            return
        equity = self._equity()
        if equity < self._initial_balance * (1.0 - self.hard_floor_pct / 100.0):
            self.flatten_all()

    def flatten_all(self) -> None:
        """Close every open position at the current market price.

        Used by the hard-floor auto-flatten (P0) and externally by the bench's
        risk checks. Bypasses the open-market check — this is an emergency exit.
        """
        for symbol, pos in list(self._positions.items()):
            if pos.quantity == 0:
                continue
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            qty = abs(pos.quantity)
            order_id = str(uuid.uuid4())
            order = PaperOrder(
                id=order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty,
            )
            self._orders[order_id] = order
            fill_px = self._market_fill_price(symbol, side) or self.market_prices.get(symbol, pos.avg_price)
            if self._can_fill(order, fill_px):
                self._execute_trade(order, fill_px)
            else:
                order.status = OrderStatus.REJECTED

    def _can_fill(self, order: PaperOrder, fill_price: float) -> bool:
        """Affordability / position / margin check shared by market and limit fills.

        BUY: must be affordable from cash (covering a short is affordable because
        the short's proceeds already sit in cash). SELL: fillable up to the
        existing long for free; any quantity beyond that opens/extends a short,
        which is rejected unless ``allow_short`` and the resulting short passes
        the margin / exposure check.
        """
        if order.side == OrderSide.BUY:
            cost = order.quantity * fill_price
            commission = cost * self.commission_bps / 10_000.0
            return cost + commission <= self._balance
        position = self._positions.get(order.symbol)
        long_qty = position.quantity if position is not None and position.quantity > 0 else 0.0
        if order.quantity <= long_qty:
            return True  # fully covered by an existing long — no short, no margin
        if not self.allow_short:
            return False  # long-only book: can't sell more than you hold
        old_qty = position.quantity if position is not None else 0.0
        resulting_qty = old_qty - order.quantity  # < 0 → net short
        return self._short_within_limits(order.symbol, resulting_qty, fill_price)

    def _short_within_limits(self, symbol: str, resulting_qty: float, fill_price: float) -> bool:
        """True if going to ``resulting_qty`` keeps short exposure within margin/cap."""
        exposure = self._short_exposure_after(symbol, resulting_qty, fill_price)
        if self.max_short_notional is not None and exposure > self.max_short_notional:
            return False
        return self.short_margin_ratio * exposure <= self._equity()

    def _equity(self) -> float:
        """Account equity: cash + mark-to-market of every position (shorts net negative)."""
        total = self._balance
        for sym, pos in self._positions.items():
            mark = self.market_prices.get(sym, pos.avg_price)
            total += pos.quantity * mark
        return total

    def _short_exposure_after(self, symbol: str, resulting_qty: float, fill_price: float) -> float:
        """Total |short| market value across the book if ``symbol`` ends at ``resulting_qty``."""
        total = 0.0
        seen = False
        for sym, pos in self._positions.items():
            if sym == symbol:
                seen = True
                qty, mark = resulting_qty, fill_price
            else:
                qty, mark = pos.quantity, self.market_prices.get(sym, pos.avg_price)
            if qty < 0:
                total += -qty * mark
        if not seen and resulting_qty < 0:
            total += -resulting_qty * fill_price
        return total

    def _execute_trade(self, order: PaperOrder, fill_price: float) -> None:
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price

        # Realized P&L: the portion of this fill that closes existing exposure,
        # priced against the position's average entry — captured *before* the
        # position is mutated below. A sell against a long, or a buy against a
        # short, realizes gain/loss; opening/adding does not.
        closed_qty = 0.0
        realized = 0.0
        existing = self._positions.get(order.symbol)
        if existing is not None:
            if order.side == OrderSide.SELL and existing.quantity > 0:
                closed_qty = min(order.quantity, existing.quantity)
                realized = closed_qty * (fill_price - existing.avg_price)
            elif order.side == OrderSide.BUY and existing.quantity < 0:
                closed_qty = min(order.quantity, -existing.quantity)
                realized = closed_qty * (existing.avg_price - fill_price)

        notional = order.quantity * fill_price
        commission = notional * self.commission_bps / 10_000.0

        if order.side == OrderSide.BUY:
            self._balance -= notional

            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                old_quantity = pos.quantity
                total_quantity = old_quantity + order.quantity

                if total_quantity > 0:
                    if old_quantity <= 0:
                        pos.avg_price = fill_price
                    else:
                        pos.avg_price = (old_quantity * pos.avg_price + order.quantity * fill_price) / total_quantity
                    pos.quantity = total_quantity
                elif total_quantity < 0:
                    pos.quantity = total_quantity
                else:
                    del self._positions[order.symbol]
            else:
                self._positions[order.symbol] = PaperPosition(order.symbol, order.quantity, fill_price)

        else:  # SELL
            self._balance += notional

            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                old_quantity = pos.quantity
                total_quantity = old_quantity - order.quantity

                if old_quantity > 0:
                    if total_quantity > 0:
                        pos.quantity = total_quantity
                    elif total_quantity == 0:
                        del self._positions[order.symbol]
                    else:
                        pos.quantity = total_quantity
                        pos.avg_price = fill_price
                else:  # was short or flat
                    if total_quantity < 0:
                        old_short_qty = abs(old_quantity)
                        new_short_qty = order.quantity
                        pos.avg_price = (old_short_qty * pos.avg_price + new_short_qty * fill_price) / (old_short_qty + new_short_qty)
                        pos.quantity = total_quantity
                    elif total_quantity == 0:
                        del self._positions[order.symbol]
                    else:
                        pos.quantity = total_quantity
                        pos.avg_price = fill_price
            else:
                self._positions[order.symbol] = PaperPosition(order.symbol, -order.quantity, fill_price)

        self._balance -= commission
        self._trade_history.append((order.symbol, fill_price, order.quantity, order.side.value))
        if closed_qty > 0:
            self._realized_pnl += realized
            self._closed_pnls.append(realized)

    def _order_result(self, order: PaperOrder) -> dict[str, Any]:
        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "filled_price": order.filled_price,
        }

    def get_position(self, symbol: str) -> PaperPosition | None:
        pos = self._positions.get(symbol)
        return PaperPosition(pos.symbol, pos.quantity, pos.avg_price) if pos else None

    def get_positions(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_price": pos.avg_price
            }
            for pos in self._positions.values()
        ]

    def get_account_value(self, market_prices: dict[str, float]) -> float:
        total_value = self._balance
        for symbol, position in self._positions.items():
            if symbol in market_prices:
                total_value += position.quantity * market_prices[symbol]
        return total_value

    def get_trade_history(self) -> list[tuple[str, float, float, str]]:
        return self._trade_history.copy()

    def get_realized_pnl(self) -> float:
        """Total realized profit/loss from closed (or reduced) positions."""
        return self._realized_pnl

    def get_win_loss(self) -> tuple[int, int]:
        """Count of closing trades that realized a gain vs. a loss.

        Break-even closes (exactly 0) count as neither, so a win rate computed
        as ``wins / (wins + losses)`` excludes them from the denominator.
        """
        wins = sum(1 for p in self._closed_pnls if p > 0)
        losses = sum(1 for p in self._closed_pnls if p < 0)
        return wins, losses

    def reset(self) -> None:
        """Reset broker state while preserving initial cash + realism config."""
        self._balance = self._initial_balance
        self._positions.clear()
        self._orders.clear()
        self._trade_history.clear()
        self._realized_pnl = 0.0
        self._closed_pnls.clear()
        self.market_prices.clear()
        self._quotes.clear()
        self._trail_peak.clear()
        self._connected = False
