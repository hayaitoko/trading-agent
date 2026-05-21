from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from trading_agent.brokers.base import (
    Broker,
    InsufficientFundsError,
    UnknownTickerError,
)
from trading_agent.models import Order, Position, Trade


class MockBroker(Broker):
    """In-memory broker with caller-supplied price function.

    Market orders fill at the current quote. Limit orders are not held in a book
    here: they fill immediately if the limit is crossed, otherwise they raise.
    Wire up a real working order book when the first strategy needs it.
    """

    def __init__(self, cash: Decimal, quote_fn: Callable[[str], Decimal]):
        self._cash = cash
        self._quote_fn = quote_fn
        self._positions: dict[str, tuple[int, Decimal]] = {}
        self._trades: list[Trade] = []

    async def get_account_value(self) -> Decimal:
        equity = sum(
            (qty * self._quote_fn(ticker) for ticker, (qty, _) in self._positions.items()),
            start=Decimal(0),
        )
        return self._cash + equity

    async def get_cash(self) -> Decimal:
        return self._cash

    async def get_positions(self) -> list[Position]:
        return [
            Position(
                ticker=ticker,
                qty=qty,
                avg_cost=avg_cost,
                current_price=self._quote_fn(ticker),
            )
            for ticker, (qty, avg_cost) in self._positions.items()
        ]

    async def get_quote(self, ticker: str) -> Decimal:
        try:
            return self._quote_fn(ticker)
        except KeyError as e:
            raise UnknownTickerError(ticker) from e

    async def place_order(self, order: Order) -> str:
        quote = await self.get_quote(order.ticker)
        fill_price = self._resolve_fill_price(order, quote)
        cost = fill_price * order.qty

        if order.side == "buy":
            if cost > self._cash:
                raise InsufficientFundsError(f"need {cost}, have {self._cash}")
            self._cash -= cost
            self._add_to_position(order.ticker, order.qty, fill_price)
        else:
            held_qty, _ = self._positions.get(order.ticker, (0, Decimal(0)))
            if order.qty > held_qty:
                raise InsufficientFundsError(
                    f"selling {order.qty} of {order.ticker} but only hold {held_qty}"
                )
            self._cash += cost
            self._reduce_position(order.ticker, order.qty)

        order_id = str(uuid4())
        self._trades.append(
            Trade(
                order_id=order_id,
                ticker=order.ticker,
                side=order.side,
                qty=order.qty,
                price=fill_price,
                executed_at=datetime.now(),
            )
        )
        return order_id

    async def cancel_order(self, order_id: str) -> None:
        # MockBroker fills synchronously, so there are no resting orders to cancel.
        return None

    async def get_trades(self, since: datetime | None = None) -> list[Trade]:
        if since is None:
            return list(self._trades)
        return [t for t in self._trades if t.executed_at >= since]

    @staticmethod
    def _resolve_fill_price(order: Order, quote: Decimal) -> Decimal:
        if order.order_type == "market":
            return quote
        if order.limit_price is None:
            raise ValueError("limit order missing limit_price")
        if order.side == "buy" and quote <= order.limit_price:
            return quote
        if order.side == "sell" and quote >= order.limit_price:
            return quote
        raise InsufficientFundsError(
            f"limit {order.limit_price} not crossed by quote {quote} ({order.side})"
        )

    def _add_to_position(self, ticker: str, qty: int, price: Decimal) -> None:
        if ticker in self._positions:
            old_qty, old_avg = self._positions[ticker]
            new_qty = old_qty + qty
            new_avg = (old_avg * old_qty + price * qty) / new_qty
            self._positions[ticker] = (new_qty, new_avg)
        else:
            self._positions[ticker] = (qty, price)

    def _reduce_position(self, ticker: str, qty: int) -> None:
        old_qty, old_avg = self._positions[ticker]
        new_qty = old_qty - qty
        if new_qty == 0:
            del self._positions[ticker]
        else:
            self._positions[ticker] = (new_qty, old_avg)
