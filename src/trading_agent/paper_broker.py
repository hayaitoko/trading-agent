"""In-memory paper broker for backtesting and dry-run.

Implements the canonical BrokerAdapter interface. Broker-internal order /
position records are named PaperOrder / PaperPosition to avoid shadowing the
models.Order / models.Position dataclasses (which carry different fields).
"""

import uuid
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


@dataclass
class PaperPosition:
    symbol: str
    quantity: float
    avg_price: float


class PaperBroker(BrokerAdapter):
    """In-memory paper broker."""

    def __init__(self, initial_balance: float = 100000.0):
        self._initial_balance: float = initial_balance
        self._balance: float = initial_balance
        self._positions: dict[str, PaperPosition] = {}
        self._orders: dict[str, PaperOrder] = {}
        self._trade_history: list[tuple[str, float, float, str]] = []
        self.market_prices: dict[str, float] = {}
        self._connected: bool = False

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

    def get_quote(self, symbol: str) -> dict[str, float]:
        """Get current quote for symbol.

        Args:
            symbol: The symbol to get quote for

        Returns:
            Dictionary with quote information containing 'price' key

        Raises:
            ValueError: If symbol is not found in market_prices
        """
        price = self.market_prices.get(symbol)
        if price is None:
            raise ValueError(f"Symbol {symbol} not found in market_prices")
        return {'price': price}

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

        order = PaperOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=order_price,
        )
        self._orders[order_id] = order

        if order_type == OrderType.MARKET:
            self._fill_market_order(order)

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

    def update_market_prices(self, prices: dict[str, float]) -> None:
        self.market_prices.update(prices)

    def _fill_market_order(self, order: PaperOrder) -> None:
        market_price = self.market_prices.get(order.symbol)
        if market_price is None:
            order.status = OrderStatus.REJECTED
            return

        if order.side == OrderSide.BUY:
            cost = order.quantity * market_price
            if cost > self._balance:
                order.status = OrderStatus.REJECTED
                return
        elif order.side == OrderSide.SELL:
            position = self._positions.get(order.symbol)
            if position is None or position.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                return

        self._execute_trade(order, market_price)

    def _execute_trade(self, order: PaperOrder, fill_price: float) -> None:
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price

        if order.side == OrderSide.BUY:
            cost = order.quantity * fill_price
            self._balance -= cost

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
            revenue = order.quantity * fill_price
            self._balance += revenue

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

        self._trade_history.append((order.symbol, fill_price, order.quantity, order.side.value))

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

    def reset(self) -> None:
        """Reset broker state while preserving initial cash balance."""
        self._balance = self._initial_balance
        self._positions.clear()
        self._orders.clear()
        self._trade_history.clear()
        self.market_prices.clear()
        self._connected = False
