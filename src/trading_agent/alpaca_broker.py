from typing import Any

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

# Import Alpaca's enums under aliased names so they do not shadow the
# project-local OrderSide / OrderType / OrderStatus enums coming from .enums.
from alpaca.trading.enums import (
    OrderSide as AlpacaOrderSide,
)
from alpaca.trading.enums import (
    OrderStatus as AlpacaOrderStatus,
)
from alpaca.trading.enums import (
    TimeInForce,
)
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from .broker_adapter import BrokerAdapter
from .enums import OrderSide, OrderType


class AlpacaBroker(BrokerAdapter):
    """
    BrokerAdapter implementation for Alpaca using alpaca-py library.
    Supports both paper and live trading for US equities.

    Accepts the canonical order_details dict produced by SignalRouter:
        {
            'symbol': str,
            'side': OrderSide (project enum) | str ('buy'/'sell'/'BUY'/'SELL'),
            'order_type': OrderType (project enum) | str ('market'/'limit'/...),
            'amount' or 'quantity': float (positive; sign-based amount also tolerated),
            'price': Optional[float],
            'time_in_force': Optional[str],
        }
    """

    TIF_MAP = {
        'day': TimeInForce.DAY,
        'gtc': TimeInForce.GTC,
        'opg': TimeInForce.OPG,
        'cls': TimeInForce.CLS,
        'ioc': TimeInForce.IOC,
        'fok': TimeInForce.FOK,
    }

    ORDER_SIDE_MAP = {
        OrderSide.BUY: AlpacaOrderSide.BUY,
        OrderSide.SELL: AlpacaOrderSide.SELL,
        'buy': AlpacaOrderSide.BUY,
        'sell': AlpacaOrderSide.SELL,
        'BUY': AlpacaOrderSide.BUY,
        'SELL': AlpacaOrderSide.SELL,
        AlpacaOrderSide.BUY: AlpacaOrderSide.BUY,
        AlpacaOrderSide.SELL: AlpacaOrderSide.SELL,
    }

    ORDER_TYPE_MAP = {
        OrderType.MARKET: 'market',
        OrderType.LIMIT: 'limit',
        'market': 'market',
        'limit': 'limit',
        'MARKET': 'market',
        'LIMIT': 'limit',
    }

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

    # --- Helpers ----------------------------------------------------------------

    @classmethod
    def _to_alpaca_side(cls, side_value: Any) -> AlpacaOrderSide:
        """Normalize any accepted side representation to alpaca's OrderSide enum."""
        if side_value not in cls.ORDER_SIDE_MAP:
            raise ValueError(f"Invalid order side: {side_value!r}")
        return cls.ORDER_SIDE_MAP[side_value]

    @classmethod
    def _to_order_type_str(cls, order_type_value: Any) -> str:
        """Normalize OrderType (project enum / string) to lowercase string."""
        if order_type_value not in cls.ORDER_TYPE_MAP:
            raise ValueError(f"Invalid order type: {order_type_value!r}")
        return cls.ORDER_TYPE_MAP[order_type_value]

    # --- BrokerAdapter interface -----------------------------------------------

    def connect(self) -> bool:
        try:
            account = self.trading_client.get_account()
            return account.status == 'ACTIVE'
        except Exception:
            return False

    def get_balance(self) -> dict[str, Any]:
        account = self.trading_client.get_account()
        positions = self.trading_client.get_all_positions()

        positions_dict = {}
        for pos in positions:
            positions_dict[pos.symbol] = {
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": pos.side,
            }

        return {
            "account_id": account.id,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "equity": float(account.equity),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "initial_margin": float(account.initial_margin),
            "maintenance_margin": float(account.maintenance_margin),
            "last_equity": float(account.last_equity),
            "daytrade_count": account.daytrade_count,
            "positions": positions_dict,
        }

    def get_quote(self, symbol: str) -> dict[str, Any]:
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quote = self.data_client.get_stock_latest_quote(request)

            if symbol in quote:
                quote_data = quote[symbol]
                bid = float(quote_data.bid_price)
                ask = float(quote_data.ask_price)
                price = (bid + ask) / 2.0

                return {
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "last": None,
                    "timestamp": quote_data.timestamp.isoformat() if quote_data.timestamp else None,
                    "price": price,
                }
            raise ValueError(f"Quote not found for symbol: {symbol}")
        except ValueError:
            # Re-raise ValueError for unknown symbols without wrapping
            raise
        except Exception as e:
            # Wrap other exceptions as ConnectionError
            raise ConnectionError(f"Error fetching quote: {str(e)}")

    def place_order(self, order_details: dict[str, Any]) -> dict[str, Any] | None:
        """Place an order using SignalRouter's canonical order_details dict."""
        if not isinstance(order_details, dict):
            raise ValueError("order_details must be a dict")

        symbol = order_details.get("symbol")
        order_type_raw = order_details.get("order_type")
        amount_raw = order_details.get("amount", order_details.get("quantity"))
        price = order_details.get("price")
        time_in_force = order_details.get("time_in_force", "day")

        if symbol is None or order_type_raw is None or amount_raw is None:
            raise ValueError("Missing required order parameters: symbol, order_type, amount/quantity")

        # Resolve side: explicit 'side' wins; fall back to sign of amount.
        side_raw = order_details.get("side")
        if side_raw is not None:
            side = self._to_alpaca_side(side_raw)
        else:
            side = AlpacaOrderSide.BUY if float(amount_raw) > 0 else AlpacaOrderSide.SELL

        quantity = abs(float(amount_raw))
        if quantity <= 0:
            raise ValueError("Order quantity must be positive")

        tif_enum = self.TIF_MAP.get(time_in_force.lower(), TimeInForce.DAY)
        order_type_str = self._to_order_type_str(order_type_raw)

        if order_type_str == 'market':
            request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=tif_enum,
            )
            order = self.trading_client.submit_order(order_data=request)
            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "amount": quantity if side == AlpacaOrderSide.BUY else -quantity,
                "price": None,
                "order_type": "market",
                "status": order.status.value,
            }
        if order_type_str == 'limit':
            if price is None:
                raise ValueError("Price is required for limit orders")
            request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                limit_price=price,
                time_in_force=tif_enum,
            )
            order = self.trading_client.submit_order(order_data=request)
            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "amount": quantity if side == AlpacaOrderSide.BUY else -quantity,
                "price": price,
                "order_type": "limit",
                "status": order.status.value,
            }
        raise ValueError(f"Unsupported order type: {order_type_raw}")

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        order = self.trading_client.get_order_by_id(order_id)
        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "qty": float(order.qty) if order.qty else 0,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "order_type": order.order_type.value,
            "side": order.side.value,
            "status": order.status.value,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
        }

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_positions(self) -> list[dict[str, Any]]:
        positions = self.trading_client.get_all_positions()
        return [
            {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": pos.side,
            }
            for pos in positions
        ]

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        timeframe_map = {
            '1Min': TimeFrame.Minute,
            '5Min': TimeFrame(5, TimeFrame.Minute.unit),
            '15Min': TimeFrame(15, TimeFrame.Minute.unit),
            '1H': TimeFrame.Hour,
            '1D': TimeFrame.Day,
        }
        mapped_timeframe = timeframe_map.get(timeframe)
        if not mapped_timeframe:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=mapped_timeframe,
            start=start_date,
            end=end_date,
        )
        bars = self.data_client.get_stock_bars(request)
        df = bars.df
        if not df.empty:
            df = df.reset_index()
        return df

    def get_orders(self, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
        status_map = {
            'open': AlpacaOrderStatus.OPEN,
            'closed': AlpacaOrderStatus.CLOSED,
            'all': None,
        }
        status_enum = status_map.get(status.lower())
        orders = self.trading_client.get_orders(status=status_enum, limit=limit)
        return [
            {
                "order_id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty) if order.qty else 0,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "order_type": order.order_type.value,
                "side": order.side.value,
                "status": order.status.value,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            }
            for order in orders
        ]
