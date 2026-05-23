from typing import Any

import ccxt

from .broker_adapter import BrokerAdapter
from .enums import OrderSide, OrderType


class CCXTBroker(BrokerAdapter):
    """Broker adapter for cryptocurrency exchanges using CCXT library"""

    SUPPORTED_EXCHANGES = ['binance', 'coinbase']

    def __init__(self, exchange_name: str, api_key: str, secret: str,
                 passphrase: str | None = None, sandbox: bool = False):
        if exchange_name not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Exchange '{exchange_name}' is not supported. "
                             f"Supported exchanges: {self.SUPPORTED_EXCHANGES}")

        self.exchange_name = exchange_name.lower()
        self.sandbox = sandbox

        try:
            if exchange_name == 'binance':
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': secret,
                    'options': {'defaultType': 'future'},
                })
            elif exchange_name == 'coinbase':
                if not passphrase:
                    raise ValueError("Passphrase is required for Coinbase")
                self.exchange = ccxt.coinbase({
                    'apiKey': api_key,
                    'secret': secret,
                    'password': passphrase,
                })
            else:
                raise ValueError(f"Unsupported exchange: {exchange_name}")

            if sandbox:
                self.exchange.set_sandbox_mode(True)

        except Exception as e:
            raise ConnectionError(f"Failed to initialize exchange connection: {str(e)}")

    def connect(self) -> bool:
        try:
            self.exchange.load_markets()
            return True
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Network error connecting to exchange: {str(e)}")
        except ccxt.ExchangeError as e:
            raise ConnectionError(f"Exchange error: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error while connecting: {str(e)}")

    # Stablecoin codes (in priority order) used to expose a single 'cash' figure.
    STABLE_CODES = ("USDT", "USDC", "USD", "BUSD", "DAI")

    def get_balance(self) -> dict[str, Any]:
        try:
            raw = self.exchange.fetch_balance()
        except ccxt.AuthenticationError as e:
            raise ConnectionError(f"Authentication failed: {str(e)}")
        except ccxt.ExchangeError as e:
            raise ConnectionError(f"Exchange error while fetching balance: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error fetching balance: {str(e)}")

        cash = 0.0
        for code in self.STABLE_CODES:
            entry = raw.get(code)
            if isinstance(entry, dict):
                total = entry.get("total")
                if total is not None:
                    cash = float(total)
                    break

        out = dict(raw)
        out["cash"] = cash
        return out

    def get_quote(self, symbol: str) -> dict[str, Any]:
        # Validate symbol first, outside try/except
        if symbol not in self.exchange.markets:
            self.exchange.load_markets()
            if symbol not in self.exchange.markets:
                raise ValueError(f"Invalid symbol: {symbol}")

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "last": ticker.get("last"),
                "timestamp": ticker.get("timestamp"),
                "datetime": ticker.get("datetime")
            }
        except ccxt.ExchangeError as e:
            raise ConnectionError(f"Exchange error fetching quote: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error fetching quote: {str(e)}")

    def place_order(self, order_details: dict[str, Any]) -> dict[str, Any] | None:
        try:
            symbol = order_details.get("symbol")
            order_type = order_details.get("order_type")
            amount = order_details.get("amount") or order_details.get("quantity")
            price = order_details.get("price")

            if not all([symbol, order_type, amount]):
                raise ValueError("Missing required order parameters: symbol, type, amount")

            if symbol not in self.exchange.markets:
                self.exchange.load_markets()
                if symbol not in self.exchange.markets:
                    raise ValueError(f"Invalid symbol: {symbol}")

            # Convert OrderType enum to string if needed
            if isinstance(order_type, OrderType):
                order_type_str = order_type.value.lower()
            else:
                order_type_str = order_type.lower()

            # Determine side (buy/sell) from OrderSide enum or amount sign
            side = order_details.get("side")
            if side is None:
                side = OrderSide.BUY if float(amount) > 0 else OrderSide.SELL
            else:
                # Ensure side is an OrderSide enum
                if isinstance(side, str):
                    side = OrderSide[side.upper()]

            # Convert OrderSide enum to string for CCXT
            side_str = side.value.lower()

            abs_amount = abs(float(amount))

            # Map to CCXT order types
            if order_type_str in ['market', 'MARKET']:
                order = self.exchange.create_order(symbol, "market", side_str, abs_amount)
            elif order_type_str in ['limit', 'LIMIT']:
                if price is None:
                    raise ValueError("Price is required for limit orders")
                order = self.exchange.create_order(symbol, "limit", side_str, abs_amount, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            return {
                "order_id": order.get("id"),
                "symbol": symbol,
                "amount": abs_amount,
                "price": price,
                "order_type": order_type_str,
                "side": side_str,
                "status": order.get("status", "open")
            }
        except ccxt.InsufficientFunds as e:
            raise ConnectionError(f"Insufficient funds: {str(e)}")
        except ccxt.InvalidOrder as e:
            raise ConnectionError(f"Invalid order parameters: {str(e)}")
        except ccxt.ExchangeError as e:
            raise ConnectionError(f"Exchange error placing order: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error placing order: {str(e)}")

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Fetch order status by ID.

        Iterates through all available markets to locate the order.

        Args:
            order_id: The order ID to fetch
        """
        try:
            # Iterate through all known symbols
            for candidate_symbol in self.exchange.symbols:
                try:
                    order = self.exchange.fetch_order(order_id, candidate_symbol)
                    return order
                except ccxt.OrderNotFound:
                    continue
                except ccxt.ExchangeError:
                    # Exchange may raise an error if the symbol does not support order queries
                    continue

            raise ValueError(f"Order {order_id} not found")
        except ccxt.OrderNotFound as e:
            raise ConnectionError(f"Order not found: {str(e)}")
        except ccxt.ExchangeError as e:
            raise ConnectionError(f"Exchange error fetching order: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error fetching order: {str(e)}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID.

        Iterates through all available markets to locate the order.

        Args:
            order_id: The order ID to cancel
        """
        try:
            # Iterate through all known symbols
            for candidate_symbol in self.exchange.symbols:
                try:
                    self.exchange.cancel_order(order_id, candidate_symbol)
                    return True
                except ccxt.OrderNotFound:
                    continue
                except ccxt.ExchangeError:
                    continue

            raise ValueError(f"Order {order_id} not found")
        except ccxt.OrderNotFound as e:
            raise ConnectionError(f"Order not found: {str(e)}")
        except ccxt.ExchangeError as e:
            raise ConnectionError(f"Exchange error cancelling order: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error cancelling order: {str(e)}")

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        try:
            if symbol:
                return self.exchange.fetch_open_orders(symbol)
            else:
                return self.exchange.fetch_open_orders()
        except ccxt.ExchangeError as e:
            raise ConnectionError(f"Exchange error fetching open orders: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error fetching open orders: {str(e)}")

    def get_positions(self) -> list[dict[str, Any]]:
        try:
            if hasattr(self.exchange, 'fetch_positions'):
                positions = self.exchange.fetch_positions()
                return positions if isinstance(positions, list) else []
            else:
                return []
        except ccxt.ExchangeError as e:
            raise ConnectionError(f"Exchange error fetching positions: {str(e)}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error fetching positions: {str(e)}")
