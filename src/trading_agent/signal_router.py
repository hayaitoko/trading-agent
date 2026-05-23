"""Routes Signal dicts to the broker (autonomous) or approval queue (approval).

A Signal is the dict shape emitted by strategies:

    {
        'asset':  str,                  # required — symbol
        'side':   'LONG'|'SHORT'|'NEUTRAL' | OrderSide | 'BUY'|'SELL',
        'type':   'market'|'limit'|... | OrderType,  # default 'market'
        'amount' or 'quantity': float,
        'price':  Optional[float],
        'reason': Optional[str],
    }

A NEUTRAL signal is a no-op (we don't try to flatten existing positions —
that's the strategy's job, by emitting SHORT to close a long).
"""

from __future__ import annotations

from typing import Any, Protocol

from .enums import Mode, OrderSide, OrderType


class _BrokerLike(Protocol):
    def place_order(self, order_details: dict[str, Any]) -> dict[str, Any] | None: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> list[dict[str, Any]]: ...


class _ApprovalQueueLike(Protocol):
    def add(self, signal: dict[str, Any]) -> Any: ...


_SIDE_MAP = {
    "LONG": OrderSide.BUY,
    "SHORT": OrderSide.SELL,
    "BUY": OrderSide.BUY,
    "SELL": OrderSide.SELL,
}


class SignalRouter:
    """Mode-aware dispatcher.

    - ``AUTONOMOUS``: call broker directly (after RiskManager has cleared the trade).
    - ``APPROVAL``: hand the signal to the approval queue.

    The router itself does **not** call RiskManager — callers wrap dispatch with
    risk checks. This keeps the router single-purpose.
    """

    def __init__(
        self,
        broker: _BrokerLike,
        approval_queue: _ApprovalQueueLike | None = None,
        global_mode: Mode = Mode.AUTONOMOUS,
        asset_modes: dict[str, Mode] | None = None,
    ) -> None:
        self.broker = broker
        self.approval_queue = approval_queue
        self.global_mode = global_mode
        self.asset_modes: dict[str, Mode] = dict(asset_modes or {})

    # --- Public API ---------------------------------------------------------

    def dispatch(self, signal: dict[str, Any]) -> dict[str, Any] | None:
        """Route a signal. Returns broker response (autonomous) or queue entry."""
        asset = signal.get("asset")
        if not asset:
            raise ValueError("Signal must contain 'asset' key")

        # NEUTRAL is the strategy saying "do nothing right now".
        side = signal.get("side")
        if isinstance(side, str) and side.upper() == "NEUTRAL":
            return None

        mode = self.asset_modes.get(asset, self.global_mode)
        if mode == Mode.AUTONOMOUS:
            return self._place(signal)
        if mode == Mode.APPROVAL:
            if self.approval_queue is None:
                raise RuntimeError("APPROVAL mode but no approval_queue is configured")
            return self.approval_queue.add(signal)
        raise ValueError(f"Unknown mode: {mode}")

    def set_asset_mode(self, asset: str, mode: Mode) -> None:
        self.asset_modes[asset] = mode

    def set_global_mode(self, mode: Mode) -> None:
        self.global_mode = mode

    def cancel_order(self, order_id: str) -> bool:
        return self.broker.cancel_order(order_id)

    def get_positions(self) -> list[dict[str, Any]]:
        return self.broker.get_positions()

    # --- Internals ----------------------------------------------------------

    def _place(self, signal: dict[str, Any]) -> dict[str, Any]:
        order = _signal_to_order(signal)
        result = self.broker.place_order(order)
        if result is None:
            raise RuntimeError(f"Broker failed to place order for {order['symbol']}")
        return result


def _signal_to_order(signal: dict[str, Any]) -> dict[str, Any]:
    """Map a strategy signal to the canonical broker order_details dict."""
    symbol = signal.get("asset")
    if not symbol:
        raise ValueError("Signal must contain 'asset' key")

    side_raw = signal.get("side")
    if side_raw is None:
        raise ValueError("Signal must contain 'side' key")
    side = _coerce_side(side_raw)

    type_raw = signal.get("type", "market")
    order_type = _coerce_order_type(type_raw)

    if "amount" in signal:
        amount = float(signal["amount"])
    elif "quantity" in signal:
        amount = float(signal["quantity"])
    else:
        raise ValueError("Signal must contain 'amount' or 'quantity' key")

    return {
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "amount": amount,
        "price": signal.get("price"),
    }


def _coerce_side(side: Any) -> OrderSide:
    if isinstance(side, OrderSide):
        return side
    if isinstance(side, str):
        key = side.upper()
        if key in _SIDE_MAP:
            return _SIDE_MAP[key]
    raise ValueError(f"Invalid side value: {side!r}")


def _coerce_order_type(order_type: Any) -> OrderType:
    if isinstance(order_type, OrderType):
        return order_type
    if isinstance(order_type, str):
        try:
            return OrderType(order_type.lower())
        except ValueError as e:
            raise ValueError(f"Invalid order type: {order_type!r}") from e
    raise ValueError(f"Invalid order type: {order_type!r}")
