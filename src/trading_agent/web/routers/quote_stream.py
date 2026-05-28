"""WebSocket /ws/quotes: realtime bridge over the in-process MessageBus.

The MessageBus already publishes ``quote.<symbol>`` and ``bar.<symbol>`` events
as the live-quote poller ticks (``feeds/live_quote.py``). This router exposes
those topics over a per-session WebSocket so the cockpit watchlist tile can
drop its 45-second polling.

Wire protocol (JSON text frames both directions):

* Client → ``{"action": "subscribe",   "symbols": ["AAPL", ...]}``
* Client → ``{"action": "unsubscribe", "symbols": ["AAPL"]}``
* Server → ``{"symbol": "AAPL", "price": 199.50, "ts": "2026-05-28T..."}``
* Server → ``{"error": "<message>"}`` on a malformed client frame.

Auth: same session cookie / Bearer token as the HTTP routes — resolved off
``app.state.db`` during the WebSocket handshake. Unauthenticated handshakes
are rejected with policy code 1008 *before* :meth:`WebSocket.accept`, so the
TestClient sees the failure as a ``WebSocketDisconnect``.

Threading note: ``MessageBus.publish`` is called from the live-quote poller
thread (or any other thread that drives the bus), not from the WS event loop.
The handler hops back onto the loop via ``loop.call_soon_threadsafe`` before
touching the per-connection ``asyncio.Queue`` — that is the only point where
the two threading worlds meet.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ...config.users import SESSION_COOKIE, resolve_session

router = APIRouter(tags=["quotes-stream"])

# Per-connection cap. The watchlist tile rarely exceeds ~25; 50 leaves head-room
# without letting a noisy client flood the bus with subscriptions.
_MAX_SUBSCRIPTIONS = 50


def _token_from_ws(websocket: WebSocket) -> str | None:
    """Same resolution rules as the HTTP ``current_user`` dependency."""
    cookie = websocket.cookies.get(SESSION_COOKIE)
    if cookie:
        return cookie
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[len("bearer ") :].strip()
    return None


def _clean_symbols(values: Iterable[Any]) -> list[str]:
    """Normalize a list of ``symbols`` into uppercase non-empty strings."""
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if not isinstance(v, str):
            continue
        sym = v.strip().upper()
        if not sym or sym in seen:
            continue
        if not sym.replace(".", "").replace("-", "").isalnum() or len(sym) > 12:
            continue
        seen.add(sym)
        out.append(sym)
    return out


@router.websocket("/ws/quotes")
async def quote_stream(websocket: WebSocket) -> None:  # noqa: C901 — single linear handler, hard to split
    """Push ``{symbol, price, ts}`` for each subscribed symbol's ``quote.*`` event."""
    db = getattr(websocket.app.state, "db", None)
    bus = getattr(websocket.app.state, "bus", None)
    user_id = resolve_session(db, _token_from_ws(websocket)) if db is not None else None
    if user_id is None:
        # Reject before accept so clients see a clean handshake failure (1008
        # policy violation, the same code starlette uses for auth issues).
        await websocket.close(code=1008)
        return
    if bus is None:
        await websocket.close(code=1011, reason="quote stream unavailable")
        return

    await websocket.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    subscribed: dict[str, Any] = {}  # symbol -> bus handler (so we can unsubscribe by identity)

    def make_handler(symbol: str) -> Any:
        def _handler(message: dict[str, Any]) -> None:
            # Called from the bus publisher's thread; hop back to the WS loop.
            payload = {
                "symbol": symbol,
                "price": message.get("price"),
                "ts": message.get("timestamp"),
            }
            try:
                loop.call_soon_threadsafe(queue.put_nowait, payload)
            except RuntimeError:
                # Loop closed — connection torn down; drop the event.
                pass

        return _handler

    def subscribe(symbols: list[str]) -> list[str]:
        added: list[str] = []
        for sym in symbols:
            if sym in subscribed:
                continue
            if len(subscribed) >= _MAX_SUBSCRIPTIONS:
                break
            handler = make_handler(sym)
            subscribed[sym] = handler
            bus.subscribe(f"quote.{sym}", handler)
            added.append(sym)
        return added

    def unsubscribe(symbols: list[str]) -> list[str]:
        removed: list[str] = []
        for sym in symbols:
            handler = subscribed.pop(sym, None)
            if handler is None:
                continue
            bus.unsubscribe(f"quote.{sym}", handler)
            removed.append(sym)
        return removed

    async def reader() -> None:
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                return
            except Exception:
                # Non-JSON frame, or transport error — tell the client and continue.
                with contextlib.suppress(Exception):
                    await websocket.send_json({"error": "invalid frame; expected JSON object"})
                continue
            if not isinstance(msg, dict):
                with contextlib.suppress(Exception):
                    await websocket.send_json({"error": "invalid frame; expected JSON object"})
                continue
            action = msg.get("action")
            syms = _clean_symbols(msg.get("symbols") or [])
            if action == "subscribe":
                added = subscribe(syms)
                with contextlib.suppress(Exception):
                    await websocket.send_json({"subscribed": added})
            elif action == "unsubscribe":
                removed = unsubscribe(syms)
                with contextlib.suppress(Exception):
                    await websocket.send_json({"unsubscribed": removed})
            else:
                with contextlib.suppress(Exception):
                    await websocket.send_json(
                        {"error": f"unknown action: {action!r}"}
                    )

    async def writer() -> None:
        while True:
            payload = await queue.get()
            try:
                await websocket.send_json(payload)
            except Exception:
                return

    reader_task = asyncio.create_task(reader())
    writer_task = asyncio.create_task(writer())
    try:
        await asyncio.wait(
            {reader_task, writer_task}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        # Tear down every bus subscription this connection registered, then
        # cancel the sibling task so the handler returns cleanly.
        for sym in list(subscribed):
            unsubscribe([sym])
        for task in (reader_task, writer_task):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        if websocket.application_state != WebSocketState.DISCONNECTED:
            with contextlib.suppress(Exception):
                await websocket.close()
