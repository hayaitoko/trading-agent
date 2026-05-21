"""Rolling in-memory network log.

Records both inbound HTTP requests to our own routes (via middleware) and
outbound calls our server makes (currently just OpenRouter chat / consolidator).
Buffer is in-memory only — not persisted, not exported. Good enough for the
'what did the agent just do' use case.
"""
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

Direction = Literal["in", "out"]


@dataclass
class NetworkEntry:
    timestamp: str
    direction: Direction
    method: str
    target: str
    status: int
    duration_ms: int
    error: str | None = None


class NetworkLog:
    def __init__(self, max_entries: int = 200):
        self._entries: deque[NetworkEntry] = deque(maxlen=max_entries)

    def record(
        self,
        *,
        direction: Direction,
        method: str,
        target: str,
        status: int,
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        self._entries.appendleft(NetworkEntry(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            direction=direction,
            method=method,
            target=target,
            status=status,
            duration_ms=duration_ms,
            error=error,
        ))

    def snapshot(self, limit: int = 50) -> list[dict]:
        return [asdict(e) for e in list(self._entries)[:limit]]

    def clear(self) -> None:
        self._entries.clear()


class NetworkLogMiddleware(BaseHTTPMiddleware):
    """Records every inbound request, skipping noisy paths."""

    SKIP_PREFIXES = ("/static",)
    SKIP_EXACT = frozenset({"/settings/api/netlog"})

    def __init__(self, app, netlog: NetworkLog):
        super().__init__(app)
        self.netlog = netlog

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self.SKIP_EXACT or any(path.startswith(p) for p in self.SKIP_PREFIXES):
            return await call_next(request)

        start = time.monotonic()
        status = 500
        error: str | None = None
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            self.netlog.record(
                direction="in",
                method=request.method,
                target=path,
                status=status,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=error,
            )
