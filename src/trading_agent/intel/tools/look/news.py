"""``news`` — recent news/social headlines from the ingest pipeline.

Tool name:      news
Args:           symbol=None (str|None), limit=10 (int)
ToolResult:     ok=True, data={"items": […], "symbol": str|None}
Latency tier:   fast
Cost class:     free
Gating flag:    always enabled (empty list when ingest store absent)
Example use:    news("TSLA") for the 10 most recent TSLA headlines.
                news(limit=20) for cross-ticker headlines.

Each item: {title, source, url, ts, ticker}

Wraps :class:`~trading_agent.ingest.store.IngestStore` (the WS-B raw_items table).
The owner_user_id scopes results — only items ingested for this user are returned.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase

_MAX_LIMIT = 100


class NewsTool(LookToolBase):
    """Read-only access to the ingest pipeline's raw_items headline feed.

    Parameters
    ----------
    db:
        :class:`~trading_agent.config.db.Database` instance.  ``None`` → empty
        result with a note.
    owner_user_id, trader_id:
        Namespace identifiers.
    """

    TOOL_META: dict[str, Any] = {
        "name": "news",
        "description": (
            "Recent news/social headlines from the ingest pipeline. "
            "Pass symbol to narrow to one ticker; omit for cross-ticker feed."
        ),
        "args": {
            "symbol": "str|None (default None)",
            "limit": "int (default 10)",
        },
        "latency": "fast",
        "cost_class": "free",
        "enabled": True,
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        db: Any = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._db = db

    def __call__(
        self,
        symbol: str | None = None,
        limit: int = 10,
    ) -> Any:
        """Return recent headlines, optionally scoped to one ticker.

        Returns
        -------
        ToolResult
            ok=True, data={"items": […], "symbol": …}

        Example
        -------
        >>> tool = NewsTool(trader_id="Alpha")
        >>> result = tool("AAPL", limit=5)
        >>> result.ok
        True
        >>> result.data["items"]
        []
        """
        if self._db is None or self.owner_user_id is None:
            return self._ok(
                {
                    "items": [],
                    "symbol": symbol,
                    "note": "news store not wired or user context absent",
                }
            )

        limit = max(1, min(int(limit), _MAX_LIMIT))
        ticker: str | None = None
        if symbol:
            ticker = symbol.strip().upper()
            if not ticker:
                ticker = None

        try:
            # Lazy-import to avoid coupling intel/ to ingest/ at module load time.
            from ....ingest.store import IngestStore  # type: ignore[import]

            IngestStore(self._db)  # ensures schema; idempotent

            if ticker:
                rows = self._db.query(
                    "SELECT source_id, ticker, text, url, ts FROM raw_items "
                    "WHERE user_id = ? AND ticker = ? "
                    "ORDER BY fetched_at DESC, id DESC LIMIT ?",
                    (self.owner_user_id, ticker, limit),
                )
            else:
                rows = self._db.query(
                    "SELECT source_id, ticker, text, url, ts FROM raw_items "
                    "WHERE user_id = ? "
                    "ORDER BY fetched_at DESC, id DESC LIMIT ?",
                    (self.owner_user_id, limit),
                )

            items = [
                {
                    "title": str(r["text"]),
                    "source": str(r["source_id"]),
                    "url": str(r["url"]),
                    "ts": str(r["ts"]),
                    "ticker": r["ticker"],
                }
                for r in rows
            ]
            return self._ok({"items": items, "symbol": ticker})

        except Exception as exc:
            return self._err("internal", f"news fetch failed: {exc}")
