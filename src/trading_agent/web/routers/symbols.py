"""Symbols router: server-side ticker search for the cockpit's quote-search tile.

The frontend used to fuzzy-match against a bundled ``static/data/symbols.json``
client-side; this router promotes that universe to a real HTTP endpoint so a
single server-owned list is the source of truth and the search can be reused
by other surfaces (manager tool-calls, etc.).

The universe loads from ``static/data/symbols.json`` at app start (the same
file the frontend ships) — *path-only*, never modified here. Search is a
prefix-first match on symbol, then case-insensitive substring on symbol/name,
capped at 25 results.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from ...config.users import current_user

router = APIRouter(tags=["symbols"])

# The universe ships with the frontend (single source of truth across UI + API).
_SYMBOLS_PATH = Path(__file__).resolve().parents[1] / "static" / "data" / "symbols.json"
_MAX_RESULTS = 25

# Cache the loaded universe — the file is small but we hit this per-keystroke.
_universe_lock = threading.Lock()
_universe_cache: list[dict[str, str]] | None = None


def _load_universe() -> list[dict[str, str]]:
    """Read the bundled symbols.json once and normalize to ``{symbol, name}``.

    The on-disk shape is ``{"s": <ticker>, "n": <name>, "x": <exchange>}``;
    we expose ``symbol`` + ``name`` (and pass ``exchange`` through for callers
    that want it). Missing file → empty universe (search returns []).
    """
    global _universe_cache
    with _universe_lock:
        if _universe_cache is not None:
            return _universe_cache
        try:
            raw = json.loads(_SYMBOLS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _universe_cache = []
            return _universe_cache
        items: list[dict[str, str]] = []
        for row in raw:
            sym = str(row.get("s") or "").strip().upper()
            if not sym:
                continue
            items.append(
                {
                    "symbol": sym,
                    "name": str(row.get("n") or "").strip(),
                    "exchange": str(row.get("x") or "").strip(),
                }
            )
        _universe_cache = items
        return _universe_cache


def _reset_cache_for_tests() -> None:
    """Tests that monkey-patch ``_SYMBOLS_PATH`` call this to force reload."""
    global _universe_cache
    with _universe_lock:
        _universe_cache = None


def _match(query: str, universe: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prefix-first symbol match, then case-insensitive substring on symbol/name.

    Ordering: exact-symbol > symbol-prefix > symbol-substring > name-substring.
    Stable within tiers (preserves the universe's curated S&P-weight ordering).
    Empty/whitespace query returns the first ``_MAX_RESULTS`` of the universe
    so the UI's "browse" interaction still has something to show.
    """
    q = (query or "").strip().upper()
    if not q:
        return universe[:_MAX_RESULTS]

    exact: list[dict[str, str]] = []
    prefix: list[dict[str, str]] = []
    sym_sub: list[dict[str, str]] = []
    name_sub: list[dict[str, str]] = []
    seen: set[str] = set()

    def _take(bucket: list[dict[str, str]], row: dict[str, str]) -> None:
        if row["symbol"] in seen:
            return
        seen.add(row["symbol"])
        bucket.append(row)

    q_lower = q.lower()
    for row in universe:
        sym = row["symbol"]
        name_lower = row["name"].lower()
        if sym == q:
            _take(exact, row)
        elif sym.startswith(q):
            _take(prefix, row)
        elif q in sym:
            _take(sym_sub, row)
        elif q_lower in name_lower:
            _take(name_sub, row)

    return (exact + prefix + sym_sub + name_sub)[:_MAX_RESULTS]


@router.get("/api/symbols")
def search_symbols(
    q: str = "",
    user_id: str = Depends(current_user),
) -> dict[str, Any]:
    """Search the bundled universe for ``q`` (prefix-first, then substring).

    Returns ``{"results": [{"symbol", "name", "exchange"}], "query": q}`` capped
    at 25. Auth-required (same as every cockpit route).
    """
    universe = _load_universe()
    return {"query": q, "results": _match(q, universe)}
