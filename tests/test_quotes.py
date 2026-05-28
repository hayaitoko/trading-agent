"""Tests for GET /api/quotes — batched latest-quote endpoint.

Covers: bench-only source (no history), history-only source (no bench), both
sources joined (price from bench, prev close from history), graceful-empty
when neither is attached, parsing of the ``?symbols=`` query, auth, caps.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.data.history import Bar, HistoryService
from trading_agent.web.app import create_cockpit_app


class _Bench:
    """Stand-in for the real Bench — only the ``snapshot()`` accessor is read."""

    def __init__(self, prices: dict[str, float], *, ts: str = "2026-05-28T15:00:00Z") -> None:
        self._prices = prices
        self._ts = ts

    def snapshot(self) -> dict[str, object]:
        return {
            "generated_at": self._ts,
            "last_prices": dict(self._prices),
        }


class _Bars:
    def __init__(self, table: dict[str, list[Bar]]) -> None:
        self._table = table
        self.calls: list[tuple[str, str, int]] = []

    def get_bars(self, symbol: str, timeframe: str, lookback: int) -> list[Bar]:
        self.calls.append((symbol, timeframe, lookback))
        return list(self._table.get(symbol, []))


def _two_bars(prev_close: float, last_close: float) -> list[Bar]:
    return [
        Bar("2026-05-27T16:00:00Z", prev_close, prev_close, prev_close, prev_close, 0),
        Bar("2026-05-28T16:00:00Z", last_close, last_close, last_close, last_close, 0),
    ]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "lukas", "password": "pw"})
    return c


def test_quotes_from_bench_with_history_change_pct(client: TestClient) -> None:
    client.app.state.bench = _Bench({"AAPL": 200.0, "MSFT": 400.0})
    client.app.state.history = HistoryService(
        _Bars({"AAPL": _two_bars(190.0, 199.0), "MSFT": _two_bars(380.0, 401.0)})
    )
    r = client.get("/api/quotes", params={"symbols": "AAPL,MSFT"})
    assert r.status_code == 200
    body = r.json()
    by_sym = {q["symbol"]: q for q in body["quotes"]}
    # Price is the bench's live tick (200.0), not the daily close (199.0).
    assert by_sym["AAPL"]["price"] == 200.0
    # change_pct uses bench price vs prev daily close.
    assert by_sym["AAPL"]["change_pct"] == pytest.approx((200.0 - 190.0) / 190.0 * 100.0)
    assert by_sym["AAPL"]["ts"] == "2026-05-28T15:00:00Z"
    assert by_sym["MSFT"]["price"] == 400.0
    assert by_sym["MSFT"]["change_pct"] == pytest.approx((400.0 - 380.0) / 380.0 * 100.0)


def test_quotes_history_only_fallback(client: TestClient) -> None:
    """No bench — price falls back to the latest daily close."""
    client.app.state.history = HistoryService(_Bars({"AAPL": _two_bars(190.0, 199.0)}))
    r = client.get("/api/quotes", params={"symbols": "AAPL"})
    assert r.status_code == 200
    q = r.json()["quotes"][0]
    assert q["symbol"] == "AAPL"
    assert q["price"] == 199.0
    assert q["change_pct"] == pytest.approx((199.0 - 190.0) / 190.0 * 100.0)
    assert q["ts"] == "2026-05-28T16:00:00Z"


def test_quotes_bench_only_no_change_pct(client: TestClient) -> None:
    """Bench tick with no history → price present, change_pct null."""
    client.app.state.bench = _Bench({"AAPL": 200.0})
    r = client.get("/api/quotes", params={"symbols": "AAPL"})
    assert r.status_code == 200
    q = r.json()["quotes"][0]
    assert q["price"] == 200.0
    assert q["change_pct"] is None


def test_quotes_unknown_symbol_returns_nulls(client: TestClient) -> None:
    """A symbol nothing has data for is included with null fields, not dropped."""
    client.app.state.history = HistoryService(_Bars({}))
    r = client.get("/api/quotes", params={"symbols": "NOPE"})
    assert r.status_code == 200
    q = r.json()["quotes"][0]
    assert q == {"symbol": "NOPE", "price": None, "change_pct": None, "ts": None}


def test_quotes_no_sources_graceful_empty(client: TestClient) -> None:
    """Neither bench nor history attached — every quote has null fields."""
    r = client.get("/api/quotes", params={"symbols": "AAPL,MSFT"})
    assert r.status_code == 200
    quotes = r.json()["quotes"]
    assert [q["symbol"] for q in quotes] == ["AAPL", "MSFT"]
    assert all(q["price"] is None for q in quotes)


def test_quotes_dedupes_and_uppercases(client: TestClient) -> None:
    client.app.state.bench = _Bench({"AAPL": 200.0})
    r = client.get("/api/quotes", params={"symbols": "aapl,AAPL, ,MSFT"})
    assert r.status_code == 200
    syms = [q["symbol"] for q in r.json()["quotes"]]
    assert syms == ["AAPL", "MSFT"]  # deduped, uppercase, blanks dropped


def test_quotes_history_error_swallowed(client: TestClient) -> None:
    """A failing history provider degrades change_pct, not the whole request."""

    class _Boom:
        def get_bars(self, *_args: object, **_kwargs: object) -> list[Bar]:
            raise RuntimeError("alpaca down")

    client.app.state.bench = _Bench({"AAPL": 200.0})
    client.app.state.history = HistoryService(_Boom())
    r = client.get("/api/quotes", params={"symbols": "AAPL"})
    assert r.status_code == 200
    q = r.json()["quotes"][0]
    assert q["price"] == 200.0
    assert q["change_pct"] is None


def test_quotes_missing_param_400(client: TestClient) -> None:
    assert client.get("/api/quotes").status_code == 400
    assert client.get("/api/quotes", params={"symbols": ""}).status_code == 400
    assert client.get("/api/quotes", params={"symbols": ", , "}).status_code == 400


def test_quotes_invalid_symbol_400(client: TestClient) -> None:
    assert client.get("/api/quotes", params={"symbols": "AAPL,@@@"}).status_code == 400


def test_quotes_too_many_400(client: TestClient) -> None:
    syms = ",".join(f"SYM{i}" for i in range(60))
    assert client.get("/api/quotes", params={"symbols": syms}).status_code == 400


def test_quotes_requires_auth(tmp_path: Path) -> None:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    c = TestClient(app)
    assert c.get("/api/quotes", params={"symbols": "AAPL"}).status_code == 401
