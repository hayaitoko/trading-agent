"""Tests for GET /api/movers — top gainers/losers ranked by |Δ%|."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.data.history import Bar, HistoryService
from trading_agent.web.app import create_cockpit_app


class _Bench:
    def __init__(
        self,
        prices: dict[str, float],
        *,
        symbols: list[str] | None = None,
    ) -> None:
        self._prices = prices
        self._symbols = symbols if symbols is not None else list(prices)

    def snapshot(self) -> dict[str, object]:
        return {
            "symbols": list(self._symbols),
            "last_prices": dict(self._prices),
        }


class _Bars:
    def __init__(self, table: dict[str, list[Bar]]) -> None:
        self._table = table

    def get_bars(self, symbol: str, timeframe: str, lookback: int) -> list[Bar]:
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


def _wire(client: TestClient, prices: dict[str, float], history_table: dict[str, list[Bar]]) -> None:
    client.app.state.bench = _Bench(prices)
    client.app.state.history = HistoryService(_Bars(history_table))


def test_movers_ranks_by_absolute_change(client: TestClient) -> None:
    _wire(
        client,
        prices={"AAPL": 210.0, "MSFT": 380.0, "TSLA": 250.0},
        history_table={
            "AAPL": _two_bars(200.0, 209.0),   # +5% (using bench price 210 vs prev 200)
            "MSFT": _two_bars(400.0, 381.0),   # −5%
            "TSLA": _two_bars(250.0, 252.5),   # 0% with bench price
        },
    )
    r = client.get("/api/movers")
    assert r.status_code == 200
    body = r.json()
    assert body["direction"] == "all"
    syms = [m["symbol"] for m in body["movers"]]
    # AAPL +5%, MSFT −5% both tied on |Δ%|, TSLA 0%. AAPL/MSFT should lead.
    assert set(syms[:2]) == {"AAPL", "MSFT"}
    assert syms[-1] == "TSLA"


def test_movers_direction_up_filter(client: TestClient) -> None:
    _wire(
        client,
        prices={"AAPL": 210.0, "MSFT": 380.0},
        history_table={
            "AAPL": _two_bars(200.0, 209.0),
            "MSFT": _two_bars(400.0, 381.0),
        },
    )
    r = client.get("/api/movers", params={"direction": "up"})
    assert r.status_code == 200
    syms = [m["symbol"] for m in r.json()["movers"]]
    assert syms == ["AAPL"]


def test_movers_direction_down_filter(client: TestClient) -> None:
    _wire(
        client,
        prices={"AAPL": 210.0, "MSFT": 380.0},
        history_table={
            "AAPL": _two_bars(200.0, 209.0),
            "MSFT": _two_bars(400.0, 381.0),
        },
    )
    r = client.get("/api/movers", params={"direction": "down"})
    assert r.status_code == 200
    syms = [m["symbol"] for m in r.json()["movers"]]
    assert syms == ["MSFT"]


def test_movers_limit_caps_result(client: TestClient) -> None:
    _wire(
        client,
        prices={f"S{i}": 100.0 + i for i in range(20)},
        history_table={f"S{i}": _two_bars(100.0, 100.0 + i) for i in range(20)},
    )
    r = client.get("/api/movers", params={"limit": 3})
    assert r.status_code == 200
    assert len(r.json()["movers"]) == 3


def test_movers_symbol_override(client: TestClient) -> None:
    """Caller-supplied universe overrides the bench's tracked list."""
    _wire(
        client,
        prices={"AAPL": 210.0, "MSFT": 380.0, "NVDA": 1100.0},
        history_table={
            "AAPL": _two_bars(200.0, 209.0),
            "MSFT": _two_bars(400.0, 381.0),
            "NVDA": _two_bars(1000.0, 1100.0),
        },
    )
    r = client.get("/api/movers", params={"symbols": "NVDA"})
    assert r.status_code == 200
    syms = [m["symbol"] for m in r.json()["movers"]]
    assert syms == ["NVDA"]


def test_movers_drops_symbols_without_prev_close(client: TestClient) -> None:
    _wire(
        client,
        prices={"AAPL": 210.0, "ZZZZ": 5.0},
        history_table={"AAPL": _two_bars(200.0, 209.0)},  # ZZZZ has no bars
    )
    r = client.get("/api/movers")
    assert r.status_code == 200
    syms = [m["symbol"] for m in r.json()["movers"]]
    assert syms == ["AAPL"]


def test_movers_no_bench_no_universe_empty(client: TestClient) -> None:
    r = client.get("/api/movers")
    assert r.status_code == 200
    assert r.json() == {"direction": "all", "movers": []}


def test_movers_validates_params(client: TestClient) -> None:
    assert client.get("/api/movers", params={"limit": 0}).status_code == 400
    assert client.get("/api/movers", params={"limit": 9999}).status_code == 400
    assert client.get("/api/movers", params={"direction": "sideways"}).status_code == 400


def test_movers_requires_auth(tmp_path: Path) -> None:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    c = TestClient(app)
    assert c.get("/api/movers").status_code == 401
