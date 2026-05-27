"""WS-J market-data router tests: real price history + fundamentals endpoints.

No network — a fake BarProvider / FundamentalsProvider stands in for Alpaca /
Finnhub so the route logic (range mapping, shapes, fail-loud on missing
provider) is exercised offline. The principle under test: data routes never
fabricate numbers — absent provider → 503, unknown symbol → 404.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_agent.config.db import Database
from trading_agent.data.history import Bar, HistoryProviderError, HistoryService
from trading_agent.web.app import create_cockpit_app


class FakeBars:
    """Records the (timeframe, lookback) it was asked for; returns canned bars."""

    def __init__(self, *, raise_for: str | None = None) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self._raise_for = raise_for

    def get_bars(self, symbol: str, timeframe: str, lookback: int) -> list[Bar]:
        self.calls.append((symbol, timeframe, lookback))
        if symbol == self._raise_for:
            raise HistoryProviderError("alpaca keys missing")
        return [
            Bar("2026-05-26T09:30:00", 100.0, 101.0, 99.5, 100.5, 1000),
            Bar("2026-05-26T09:35:00", 100.5, 102.0, 100.0, 101.5, 1200),
            Bar("2026-05-26T09:40:00", 101.5, 103.0, 101.0, 102.0, 900),
        ]


class FakeFundamentals:
    def get_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        if symbol == "NOPE":
            return None
        return {"name": "Apple Inc.", "sector": "Tech", "market_cap": 3.2e12, "pe": 31.5}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    app.state.history = HistoryService(FakeBars(), fundamentals_provider=FakeFundamentals())
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "lukas", "password": "pw"})
    return c


def test_history_returns_real_bars_and_change(client: TestClient) -> None:
    r = client.get("/api/history/AAPL", params={"range": "1D"})
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "AAPL"
    assert body["timeframe"] == "5Min"
    assert len(body["bars"]) == 3
    assert body["bars"][0] == {"t": "2026-05-26T09:30:00", "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 1000.0}
    assert body["first"] == 100.5 and body["last"] == 102.0
    assert body["change"] == pytest.approx(1.5)
    assert body["change_pct"] == pytest.approx(1.5 / 100.5 * 100)


def test_history_range_maps_to_timeframe(client: TestClient) -> None:
    client.get("/api/history/AAPL", params={"range": "1Y"})
    bars: FakeBars = client.app.state.history.bar_provider
    assert ("AAPL", "1D", 252) in bars.calls


def test_history_ytd_is_dynamic(client: TestClient) -> None:
    r = client.get("/api/history/AAPL", params={"range": "YTD"})
    assert r.status_code == 200
    assert r.json()["timeframe"] == "1D"


def test_history_bad_range_400(client: TestClient) -> None:
    assert client.get("/api/history/AAPL", params={"range": "7Q"}).status_code == 400


def test_history_invalid_symbol_400(client: TestClient) -> None:
    assert client.get("/api/history/@@@").status_code == 400


def test_history_provider_error_is_503_not_fake(client: TestClient) -> None:
    client.app.state.history = HistoryService(FakeBars(raise_for="AAPL"))
    assert client.get("/api/history/AAPL").status_code == 503


def test_history_no_service_503(tmp_path: Path) -> None:
    app = create_cockpit_app(Database(tmp_path / "c.db"))  # no app.state.history
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "u", "password": "pw"})
    assert c.get("/api/history/AAPL").status_code == 503


def test_fundamentals_returns_real_data(client: TestClient) -> None:
    r = client.get("/api/fundamentals/AAPL")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "AAPL" and body["name"] == "Apple Inc." and body["pe"] == 31.5


def test_fundamentals_no_provider_503(tmp_path: Path) -> None:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    app.state.history = HistoryService(FakeBars())  # no fundamentals provider
    c = TestClient(app)
    c.post("/api/auth/signup", json={"username": "u", "password": "pw"})
    assert c.get("/api/fundamentals/AAPL").status_code == 503


def test_fundamentals_unknown_symbol_404(client: TestClient) -> None:
    assert client.get("/api/fundamentals/NOPE").status_code == 404


def test_market_routes_require_auth(tmp_path: Path) -> None:
    app = create_cockpit_app(Database(tmp_path / "c.db"))
    app.state.history = HistoryService(FakeBars(), fundamentals_provider=FakeFundamentals())
    c = TestClient(app)  # not logged in
    assert c.get("/api/history/AAPL").status_code == 401
    assert c.get("/api/fundamentals/AAPL").status_code == 401
