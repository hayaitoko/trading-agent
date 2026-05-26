"""WS-A · Data & history tests.

Covers the HistoryService (typed bars + fundamentals, TTL caching, the
budget-bounded context block), the provider adapters (Alpaca with an injected
fake client; Finnhub/Polygon over ``httpx.MockTransport`` — no live calls), the
``build_history_service`` resolution, and the optional injection into LLMTrader
(richer block when present, 30-close fallback otherwise).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from trading_agent.data.history import (
    PRESETS,
    Bar,
    HistoryDepth,
    HistoryService,
    build_history_service,
    resolve_depth,
)
from trading_agent.data.providers.alpaca import AlpacaBarProvider
from trading_agent.data.providers.finnhub import FinnhubProvider
from trading_agent.data.providers.polygon import PolygonProvider
from trading_agent.llm.openrouter import ChatResult
from trading_agent.llm.trader import LLMTrader

# --- fakes -------------------------------------------------------------------


class FakeBarProvider:
    """Deterministic synthetic bars; records calls so caching can be asserted."""

    def __init__(self, empty: bool = False) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.empty = empty

    def get_bars(self, symbol: str, timeframe: str, lookback: int) -> list[Bar]:
        self.calls.append((symbol, timeframe, lookback))
        if self.empty:
            return []
        return [
            Bar(
                timestamp=f"2026-01-{(i % 28) + 1:02d}T10:30:00+00:00",
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.0 + i,
                volume=1000.0 + i,
            )
            for i in range(lookback)
        ]


class FakeFundamentals:
    def __init__(self, data: dict[str, Any] | None) -> None:
        self.data = data
        self.calls = 0

    def get_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        self.calls += 1
        return self.data


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class FakeChatClient:
    """Duck-typed chat client capturing the messages it is handed."""

    def __init__(self, content: str = "{}") -> None:
        self.content = content
        self.messages: list[dict[str, str]] = []

    def chat(self, model: str, messages: list[dict[str, str]], **kw: Any) -> ChatResult:
        self.messages = messages
        return ChatResult(content=self.content, model=model, usage={})


ACCOUNT = {"cash": 12345.0, "positions": [{"symbol": "AAPL", "qty": 3}]}


# --- bars + caching ----------------------------------------------------------


def test_bars_return_typed_and_are_cached() -> None:
    clock = _Clock()
    provider = FakeBarProvider()
    svc = HistoryService(provider, time_fn=clock, bar_ttl=100.0)

    first = svc.bars("AAPL", "1D", 5)
    assert len(first) == 5 and all(isinstance(b, Bar) for b in first)
    # second call within TTL is served from cache (provider not hit again)
    svc.bars("AAPL", "1D", 5)
    assert provider.calls == [("AAPL", "1D", 5)]


def test_bars_cache_expires_after_ttl() -> None:
    clock = _Clock()
    provider = FakeBarProvider()
    svc = HistoryService(provider, time_fn=clock, bar_ttl=100.0)
    svc.bars("AAPL", "1D", 5)
    clock.t += 101.0  # past TTL
    svc.bars("AAPL", "1D", 5)
    assert len(provider.calls) == 2


# --- fundamentals ------------------------------------------------------------


def test_fundamentals_none_without_provider() -> None:
    svc = HistoryService(FakeBarProvider())
    assert svc.fundamentals("AAPL") is None


def test_fundamentals_returns_dict_and_caches() -> None:
    fund = FakeFundamentals({"symbol": "AAPL", "pe": 30.0})
    svc = HistoryService(FakeBarProvider(), fundamentals_provider=fund, fundamentals_ttl=1e9)
    assert svc.fundamentals("AAPL") == {"symbol": "AAPL", "pe": 30.0}
    svc.fundamentals("AAPL")
    assert fund.calls == 1  # cached


# --- context block -----------------------------------------------------------


def test_context_block_has_long_recent_fundamentals_and_fits_budget() -> None:
    fund = FakeFundamentals(
        {"symbol": "AAPL", "name": "Apple", "pe": 30.0, "market_cap": 3.0e12}
    )
    svc = HistoryService(FakeBarProvider(), fundamentals_provider=fund, depth="standard")
    block = svc.context_block(["AAPL", "MSFT"], ACCOUNT)

    assert "Cash available: 12,345.00" in block
    assert "Tradable symbols: AAPL, MSFT" in block
    assert "=== AAPL ===" in block and "=== MSFT ===" in block
    assert "Long view" in block and "Downsampled closes" in block
    assert "Recent (" in block
    assert "Fundamentals:" in block and "P/E=30.00" in block
    assert block.rstrip().endswith("Return your JSON decision now.")
    # stays within a sane token budget at default depth
    assert len(block) <= svc.max_chars


def test_context_block_downsamples_long_view() -> None:
    depth = HistoryDepth(long_timeframe="1D", long_lookback=300, downsample_long_to=25,
                         recent_timeframe="1D", recent_lookback=5)
    svc = HistoryService(FakeBarProvider(), depth=depth)
    block = svc.context_block(["AAPL"], ACCOUNT)
    # the "Downsampled closes: [...]" line must carry no more than N points
    line = next(ln for ln in block.splitlines() if "Downsampled closes" in ln)
    closes = json.loads(line.split(":", 1)[1].strip())
    assert len(closes) <= 25 < 300


def test_context_block_handles_empty_history() -> None:
    svc = HistoryService(FakeBarProvider(empty=True))
    block = svc.context_block(["AAPL"], ACCOUNT)
    assert "(no history available)" in block


def test_context_block_budget_trims_when_oversized() -> None:
    syms = ["AAPL", "MSFT", "NVDA"]
    untrimmed = HistoryService(FakeBarProvider(), depth="deep", max_chars=10**9)
    big = untrimmed.context_block(syms, ACCOUNT)
    # A tight budget forces the refit loop to shrink the recent window/long view.
    tight = HistoryService(FakeBarProvider(), depth="deep", max_chars=800)
    small = tight.context_block(syms, ACCOUNT)
    assert len(small) < len(big)
    assert "Return your JSON decision now." in small  # still well-formed


def test_depth_presets_scale_output() -> None:
    shallow = HistoryService(FakeBarProvider(), depth="shallow").context_block(["AAPL"], ACCOUNT)
    deep = HistoryService(FakeBarProvider(), depth="deep").context_block(["AAPL"], ACCOUNT)
    assert len(deep) > len(shallow)


def test_resolve_depth_variants() -> None:
    assert resolve_depth("deep") is PRESETS["deep"]
    assert resolve_depth("nonsense") is PRESETS["standard"]
    custom = resolve_depth({"recent_lookback": 7})
    assert custom.recent_lookback == 7
    assert custom.long_lookback == PRESETS["standard"].long_lookback  # untouched
    assert resolve_depth(PRESETS["off"]) is PRESETS["off"]


# --- LLMTrader injection -----------------------------------------------------


class _SentinelHistory:
    def context_block(self, symbols: Any, account: dict[str, Any]) -> str:
        return "RICH-CONTEXT-BLOCK"


def test_trader_uses_history_when_injected() -> None:
    client = FakeChatClient()
    trader = LLMTrader("m", client, symbols=["AAPL"], history=_SentinelHistory())  # type: ignore[arg-type]
    trader.decide({"cash": 1000, "positions": []})
    assert client.messages[1]["content"] == "RICH-CONTEXT-BLOCK"


def test_trader_falls_back_to_closes_without_history() -> None:
    client = FakeChatClient()
    trader = LLMTrader("m", client, symbols=["AAPL"], lookback=5)
    for px in (100, 101, 102):
        trader.observe({"symbol": "AAPL", "close": px})
    trader.decide({"cash": 1000, "positions": []})
    user_msg = client.messages[1]["content"]
    assert "close prices" in user_msg and "102" in user_msg
    assert "RICH-CONTEXT-BLOCK" not in user_msg


def test_trader_uses_real_history_service_end_to_end() -> None:
    client = FakeChatClient()
    svc = HistoryService(FakeBarProvider(), depth="shallow")
    trader = LLMTrader("m", client, symbols=["AAPL"], history=svc)
    trader.decide(ACCOUNT)
    assert "Long view" in client.messages[1]["content"]


# --- Alpaca adapter (offline, injected fake client) --------------------------


class _FakeBarRow:
    def __init__(self, i: int) -> None:
        from datetime import UTC, datetime

        self.timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        self.open = 10.0 + i
        self.high = 11.0 + i
        self.low = 9.0 + i
        self.close = 10.5 + i
        self.volume = 100 + i


class _FakeBarSet:
    def __init__(self, symbol: str, n: int) -> None:
        self.data = {symbol: [_FakeBarRow(i) for i in range(n)]}


class _FakeDataClient:
    def __init__(self, n: int) -> None:
        self.n = n
        self.requests: list[Any] = []

    def get_stock_bars(self, request: Any) -> _FakeBarSet:
        self.requests.append(request)
        return _FakeBarSet("AAPL", self.n)


def test_alpaca_provider_maps_and_slices_to_lookback() -> None:
    client = _FakeDataClient(n=50)
    provider = AlpacaBarProvider(data_client=client)
    bars = provider.get_bars("AAPL", "1D", 10)
    assert len(bars) == 10  # sliced to the tail of 50
    assert all(isinstance(b, Bar) for b in bars)
    assert bars[-1].close == 10.5 + 49


def test_alpaca_provider_rejects_unknown_timeframe() -> None:
    provider = AlpacaBarProvider(data_client=_FakeDataClient(n=1))
    with pytest.raises(Exception):
        provider.get_bars("AAPL", "3Y", 10)


def test_alpaca_provider_zero_lookback_is_empty() -> None:
    provider = AlpacaBarProvider(data_client=_FakeDataClient(n=5))
    assert provider.get_bars("AAPL", "1D", 0) == []


# --- Finnhub / Polygon adapters (offline via MockTransport) ------------------


def _finnhub_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/profile2"):
            return httpx.Response(
                200, json={"name": "Apple Inc", "finnhubIndustry": "Technology",
                           "marketCapitalization": 3_000_000.0}
            )
        return httpx.Response(
            200,
            json={"metric": {"peTTM": 29.5, "epsTTM": 6.1,
                             "dividendYieldIndicatedAnnual": 0.5, "beta": 1.2,
                             "52WeekHigh": 200.0, "52WeekLow": 120.0}},
        )

    return httpx.MockTransport(handler)


def test_finnhub_provider_maps_canonical_fields() -> None:
    provider = FinnhubProvider(api_key="fh-key", transport=_finnhub_transport())
    fund = provider.get_fundamentals("AAPL")
    assert fund is not None
    assert fund["name"] == "Apple Inc"
    assert fund["sector"] == "Technology"
    assert fund["pe"] == 29.5 and fund["eps"] == 6.1
    assert fund["week52_high"] == 200.0
    assert fund["source"] == "finnhub"


def test_finnhub_provider_none_without_key() -> None:
    assert FinnhubProvider(api_key="").get_fundamentals("AAPL") is None


def test_finnhub_provider_none_on_http_error() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(403, json={"error": "no"}))
    provider = FinnhubProvider(api_key="fh-key", transport=transport)
    assert provider.get_fundamentals("AAPL") is None


def test_polygon_provider_maps_reference_data() -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"results": {"name": "Microsoft", "sic_description": "Software",
                              "market_cap": 2.5e12}},
        )
    )
    provider = PolygonProvider(api_key="pg-key", transport=transport)
    fund = provider.get_fundamentals("MSFT")
    assert fund is not None
    assert fund["name"] == "Microsoft" and fund["sector"] == "Software"
    assert fund["market_cap"] == 2.5e12
    assert fund["pe"] is None  # not on the free reference endpoint
    assert fund["source"] == "polygon"


def test_polygon_provider_none_without_key() -> None:
    assert PolygonProvider(api_key="").get_fundamentals("MSFT") is None


# --- build_history_service resolution ----------------------------------------


class _FakeSettings:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def get(self, user_id: str, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def test_build_history_service_resolves_depth_and_fundamentals_from_settings() -> None:
    settings = _FakeSettings({"history_depth": "deep", "fundamentals_provider": "finnhub"})
    svc = build_history_service(
        settings=settings,  # type: ignore[arg-type]
        user_id="u1",
        bar_provider=FakeBarProvider(),  # keep offline; don't construct Alpaca
    )
    assert svc.depth is PRESETS["deep"]
    assert isinstance(svc.fundamentals_provider, FinnhubProvider)


def test_build_history_service_defaults_when_no_settings() -> None:
    svc = build_history_service(bar_provider=FakeBarProvider())
    assert svc.depth is PRESETS["standard"]
    assert svc.fundamentals_provider is None
