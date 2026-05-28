# Situation Enrichment + Forecast Surface

**Status:** WS-Situation A0 ✅ · A1 ✅ · A2 ✅ · B0 🔵 · B1 🔵 · C0 🔵 · C1 🔵  
**Branch:** `feat/engine-realism`  
**Legend:** ✅ exists · 🟡 partial · 🔵 planned  
**Plan reference:** `~/.claude/plans/helm-situation-forecast.md`

---

## 1. Purpose

The existing situation layer (P3) gives each trader a regime label + social
metrics + calendar events.  This workstream adds **seven new data sources**
(GDELT macro tone, Polymarket + Kalshi forward odds, Alpaca-native options IV,
Substack/SeekingAlpha RSS, Bluesky list/author feeds) and surfaces a
**forecast cone** combining empirical realized vol, options IV, and
prediction-market implied probabilities.

Design principle: **tools, not stuffed context.**  Each new source is a LOOK
tool the trader calls when it decides it needs that information — not context
injected unconditionally into every turn.  This makes the data cheaper (tokens
only when used), more agentic (the trader chooses what to look at), and more
measurable (tool-call provenance is recorded in the turn trace and consumed
by the P6 calibration experiment).

**Zero new credentials.**  Every source verified key-free in the recon report
(`~/.claude/plans/helm-situation-recon.md`).  Existing `ALPACA_API_KEY`
covers the options IV upgrade.  Default-off feature flags on every new path —
flag-off-during-failure path logged but never crashes the situation block.

---

## 2. Component map

| Component | File | Status |
|---|---|---|
| **GDELT provider** | `src/trading_agent/data/providers/gdelt.py` | ✅ A0 |
| **Prediction markets provider** | `src/trading_agent/data/providers/prediction_markets.py` | ✅ A1 |
| **OptionQuote IV + Greeks** | `src/trading_agent/instruments/options.py` | ✅ A2 |
| **Alpaca snapshot IV passthrough** | `src/trading_agent/instruments/options_chain.py` | ✅ A2 |
| **world_events LOOK tool** | `src/trading_agent/intel/tools/look/world_events.py` | ✅ A0+wiring |
| **prediction_market_odds LOOK tool** | `src/trading_agent/intel/tools/look/prediction_market_odds.py` | ✅ A1+wiring |
| **options_iv LOOK tool** | `src/trading_agent/intel/tools/look/options_iv.py` | ✅ A2+wiring |
| **forecast LOOK tool** | `src/trading_agent/intel/tools/look/forecast.py` | 🔵 Track C |
| Substack RSS (config seed) | `config/ingest_sources.*` | 🔵 Track B |
| Bluesky list/author fetcher | `src/trading_agent/ingest/fetchers/bluesky.py` | 🔵 Track B |
| Situation layer integration | `src/trading_agent/intel/situation.py` | 🔵 Track C |
| Forecast cone compute | `src/trading_agent/intel/forecast.py` | 🔵 Track C |
| Forecast API router | `src/trading_agent/web/routers/forecast.py` | 🔵 Track C |
| Cockpit forecast tile | `web/static/cockpit.html` | 🔵 Track C |

---

## 3. Data Sources

### 3a. GDELT — Macro/geopolitical regime feed

**Status:** ✅ A0  
**File:** `src/trading_agent/data/providers/gdelt.py`

**Purpose:** 15-minute macro/geopolitical signal from the GDELT Global
Knowledge Graph.  Three methods surface different facets of the same DOC 2.0
API: mention-volume velocity (regime acceleration), average tone trend
(sentiment direction), and top article headlines (qualitative context for the
world-events tile).

**Endpoint:** `https://api.gdeltproject.org/api/v2/doc/doc`  
**Auth:** None — fully public, no API key.  
**Cadence:** Updated every 15 minutes; rolling 3-month window.  
**Rate limits:** None published; GDELT is a public research service.  

**Provider API:**

```python
class GDELTProvider:
    def timeline_volume(self, theme: str, timespan: str = "24h") -> list[GDELTBin]: ...
    def timeline_tone(self, theme: str, timespan: str = "24h") -> list[GDELTBin]: ...
    def top_articles(self, theme: str, n: int = 10) -> list[GDELTArticle]: ...

@dataclass(frozen=True)
class GDELTBin:
    bucket_start: datetime   # UTC
    value: float             # volume count or tone average
    unit: str                # "mentions" | "tone"

@dataclass(frozen=True)
class GDELTArticle:
    title: str; url: str; published: datetime; source_domain: str; tone: float
```

**Caching:** In-process 900-second (15-min) cache keyed on
`(method, theme, timespan, n)`.  Matches GDELT's update cadence — no stale
amplification.

**Failure mode:** `GDELTProviderError` (subclass of `RuntimeError`) on any
network error or HTTP non-2xx.  Fail-loud per WS-J discipline.  The
`world_events` LOOK tool catches this and returns a structured
`ToolError(kind="network_error")` to the trader.

**GKG themes of interest for macro regime:**
- `WAR` — geopolitical conflict velocity
- `ELECTION` — electoral uncertainty
- `EPU_POLICY_*` — economic policy uncertainty (Fed, fiscal, trade)
- `NATURAL_DISASTER`, `ECON_BANKRUPTCY`, `ECON_DEBT` — tail risks

**Gating flag:** `SITUATION_GDELT` (user_settings, default `False`).

**Citations:**
- [GDELT DOC 2.0 blog](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- [GKG Codebook V2.1](http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf)

---

### 3b. Prediction Markets — Polymarket + Kalshi forward odds

**Status:** ✅ A1  
**File:** `src/trading_agent/data/providers/prediction_markets.py`

**Purpose:** Forward probabilities for macro events from two regulated/liquid
prediction market venues.  The `event_odds()` method reconciles where both
venues list the same event and returns a unified `EventOdds` list.

**Polymarket endpoints:**
- Gamma: `https://gamma-api.polymarket.com/events?closed=false&limit=N`
  (event discovery + nested market prices)
- CLOB: `https://clob.polymarket.com/price` (per-market precise pricing)  
**Auth:** None — public read endpoints.  Rate limits: 4000 req/10s overall;
`/events` 500/10s.  US IP geoblock applies only to order submission, not reads.

**Kalshi endpoint:**
- `https://external-api.kalshi.com/trade-api/v2/markets` (and `/events`)  
**Auth:** None for public market-data reads.  429 backoff applied.

**Liquidity filter:** Polymarket markets with `liquidity < $1,000` OR
`volume_24h < $500` are excluded — thin markets are easy to manipulate and
not reliable signals.

**Provider API:**

```python
class PredictionMarketsProvider:
    def event_odds(self, category: str, query: str | None = None,
                   *, min_liquidity: float = 1_000.0) -> list[EventOdds]: ...
    def by_id(self, venue: Literal["polymarket","kalshi"], event_id: str) -> EventOdds | None: ...

@dataclass(frozen=True)
class EventOdds:
    venue: Literal["polymarket","kalshi"]
    event_id: str; title: str; outcomes: list[str]
    prices: list[float]            # parallel to outcomes, 0.0–1.0
    liquidity: float; volume_24h: float
    end_date: datetime | None; restricted: bool
```

**Reconciliation logic:** When both venues list events with overlapping
keywords (e.g. both have a "Fed rate decision" event), the provider returns
both as separate `EventOdds` rows (venue-tagged) rather than silently merging
them.  The trader can observe the spread.

**Failure mode:** `PredictionMarketsProviderError` on network/4xx.
429 → exponential backoff (up to 3 retries, max 8 s).  Restricted events
(`restricted: true` on Polymarket) are skipped without crashing.

**Caching:** 60-second in-process cache (prediction markets update continuously
but our polling cadence is at most per-turn, not sub-second).

**Gating flag:** `SITUATION_PREDICTION_MARKETS` (user_settings, default `False`).

**Citations:**
- [Polymarket CLOB auth docs](https://docs.polymarket.com/developers/CLOB/authentication)
- [Polymarket rate limits](https://docs.polymarket.com/quickstart/introduction/rate-limits)
- [Polymarket geoblock reference](https://docs.polymarket.com/api-reference/geoblock)
- [Kalshi quick start](https://docs.kalshi.com/getting_started/quick_start_market_data)
- [Kalshi rate limits](https://docs.kalshi.com/getting_started/rate_limits)

---

### 3c. Options Implied Volatility — Alpaca passthrough

**Status:** ✅ A2  
**File:** `src/trading_agent/instruments/options.py` + `options_chain.py`

**Purpose:** Surface per-contract IV and Greeks that Alpaca already computes
server-side.  Zero new math required — the `OptionsSnapshot` returned by
`OptionHistoricalDataClient` carries `implied_volatility` and `greeks`
(delta/gamma/theta/vega/rho); this phase simply passes them through into the
`OptionQuote` dataclass rather than discarding them.

**Additive fields on `OptionQuote`:**

```python
@dataclass
class OptionQuote:
    contract: OptionContract
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    implied_vol: float | None = None   # NEW — Alpaca OptionsSnapshot.implied_volatility
    greeks: dict[str, float] | None = None  # NEW — {"delta","gamma","theta","vega","rho"}
```

**No breaking change** — all existing callers of `OptionQuote` are unaffected
(new fields default `None`).

**Newton-Raphson IV solver:** explicitly deferred per recon §4.  Alpaca's
paper feed returns `implied_volatility` populated on the indicative snapshot.
Build the fallback BS-inversion solver only if real-data testing shows IV is
`None` on the indicative feed.

**Auth:** Uses existing `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`.  Available on
the free paper-trading tier without OPRA subscription.

**Gating flag:** `SITUATION_OPTIONS_IV` (user_settings, default `False`).

**Citations:**
- [Alpaca-py option models](https://alpaca.markets/sdks/python/api_reference/data/models.html)
- [Alpaca options trading docs](https://docs.alpaca.markets/us/docs/options-trading)

---

## 4. Feature Flags (operator reference)

| Flag key | Type | Default | Effect when ON | Dependencies |
|---|---|---|---|---|
| `SITUATION_GDELT` | bool | `False` | Enables `world_events` LOOK tool; `GDELTProvider` constructed once per turn | None |
| `SITUATION_PREDICTION_MARKETS` | bool | `False` | Enables `prediction_market_odds` LOOK tool | None |
| `SITUATION_OPTIONS_IV` | bool | `False` | Enables `options_iv` LOOK tool; requires `ALPACA_API_KEY` | Options chain provider |

All flags live in `user_settings` (see `src/trading_agent/config/settings_store.py`).

**Default-off rationale:** Per WS-J and Track A plan discipline, new data
sources ship disabled.  The byte-identical-prompt test on C0 is the gate
that proves flag-off leaves the existing situation block unchanged.

**Turning a flag ON:** Update user_settings via the settings API or directly:
```
PATCH /api/settings  {"SITUATION_GDELT": true}
```

**Turning a flag OFF mid-session:** The tool immediately returns
`ToolError(kind="disabled", message="enable in trader settings")` — no
cached data is leaked, no partial results.

---

## 5. Forecast Cone (Track C — planned 🔵)

_Section reserved for Track C.  The forecast cone combines:_

1. **Empirical realized vol** — from `HistoryService` 30-day normalized window
   (σ computed from log returns on OHLCV bars).
2. **Options IV** — annualized via `iv * sqrt(t / 252)` where `t` is the
   forward horizon in days; taken from `OptionQuote.implied_vol` (A2).
3. **Prediction-market implied move** — absolute fractional move implied by
   any `EventOdds` event that references the ticker (A1).

The cone is instrument-agnostic: for equity the full three-component blend
applies; for crypto (no options chain), `iv_sigma` is `None` and the cone
renders from empirical-σ plus any matching Polymarket BTC/ETH event.

_Anti-overconfidence note:_ the cone is an **envelope of historical outcomes
at 1σ**, not a single-line prediction.  The horizon boundaries are statistical
bands, not guarantees.  Track C ships explicit UI copy to this effect.

---

## 6. API Reference (Track C — planned 🔵)

_Reserved for `GET /api/forecast?symbol=…&horizon=5|10|30` endpoint spec._

---

## 7. Cockpit Tile (Track C — planned 🔵)

_Reserved for the `forecastCone` tile type in the Forecast tile group._
