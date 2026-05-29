# Situation Enrichment + Forecast Surface

**Status:** WS-Situation A0 ✅ · A1 ✅ · A2 ✅ · B0 ✅ · B1 ✅ · C0 ✅ · C1 ✅  
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
| **forecast LOOK tool** | `src/trading_agent/intel/tools/look/forecast.py` | ✅ C1 |
| Substack + SA RSS seeds | `src/trading_agent/ingest/seed_sources.py` | ✅ B0 |
| Bluesky list/author fetcher | `src/trading_agent/ingest/fetchers/bluesky.py` | ✅ B1 |
| Universe listener (SA ticker auto-seed) | `src/trading_agent/web/routers/requests.py` | ✅ C2 |
| Forecast cone compute | `src/trading_agent/intel/forecast.py` | ✅ C1 |
| Forecast API router | `src/trading_agent/web/routers/forecast.py` | ✅ C1 |
| Cockpit forecast tile | `web/static/cockpit.html` | ✅ C1 |

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

### 3d. Substack Finance Newsletters (B0)

**Status:** ✅ B0  
**File:** `src/trading_agent/ingest/seed_sources.py`

**Purpose:** Seed 10 hand-picked finance Substacks into the ingest pipeline as
standard RSS sources.  No new fetcher code — these rows are consumed by the
existing `RssSource` adapter unchanged.

**Endpoint pattern:** `https://{slug}.substack.com/feed` (RSS 2.0)  
**Auth:** None.  
**Rate:** No published limit.  At the default 60-second ingest cadence, well
within polite use.  
**Coverage:** Post title + excerpt for free posts; paywalled posts appear as
title + teaser only (sufficient for brief generation).

**Deployment note:** Substack uses Cloudflare IUAM protection that blocks
datacenter IP ranges (this is a well-known constraint, not a bug in our
configuration).  Feeds work correctly from residential and homelab networks.
On datacenter deployments, configure an egress proxy or reduce
`INGEST_CADENCE_SECONDS` for Substack sources.

**Seeds (10 publications):**

| Source name | Slug | Primary signal |
|---|---|---|
| Net Interest (Rubinstein) | `netinterest` | Banking, fintech, financial history |
| The Macro Tourist (Muir) | `themacrotourist` | Rates, FX, commodities, vol |
| Doomberg | `doomberg` | Energy + commodity macro, industrial policy |
| Marc Rubinstein (alt) | `rubinstein` | Banking + financial analysis alt feed |
| Kevin Muir (alt) | `kevinmuir` | Macro commentary, daily observations |
| Garrett Baldwin | `garrettbaldwin` | Options flow, vol strategy, derivatives |
| Junk Bond Investor | `junkbondinvestor` | HY credit, leveraged finance, distress |
| Pragmatic Capitalist (Roche) | `pragcapitalist` | Monetary realism, portfolio construction |
| Kyla's Newsletter (Scanlon) | `kylascan` | Economic narratives, consumer sentiment |
| Epsilon Theory (Hunt) | `epsilontheory` | Game theory, narrative analysis, long-cycle |

**Adding new publications:** one `/api/sources` POST or `seed_finance_sources()`
call — no code change required.

**Failure mode:** `SourceError` on HTTP 4xx/5xx or XML parse error.  The
worker isolates per-source failures; a blocked Substack never stalls SA or
Bluesky sources.

---

### 3e. Seeking Alpha Public RSS (B0)

**Status:** ✅ B0  
**File:** `src/trading_agent/ingest/seed_sources.py`

**Purpose:** Seed three SA global RSS feeds plus per-ticker combined feeds for
the default watchlist symbols.  Like Substack, these are consumed by the
existing `RssSource` adapter — zero new code.

**Auth:** None — documented public feeds at https://about.seekingalpha.com/feeds.

**Live smoke (2026-05-28):**
- `market_currents.xml` → 200 OK, 7 items (example: "First Solar soars to multiyear high…")
- `api/sa/combined/SPY.xml` → 200 OK, 30 items
- `feed.xml` → 200 OK, 30 items
- `sector/transcripts.xml` → 200 OK, 20 items

**Global seeds (3 feeds):**

| Source name | URL | Primary signal |
|---|---|---|
| SA Market Currents | `seekingalpha.com/market_currents.xml` | Real-time market-moving news |
| SA Latest Analysis | `seekingalpha.com/feed.xml` | Analyst opinions, earnings previews |
| SA Transcripts | `seekingalpha.com/sector/transcripts.xml` | Earnings call awareness |

**Per-ticker combined feed:**  
URL: `https://seekingalpha.com/api/sa/combined/{TICKER}.xml`  
Default tickers seeded: SPY, AAPL, MSFT, NVDA, TSLA.

To register a new ticker:
```python
from trading_agent.ingest.seed_sources import seed_sa_ticker
seed_sa_ticker(db, user_id, "AMZN")
```

**Gap — no auto-registration on watch_symbol (B0):** When a trader calls
`watch_symbol`, the new symbol is not automatically added to the SA per-ticker
feed list.  The correct fix is a `universe_listener` in the ingest worker that
calls `seed_sa_ticker` whenever a symbol joins a trader's universe.  Tracked as
a Track C / WS-Agent integration follow-up.

---

### 3f. Bluesky List + Author Feeds (B1)

**Status:** ✅ B1  
**File:** `src/trading_agent/ingest/fetchers/bluesky.py`

**Purpose:** Extend the existing Bluesky fetcher with two new source kinds that
pull curated finance voices from Bluesky starter-pack lists and individual
author feeds.  All new kinds feed through the same compact-metrics aggregation
path as the existing cashtag kind — raw post text is never forwarded to the model.

**Endpoint base:** `https://public.api.bsky.app/xrpc/`  
**Auth:** None — public AppView, fully unauthenticated.

**New source kinds:**

| Kind | Config keys | XRPC call |
|---|---|---|
| `bluesky_list` | `{"list_uri": "at://..."}` | `app.bsky.feed.getListFeed` |
| `bluesky_author` | `{"handle": "user.bsky.social"}` | `app.bsky.feed.getAuthorFeed` |

**Starter-pack resolution:** Call `app.bsky.graph.getStarterPack` once per
pack URL to obtain the backing list AT-URI.  Persist the resolved URI in the
`config_json` row so no re-resolution occurs on subsequent fetch ticks.
`resolve_starter_pack(client, pack_url) → str` is a module-level helper in
`bluesky.py`.

**Seeds — starter-pack lists (5):**

| Source name | Starter-pack URL |
|---|---|
| Bluesky: Fintwit Starter Pack | `bsky.app/starter-pack/alexbhturnbull.bsky.social/3lbgeejdteh2u` |
| Bluesky: FinTwit (Kelly) | `bsky.app/starter-pack/stevenkelly49.bsky.social/3laptmzbdhg2e` |
| Bluesky: Finance News + Analysis (Woodley) | `bsky.app/starter-pack/kylewoodley.bsky.social/3lbcvvhwm2v2q` |
| Bluesky: Finance Investing Econ (Roche) | `bsky.app/starter-pack/cullenroche.bsky.social/3lbgrvn57r424` |
| Bluesky: Investment + Financial Media (Lowe) | `bsky.app/starter-pack/thelowegroup.bsky.social/3lbv3ofuavt2f` |

**Seeds — author handles (10):**

> **Reconciliation note (C2):** doc previously listed `joeweisenthal.bsky.social`,
> `arsorkin.bsky.social`, and `yahoofinance.bsky.social`.  Code (B1's seed list,
> verified live 2026-05-28) uses `weisenthal.bsky.social`, `andrewrsorkin.bsky.social`,
> and `markgurman.bsky.social` (Bloomberg tech reporter replacing Yahoo Finance).
> Code is the source of truth; this table now matches code.

| Handle | Signal |
|---|---|
| `carlquintanilla.bsky.social` | CNBC anchor, breaking market news |
| `weisenthal.bsky.social` | Bloomberg Odd Lots — macro, yields, heterodox economics |
| `andrewrsorkin.bsky.social` | NYT DealBook, M&A, tech finance |
| `bencasselman.bsky.social` | NYT economics, jobs data, macro |
| `heatherlong.bsky.social` | WaPo economics editor |
| `cullenroche.bsky.social` | Pragmatic Capitalist, monetary realism |
| `markgurman.bsky.social` | Bloomberg tech reporter — Apple, big tech, supply chain |
| `conorsen.bsky.social` | BofA research, macro strategy |
| `jasonfurman.bsky.social` | Harvard economist, fiscal policy |
| `steveliesman.bsky.social` | CNBC Fed reporter, monetary policy |

**Aggregation:** All three `bluesky*` kinds (cashtag, list, author) produce
compact `bluesky_metrics` dicts: mention count, sentiment distribution, top
cashtags found in fetched posts.  Raw post text never leaves the ingest layer.

**Failure mode:** `SourceError` on network/4xx.  Starter-pack resolution
failure on first fetch is logged and raises `SourceError` (the source is skipped
that cycle; resolved URIs already persisted in config survive restarts).
429 on the public AppView → logged, source skipped that cycle.

**MONEY IS REAL compliance:** Aggregated metrics flowing through these sources
contain no account-status strings.  The ingest layer is upstream of any model
call; the compact-metrics format carries only counts and float scores.

---

## 4. Feature Flags (operator reference)

| Flag key | Type | Default | Effect when ON | Dependencies |
|---|---|---|---|---|
| `SITUATION_GDELT` | bool | `False` | Enables `world_events` LOOK tool; `GDELTProvider` constructed once per turn | None |
| `SITUATION_PREDICTION_MARKETS` | bool | `False` | Enables `prediction_market_odds` LOOK tool | None |
| `SITUATION_OPTIONS_IV` | bool | `False` | Enables `options_iv` LOOK tool; requires `ALPACA_API_KEY` | Options chain provider |
| `SITUATION_FORECAST` | bool | `False` | Enables `forecast` LOOK tool + `/api/forecast` endpoint | Requires `HistoryService`; degrades without IV/PM providers |
| `INGEST_SEEDS_ENABLED` | env str | `"1"` (on) | When `"0"`, `seed_finance_sources` and `seed_sa_ticker` are no-ops | None; set to `"0"` in test environments |

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

## 5. Forecast Cone (✅ C1)

**Implementation:** `src/trading_agent/intel/forecast.py` — `ForecastCone`,
`ConePoint` dataclasses + `build_forecast()`.

### 5a. Anti-overconfidence contract

**This is an envelope, never a point estimate.**  The cone shows where price
*could* be over a forward horizon at ±1σ of historical/implied vol.  The **mid
line is flat** (current price repeated) — there is no drift estimate.  Callers
and UI must present this framing explicitly.  The `forecast()` tool description
and the cockpit tile both display an `⚠ envelope only` warning.

### 5b. Sigma components

Three optional inputs — all absent is valid (returns degenerate cone with no points):

| Component | Source | Condition |
|---|---|---|
| `empirical_sigma` | `HistoryService.get_bars()` 30D log-return std × √252 | Always attempted; None if <5 bars |
| `iv_sigma` | Mean near-money implied vol from `options_chain.get_chain()` | `SITUATION_OPTIONS_IV` flag on; None for crypto/no-options |
| `pm_implied_move` | Probability-weighted distance from matched PM event | `SITUATION_PREDICTION_MARKETS` flag on; None if no matching event |

**Combined sigma:** `max(available_sigmas)` — conservative; the widest cone wins.

### 5c. Cone math (log-normal)

```
t_years = t / 252
band(t) = combined_sigma * sqrt(t_years)
hi(t)   = current_price * exp(+band)
lo(t)   = current_price * exp(-band)
mid(t)  = current_price  (flat — no drift)
```

Points are generated for `t ∈ [0, horizon_days]` (inclusive, 1 per day).

### 5d. Instrument-agnostic degradation

- **Crypto (BTC/USD, ETH/USD):** no options chain → `iv_sigma = None`.  Cone
  renders from `empirical_sigma` + optional `pm_implied_move`.
- **Thinly traded equities:** options chain may return no IV data → `iv_sigma = None`.
- **No history data:** `empirical_sigma = None`, cone has `points = []`.

---

## 6. API Reference

### `GET /api/forecast`

**Auth:** current_user header (standard).  
**File:** `src/trading_agent/web/routers/forecast.py`

**Query parameters:**

| Param | Type | Required | Description |
|---|---|---|---|
| `symbol` | string | yes | Ticker or instrument (e.g. `AAPL`, `SPY`, `BTC/USD`) |
| `horizon` | int | no (default 30) | Forward horizon in trading days: `5`, `10`, or `30` |

**Response:** `ForecastCone.to_dict()` JSON:

```json
{
  "symbol": "AAPL",
  "horizon_days": 30,
  "current_price": 185.40,
  "empirical_sigma": 0.18,
  "iv_sigma": 0.21,
  "pm_implied_move": null,
  "combined_sigma": 0.21,
  "components_used": ["empirical", "iv"],
  "points": [
    {"t": 0, "lo": 185.40, "mid": 185.40, "hi": 185.40},
    {"t": 1, "lo": 184.93, "mid": 185.40, "hi": 185.87},
    ...
    {"t": 30, "lo": 171.49, "mid": 185.40, "hi": 200.41}
  ]
}
```

**Degenerate response** (no history, all flags off):

```json
{
  "symbol": "XYZ",
  "horizon_days": 30,
  "current_price": null,
  "empirical_sigma": null,
  "iv_sigma": null,
  "pm_implied_move": null,
  "combined_sigma": null,
  "components_used": [],
  "points": []
}
```

---

## 7. Cockpit Tile (✅ C1)

**Tile type:** `forecastCone`  
**Group:** `Forecast` (new group — supports future crypto-aware and multi-symbol tiles)  
**Default:** off (not in any default tab layout; operator adds via tile palette)

**Options:**

| Key | Type | Description |
|---|---|---|
| `symbol` | string | Symbol to forecast (e.g. `AAPL`). Default `SPY`. |
| `horizon` | string | `"5"`, `"10"`, or `"30"`. Default `"30"`. |

**Tile behavior:**

- Calls `GET /api/forecast?symbol=…&horizon=…` on mount and every 5 min.
- Renders a 5/10/30D horizon selector (no page reload on switch).
- Displays `⚠ envelope only` tooltip — anti-overconfidence framing.
- When `combined_sigma` is null (all data absent), shows an actionable message:
  `"Enable SITUATION_FORECAST in settings."`.
- Renders cone on a `lightweight-charts` line + area series (hi/lo fill + mid line).
  Falls back to inline SVG sparkline when LightweightCharts is not loaded.

**MANAGER FRUGALITY:** no LLM calls — tile reads cached `/api/forecast` only.
