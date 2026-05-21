# DESIGN — trading-agent

> Architecture and design decisions. Sections are numbered; don't renumber when adding (per Lukas's convention). New material goes at the end or as a sub-section.

## 1. Mission

A personal trading agent that uses forum sentiment as its primary signal source, with a built-in model evaluation harness as a first-class product surface. Single-user, paper-trading only, runs in a homelab.

The system answers two operational questions:

- **"What should I trade?"** — sentiment from forums + signal aggregation + strategy + broker.
- **"Which model decides best?"** — multi-account architecture, each account bound to its own LLM, leaderboard compares them on shared signal feed.

## 2. Two-product framing

These aren't two systems. They're the same system viewed from two sides. The multi-account architecture from §5 makes evaluation a structural property, not a feature bolted on later.

- Trading-agent surface: dashboard, accounts, trades, today, signals, strategy, notes.
- Model-eval surface: eval leaderboard, strategy-binding-per-account, model dropdown on account creation.

Settings, chat sidebar, and notes are shared between both.

## 3. Architecture (high level)

```
        ┌────────────┐
        │  Forums    │  (Reddit, StockTwits, Seeking Alpha, …)
        └─────┬──────┘
              │
              ▼
        ┌────────────┐
        │  Scrapers  │  Scraper ABC → poll() → AsyncIterator[Post]
        └─────┬──────┘
              │
              ▼
        ┌────────────┐
        │  Post log  │  SQLite (planned in v0.1)
        └─────┬──────┘
              │
              ▼
        ┌────────────┐
        │  Sentiment │  per-post scorer (VADER or LLM)
        └─────┬──────┘
              │
              ▼
        ┌────────────┐
        │  Aggregator│  per-ticker rolling sentiment + velocity
        └─────┬──────┘
              │           (chat tools read here)
              ▼
        ┌────────────┐
        │  Strategy  │  Strategy ABC → decide(signals, portfolio, model) → list[Order]
        └─────┬──────┘
              │
              ▼
        ┌────────────┐
        │  Broker    │  Broker ABC → place_order, get_positions, ...
        └─────┬──────┘
              │
              ▼
        ┌────────────┐
        │  Account   │  (broker + model + starting_cash + enabled)
        └─────┬──────┘
              │
              ▼
        ┌────────────┐
        │   Eval     │  leaderboard by P&L vs starting_cash + (planned: Sharpe, drawdown, hit rate)
        └────────────┘
```

The boundaries between stages are ABCs in `src/trading_agent/`. Each stage transforms one wire format into the next (see §4).

## 4. Layer responsibilities and wire formats

### 4.1 Scraper layer

- **Contract** (`src/trading_agent/scrapers/base.py`): `class Scraper(ABC): poll(since) -> AsyncIterator[Post]`. Async generator so a single scraper can stream many posts; the runner drains the iterator and persists each.
- **State the scraper owns**: source name, polling cursor (since-watermark). Optional dedup cache (or rely on storage layer to dedup).
- **State the scraper does NOT own**: post storage, sentiment, tickers. Those are downstream stages.

### 4.2 Post log

- Persisted in SQLite (`storage/db.py`, v0.1). Single `posts` table with `(source, post_id, …)`. Unique on `(source, post_id)` for dedup.
- Tickers extracted on insert (via `sentiment/tickers.py`) and stored as a JSON column. Sentiment populated by a separate task (§4.3) once a scorer is wired.

### 4.3 Sentiment layer

- **Contract** (`src/trading_agent/sentiment/scorer.py`, v0.2): `class SentimentScorer(Protocol): score(text) -> float`. Range `[-1, 1]`.
- **Impls**: `VaderScorer` (deterministic, free, ~OK for finance). `ClaudeScorer` (uses chat client). Picked via `settings.sentiment_backend`.
- **Where it runs**: background task that pulls unscored posts from the store, scores, writes back. Decoupled from scraping so the scorer can be swapped without restarting scrapers.

### 4.4 Aggregator

- Per-ticker rolling sentiment over configured windows (default 1h, 24h, 7d).
- Mention velocity (count per hour).
- Output: in-memory map `{ticker: Signal}`. Persisted to JSON for survivability.
- **Contract**: read-only consumer of post log. Doesn't mutate anything upstream.

### 4.5 Strategy

- **Contract** (`src/trading_agent/strategy/base.py`, v0.6): `class Strategy(ABC): decide(signals, portfolio, account_model) -> list[Order]`. Synchronous and stateless per call. Per-account state belongs to the account, not the strategy.
- **First impls**: `naive_momentum` (threshold-based, no LLM), `llm_advisor` (uses the account's bound model to reason about signals + portfolio).
- **Per-account params**: stored on the account spec (`accounts.json`). Editable on `/strategy/`.

### 4.6 Broker

- **Contract** (`src/trading_agent/brokers/base.py`): async methods for `place_order`, `cancel_order`, `get_positions`, `get_quote`, `get_account_value`, `get_cash`, `get_trades`. Errors raise `BrokerError` subclasses (`InsufficientFundsError`, `UnknownTickerError`).
- **Impls today**: `MockBroker` (in-memory, callable `quote_fn`).
- **Impls planned**: `InvestopediaBroker` (Playwright), `AlpacaBroker` (paper API).
- **State**: each impl owns its state. MockBroker = in-memory dict. InvestopediaBroker = reads from live simulator on every call (no local state). AlpacaBroker = same (API is source of truth).

### 4.7 Account

- `dataclass Account` (`src/trading_agent/accounts.py`): bundles `id`, `name`, `broker`, `starting_cash`, `enabled`, `model`.
- **The model field is the linchpin of §2.** Each account is bound to one LLM (OpenRouter id). Strategy uses this model when reasoning. Eval compares accounts by P&L → models compared as a side-effect.
- Persisted as `AccountSpec` (no broker handle; just metadata) to `accounts.json`.

### 4.8 Wire formats (`src/trading_agent/models.py`)

All frozen dataclasses:

- `Post(source, post_id, author, text, url, created_at, score, num_comments)` — output of scrapers, input to everything downstream.
- `Signal(ticker, sentiment, confidence, mentions, window_start, window_end, sources)` — output of aggregator, input to strategy.
- `Order(ticker, side, qty, order_type, limit_price, time_in_force)` — output of strategy, input to broker.
- `Position(ticker, qty, avg_cost, current_price)` — read from broker for display + strategy.
- `Trade(order_id, ticker, side, qty, price, executed_at)` — read from broker, audit trail.

Decimals (not floats) for any money/price/qty. Datetimes are timezone-aware (UTC where the source allows).

## 5. State model

Three storage tiers:

| Tier | Examples | When |
|---|---|---|
| In-memory only | MockBroker positions, NetworkLog, Aggregator output | Volatile state that's cheap to recompute |
| Flat file | `accounts.json`, `trading_agent_secrets.json`, `chat_history.json`, `consolidator_config.json`, `notes/*.md` | Singleton or small-set config; human-editable |
| SQLite | Post log (v0.1), equity snapshots (v0.5), strategy decision log (v0.6) | Tabular data with growth |

All persistent state lives under `$TRADING_AGENT_DATA` (defaults to cwd locally, `/data` in Docker).

No Postgres. No Redis. No external state stores. If we outgrow SQLite (>10⁸ rows), revisit.

## 6. The chat subsystem (`src/trading_agent/chat/`)

A persistent left-rail sidebar visible on every page. Operates as a tool-use loop over an OpenRouter-gated set of LLMs.

### 6.1 Tool-use loop

```
user msg → send to LLM with tools + system prompt + history
            │
            ├── LLM returns text → append, done
            │
            └── LLM returns tool_calls
                    │
                    ▼
                execute each tool, append results
                    │
                    └── back to top (max MAX_TOOL_ITERATIONS=6)
```

Implemented in `chat/service.py`. Stateless per call; full history sent each time (cache_control on system prompt keeps the prefix cached for Anthropic models).

### 6.2 Tools available to the chat assistant

- `list_accounts`, `get_account`, `get_trades` — read account state.
- `list_notes`, `read_note`, `search_notes` — read the agent's memory.

Future:
- `list_recent_posts`, `posts_for_ticker` (v0.1)
- `get_signal(ticker)` (v0.3)
- `explain_trade(trade_id)` (v0.6, reads `strategy_log/`)

### 6.3 Provider routing

All routes through OpenRouter (`openrouter.ai/api/v1/chat/completions`). One API key, eight model options. Anthropic-family models get:
- `provider.order: ["anthropic"]` (so prompt cache keys stay stable across requests)
- `cache_control: {"type": "ephemeral"}` on the system prompt + tool defs

Other providers ignore those fields; they pass through unchanged.

## 7. The notes subsystem (`src/trading_agent/notes/`)

Markdown corpus that is both human-editable and the chat assistant's memory.

### 7.1 Convention: timestamps are first-class

- Frontmatter on every note: `title`, `created`, `updated` (ISO dates).
- Inline `(as of YYYY-MM-DD)` markers on every time-bound claim in the body.
- Old observations stay. Information value comes from provenance, not freshness.

### 7.2 Directory shape

```
notes/
├── companies/      per-ticker research
├── sectors/        industry-level trends
├── macro/          Fed, inflation, geopolitics
├── general/        anything else
├── .history/<ts>/  timestamped backups of every overwrite or delete
└── .consolidator/log.md  consolidator run log
```

Reserved dirs (`.history`, `.consolidator`) are not user-editable. Storage layer enforces this.

### 7.3 The consolidator

A background task that runs on a configurable timer. Reads all notes, calls the configured LLM with a strict system prompt: **curate, don't prune**. Allowed ops: add missing frontmatter, add missing `(as of …)` markers, merge structurally duplicate notes (two NVDA notes → one), improve formatting, add cross-links. Forbidden ops: delete information, mark-stale-because-old, rewrite claims to change meaning.

Every edit gets a `.history/<ts>/` backup. Disabled by default. Lives at `notes/consolidator.py`.

## 8. The evaluation surface (`/eval/`)

Ranks accounts by P&L delta vs starting cash. Real today.

Planned columns light up when data exists:

- **Sharpe ratio** — needs equity-curve snapshots over time (v0.5).
- **Max drawdown** — same.
- **Win rate** — needs closed-trade pairing (v0.7).
- **Average hold time** — same.
- **Head-to-head signal response** — needs per-account decision log (v0.6).

The eval page is the central reason the multi-account architecture exists; making it useful is the long-term north star.

## 9. The web layer (`src/trading_agent/web/`)

FastAPI + Jinja + htmx + EasyMDE + Lucide + Tailwind CDN.

- **Routing**: one file per nav section under `web/routes/`. Each registers a router in `web/app.py`.
- **Templates**: `web/templates/`. `base.html` owns the palette, fonts, theme toggle, header, chat sidebar inclusion, and the CSS for `.surface`, `.ledger`, `.mark`, `.lift`, tooltips, animations.
- **Static**: `web/static/{chat,notes}.js`. No bundler; vanilla ES modules via `<script src>`.
- **State**: `app.state` carries `app_state` (the AppState dataclass), `templates` (Jinja2Templates), `netlog`, `chat_service`, `consolidator`, `notes_dir`.

### 9.1 Middleware

`NetworkLogMiddleware` records every inbound request to the in-memory `NetworkLog`, skipping `/static/*` and the netlog endpoint itself.

### 9.2 Lifespan

FastAPI lifespan starts the consolidator's background loop on app start, stops it on shutdown. Disabled via `start_consolidator=False` for tests.

## 10. Theming and aesthetic

### 10.1 Palette as CSS variables

All colors defined as RGB triples on `:root` and overridden on `[data-theme="dark"]`. Tailwind reads them via `rgb(var(--token) / <alpha-value>)` so utilities like `bg-paper-base/50` still work.

- `--paper-*` (base, elevated, edge, recess, shadow) — surfaces.
- `--ink-*` (100 down to 10) — text and dividers.
- `--vermillion` (default, soft, deep) — single accent.
- `--gain`, `--loss`, `--warning` — semantic.

### 10.2 Type stack

- Display: **Bodoni Moda** (variable opsz + wght, Didone, high contrast).
- Body: **Manrope** (variable, weight 500 default).
- Mono: **IBM Plex Mono** (tnum + zero features).

No Inter. No Space Grotesk. Lukas's standing preference: distinctive over generic.

### 10.3 Standing rules

- No em dashes in prose.
- No hex literals in templates.
- Status indicators are squares (`.mark`), not pills.
- Tables use the `.ledger` pattern (hairlines above + below header, no full borders).
- Placeholder pages get a `†` in the nav and the `_status_banner.html` partial.
- One animation per page-load (`.reveal-stack > *` staggered fade-up); hover lift on cards; pulse on `.mark-active`. Nothing else.

## 11. Security and compliance

### 11.1 Hard wall

This codebase must stay sandboxed from anything connected to:

- LPL ClientWorks
- Orion
- Black Diamond
- Anything owned by Advantage Wealth Advisors (Lukas's family firm)

Paper trading only. No real-money broker credentials in `.env`, `trading_agent_secrets.json`, or anywhere else. Stated in `README.md` and enforced by code review.

### 11.2 Secret handling

- `trading_agent_secrets.json` stores OpenRouter / Reddit / StockTwits / Investopedia (paper) credentials. Gitignored.
- Settings page renders secret fields as `type=password`. Existing secret values are shown redacted (`*` for length). Blank submission keeps the existing value, so secrets never round-trip through HTML.
- Never log a secret. Never include a secret in chat history. Never expose a secret via the netlog (the OpenRouter target shows `openrouter · model_id`, not the auth header).

### 11.3 No auth

Single-user by design. If the system grows multi-user, a full auth/session/RBAC layer needs designing first; don't bolt it on.

## 12. Why X, not Y (alternatives considered)

### 12.1 Browser-control broker first, not Alpaca

User explicitly chose Playwright-driving-Investopedia (v0.4) over Alpaca's paper API. Rationale: final deployment home may not have API access; building browser control first ensures that path works. Alpaca remains v1.0 as a sibling under the same `Broker` ABC.

### 12.2 OpenRouter as the single LLM gateway

Considered: per-provider SDKs (`anthropic`, `openai`, `google-genai`). Rejected because:
- Three SDKs to maintain.
- Different schemas for tool use (Anthropic vs OpenAI).
- The eval-by-model framing demands one swappable surface; per-SDK breaks that.

OpenRouter's OpenAI-compatible API gives one schema, one auth, and passthrough of provider-specific knobs (cache_control, provider pinning).

### 12.3 SQLite for the post log, not Postgres

Single-user homelab. SQLite scales to ~10⁸ rows comfortably. No network setup, no replication, no operational overhead. Backup is `cp data.db backup.db`. Postgres at this stage is over-engineering.

### 12.4 Flat JSON for accounts and chat history, not SQLite

The volume is small (≤ tens of accounts, ≤ thousands of chat messages). Human-editable JSON wins on simplicity. Switch to SQLite if a user starts running hundreds of accounts.

### 12.5 EasyMDE for the notes editor, not CodeMirror 6

Considered CodeMirror 6 + manual toolbar. EasyMDE wraps CodeMirror with a markdown-aware toolbar pre-built, plus side-by-side preview, plus standard keybinds. The CSS overrides to fit the editorial aesthetic are tractable. CodeMirror 6 is the right answer if/when we need code editing (Python, SQL) inside the app.

### 12.6 Lucide icons over Heroicons / Phosphor / inline SVGs

Lucide has the panel-toggle icons we needed for the sidebar, looks coherent across the icon set we use (sun, moon, folder, file-text, paperclip, send, arrow-up-right, etc.), and the CDN-friendly distribution avoids a build step.

### 12.7 Tailwind via CDN, not a build pipeline

No build = faster iteration, no node_modules, no build-time secrets. Cost: production CSS bundle is larger than a build-optimized PostCSS run. For a single-user homelab dashboard this trade is fine. Revisit if we ever ship publicly.

### 12.8 Light theme as default, not dark

Most finance dashboards default to dark. Light is the more distinctive call and pairs with the editorial-finance aesthetic (newspaper-on-the-table feel). Dark theme exists; both are equally polished.

## 13. Forward debt and risks

Tracked in `HANDOFF.md §Known Risks` and `PLAN.md §Cross-cutting work`. Briefly:

- CDN dependence (Tailwind, htmx, Chart.js, Lucide, EasyMDE, Google Fonts). Offline = degraded.
- MockBroker positions reset on restart. Goes away when InvestopediaBroker lands.
- EasyMDE editor's dark-mode pass is incomplete; the editor surface still has light-theme color literals.
- No type checker. `mypy` or `pyright` (strict-ish) is on the cross-cutting list.
- No coverage gate. Soft target 85% but not enforced.
- Mobile layout breaks below ~900px. Desktop-only by intent.

If any of these blocks a feature, surface it before working around it.
