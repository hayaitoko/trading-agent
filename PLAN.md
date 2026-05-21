# PLAN — trading-agent

> Round-by-round backlog for the Artoo orchestrator. Each milestone is a coherent batch the planner can derive 1–4 tasks from per round. Don't ship the next milestone until the previous one's tests + acceptance are green.

## Active milestone

**v0.1 — Reddit scraper + post log**

## Milestone overview

```
v0   (shipped)  scaffold, UI, chat, notes, eval — no engine wired
v0.1 (next)     RedditScraper + post log + tickers extracted
v0.2            StockTwitsScraper + sentiment scoring
v0.3            signal aggregator (rolling sentiment + mention velocity)
v0.4            InvestopediaBroker via Playwright
v0.5            runner loop + post snapshots (eval gets equity curves)
v0.6            Strategy ABC + naive_momentum + per-account strategy binding
v0.7            backtest harness against historical post log
v0.8            eval metrics: Sharpe, max drawdown, win rate, head-to-head
v0.9            Telegram alerts + global kill switch
v1.0            Alpaca paper broker (sibling to InvestopediaBroker)
```

The order matters. Signal quality must be validated before strategy iteration. Most of the leverage is in v0.3 and v0.7.

---

## v0.1 — Reddit scraper + post log

### Goal
Pull posts and comments from r/wallstreetbets, r/stocks, r/options into a persistent log. Each row has tickers extracted. The Today page populates from this log.

### Deliverables
1. `src/trading_agent/scrapers/reddit.py` — `RedditScraper(Scraper)`. Uses PRAW (already-listed dep target). Reads creds from `state.secrets` (`reddit_client_id`, `reddit_client_secret`, `reddit_user_agent`). Polls a configured list of subs.
2. `src/trading_agent/storage/db.py` — SQLite schema for `posts` table (source, post_id, author, text, url, created_at, score, num_comments, tickers JSON column, scraped_at). Singleton-ish `PostStore` with `add(post)`, `recent(limit)`, `since(datetime)`, `for_ticker(symbol)`.
3. Pre-flight wiring: `pyproject.toml` adds `praw`, deps installed. `state.post_store` accessible to routes.
4. `src/trading_agent/runner/scraper_loop.py` (new package). Periodically polls each `Scraper` and writes new `Post` rows. Background task via the existing FastAPI lifespan. Configurable interval (default 5min) via env var.
5. Today page (`templates/today.html`) — replace placeholder with a live feed of the last 200 posts. Filters: source dropdown, sentiment placeholder (lights up in v0.2), ticker substring filter. Each row links to the source URL.
6. Tests:
   - `tests/test_reddit_scraper.py` — uses a fake `praw.Reddit` to verify deduplication and ticker extraction wiring.
   - `tests/test_post_store.py` — SQLite CRUD round trip, tmp_path-isolated.
   - `tests/test_today_page.py` — Today page renders rows from a seeded post_store.
7. Chat tools: `list_recent_posts(limit)`, `posts_for_ticker(symbol, days_back)`.

### Acceptance
- Without creds: scraper stays idle, Today page shows zero rows + a clear "no creds configured" message.
- With creds: scraper polls every interval, new posts appear at `/today/` within 1 cycle.
- Chat can answer "What's the latest from WSB?" by calling `list_recent_posts`.
- Tests still 100% green.

### Tasks (suggested split)
- **T1**: PostStore + SQLite schema + tests.
- **T2**: RedditScraper + dedup + tests.
- **T3**: Runner loop + lifespan wiring.
- **T4**: Today page template + chat tools.

### Non-goals
- Sentiment scoring (v0.2).
- Cross-subreddit aggregation logic (v0.3).
- Backfill of historical posts.

---

## v0.2 — StockTwits + sentiment scoring

### Goal
Add a second source and assign each post a sentiment value `[-1, 1]`.

### Deliverables
1. `src/trading_agent/scrapers/stocktwits.py` — `StockTwitsScraper(Scraper)`. Uses the public sentiment endpoint when available (free tier).
2. `src/trading_agent/sentiment/scorer.py` — `SentimentScorer` protocol + two impls:
   - `VaderScorer` (vaderSentiment dep, deterministic, free).
   - `ClaudeScorer` (uses the existing `chat.client.call_model` with a tight one-shot prompt; per-post cost gates this behind config).
3. Background sentiment-scoring task. Picks up unscored posts from the store, scores them, writes back.
4. Today page: sentiment column lights up, color-coded.
5. Settings: add `sentiment_backend` field (`vader` | `claude`) with explanation.
6. Tests for both scorers (vader is deterministic; Claude uses a fake caller).

### Non-goals
- Per-ticker aggregation across posts (v0.3).
- finBERT or any other paid/heavy model (revisit if VADER misses badly).

---

## v0.3 — signal aggregator

### Goal
Per-ticker rolling sentiment + mention velocity. The Signals page populates.

### Deliverables
1. `src/trading_agent/signal/aggregator.py` — `SignalAggregator`. Reads from PostStore, computes per-ticker:
   - rolling mean sentiment over window (default 1h, 24h, 7d)
   - mention count per hour (velocity)
   - top contributing posts
2. Periodic recompute or on-demand. Stored in memory + persisted to `signals.json` for survivability.
3. Signals page renders the aggregator output: ticker grid, sentiment bar, mention/hr, click-through to source posts.
4. Chat tool: `get_signal(ticker)`.

### Acceptance
- Signals page replaces the placeholder; status banner gone.
- A signal computed at moment X is reproducible from the same post corpus.

---

## v0.4 — InvestopediaBroker

### Goal
First real broker, sitting behind the existing `Broker` ABC.

### Deliverables
1. `src/trading_agent/brokers/investopedia.py` — `InvestopediaBroker(Broker)`. Uses Playwright (`playwright[chromium]`).
2. Login flow with session persistence (`storage_state.json` gitignored).
3. `place_order(Order)` → drives the simulator's order form.
4. `get_positions()`, `get_cash()`, `get_quote(ticker)` → read from the portfolio page.
5. `get_trades()` → read order history page.
6. Defensive selectors (data-testid where possible; fail loud on missing).
7. Tests: at least one integration test that uses a recorded Playwright trace, plus unit tests for the form-builder logic.
8. Accounts UI: option to pick broker type when creating an account (currently always Mock).

### Non-goals
- Real-money brokers. Hard wall.
- Alpaca (that's v1.0).

---

## v0.5 — runner loop + snapshots

### Goal
Tie scrapers + signals + brokers together with periodic snapshots so eval has time-series data.

### Deliverables
1. `src/trading_agent/runner/loop.py` — main orchestrator. Per-account: read signals, decide (Strategy in v0.6 — for now: passthrough/no-op), execute orders.
2. Equity snapshots: every account's `total` written to `snapshots.db` (SQLite) every N minutes.
3. Eval page: equity curves chart populates (Chart.js, already loaded).
4. Kill switch wired (UI in v0.8 builds on this hook).

---

## v0.6 — Strategy ABC + first impl

### Goal
Actual decision-making code, tied to per-account model.

### Deliverables
1. `src/trading_agent/strategy/base.py` — `Strategy` ABC. `decide(signals, portfolio, model) -> list[Order]`.
2. `src/trading_agent/strategy/naive_momentum.py` — first concrete: buy on positive sentiment + velocity above threshold, exit on decay.
3. `src/trading_agent/strategy/llm_advisor.py` — wraps `chat.client.call_model` so an account's bound LLM is asked "given these signals and this portfolio, what orders?" Tool use to fetch more data.
4. Strategy page is no longer a placeholder — bind a strategy to each account, tune per-account params (sentiment threshold, position size %, max drawdown).
5. Per-account decision log under `notes/.strategy_log/<account_id>/<date>.md` so the chat can answer "why did this account buy X?"

### Acceptance
- Strategy page editable. Changes persist to `accounts.json` (Account gains `strategy_id` and `strategy_params`).
- A run with `naive_momentum` produces orders the eval can rank.
- A run with `llm_advisor` records its reasoning under `.strategy_log/`.

---

## v0.7 — backtest harness

### Goal
Replay historical post log against historical price data so we can score strategies offline.

### Deliverables
1. Historical price fetcher (yfinance or Polygon free tier). Cached locally.
2. `src/trading_agent/runner/backtest.py` — feeds posts to aggregator in chronological order, runs strategies, computes equity curves.
3. Backtest controls on the Eval page: pick date range, strategy, model, run.
4. Per-trade closure tracking (which buy matched which sell).

---

## v0.8 — eval metrics + alerts + kill switch

### Goal
The eval surface goes from a leaderboard to a real comparison tool, and the user gets paged when something happens.

### Deliverables
1. Sharpe ratio, max drawdown, win rate, average hold time (from per-trade closures landed in v0.7).
2. Head-to-head signal view: pick a signal, see which models acted on it and what they did.
3. Telegram alerts via existing `TELEGRAM_BOT_TOKEN` (in Artoo's salvage list): trade fires, daily P&L digest, kill switch trips.
4. Global kill switch UI (header pill?). Stops all strategy execution across accounts. Persisted in `state` and checked by the runner loop.

---

## v0.9 — Polish + observability

### Goal
Production-grade for a single user.

### Deliverables
1. Real metrics endpoint (`/metrics` Prometheus-style or Grafana-friendly JSON).
2. Log rotation for `consolidator/log.md`, `strategy_log/*`.
3. Backup/restore commands for `notes/` and the SQLite stores.
4. Healthcheck route for Docker.

---

## v1.0 — Alpaca paper broker

### Goal
Second real broker. Pure API (no browser), as a sibling to Investopedia.

### Deliverables
1. `src/trading_agent/brokers/alpaca.py` — `AlpacaBroker(Broker)`. Uses `alpaca-py`.
2. Settings: Alpaca paper API key + secret.
3. Accounts UI: third broker option.
4. The eval page can compare an Investopedia account vs an Alpaca account running the same strategy + model — the broker layer itself becomes a variable.

### Non-goals
- Alpaca live (real-money) trading. Hard wall stays.

---

## Cross-cutting work (no specific milestone)

- **Type checking**: introduce `mypy` or `pyright` in strict-ish mode. Currently we rely on runtime + tests.
- **Coverage report**: add `coverage` to dev deps, target 85% line coverage as a soft floor.
- **i18n** of dates/numbers if the project ever ships beyond a single user.
- **Mobile responsiveness**: currently desktop-only. Not a priority but the editorial layout breaks below ~900px.
- **Auth**: out of scope while this is single-user. Becomes mandatory if multi-user lands (not on this roadmap).
