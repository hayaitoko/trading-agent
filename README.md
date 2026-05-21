# trading-agent

Autonomous stock-trading agent. Scrapes investment forums for short-term sentiment signals on individual tickers, runs them through a signal pipeline, and executes paper trades through a pluggable broker backend.

**Status:** v0. Architecture scaffold. No live scrapers, no live broker yet.

## Architecture

```
forums ─► scrapers ─► sentiment ─► signal ─► strategy ─► broker
(reddit,   (Scraper   (ticker     (per-ticker (rules /    (Broker
 stocktwits, ABC)      extract +   aggregate)  ML later)   ABC:
 ...)                  scoring)                            mock,
                                                           investopedia,
                                                           future)
```

Two interfaces carry the abstraction:

- **`Broker` ABC** (`src/trading_agent/brokers/base.py`): `place_order`, `get_positions`, `get_quote`, etc. Concrete impls: `MockBroker` for tests/dry runs, `InvestopediaBroker` (planned) drives the Investopedia stock simulator via Playwright. Future impls (a price-replay sim, Alpaca, etc.) just implement the same interface.
- **`Scraper` ABC** (`src/trading_agent/scrapers/base.py`): `poll()` yields `Post` objects. One impl per forum source.

Models in `src/trading_agent/models.py` are the wire format between layers: `Post`, `Signal`, `Order`, `Position`, `Trade`.

## Compliance boundary

This project must stay fully sandboxed from any real brokerage account, including read-only access to LPL ClientWorks, Orion, Black Diamond, or anything connected to Advantage Wealth Advisors. Paper trading only. No credentials for real broker accounts in `.env` or anywhere else in this repo.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv venv
uv pip install -e ".[dev]"
uv run pytest
```

## Web UI

```powershell
uv run trading-agent-web
# open http://127.0.0.1:8765
```

Single user, no auth. Multiple paper accounts run in parallel against the same signal feed so you can A/B test strategies side by side.

Built pages:
- **Dashboard**: per-account cash, equity, positions, today's trades. Has a `+ account` shortcut.
- **Accounts**: list, add (htmx inline form), enable/disable, delete. Persisted to `accounts.json`.
- **Trades**: per-account full execution history.
- **Settings**: API credentials for OpenRouter, Reddit, StockTwits, Investopedia. Persisted to `trading_agent_secrets.json`. Secrets are masked in the UI and never round-tripped through HTML.

Persistent chat sidebar (left rail, on every page):
- Model dropdown (Claude Opus/Sonnet/Haiku, GPT-5, Grok 4, Gemini 3.1 Pro, DeepSeek v4, Kimi K2.6). Choice persisted in `localStorage`.
- Approximate context counter (`chars/3 = tokens`) showing % of the selected model's context window.
- Reset button clears the conversation.
- Paste a screenshot directly into the textarea or use the file picker. Images are base64-inlined into the request.
- Routes through **OpenRouter** so one API key covers every model. Anthropic models get `cache_control` on the system prompt and `provider.order: [anthropic]` for cache consistency across requests.
- Conversation persists server-side in `chat_history.json` so reloads (and other browsers pointed at the same server) see the same context.
- Tool use: `list_accounts`, `get_account`, `get_trades`. Tools that need scrapers/signals/news will light up as those layers ship.

Planned (nav present but disabled):
- **Today**: chronological feed of ingested posts + sentiment scores + source links + which trade each post fed into. The audit trail. Needs scrapers to be wired.
- **Signals**: per-ticker rolling sentiment and mention velocity, drill down to source posts. Needs the signal aggregator.
- **Strategy**: per-account knobs (sentiment threshold, position size, max drawdown circuit breaker). Needs the strategy layer.

A kill switch, compare-mode, and Telegram alerts are planned alongside the runner loop in v0.5.

`accounts.json` and `trading_agent_secrets.json` are gitignored and only contain non-broker state (account metadata, API keys). `MockBroker` positions and trades are in-memory only and reset on restart — that limitation goes away when `InvestopediaBroker` lands, since it reads state from the live simulator.

## Roadmap

1. **v0** (this commit): architecture scaffold. Models, ABCs, `MockBroker`, ticker extraction, tests.
2. **v0.1**: `RedditScraper` (PRAW) for r/wallstreetbets, r/stocks, r/options. SQLite post log.
3. **v0.2**: `StockTwitsScraper`. Sentiment scoring (start with VADER or a Claude API call; revisit with finBERT if needed).
4. **v0.3**: signal aggregator: per-ticker rolling sentiment + mention velocity.
5. **v0.4**: `InvestopediaBroker` via Playwright. Login flow, order placement, position read.
6. **v0.5**: main runner loop. Backtest harness against historical posts and prices.
7. **v0.6**: first naive strategy + eval metrics (Sharpe, max drawdown, hit rate vs random).

The order matters: signal quality has to be validated before strategy iteration. Most of the work lives in steps 3–4 and the eval harness in step 6.
