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

## Roadmap

1. **v0** (this commit): architecture scaffold. Models, ABCs, `MockBroker`, ticker extraction, tests.
2. **v0.1**: `RedditScraper` (PRAW) for r/wallstreetbets, r/stocks, r/options. SQLite post log.
3. **v0.2**: `StockTwitsScraper`. Sentiment scoring (start with VADER or a Claude API call; revisit with finBERT if needed).
4. **v0.3**: signal aggregator: per-ticker rolling sentiment + mention velocity.
5. **v0.4**: `InvestopediaBroker` via Playwright. Login flow, order placement, position read.
6. **v0.5**: main runner loop. Backtest harness against historical posts and prices.
7. **v0.6**: first naive strategy + eval metrics (Sharpe, max drawdown, hit rate vs random).

The order matters: signal quality has to be validated before strategy iteration. Most of the work lives in steps 3–4 and the eval harness in step 6.
