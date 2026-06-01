# trading-agent

A Python trading-agent framework supporting crypto (Binance/Coinbase via ccxt) and
US equities (Alpaca, paper or live). Switchable globally and per-asset between two
modes:

- **AUTONOMOUS** — the agent executes trades directly. Hard kill switch (env var
  or file flag) + per-strategy circuit breakers (max-daily-loss,
  max-position-size, max-trades-per-hour, max-open-positions).
- **APPROVAL** — proposed trades land in a persistent queue; a human approves or
  rejects each one before execution. Proposals time out automatically.

## Agent model

Each trader is a **ReAct-style tool-using agent** (`AgentTrader` in `llm/trader.py`),
not a prompt-stuffed structured-output pipeline.  The agent runs a tool-call loop
each turn — calling LOOK tools (history, news, research, memory, market state), NOTE
tools (reflect, remind_me, watchpoint), and ACT tools (trade, trade_batch) — and
exits when it emits a terminal action (trade / hold / pass / done_for_day).  The
operator sees the full tool-call trace per turn in the cockpit.  New traders receive
three guided orientation turns via tutorial mode before operating freely.  See
[`design/TRADER-AGENT.md`](design/TRADER-AGENT.md) for the full agent specification.

## Situation + forecast surface

Seven real-time data sources enrich trader decisions without stuffing raw text
into prompts.  Each source is exposed as a callable LOOK tool, gated by a
per-user feature flag (all default off):

| Tool | Flag | Source |
|---|---|---|
| `world_events()` | `SITUATION_GDELT` | GDELT macro/geopolitical themes |
| `prediction_market_odds()` | `SITUATION_PREDICTION_MARKETS` | Polymarket + Kalshi implied probabilities |
| `options_iv()` | `SITUATION_OPTIONS_IV` | Alpaca options IV + Greeks |
| `forecast()` | `SITUATION_FORECAST` | 1σ price-cone (realized vol + IV + PM) |

News, research briefs, and social metrics (Substack/SeekingAlpha RSS, Bluesky
list and author feeds) flow through the ingest pipeline into the `news()` and
`situation()` LOOK tools automatically.  The forecast cone (`GET /api/forecast`)
combines empirical realized vol, options IV, and prediction-market implied move
into a ±1σ price envelope — never a directional point estimate.  See
[`design/SITUATION-FORECAST.md`](design/SITUATION-FORECAST.md) for the full
source map, feature-flag reference, and forecast cone math.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Verify the build
.venv/bin/pytest -q
.venv/bin/mypy src/trading_agent/
.venv/bin/ruff check src/ tests/

# End-to-end paper-trading demo (no creds, no internet)
.venv/bin/python -m trading_agent.scripts.demo --bars 300
# or after install:
trading-agent-demo --bars 300
```

The demo wires up: `MeanReversionStrategy` → `SignalRouter` (autonomous) → `RiskManager`
→ `PaperBroker`, replaying a synthetic mean-reverting price series through
`CsvReplayFeed`. Every trade is recorded to both a per-day JSONL file at
`data/audit.YYYY-MM-DD.jsonl` and the SQLite `audit_log` table.

## Layout

```
src/trading_agent/
  broker_adapter.py        # Frozen ABC: canonical order_details dict contract
  paper_broker.py          # Reference in-memory broker
  alpaca_broker.py         # US equities via alpaca-py
  ccxt_broker.py           # Crypto via ccxt (Binance, Coinbase)
  signal_router.py         # Autonomous / approval dispatch
  risk_manager.py          # Kill switch + circuit breakers
  approval_queue.py        # SQLite-backed proposal queue with timeouts
  strategy.py              # Strategy ABC
  strategies/
    mean_reversion.py      # Placeholder: 20-SMA, -2σ entry, SMA exit
  data_feed.py             # DataFeed ABC + in-process MessageBus
  feeds/
    csv_replay.py          # CSV / synthetic replay feed
    live_quote.py          # Real quotes -> PaperBroker (live-data paper mode)
  broker_factory.py        # Build Alpaca/CCXT brokers from env (paper default)
  market_hours.py          # US equity RTH gate for PaperBroker
  config.py                # YAML global + TOML per-strategy + ${VAR} refs
  strategy_loader.py       # Resolves and loads per-strategy TOML
  db.py                    # SQLite WAL connection manager
  audit.py                 # JSONL + audit_log dual sink
  logging_config.py        # Structured JSON logging to stdout
  models.py / enums.py     # Signal, Order, Position dataclasses + enums
  scripts/demo.py          # End-to-end demo entry point
```

## Paper trading on real market data

The `PaperBroker` is a simulator, but it can fill against **real** prices — two ways,
modeled on Investopedia's stock simulator (real quotes, fake money):

**Path A — Alpaca paper account (broker-hosted, most realistic).** Real data plus
Alpaca's own server-side fills (spread, partial fills, market hours). Get free paper
keys at <https://alpaca.markets>, put them in a gitignored `.env` (see `.env.example`):

```python
from trading_agent.broker_factory import build_alpaca_broker
broker = build_alpaca_broker()          # paper=True by default
broker.connect()                         # routes orders to Alpaca's paper account
```

**Path B — real quotes into the local PaperBroker (self-contained).** A `LiveQuoteFeed`
polls any broker's `get_quote()` and feeds bid/ask/last into the PaperBroker, so market
orders fill at the ask/bid and resting limit orders match on real price moves. Crypto
public tickers need no API key:

```python
from trading_agent.broker_factory import build_ccxt_broker
from trading_agent.data_feed import MessageBus
from trading_agent.feeds import LiveQuoteFeed
from trading_agent.paper_broker import PaperBroker

source = build_ccxt_broker("binance")            # read-only public ticker source
paper = PaperBroker(initial_balance=10_000, slippage_bps=2.0, commission_bps=1.0)
paper.connect()
feed = LiveQuoteFeed(MessageBus(), source, ["BTC/USDT"], paper_broker=paper, poll_interval=5.0)
feed.poll_once()                                  # or: await feed.run()
```

`PaperBroker` realism knobs (all default off — behaviour unchanged unless set):
`slippage_bps`, `commission_bps`, `is_market_open` (e.g. `market_hours.us_equity_clock()`),
plus bid/ask-aware fills and limit-order matching via `update_quote()`.

> ⚠️ Live (real-money) trading has never been exercised — only paper + mocked SDKs.
> Smoke-test against an Alpaca **paper** account before ever setting `paper=False`.

## Configuration

- `config.yaml` — global mode, risk limits, kill-switch file path, credential
  env-var refs. See the file in this repo for the schema.
- `strategies/config/<name>.toml` — per-strategy parameters (e.g.
  `mean_reversion.toml`).

Both files support `${ENV_VAR}` and `${ENV_VAR:default}` substitution. Missing
required env vars raise `ConfigError`.

## Kill switch

Any of these will block all `RiskManager.check_*` from returning `False`:

1. `rm.activate_kill_switch()` — programmatic
2. `TRADING_AGENT_KILL_SWITCH=1` env var
3. `data/.kill_switch` file exists (path configurable per RiskManager)

## Tests

```bash
.venv/bin/pytest --cov=trading_agent --cov-report=term-missing
```

Coverage on the SPEC-mandated modules:

| Module           | Coverage |
| ---------------- | -------- |
| `risk_manager`   | 99%      |
| `signal_router`  | 97%      |
| `paper_broker`   | 83%      |
| `approval_queue` | 100%     |
| `audit`          | 100%     |
| `config`         | 100%     |

The broker contract suite (`tests/test_broker_adapter_contract.py`) runs the
same assertions against `PaperBroker`, `AlpacaBroker` (mocked `TradingClient`),
and `CCXTBroker` (mocked ccxt Exchange) — the only place all three are
exercised together against the frozen `BrokerAdapter` ABC.

## Deployment

### systemd user service (Raspberry Pi / homelab)

Create `~/.config/systemd/user/trading-cockpit.service`:

```ini
[Unit]
Description=Trading-agent cockpit (multi-user paper trading)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/trading-agent
ExecStart=/path/to/trading-agent/.venv/bin/trading-agent-serve \
    --cockpit \
    --host 0.0.0.0 \
    --port 8000 \
    --data-dir data \
    --owner <your-username>
Restart=on-failure
RestartSec=10

# --- required ---
Environment=OPENROUTER_API_KEY=sk-or-...
Environment=ALPACA_API_KEY=...
Environment=ALPACA_SECRET_KEY=...

# --- optional intelligence layer ---
# Bind the trader-intelligence layer to this user (id or username).
# Falls back to TRADING_AGENT_OWNER_ID env var, then a lone signed-up user.
# Environment=TRADING_AGENT_OWNER_ID=<user-id>

# Set to the base URL of a local Ollama-compatible embedding endpoint.
# Required for research-brief recall and per-trader memory (WS-D).
# Example: http://localhost:11434 (nomic-embed-text model)
# Without this, traders run history-only — research + memory stay dark.
# Environment=TRADING_AGENT_EMBED_ENDPOINT=http://localhost:11434

# --- optional feature flags ---
# Gate decision turns to US regular trading hours (US/Eastern).
# Set to 1 for a production paper test; leave unset for free-running dev.
# Environment=TRADING_AGENT_SCHEDULER=1

# Enable the POST /api/dev/fire-turn endpoint (development only).
# Remove in production.
# Environment=TRADING_AGENT_DEV_TRIGGER=1

# Number of guided tutorial turns each new trader gets before trading freely.
# 0 (default) disables tutorial mode — traders trade from their first RTH turn.
# Set >0 to re-enable onboarding after a restart.
# Environment=TRADING_AGENT_TUTORIAL_TURNS=3

# Hard kill switch — set to 1 to halt all trades immediately (all
# RiskManager.check_* methods return True). Unset to resume.
# Alternatively, create the file at data/.kill_switch (path configurable).
# Environment=TRADING_AGENT_KILL_SWITCH=1

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now trading-cockpit.service
journalctl --user -u trading-cockpit.service -f
```

### Key environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | LLM routing (OpenRouter). Required for add-trader; without it, cockpit read surfaces work but no new traders can be added. |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Yes (live bars) | Alpaca paper account. Without these, the cockpit falls back to a synthetic mean-reverting price feed. |
| `TRADING_AGENT_OWNER_ID` | Recommended | User id (or username) to bind the intelligence layer. Overridden by `--owner` CLI flag. |
| `TRADING_AGENT_EMBED_ENDPOINT` | Recommended | Local Ollama-compatible embed base URL (e.g. `http://localhost:11434`). Required for WS-D memory + research recall. |
| `TRADING_AGENT_SCHEDULER` | Optional | Set to `1` to gate turns to US RTH via `MarketScheduler`. Leave unset for free-running cadence (development). |
| `TRADING_AGENT_DEV_TRIGGER` | Optional | Set to `1` to expose `POST /api/dev/fire-turn` (development only). |
| `TRADING_AGENT_TUTORIAL_TURNS` | Optional | Guided orientation turns for new traders. Default `0` (disabled). |
| `TRADING_AGENT_KILL_SWITCH` | Optional | Set to `1` to halt all trades immediately. Equivalent to touching `data/.kill_switch`. |

### `--owner` flag

The `--owner` flag (or `TRADING_AGENT_OWNER_ID`) binds the trader-intelligence
layer to a specific user so traders have access to their research briefs, memory,
and advisor notes. Without it, traders run in history-only mode (price bars +
tool calls, no recall). Set it to the username or `user_id` of an account that
has been registered via the cockpit.

### Embed endpoint requirement

Memory and research recall depend on a local embedding model. Configure a
`local` type endpoint in the cockpit settings UI (or via the API) pointing at an
Ollama-compatible server. The recommended model is `nomic-embed-text` (Ollama).
Without a working embed endpoint, `memory_search` and `research_brief` return
`ToolError(kind="unavailable")` — the trader continues operating normally on
price history alone.

## Out of scope

Docker, strategy R&D. The single placeholder mean-reversion strategy exists
only to prove the framework end-to-end.
