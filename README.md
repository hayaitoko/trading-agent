# trading-agent

A Python trading-agent framework supporting crypto (Binance/Coinbase via ccxt) and
US equities (Alpaca, paper or live). Switchable globally and per-asset between two
modes:

- **AUTONOMOUS** — the agent executes trades directly. Hard kill switch (env var
  or file flag) + per-strategy circuit breakers (max-daily-loss,
  max-position-size, max-trades-per-hour, max-open-positions).
- **APPROVAL** — proposed trades land in a persistent queue; a human approves or
  rejects each one before execution. Proposals time out automatically.

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
  config.py                # YAML global + TOML per-strategy + ${VAR} refs
  strategy_loader.py       # Resolves and loads per-strategy TOML
  db.py                    # SQLite WAL connection manager
  audit.py                 # JSONL + audit_log dual sink
  logging_config.py        # Structured JSON logging to stdout
  models.py / enums.py     # Signal, Order, Position dataclasses + enums
  scripts/demo.py          # End-to-end demo entry point
```

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

## Out of scope

Deployment infra, Docker, web UI, strategy R&D. The single placeholder
mean-reversion strategy exists only to prove the framework end-to-end.
