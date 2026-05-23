# Project Progress

**Status: v1 feature-complete (2026-05-22).** SPEC-compliant. `pytest` 174 passed,
`mypy` clean, `ruff` clean. End-to-end demo runs the full pipeline on synthetic data.

Last verified: 2026-05-23 — `174 passed` in 1.37s.

---

## What's implemented

All SPEC core abstractions are built and tested.

| Module | File | Status |
|---|---|---|
| BrokerAdapter ABC | `src/trading_agent/broker_adapter.py` | done |
| PaperBroker (in-memory) | `src/trading_agent/paper_broker.py` | done — 83% cov |
| AlpacaBroker (US equities) | `src/trading_agent/alpaca_broker.py` | done — mocked in tests, never run live |
| CCXTBroker (crypto) | `src/trading_agent/ccxt_broker.py` | done — mocked in tests, never run live |
| RiskManager | `src/trading_agent/risk_manager.py` | done — 99% cov; kill switch + circuit breakers |
| SignalRouter | `src/trading_agent/signal_router.py` | done — 97% cov |
| DataFeed / pub-sub | `src/trading_agent/data_feed.py` | done |
| CSV replay + synthetic bars | `src/trading_agent/feeds/csv_replay.py` | done |
| ApprovalQueue (SQLite) | `src/trading_agent/approval_queue.py` | done — consumer not wired (see gaps) |
| AuditLogger (JSONL + SQLite) | `src/trading_agent/audit.py` | done |
| DatabaseManager (SQLite WAL) | `src/trading_agent/db.py` | done |
| Config (YAML global + TOML per-strategy) | `src/trading_agent/config.py`, `strategy_loader.py` | done |
| Structured JSON logging | `src/trading_agent/logging_config.py` | done |
| MeanReversion strategy (placeholder) | `src/trading_agent/strategies/mean_reversion.py` | done |
| End-to-end demo | `src/trading_agent/scripts/demo.py` (`trading-agent-demo`) | done |

Coverage on the three SPEC-mandated modules exceeds the ≥80% target:
RiskManager 99%, SignalRouter 97%, PaperBroker 83%.

## Operating modes

- **AUTONOMOUS** — SignalRouter calls `broker.place_order` directly. Gated by
  RiskManager: kill switch (env var `+ data/.kill_switch` file flag `+` programmatic
  flag), max-daily-loss, max-position-size, max-trades-per-hour, max-open-positions.
- **APPROVAL** — SignalRouter enqueues proposals on ApprovalQueue with a TTL.
  A consumer approves/rejects; expired proposals auto-cancel. **The consumer is not
  yet built** — the queue exposes a clean `pending()/approve()/reject()` API but
  nothing currently reads it. Intended consumer: a web-UI notification center
  (originally specced as the Artoo Telegram bot; redirected to web UI 2026-05-23).

## Known gaps / decisions made unattended

- **Approval consumer unwired.** No Telegram, no web UI hook yet. `ApprovalQueue` is
  ready for a poller / notification center.
- **No live trading ever executed.** Alpaca + CCXT are only exercised against mocks.
  Smoke-test against Alpaca *paper* before trusting AUTONOMOUS mode.
- **mypy ignores `alpaca_broker` + `ccxt_broker`** (see `[tool.mypy.overrides]` in
  pyproject.toml). The SDKs return `X | str` unions that need excessive isinstance
  guards. Those two files are not statically type-checked.
- **CCXT `get_balance` synthesizes a `cash` key** from the first available stablecoin
  total (USDT > USDC > USD > BUSD > DAI). Convenience for the rest of the system,
  which expects a single cash figure — verify against a real account before live use.
- **Not a local git repo.** No `.git` here. The GitHub repo `hayaitoko/trading-agent`
  currently holds a *different, earlier* trading project (web dashboard + forum
  sentiment scraping), not this framework — needs reconciliation before any push.

## Build history (condensed)

This framework was built overnight 2026-05-21 → 05-22 by the Artoo orchestrator. The
full round-by-round log (rounds 0–37) has been retired from this file.

Summary of that run: the orchestrator stalled for ~15 rounds in a reviewer-opinion
oscillation on the broker interface (the `amount` vs `quantity` / enum-vs-string
churn) because no executable verify step existed — reviewers hallucinated bugs and
contradicted each other. It then spent rounds 18–28 in identical planner-level
diagnostic rounds producing zero code, briefly recovered at rounds 29–35 (added
verify.yml, froze the BrokerAdapter contract, started the real test suite), and
finally halted at rounds 36–37 on an OpenRouter daily-limit 403.

The build was completed by hand afterward: rewrote RiskManager / SignalRouter /
MeanReversion, finalized the frozen BrokerAdapter contract, wrote the parametrized
broker-contract test suite, and built ApprovalQueue / CsvReplayFeed / AuditLogger /
demo net-new — landing at 174 passing tests, clean mypy, clean ruff.

Lesson recorded: the orchestrator's file-level halt diagnoses can be hallucinated —
run the real toolchain before trusting them.
