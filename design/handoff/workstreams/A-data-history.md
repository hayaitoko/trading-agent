# WS-A · Data & history (Wave 1, parallel)

**Goal:** stop the traders trading blind. Today each model sees only the last 30 *close* prices
(`llm/trader.py:143` `_build_context`). Give agents real historical depth + fundamentals, and feed a
smarter context block.

**Depends on:** WS-0 (db, settings). **Blocks:** richer WS-C/E context (nice-to-have, not hard).

**Owns (create/edit):**
- `data/providers/` — `alpaca.py` (historical bars via the existing Alpaca data key — same key that
  feeds live books), and a stub `finnhub.py`/`polygon.py` adapter for fundamentals behind one
  interface. Provider choice comes from `user_settings`/env, not hardcoded.
- `data/history.py` — `HistoryService` per `CONTRACTS.md`: `bars(symbol,timeframe,lookback)`,
  `fundamentals(symbol)`, and `context_block(symbols, account)` that returns a **downsampled long
  view + dense recent window** (e.g. daily for 1–2y + intraday last N days) — configurable depth from
  `user_settings` (`history_depth`). Do NOT dump raw years of bars into the prompt; summarize/feature.
- Edit `llm/trader.py` **only** to let `_build_context` optionally delegate to
  `HistoryService.context_block` when one is injected (keep the 30-close path as fallback; keep the
  signature backward-compatible so `bench` tests still pass).

**Steps:** Alpaca historical adapter → HistoryService with caching (history rarely changes intraday)
→ context_block (downsample + recent window, full OHLCV not just close) → wire as optional injection
into LLMTrader → tests with a fake provider.

**Acceptance:**
- `HistoryService.bars/fundamentals` return typed data from a fake/recorded provider in tests (no live
  calls in CI).
- `context_block` stays within a sane token budget at default depth; depth is configurable.
- LLMTrader uses it when injected, falls back to 30-close otherwise; existing bench tests still green.
- ruff + mypy green.

**Out of scope:** news/social (that's WS-B/C). The UI history-depth control (WS-G/Settings) — just
read the setting.
