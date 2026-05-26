# Cockpit web UI — redesign handoff

## Task
Use the `/frontend-design` skill, then design + build a NEW unified web UI ("cockpit")
for trading-agent. We are SCRAPPING both existing UIs. First mockup pass already exists
at `design/cockpit.html` but it violates the skill (system fonts: -apple-system/Inter/Roboto)
— treat it as a throwaway reference for layout + mock data, not the design.

## Repo / env
- Repo: `/home/hayai/projects/trading-agent-build-a-python-trading`  (git main, clean, pushed)
- venv: `.venv` (py3.12). `.venv/bin/pytest` → 269 pass, ruff + mypy green.
- Host is `artoo` at `10.0.0.26`. The browser is on a DIFFERENT LAN machine, so any dev
  server MUST bind `0.0.0.0`, never `127.0.0.1` (that was why nothing loaded earlier).
- Serve the mockup: `.venv/bin/python -m http.server 8090 --bind 0.0.0.0 --directory design`
  → open `http://10.0.0.26:8090/cockpit.html`

## Being scrapped
- `src/trading_agent/web/static/index.html` — notification center (`trading-agent-serve`, :8000)
- `src/trading_agent/web/static/bench.html` — model bench (`trading-agent-bench`, :8050)
- The FastAPI apps `web/app.py` + `web/bench_app.py` get rewritten.
- KEEP (logic, not UI): `web/notifications.py` (NotificationCenter), `web/market_watch.py`
  (MarketMoveWatcher) — pure read-model logic, reusable.

## Layout (from Lukas's wireframe — firm constraint)
- LEFT RAIL (persistent): `model selector` top-left; `context monitor` + `context wipe
  button` top-right of the rail; `chat box` fills the rest below.
- TOP: ~5 tabs.
- MAIN: a grid of cards, 3 across — EACH CARD = ONE ACCOUNT.
- Concept: each "account" = one LLM's isolated paper book (a bench competitor); one Alpaca
  data key feeds N in-process PaperBroker books. Left-rail chat = talk to the selected model
  about its book; context monitor = that chat's token usage; wipe = clear the conversation.
- Proposed tab set (confirm with Lukas): Accounts (the grid) / Leaderboard / Approvals /
  Risk / Audit.

## Color scheme picker (Lukas asked for this) — 4 palettes to offer + persist choice
1. Teal Coral: #091d26 #102937 #124d54 #094044  accent #f9744b / #d84f2a  light #b1d9cf #d6c4b0 #ece6df
2. Deep Blue:  #0c0c0c #212121  blue #002756 #00447c  teal #035b7a #0b7b9e  light #c6c3b6 #dddace
3. Slate:      #11161C #2F3B49 #6E7F90 #CFD8E3 #F6F8FB
4. Steel Lavender: #1B1D23 #515563 #A6A0C4 #D9D6EE #F4F3FB

## Backend surface to build against
Existing HTTP routes (will be rebuilt, but show what the read/action model is):
```
serve  GET /api/notifications · GET /api/health
       POST /api/approvals/{id}/approve · /reject
bench  GET /api/bench · GET /api/bench/models
       POST /api/bench/competitors · DELETE /api/bench/competitors/{name}
       POST /api/bench/cadence · /start · /stop · /tick
```
Core building blocks (all live, tested):
- `bench/bench.py` Bench: add/remove_competitor, observe_bar/observe_quote, run_decisions,
  leaderboard(), recent_decisions(), snapshot()
- `bench/controller.py` BenchController: add_model, set_cadence, start/stop, tick_now,
  available_models() (cached OpenRouter menu), status()
- `approval_queue.py` ApprovalQueue: add/approve/reject/pending/get (SQLite, thread-safe)
- `web/notifications.py` NotificationCenter.snapshot(); `web/market_watch.py` MarketMoveWatcher
- `risk_manager.py` RiskManager: kill switch, check_*, limits, exposure
- `paper_broker.py` PaperBroker; `llm/openrouter.py` OpenRouterClient.chat/list_models;
  `llm/trader.py` LLMTrader / StrategyTrader
- Featured OpenRouter slugs (verified): anthropic/claude-opus-4.7, anthropic/claude-sonnet-4.6,
  moonshotai/kimi-k2.6, deepseek/deepseek-v4-pro, google/gemini-3.5-flash, z-ai/glm-5.1, x-ai/grok-4.3

## Open questions to confirm before/while building
1. Exact tab set + what each shows.
2. Static mockup first (mock data) vs wire to the live backend now.
3. Stack: keep vanilla-JS-over-FastAPI, or move to SvelteKit (like the agent-interface project)?
4. Keys are fine to use per Lukas (OpenRouter + Alpaca paper live). Running bench rounds costs
   a few cents/model — don't autostart; gate behind an explicit click.

Full backend function inventory was dumped via AST earlier; re-run if needed:
`.venv/bin/python` AST walk over `src/trading_agent/**.py`.
