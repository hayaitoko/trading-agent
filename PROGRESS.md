# PROGRESS — trading-agent

> Round-by-round log of completed work. Append-only. The Artoo orchestrator reads this plus `PLAN.md` to plan the next batch of tasks.

## Pre-Artoo (built in single-Claude sessions, May 20–21, 2026)

### Round 0.0 — Initial scaffold
*Commit: `9d9fd73`*

- `pyproject.toml` (Python 3.13, uv-managed).
- `Broker` ABC + `MockBroker` (in-memory, callable quote_fn).
- `Scraper` ABC (no concrete impls yet).
- Frozen dataclass wire-format models: `Post`, `Signal`, `Order`, `Position`, `Trade`.
- Ticker extraction (cashtag + bareword with `COMMON_WORDS` filter).
- 14 tests.

### Round 0.1 — Dashboard + multi-account
*Commits: `614f060`, `86faedf`*

- FastAPI app factory, Jinja templates, Tailwind via CDN.
- `Account` dataclass bundles broker + name + id + starting_cash + enabled.
- `AppState` with in-memory accounts dict + JSON persistence (`accounts.json`).
- Dashboard renders per-account cards (cash, equity, positions, today's trades).
- `/accounts/` page with htmx CRUD (add via inline form, toggle enable/disable, delete with confirm).
- `/trades/` page with per-account execution history.
- `/settings/` page for API credentials (OpenRouter, Reddit, StockTwits, Investopedia). Secrets masked in UI, blank submission keeps existing value.
- Demo seeds two paper accounts on first run.
- 34 tests.

### Round 0.2 — Chat sidebar + tool use
*Commit: `f11a664`*

- `src/trading_agent/chat/` package: `models.py` (ChatMessage, ModelSpec, MODELS list), `history.py` (JSON load/save), `tools.py` (TOOL_SCHEMAS + execute + SYSTEM_PROMPT), `client.py` (OpenRouter HTTP wrapper), `service.py` (tool-use loop).
- Persistent left-rail chat sidebar on every page.
- 8-model dropdown (Claude Opus/Sonnet/Haiku, GPT-5, Grok 4, Gemini 3.1 Pro, DeepSeek v4 Pro, Kimi K2.6).
- Approximate context counter via `chars/3 = tokens`.
- Reset button, paste-image support, Enter to send / Shift+Enter newline.
- OpenRouter provider pinning to `anthropic` for Claude models + `cache_control: ephemeral` on system prompt.
- `OPENROUTER_API_KEY` added to settings.
- Chat tools: `list_accounts`, `get_account`, `get_trades`.
- Server-side history persisted to `chat_history.json` (gitignored).
- 57 tests.

### Round 0.3 — Collapsible sidebar + placeholder pages
*Commit: `33a979f`*

- Sidebar collapse button (initially `«` text, later replaced with Lucide icon).
- Placeholder pages `/today/`, `/signals/`, `/strategy/` render real templates with status banners and structural skeletons (not disabled nav labels). Marked with `†` in nav.
- Strategy page iterates real accounts so cards have content even pre-strategy-layer.
- 61 tests.

### Round 0.4 — Notes + memory consolidator
*Commit: `f26d0f7`*

- `src/trading_agent/notes/storage.py`: read/write/list/delete with path validation (no traversal, no reserved dirs, must be `.md`). Every write takes a timestamped backup to `notes/.history/<ts>/<path>`.
- Default structure auto-created on first run: `companies/`, `sectors/`, `macro/`, `general/`. `general/README.md` documents conventions.
- `src/trading_agent/notes/consolidator.py`: `Consolidator` class with `run_once` + background `_loop`. System prompt enforces "provenance over freshness" — no deletions, no marking-stale-because-old. Permitted ops: add missing frontmatter, add missing `(as of YYYY-MM-DD)` markers, merge structurally duplicate notes, improve formatting.
- `/notes/` page with file tree + textarea editor + markdown preview, plus consolidator status strip (last/next run, model, run-now button, config panel).
- Consolidator config disabled by default. Persisted to `consolidator_config.json`.
- Chat tools: `list_notes`, `read_note`, `search_notes`.
- 91 tests.

### Round 0.5 — Editorial finance aesthetic redesign
*Commit: `77690d1`*

- Light parchment palette over warm dotted grain.
- Bodoni Moda display serif, Atkinson Hyperlegible body, IBM Plex Mono data.
- Single vermillion accent (`#B6361A`), used sparingly.
- Status indicators become small squares (`.mark`), not pills.
- `.ledger` tables: hairlines above/below header, no full borders.
- Page-load reveal: 60ms staggered fade-up on `.reveal-stack > *`.
- Placeholder pages marked with `†` (dagger).

### Round 0.6 — Per-account models + Eval page + EasyMDE
*Commit: `7c65dc5`*

- `Account.model` field with default `anthropic/claude-sonnet-4.6`. AccountSpec persistence with back-compat fallback.
- Accounts form gets a model dropdown. Accounts table gets a Model column. Dashboard cards show bound model in vermillion under id.
- New `/eval/` route — standing leaderboard ranking accounts by P&L vs starting cash. Sharpe / drawdown / win-rate flagged with `‡` banner pending v0.5+.
- Demo accounts now bound to different models: Paper Aggressive → Opus 4.7, Paper Conservative → Sonnet 4.6.
- Notes editor: raw textarea → EasyMDE (markdown with toolbar + Ctrl+B/I/K + side-by-side preview). Heavy CSS overrides for editorial palette.
- README reframed as dual-product (trading agent + model evaluation harness).
- 97 tests.

### Round 0.7 — Dark mode + Manrope + Lucide + network log
*Commit: `0059621`*

- Light/dark theme toggle. CSS custom properties as RGB triples drive whole palette. `data-theme` attr on `<html>`, applied before paint. Dark variant tuned warm (library at night).
- Sun/moon Lucide icons in header. Preference persisted to localStorage.
- Font swap: Atkinson Hyperlegible → Manrope. Default body weight 500.
- Sidebar collapse uses Lucide `panel-left-close` / `panel-left-open`.
- `src/trading_agent/web/netlog.py`: 200-entry rolling buffer. `NetworkLogMiddleware` records inbound (skipping `/static/*` and the netlog endpoint itself). `chat/client.py` records outbound OpenRouter calls.
- `/settings/api/netlog` endpoint. Settings page grows a network log section with live table (auto-refresh every 5s, pauses when tab hidden, color-coded status).
- Icons in notes tree (folder for dirs, file-text for `.md` files). Reset/attach/send/refresh icons added throughout.
- 103 tests.

### Round 0.8 — Dropdown dark fix + tooltips + animations + Docker + handoff docs
*Commit: pending (this round)*

- `<select>` and `<option>` explicitly themed so dark-mode dropdowns are readable (Chromium 105+, Firefox, Safari 17+ support).
- Hover lift on cards (`.lift` class on dashboard / strategy / trades cards). Subtle drop shadow plus vermillion glow in dark.
- Soft pulse animation on `.mark-active`.
- CSS-only tooltips via `data-tip` attribute (`data-tip-below` variant for elements near the top of viewport).
- Tooltips added to: theme toggle, chat collapse, chat reset, chat attach/send, account toggle/delete, account form labels, consolidator run/configure, settings save, netlog refresh, notes save/delete/new.
- Settings page becomes 2-column at `lg`: credential form on left, network log on right.
- Docker: multi-stage `Dockerfile` (python:3.13-slim + uv), `docker-compose.yml` with named volume mounted at `/data`, `.dockerignore`.
- `demo.py` reads `TRADING_AGENT_DATA`, `TRADING_AGENT_HOST`, `TRADING_AGENT_PORT`.
- Handoff docs: `HANDOFF.md` (user-written, session state + risks + rounds), `PLAN.md` (milestone roadmap), `AGENTS.md` (constraints + conventions), `PROGRESS.md` (this file).
- 103 tests (same count; no test additions needed for these UI-only changes).

---

## Next round (planned, see PLAN.md §v0.1)

**v0.1 — Reddit scraper + post log.** Four tasks suggested:
1. PostStore + SQLite schema + tests.
2. RedditScraper + dedup + tests.
3. Runner loop + lifespan wiring.
4. Today page (replace placeholder) + chat tools (`list_recent_posts`, `posts_for_ticker`).

Acceptance: scraper polls every configured interval, posts appear at `/today/` within one cycle, chat can answer "What's the latest from WSB?", tests still 100% green.

---

## Quick stats

- Lines of code (src/, excl. tests): ~3,200
- Test count: 103
- Templates: 14
- Routes: 9 page routes + chat + notes API + settings/netlog API + consolidator API
- Models supported in chat dropdown: 8
- Time investment so far: see `time spent log`
