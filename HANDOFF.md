# Trading Agent Handoff for Artoo

## Current State

This is a Python 3.13 FastAPI/Jinja trading dashboard with htmx, Tailwind CDN,
Lucide icons, Chart.js, EasyMDE for notes, and OpenRouter-backed chat. The app is
single-user and paper-trading only. Do not connect it to real brokerage accounts.

Recent work in this tree:

- Added warm light/dark theme tokens in `src/trading_agent/web/templates/base.html`.
- Added native select/dropdown dark-mode styling so model pickers are readable.
- Added CSS-only tooltips via `data-tip`.
- Added card hover lift/scale with `.lift`.
- Moved Settings into a side-by-side layout: credential form left, network log right.
- Added a rolling in-memory network log endpoint at `/settings/api/netlog`.
- Added Docker packaging: `Dockerfile`, `.dockerignore`, and `docker-compose.yml`.
- Runtime state is controlled by `TRADING_AGENT_DATA`; Docker uses `/data`.

There are uncommitted changes in the repo. Treat them as user/previous-agent work
and do not revert them.

## Run Commands

Local development:

```powershell
uv venv
uv pip install -e ".[dev]"
uv run pytest
uv run trading-agent-web
```

Open the app at `http://127.0.0.1:8765`.

Docker:

```powershell
docker compose up --build
```

Open `http://127.0.0.1:8765`. Persistent container data lives in the `ta-data`
named volume.

## Agent Instructions

- Keep the app paper-trading only. Never add real brokerage integrations without
  an explicit safety design and user approval.
- Prefer small, test-backed changes. Update or add tests in `tests/` when routes,
  persistence, or rendered contracts change.
- Preserve the current visual direction: editorial, warm paper, quiet dashboard,
  square edges, minimal ornament. Avoid marketing-page patterns.
- Use Lucide icons for controls when an icon is appropriate.
- Add tooltips to compact controls, icon buttons, destructive actions, and places
  where state is not self-explanatory. Do not tooltip every static text label.
- Keep dropdowns readable in dark mode. Validate account model picker, chat model
  picker, notes consolidator model picker, and disabled placeholder selects.
- Settings should keep credentials and network log side-by-side on desktop. It can
  stack on mobile.
- Do not store secrets in HTML responses. Existing behavior redacts secret fields
  and keeps blank secret submissions unchanged.
- Use `TRADING_AGENT_DATA` for state paths in runtime/container work. Do not write
  persistent app state into the source tree from the container.

## Suggested Rounds

### Round 1: Finish UI Polish

Goal: make the existing app feel cohesive in light and dark mode.

- Audit all selects in dark mode:
  - Accounts model picker
  - Chat model picker
  - Notes consolidator model picker
  - Today placeholder disabled selects
- Verify the EasyMDE notes editor remains readable in both themes. The current
  editor overrides are mostly fixed colors; dark-mode editor support is still a
  likely follow-up.
- Tune `.lift` only on account/dashboard surfaces where hover movement helps.
- Add missing `data-tip` attributes to dense controls, especially icon-only or
  destructive buttons.
- Confirm mobile layout has no overlapping text or horizontal overflow.

Suggested checks:

```powershell
uv run pytest
uv run trading-agent-web
```

Manual browser check:

- Toggle theme.
- Visit Dashboard, Accounts, Settings, Notes.
- Open each dropdown in dark mode.
- Hover account cards and compact controls.

### Round 2: Container Hardening

Goal: make Docker the normal way to run a local instance.

- Build and run with `docker compose up --build`.
- Verify `/data/accounts.json`, `/data/trading_agent_secrets.json`,
  `/data/chat_history.json`, and `/data/notes/` persist across restarts.
- Add a healthcheck only if the final image has a lightweight HTTP client
  available or one is intentionally installed.
- Decide whether demo seeding should run in containers by default or be controlled
  by an env var.
- Document volume reset commands carefully, but do not automate destructive volume
  deletion.

### Round 3: Network Log UX

Goal: make the network log useful as an operator panel.

- Add client-side filters for direction, status class, and search target.
- Add pause/resume auto-refresh.
- Keep the table compact; avoid wrapping controls into a second large card.
- Consider exposing outbound OpenRouter request metadata without leaking prompts,
  credentials, or message content.

### Round 4: Trading/Data Layers

Goal: move beyond UI scaffolding.

- Implement `RedditScraper` and post persistence before strategy logic.
- Add `StockTwitsScraper`.
- Add signal aggregation and tests around ticker extraction, scoring windows, and
  duplicate handling.
- Only after signal quality is inspectable, build strategy controls and runner.

### Round 5: Evaluation Harness

Goal: make model/account comparison meaningful.

- Persist mock broker positions/trades or add a replayable paper broker store.
- Add metrics: return, max drawdown, hit rate, turnover, cash drag.
- Add compare mode on `/eval/`.
- Keep all account actions auditable.

## Known Risks

- The app loads Tailwind, htmx, Chart.js, Lucide, fonts, and EasyMDE from CDNs.
  Containers still need internet access for a fully styled UI unless assets are
  vendored.
- Native `<select>` popup styling differs by browser/OS. CSS improves it, but
  final validation must happen in the target browser.
- `MockBroker` state is in memory; account metadata persists, positions/trades do
  not survive restart yet.
- Notes editor dark mode likely needs a dedicated pass because EasyMDE uses its
  own nested DOM and current overrides are mostly fixed light-theme colors.
