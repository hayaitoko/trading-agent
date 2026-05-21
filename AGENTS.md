# AGENTS — instructions for any agent working on trading-agent

> Constraints, conventions, and forbidden moves. Apply to every coding round.
> If a rule conflicts with a user instruction in a specific session, the user wins; revert this doc after.

## Core principles

1. **Tests green at the end of every task.** Never merge a task that drops a test. Current floor: 103 passing in `pytest -q`. If you have to disable a test, you have to explain why in the commit body.
2. **Ruff clean at the end of every task.** Run `uv run ruff check .` before declaring done. Use `uv run ruff check --fix .` for the auto-fixable, hand-fix the rest.
3. **Provenance over freshness.** Never delete information in `notes/` or in a log. Timestamp it. The consolidator's system prompt makes this explicit; honor it in any new feature that writes to memory.
4. **Pick a path, defend it, let the user redirect.** Don't ship a menu of options where one good answer would do. Comments explaining trade-offs are welcome.
5. **No half-finished implementations.** Either a feature works behind the abstraction it claims to be, or it raises `NotImplementedError` and the caller never reaches it. No silent fakes.
6. **No emojis in files, no em dashes (—) in prose.** Lukas hates em dashes. Use commas, colons, parens, or hyphen-hyphen instead.

## Testing requirements

- **Async tests**: `pytest-asyncio` is in `asyncio_mode = "auto"`. `async def test_*` functions are awaited automatically. Don't call `asyncio.run()` inside an async test — it'll error because the loop is already running.
- **TestClient**: FastAPI's `TestClient` is sync; tests using it are `def test_*`, not `async`.
- **State isolation**: use `tmp_path` for any test that touches the filesystem. Don't rely on `accounts.json` etc. existing at the repo root.
- **Mocking OpenRouter**: pass a fake `model_caller` to `ChatService` or `Consolidator`. Look at `tests/test_chat.py::fake_caller` for the pattern. Never hit the real API in tests.
- **Smoke before commit**: for non-trivial UI changes, boot the demo server and curl each affected route. The `until` + `curl` pattern at the top of `tests/` smoke sections works.

## Commit conventions

- Subject line: `<area>: <imperative>` (lowercase, no period). Examples: `chat: persistent sidebar with multi-model OpenRouter + tool use`, `notes: agent memory with timestamp convention + curation consolidator`.
- Body: bullet what changed and why. Always include test count + ruff status at the end (`103/103 tests pass. Ruff clean.`).
- One commit per coherent feature. Multiple small commits are fine; squash before push only if asked.
- Always `git pull --rebase origin main` before pushing — Lukas occasionally commits via the GitHub web UI.

## Aesthetic

The look is **editorial finance**, light parchment + warm dark.

- Display serif: **Bodoni Moda** (variable, weights 400–900, optical sizing).
- Body sans: **Manrope** (variable, weight 500 default — heavier than typical 400).
- Mono: **IBM Plex Mono**.
- Color tokens via CSS custom properties (`--paper-base`, `--ink-100`, `--vermillion`, etc.). Tailwind reads them via `rgb(var(--name) / <alpha-value>)`. Never hardcode hex in templates; always reference the token.
- Single accent: `--vermillion`. Used sparingly — live state, active links on hover, primary buttons, text selection. Adding more accents weakens the brand.
- Status indicators are small squares (`.mark`), not pills. Active = solid vermillion, paused = outlined ink, warn = solid amber.
- Tables use `.ledger` (hairlines above + below the header, no full borders, no zebra). Numbers are right-aligned mono with `tnum`.
- Placeholder pages use the daggered `†` convention in the nav and the `_status_banner.html` partial.
- Animations: one orchestrated page-load fade-up via `.reveal-stack > *`, hover lift on cards via `.lift`, soft pulse on `.mark-active`. No scattered micro-animations.
- Tooltips: add `data-tip="..."` (CSS-only). Add `data-tip-below` for elements near the top of the viewport so the tooltip doesn't get cropped.

When in doubt, look at `dashboard.html` and `trades.html` — they're the canonical examples.

## What never to break

- **Tests**: 103 passing.
- **Editorial palette**: no hex literals in templates; always `rgb(var(--token))` or Tailwind `bg-paper-base` style.
- **Compliance wall**: nothing in this repo ever touches LPL ClientWorks, Orion, Black Diamond, or anything connected to Advantage Wealth Advisors. Paper trading only. No real-money broker creds in `.env` or anywhere else.
- **Notes consolidator is non-destructive**: the system prompt forbids deletion. Don't add a "prune stale" path. Adding metadata, merging duplicates, improving formatting — all fine.
- **MockBroker is in-memory only**: don't try to persist its state. The Account spec (id/name/starting_cash/model/enabled) is what persists.
- **Chat history persists server-side** in `chat_history.json`. If you make it client-side localStorage, you break multi-browser. Don't.
- **OpenRouter is the only LLM gateway.** No direct provider SDKs. This keeps the swap-the-model story coherent and keeps the network log informative.

## File ownership map

| Layer | Owner | Touch when |
|---|---|---|
| `src/trading_agent/models.py` | Wire format | Adding a new dataclass that crosses layer boundaries |
| `src/trading_agent/brokers/base.py` | `Broker` ABC contract | Adding a method ALL impls must support |
| `src/trading_agent/brokers/<impl>.py` | Concrete broker | Building a new broker (Investopedia, Alpaca, sim) |
| `src/trading_agent/scrapers/base.py` | `Scraper` ABC | Almost never — `poll()` is the only contract |
| `src/trading_agent/scrapers/<source>.py` | Forum-specific | Adding a source |
| `src/trading_agent/chat/tools.py` | Chat tools the LLM can call | Adding a tool the chat assistant should have access to. Update `TOOL_SCHEMAS` + `execute()` + `SYSTEM_PROMPT` together. |
| `src/trading_agent/chat/client.py` | OpenRouter HTTP | Almost never — touching this affects every model call |
| `src/trading_agent/notes/storage.py` | Markdown file I/O | Almost never — path validation and history backups are load-bearing |
| `src/trading_agent/notes/consolidator.py` | Memory curation prompt + loop | Tuning the consolidator |
| `src/trading_agent/web/routes/*.py` | One file per nav section | Adding endpoints |
| `src/trading_agent/web/templates/*.html` | Page templates | UI changes |
| `src/trading_agent/web/templates/base.html` | Layout + CSS variables + theme | Palette, fonts, header, theme toggle |
| `src/trading_agent/web/static/{chat,notes}.js` | Client-side editor + chat logic | UI interactions |
| `src/trading_agent/web/app.py` | App factory + middleware + lifespan | Registering a new route or background task |

## When to ask vs. when to just execute

- **Just execute** if: the change is mechanical (rename a field, add a column, write a test), the scope is one file or one tightly coupled set, or the user already gave clear direction in the issue.
- **Ask first** if: there's an aesthetic decision (light/dark variants, font swap), a data shape decision (new DB schema, breaking change to `accounts.json`), or a UX decision the user might reasonably want differently (button location, copy tone).
- Ask **before** doing destructive operations on `notes/`, `accounts.json`, or any state file. Even with the `.history/` backup.

## How rounds are scoped

The Artoo orchestrator splits a milestone into 1–4 tasks per round. Aim for:

- Each task < 60K chars of context (prompts + diffs).
- Each task has a clear acceptance test in `PLAN.md`.
- Tasks within a round are independent — a coder running task B should not need task A's output.
- If a task spans both backend and a new template, that's still one task; the integration is the point.

## Forbidden

- Calling provider SDKs directly (`anthropic`, `openai`, `google-generativeai`). Use `chat.client.call_model` through OpenRouter.
- Adding a database other than SQLite without a migration plan.
- Adding auth without an explicit user ask. This is a single-user homelab tool.
- Changing the URL structure of `/eval/`, `/notes/`, `/accounts/`, `/settings/` — these are linked from the README and from notes the chat may have written.
- Removing the `†` placeholder convention without first promoting the corresponding page to a real implementation.
- Committing `accounts.json`, `trading_agent_secrets.json`, `chat_history.json`, or anything under `notes/` (other than the source `general/README.md` if you're seeding via a migration).

## Pre-commit checklist

1. `uv run pytest -q` — 103+ passing.
2. `uv run ruff check .` — clean.
3. Tooltips on new interactive elements.
4. Templates use CSS variables, not hex literals.
5. No em dashes in prose.
6. Commit message follows convention (subject + bulleted body + test+ruff status).
7. `git pull --rebase origin main` before push.
