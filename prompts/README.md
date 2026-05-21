# prompts/

System prompts injected by the Artoo orchestrator for each role in a round.

## Role flow

```
PLANNER → (1-4 tasks) → CODER × N → REVIEWER → FIXER → ORCH_REVIEW → PROGRESS.md
```

| File | Role | Model (Artoo default) | Input | Output |
|---|---|---|---|---|
| `planner.md` | Plan the next round from the active milestone | Kimi K2.6 | HANDOFF.md, PROGRESS.md, PLAN.md, AGENTS.md, DESIGN.md | JSON task list |
| `coder.md` | Implement one task | DeepSeek v4-pro (large), v4-flash (small/med) | Task JSON + referenced files | Full file contents per touched path |
| `reviewer.md` | Find bugs across all coder outputs | Kimi K2.6 (+ HY3 cross-check) | All coder outputs + original files | JSON bug report |
| `fixer.md` | Apply fixes from review | DeepSeek v4-pro | Bug report + current files | Updated file contents |
| `orchestrator.md` | Meta — round gating, retries, escalation | Claude Sonnet 4.6 (Artoo boss) | Full round packet | Decision + PROGRESS.md entry |

## Conventions used by every prompt

- The canonical state files an agent should treat as ground truth: `HANDOFF.md`, `PROGRESS.md`, `PLAN.md`, `AGENTS.md`, `DESIGN.md`.
- The codebase rules an agent must follow: enumerated in `AGENTS.md`. Every prompt below tells the worker to honor these rules; the planner enforces them at planning time, the reviewer at review time.
- Prompt size guard: planner outputs are sized so a single coder task stays under ~60K characters of context (per Artoo's documented constraint).
- No prompt depends on a specific previous run; each round is self-contained given the state files.

## How to update these prompts

If a rule or convention changes, update `AGENTS.md` first. The prompts here reference AGENTS.md by section, so a behavior change downstream just requires updating the rule, not the prompt. Don't add per-round prompts here; per-round briefs live in `PLAN.md` under each milestone.
