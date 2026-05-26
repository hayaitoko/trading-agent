# trading-agent — Product Vision

> **This is its own product.** It is **not** `agent-interface` (that's Artoo's
> development webapp). This project has its own dedicated web UI, described below.

## North star

A fully autonomous **agentic trading harness** that researches the market on its
own, keeps clean organized notes, and uses them to make **informed trades, fast** —
all operated through a single polished web UI with human-in-the-loop control where
the operator wants it.

## The autonomous loop

```
  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ 1. INGEST   │──▶│ 2. DISTILL   │──▶│ 3. REMEMBER  │──▶│ 4. DECIDE &  │
  │ collect news│   │ clean notes, │   │ notes the    │   │    TRADE     │
  │ + info from │   │ organized by │   │ agent queries│   │ skills pull  │
  │ live sources│   │ SECTOR →     │   │ before acting│   │ notes → fast │
  │             │   │ COMPANY      │   │ (RAG/vault)  │   │ informed     │
  └─────────────┘   └──────────────┘   └──────────────┘   │ trades       │
        ▲                                                  └──────┬───────┘
        │                                                         │
        │              ┌──────────────────────────────┐          │
        └──────────────│ 5. SUPERVISE (operator + UI)  │◀─────────┘
                       │ autonomous OR manual-approval │
                       │ per account                   │
                       └──────────────────────────────┘
```

1. **Ingest** — agentically collect news and information from configured sites/sources.
2. **Distill** — compile clean notes, organized **by sector, then by company**.
3. **Remember** — store notes in a queryable, **timestamped** memory the agent reads
   before acting. Retrieval is **recency-aware** so trades use the *freshest* info
   (recommended store under *Decisions* below).
4. **Decide & trade** — skills let the agent pull the relevant notes and place
   trades **quickly**, across one or more accounts.
5. **Supervise** — the operator watches and steers via the UI; some accounts trade
   autonomously, others require manual approval.

## Design principles

- **Freshness is first-class.** Every ingested item and every note carries ISO-8601
  timestamps (`published_at` when known, plus `ingested_at`). Retrieval **prefers the
  most recent** info; stale notes are down-ranked or expired. The trading agent must
  always act on the **latest** available information.
- **Human-readable *and* machine-queryable memory.** Notes are browsable by a human
  (the memory page) *and* fast to retrieve semantically by the agent.
- **Human-in-the-loop where it matters.** Per-account autonomous vs. manual-approval.
- **Auditable.** Every trade traces back to the notes/news that motivated it.

## Web UI

A single cohesive app (purpose-built for this project):

- **Chat window** — docked on the **right**; converse with / direct the agent.
- **Accounts page (main)** — multiple accounts laid out nicely; per-account state,
  positions/P&L, and a **"manual approval" toggle** per account.
- **Right pop-out menu** — surfaces:
  - **alerts**
  - **approval requests** from accounts with manual-approval enabled
  - a **ticker of incoming information** (live news flow)
- **News / Sources page** — refine the sources it pulls from, and view the news
  it's bringing in.
- **Memory page** — the notes / RAG (Obsidian?) the agent acts on when it trades.
- **Settings page** — API keys and network activity / ingress.

## What already exists vs. what's net-new

**Built this session (building blocks, currently standalone — to be folded in):**
- `llm/` + `bench/` — LLM traders + a multi-model evaluation bench. This is the
  prototype of the **decision/skills engine** (step 4) and a way to compare models.
- `web/` alerts + `approval_queue.py` — the **alerts + manual-approval requests**
  that belong in the pop-out menu (step 5).
- `paper_broker.py` (N isolated books), `alpaca_broker.py`, `ccxt_broker.py`,
  `risk_manager.py`, `audit.py` — **execution + safety** layer (multi-account, paper/live).

**Net-new (not built yet):**
- News/info **ingestion** + **source management** (step 1).
- **Note compilation** organized by sector → company (step 2).
- **Memory / RAG** store the agent reads before trading (step 3).
- **Multi-account** UI with per-account autonomous/manual toggle.
- The **unified web UI** (chat, accounts, pop-out, news, memory, settings).

## Decisions

- **Memory store → hybrid, Qdrant-led.** Keep **timestamped markdown notes**
  (organized sector → company, Obsidian-readable) as the human-facing source of
  truth, and build a **Qdrant** vector index *on top* for the agent's retrieval.
  Payload carries `sector`, `company`, `source`, `url`, `published_at`,
  `ingested_at` so retrieval combines **semantic relevance + recency + filters**.
  *Rationale:* pure Obsidian doesn't scale for retrieval and bloats; a pure vector
  store is opaque to the operator. Hybrid gives readability *and* fast,
  recency-aware recall. Apply hygiene from day one (dedup, rollups, expiry).

## Open questions (to pin down before/while building)

1. **"Accounts"** — real Alpaca accounts, internal books per model/strategy, or a mix?
   (The bench currently uses one data key → N internal paper books.)
2. **Paper vs live** money for autonomous trading, and the guardrails for going live.
3. **Chat scope** — Q&A only, or can it direct trades / change config / steer research?
4. **News sources** — which sites, and how (scraping, RSS, news APIs)?
5. **UI stack** — purpose-built; vanilla-JS SPA (like the current pages) vs a framework.
6. **"Quickly"** — target decision-to-order latency / cadence.

## Status

- Framework v1 (brokers, risk, signal routing, strategies, demo) — complete.
- Alerts/notification center + approval queue — complete (standalone page).
- Model bench (LLM traders + leaderboard, OpenRouter w/ ZDR) — complete (standalone page).
- All of the above **uncommitted** as of writing.
- The autonomous research → notes → memory loop and the unified UI — **not started**.
