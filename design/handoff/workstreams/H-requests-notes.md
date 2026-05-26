# WS-H · Stock-requests + advisor notes (Wave 2, parallel)

**Goal:** make two cockpit features real — (1) the notification center's **stock-requests** (a trader
asks to trade a symbol outside its universe → operator allows → universe updates), and (2) **advisor
notes** (per-trader and per-ticker notes the human writes).

**Depends on:** WS-0 (`notes`, `stock_requests` tables, notification center), `approval_queue.py`
(EXISTS), per-trader universe concept (coordinate with WS-A/bench). **Blocks:** WS-G2 (Notifications +
notes UI).

**Owns (create/edit):**
- `notes.py` — get/put notes keyed by `(user_id, scope, ref)` where scope ∈ {trader,ticker}; backs the
  account-window notes box and the ticker-card notes. Fill `web/routers/notes.py`:
  `GET/PUT /api/notes?scope=&ref=`.
- `requests.py` — a trader emits a `StockRequest{user_id,trader_id,symbol,reason}` when it wants a
  symbol outside its `universe`. Persist to `stock_requests`; surface in the **notification center**
  (`web/notifications.py`) as a `request`-type item. On **allow**: add the symbol to that trader's
  universe (the tradable-symbols set the trader/bench reads) and mark fulfilled; on **decline**: mark
  declined. Fill `web/routers/requests.py`: `GET /api/requests`, `POST /api/requests/{id}/allow|decline`.
- Extend `web/routers/notifications.py` (stubbed by WS-0) to merge: market-move alerts
  (`web/market_watch.py`), fills, blocks, and these stock-requests — matching the cockpit `NOTIFS`
  shape (types: request|alert|fill|block, unread flag).
- Define where a trader's `universe` lives (per `(user_id, trader_id)`); coordinate with bench so the
  trader's tradable-symbols actually reflect it.

**Steps:** notes store+router → universe storage → stock_requests model+router (allow updates universe)
→ wire requests + alerts + fills into the notifications router → tests.

**Acceptance:**
- Note round-trips per (user,scope,ref); isolated per user.
- A request → appears in `GET /api/notifications` as a request → allow adds the symbol to that trader's
  universe (verified) → status flips; decline leaves universe unchanged.
- Notifications endpoint merges request/alert/fill/block with unread counts in the cockpit's shape.
- ruff + mypy green.

**Out of scope:** the chat/manager (WS-E). Approval-queue internals (reuse it as-is).
