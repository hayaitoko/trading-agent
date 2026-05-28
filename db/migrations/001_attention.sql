-- Migration 001: attention_queue table for WS-Agent A2 NOTE toolkit.
--
-- Stores pending reminders and watchpoints set by trader agents.
-- Rows are soft-fired (fired_at IS NOT NULL) rather than hard-deleted,
-- preserving a full audit trail of what the agent deferred.
--
-- The partial index covers the hot scheduler path: scan unfired rows
-- per trader without a full-table scan.
--
-- Idempotent: uses CREATE TABLE/INDEX IF NOT EXISTS throughout.

CREATE TABLE IF NOT EXISTS attention_queue (
    id           INTEGER PRIMARY KEY,
    trader_id    TEXT    NOT NULL,
    kind         TEXT    NOT NULL,       -- 'reminder' | 'watchpoint'
    payload_json TEXT    NOT NULL,       -- JSON: {symbol?, when_unix?, condition?, why}
    created_at   INTEGER NOT NULL,       -- Unix seconds UTC
    expires_at   INTEGER NOT NULL,       -- Unix seconds UTC
    fired_at     INTEGER,               -- NULL = unfired; set when condition trips or reminder elapses
    fire_reason  TEXT,                   -- why it fired ('elapsed', 'price_sigma', 'news_rate',
                                         --   'realized_vol', 'approval_queue', 'condition', 'expired')
    FOREIGN KEY (trader_id) REFERENCES traders(id) ON DELETE CASCADE
);

-- Partial index: scheduler polls only unfired rows, narrowed by trader.
CREATE INDEX IF NOT EXISTS idx_attention_pending
    ON attention_queue(trader_id, fired_at)
    WHERE fired_at IS NULL;
