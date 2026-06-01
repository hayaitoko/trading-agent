"""Analyst-digest tier (WS-Digest).

A cheap-model pass distils slow-data signals (news, world events, prediction
markets, research briefs, situation) into a compact, token-budgeted per-universe
digest record **once**, then pushes it into the trader's first-look context so
the expensive decider stops paying for repeated gather calls.

Key components:
  - :class:`~.store.DigestStore` — SQLite + best-effort vector index.
  - :class:`~.compiler.DigestCompiler` — cheap-model distillation pass.
  - :class:`~.daemon.DigestDaemon` — background compile cadence + event-wake.

Flag: ``intelligence_flags["digest_mode"]`` (per-trader) or the user-level
``digest_mode`` setting.  Default OFF — behaviour is byte-for-byte identical to
the pre-digest baseline when the flag is off.
"""
