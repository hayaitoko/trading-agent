"""FastAPI routers — one module per workstream (CONTRACTS.md §HTTP route table).

All routers are fully implemented. ``config`` (WS-0) handles auth, settings,
endpoints, and sources; every other router ships its complete implementation
with graceful degradation when the relevant ``app.state`` dependency is absent.
"""
