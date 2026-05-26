"""FastAPI routers — one module per workstream (CONTRACTS.md §HTTP route table).

WS-0 ships ``config`` fully implemented (auth, settings, endpoints, sources) and
every other router as 501 stubs carrying the correct method/path + ``current_user``
dependency, so each owning stream just fills the body.
"""
