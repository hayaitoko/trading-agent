"""Data-provider adapters behind the WS-A interfaces.

Each module here implements :class:`~trading_agent.data.history.BarProvider` or
:class:`~trading_agent.data.history.FundamentalsProvider`. They are imported
on demand (not from this ``__init__``) so the heavy/optional dependencies
(alpaca-py, httpx) only load when a provider is actually used.
"""
