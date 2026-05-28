"""Agent tool catalog sub-packages.

Each sub-package (``look``, ``note``, ``act``) owns one category of tools.
Tools are registered into :class:`~trading_agent.llm.trader.AgentTrader`
by wave:

  A0  — built-ins (list_tools, memory_search, hold, pass)
  A1  — LOOK catalog  (intel/tools/look/)
  A2  — NOTE catalog  (intel/tools/note/)    ← this wave
  A3  — ACT catalog   (intel/tools/act/)
"""
