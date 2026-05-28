"""LOOK toolkit — read-only information tools for the agent trader (A1).

One module per tool; all return :class:`~trading_agent.intel.tool_envelope.ToolResult`.

Enabled tools (wrap existing services):
  list_tools, recent_turns, history, news, research_brief, request_research,
  situation, watchlist, account_state, memory_search, advisor_notes, ask_manager

Disabled stubs (provider lands in WS-Situation+Forecast):
  world_events, prediction_market_odds, options_iv, forecast

Import the individual modules to access tool callables:

    from trading_agent.intel.tools.look.list_tools import ListToolsTool
    from trading_agent.intel.tools.look.advisor_notes import AdvisorNotesTool
    …
"""
