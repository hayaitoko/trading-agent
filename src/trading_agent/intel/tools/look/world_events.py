"""``world_events`` — GDELT-based global event feed (gated by SITUATION_GDELT).

Tool name:      world_events
Args:           theme=None (str|None), timespan="24h" (str)
ToolResult:     ok=True,  data={"bins": [...], "articles": [...]}
                ok=False, error=ToolError(kind="disabled", …) when flag off
                ok=False, error=ToolError(kind="network_error", …) on provider failure
Latency tier:   medium (15-min cache inside GDELTProvider)
Cost class:     free
Gating flag:    SITUATION_GDELT (user_settings, default False)

Purpose
-------
Wraps :class:`~trading_agent.data.providers.gdelt.GDELTProvider` to expose
GDELT macro/geopolitical signals as a LOOK tool.  The trader calls this tool
when it wants to understand the current global macro environment:

  world_events()                 → WAR + ELECTION themes, 24h window
  world_events(theme="EPU_POLICY_*")  → economic-policy-uncertainty focus
  world_events(theme="WAR", timespan="48h") → extended look-back

Returns ``bins`` (volume or tone timeline) and ``articles`` (recent headlines).
Both are absent / empty list when the provider is disabled or unavailable.

The trader should read this tool when:
  - Deciding whether macro risk warrants a more defensive posture.
  - Checking if a geopolitical event is driving unusual volume in a sector.
  - Seeking qualitative context before a large position.

Money-is-real note: this tool never discloses paper/live status, account
value, or position details.  It is pure external data.
"""

from __future__ import annotations

from typing import Any

from ._base import LookToolBase

# Default GDELT themes queried when theme=None
_DEFAULT_THEMES = ["WAR", "ELECTION", "EPU_POLICY_GOVERNMENT_SPENDING"]


class WorldEventsTool(LookToolBase):
    """GDELT-based global macro/geopolitical event feed.

    Parameters
    ----------
    owner_user_id, trader_id:
        Namespace identifiers (used for settings lookup).
    settings_store:
        Duck-typed: ``get(user_id, key, default)`` — used to check the
        ``SITUATION_GDELT`` flag.  ``None`` → tool behaves as disabled.
    gdelt_provider:
        Pre-constructed :class:`~trading_agent.data.providers.gdelt.GDELTProvider`.
        ``None`` → tool returns disabled error even when flag is on.
    """

    TOOL_META: dict[str, Any] = {
        "name": "world_events",
        "description": (
            "Global macro and geopolitical event feed from GDELT. "
            "Returns a mention-volume timeline and recent headlines for a "
            "theme (e.g. WAR, ELECTION, EPU_POLICY_*). "
            "Use to check whether global events warrant a regime shift in your posture. "
            "Enable via SITUATION_GDELT in trader settings."
        ),
        "args": {
            "theme": "str|None (default None — queries WAR + ELECTION + EPU_POLICY)",
            "timespan": "str (default '24h'; e.g. '48h', '7d')",
        },
        "latency": "medium",
        "cost_class": "free",
        "enabled": True,  # wired — returns disabled error when flag off
        "disabled_reason": None,
    }

    def __init__(
        self,
        *,
        owner_user_id: str | None = None,
        trader_id: str,
        settings_store: Any = None,
        gdelt_provider: Any = None,
    ) -> None:
        super().__init__(owner_user_id=owner_user_id, trader_id=trader_id)
        self._settings = settings_store
        self._gdelt = gdelt_provider

    def __call__(
        self,
        theme: str | None = None,
        timespan: str = "24h",
    ) -> Any:
        """Fetch GDELT macro event timeline and recent headlines.

        Parameters
        ----------
        theme:
            GDELT GKG theme string.  ``None`` → queries default macro themes
            (WAR, ELECTION, EPU_POLICY_GOVERNMENT_SPENDING).
        timespan:
            Rolling lookback window.  ``"24h"`` (default), ``"48h"``, ``"7d"``.

        Returns
        -------
        ToolResult
            ok=True, data={"theme": str, "timespan": str,
              "bins": [{"bucket_start": str, "value": float, "unit": str}, ...],
              "articles": [{"title": str, "url": str, "published": str,
                            "source_domain": str, "tone": float}, ...]}

            ok=False, error=ToolError(kind="disabled")
                When SITUATION_GDELT flag is off.

            ok=False, error=ToolError(kind="network_error", message=...)
                On provider network/HTTP failure.

        Example
        -------
        >>> tool = WorldEventsTool(trader_id="Alpha")
        >>> result = tool()
        >>> result.ok
        False
        >>> result.error.kind
        'disabled'
        """
        # Check feature flag
        if not self._flag_enabled():
            return self._err(
                "disabled",
                "world_events: enable SITUATION_GDELT in trader settings to use this tool.",
            )

        if self._gdelt is None:
            return self._err("disabled", "world_events: GDELT provider not initialised.")

        # Determine theme(s) to query
        if theme is None:
            # Query each default theme and merge bins
            all_bins = []
            all_articles = []
            for t in _DEFAULT_THEMES:
                try:
                    bins = self._gdelt.timeline_volume(t, timespan)
                    all_bins.extend([_bin_to_dict(b) for b in bins])
                except Exception as exc:  # noqa: BLE001
                    return self._err("network_error", f"GDELT error for theme {t}: {exc}")
            try:
                all_articles = [_article_to_dict(a) for a in self._gdelt.top_articles("WAR", n=10)]
            except Exception as exc:  # noqa: BLE001
                return self._err("network_error", f"GDELT articles error: {exc}")
            resolved_theme = ",".join(_DEFAULT_THEMES)
            return self._ok(
                {
                    "theme": resolved_theme,
                    "timespan": timespan,
                    "bins": all_bins,
                    "articles": all_articles,
                }
            )
        else:
            try:
                bins = self._gdelt.timeline_volume(theme, timespan)
                articles = self._gdelt.top_articles(theme, n=10)
            except Exception as exc:  # noqa: BLE001
                return self._err("network_error", f"GDELT error: {exc}")
            return self._ok(
                {
                    "theme": theme,
                    "timespan": timespan,
                    "bins": [_bin_to_dict(b) for b in bins],
                    "articles": [_article_to_dict(a) for a in articles],
                }
            )

    # ------------------------------------------------------------------

    def _flag_enabled(self) -> bool:
        if self._settings is None:
            return False
        try:
            return bool(
                self._settings.get(self.owner_user_id or "", "SITUATION_GDELT", False)
            )
        except Exception:  # noqa: BLE001
            return False


def _bin_to_dict(b: Any) -> dict[str, Any]:
    return {
        "bucket_start": b.bucket_start.isoformat(),
        "value": b.value,
        "unit": b.unit,
    }


def _article_to_dict(a: Any) -> dict[str, Any]:
    return {
        "title": a.title,
        "url": a.url,
        "published": a.published.isoformat(),
        "source_domain": a.source_domain,
        "tone": a.tone,
    }
