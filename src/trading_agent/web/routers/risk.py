"""Risk router (WS-I): emergency stop + per-user risk limits.

The edited limits are the operator's, so they persist per ``user_settings``
(``risk_limits`` — the same key the cockpit's Settings tab reads). When a live
global :class:`~trading_agent.risk_manager.RiskManager` is attached at
``app.state.risk`` (the serve process), the kill switch reflects/toggles the real
engine and limit edits are mapped onto it best-effort; without one the routes
still round-trip against ``user_settings`` so the cockpit's toggle and limit
editor work in any deployment.

Cockpit limit keys map onto :class:`RiskLimits` fields:
``dailyLoss``→max_daily_loss · ``maxPos``→max_position_size ·
``tradesHour``→max_trades_per_hour · ``openPos``→max_open_positions.
(``perTrade``/``exposure`` have no engine field; they are persisted only.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from ...config.settings_store import SettingsStore
from ...config.users import current_user
from ...risk_manager import RiskLimits

if TYPE_CHECKING:
    from ...risk_manager import RiskManager

router = APIRouter(tags=["risk"])


class KillIn(BaseModel):
    active: bool | None = None


def _settings(request: Request) -> SettingsStore:
    return request.app.state.settings  # type: ignore[no-any-return]


def _risk(request: Request) -> RiskManager | None:
    return getattr(request.app.state, "risk", None)


def _apply_limits(rm: RiskManager, limits: dict[str, Any]) -> None:
    """Map the cockpit's limit keys onto the live RiskManager's RiskLimits."""
    cur = rm.limits

    def pick(key: str, default: Any, cast: Any) -> Any:
        try:
            return cast(limits[key])
        except (KeyError, TypeError, ValueError):
            return default

    rm.limits = RiskLimits(
        max_daily_loss=pick("dailyLoss", cur.max_daily_loss, float),
        max_position_size=pick("maxPos", cur.max_position_size, float),
        max_trades_per_hour=pick("tradesHour", cur.max_trades_per_hour, int),
        max_open_positions=pick("openPos", cur.max_open_positions, int),
    )


@router.get("/api/risk")
def risk(request: Request, user_id: str = Depends(current_user)) -> dict[str, Any]:
    """Current emergency-stop state + the user's saved limits."""
    settings = _settings(request)
    rm = _risk(request)
    kill = rm.kill_switch_active if rm is not None else bool(settings.get(user_id, "risk_kill", False))
    return {"kill": kill, "limits": settings.get(user_id, "risk_limits", {}) or {}}


@router.put("/api/risk/limits")
def put_limits(
    request: Request, body: dict[str, Any], user_id: str = Depends(current_user)
) -> dict[str, Any]:
    """Persist edited limits per user, and apply them to the live engine if present."""
    settings = _settings(request)
    stored = dict(settings.get(user_id, "risk_limits", {}) or {})
    stored.update(body)
    settings.set(user_id, "risk_limits", stored)
    rm = _risk(request)
    if rm is not None:
        _apply_limits(rm, stored)
    return {"limits": stored}


@router.post("/api/risk/kill")
def kill(
    request: Request, body: KillIn | None = None, user_id: str = Depends(current_user)
) -> dict[str, Any]:
    """Toggle the emergency stop (defaults to engaging it)."""
    active = bool(body.active) if body is not None and body.active is not None else True
    settings = _settings(request)
    settings.set(user_id, "risk_kill", active)
    rm = _risk(request)
    if rm is not None:
        rm.activate_kill_switch() if active else rm.deactivate_kill_switch()
    return {"kill": active}
