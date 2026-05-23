"""Loads per-strategy TOML parameter files."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from .config import _substitute_env_vars  # type: ignore[attr-defined]


class StrategyConfigError(Exception):
    """Raised when strategy config cannot be loaded or parsed."""


def _default_config_dir() -> Path:
    """Project-root/strategies/config relative to this module."""
    # .../src/trading_agent/strategy_loader.py → project root is 3 dirs up.
    return Path(__file__).resolve().parents[2] / "strategies" / "config"


def _resolve_config_dir(config_dir: Path | str | None) -> Path:
    if config_dir is not None:
        return Path(config_dir)
    env = os.getenv("TRADING_AGENT_CONFIG_DIR")
    return Path(env) if env else _default_config_dir()


def load_strategy_parameters(
    strategy_name: str, config_dir: Path | str | None = None
) -> dict[str, Any]:
    """Load ``<strategy_name>.toml`` from the resolved config directory."""
    base = _resolve_config_dir(config_dir)
    config_path = base / f"{strategy_name}.toml"
    if not config_path.exists():
        raise StrategyConfigError(f"Strategy configuration file not found: {config_path}")

    try:
        content = _substitute_env_vars(config_path.read_text())
        return tomllib.loads(content)
    except tomllib.TOMLDecodeError as e:
        raise StrategyConfigError(f"Error parsing TOML in {config_path}: {e}") from e


def load_all_strategy_parameters(
    config_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load every ``*.toml`` in the resolved config directory."""
    base = _resolve_config_dir(config_dir)
    if not base.exists():
        raise StrategyConfigError(f"Configuration directory not found: {base}")

    strategies: dict[str, dict[str, Any]] = {}
    for path in sorted(base.glob("*.toml")):
        try:
            content = _substitute_env_vars(path.read_text())
            strategies[path.stem] = tomllib.loads(content)
        except tomllib.TOMLDecodeError as e:
            raise StrategyConfigError(f"Error parsing TOML in {path}: {e}") from e
    return strategies
