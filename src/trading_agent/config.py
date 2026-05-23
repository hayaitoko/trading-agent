"""Configuration loading: global YAML + per-strategy TOML with env-var refs."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Union

import yaml

PathLike = Union[str, Path]

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or is invalid."""


def load_config(config_path: PathLike) -> dict[str, Any]:
    """Load global YAML config, substituting ``${VAR}`` / ``${VAR:default}`` refs."""
    try:
        with open(config_path) as f:
            content = f.read()
    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {config_path}") from e

    content = _substitute_env_vars(content)
    try:
        config = yaml.safe_load(content) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML configuration: {e}") from e
    if not isinstance(config, dict):
        raise ConfigError(f"Top-level YAML config must be a mapping, got {type(config).__name__}")
    return config


def load_strategy_config(config_path: PathLike) -> dict[str, Any]:
    """Load per-strategy TOML config with env-var refs."""
    try:
        with open(config_path) as f:
            content = f.read()
    except FileNotFoundError as e:
        raise ConfigError(f"Strategy configuration file not found: {config_path}") from e

    content = _substitute_env_vars(content)
    try:
        return tomllib.loads(content)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Error parsing TOML configuration: {e}") from e


def _substitute_env_vars(content: str) -> str:
    """Substitute ``${VAR_NAME}`` and ``${VAR_NAME:default}`` references."""

    def replace(match: re.Match[str]) -> str:
        spec = match.group(1)
        if ":" in spec:
            name, default = spec.split(":", 1)
        else:
            name, default = spec, ""
        return os.environ.get(name, default)

    return _ENV_VAR_RE.sub(replace, content)


def get_credentials(config: dict[str, Any]) -> dict[str, str]:
    """Pull ``credentials`` block out of a loaded config. Raises if a required
    ``${VAR}`` reference is unset.
    """
    out: dict[str, str] = {}
    for key, value in config.get("credentials", {}).items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            spec = value[2:-1]
            name = spec.split(":", 1)[0] if ":" in spec else spec
            env_value = os.environ.get(name)
            if not env_value:
                raise ConfigError(f"Required environment variable not set: {name}")
            out[key] = env_value
        else:
            out[key] = str(value)
    return out
