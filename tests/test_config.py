"""Tests for trading_agent.config: YAML/TOML loaders and env-var substitution."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from trading_agent.config import (
    ConfigError,
    _substitute_env_vars,
    get_credentials,
    load_config,
    load_strategy_config,
)

# --- _substitute_env_vars -----------------------------------------------------


def test_substitute_env_vars_plain_ref():
    with patch.dict(os.environ, {"FOO": "bar"}, clear=False):
        assert _substitute_env_vars("hello ${FOO}") == "hello bar"


def test_substitute_env_vars_with_default_used():
    with patch.dict(os.environ, {}, clear=True):
        assert _substitute_env_vars("x=${MISSING:fallback}") == "x=fallback"


def test_substitute_env_vars_with_default_overridden():
    with patch.dict(os.environ, {"SET": "real"}, clear=False):
        assert _substitute_env_vars("x=${SET:fallback}") == "x=real"


def test_substitute_env_vars_missing_no_default_yields_empty():
    with patch.dict(os.environ, {}, clear=True):
        assert _substitute_env_vars("x=${NOPE}") == "x="


def test_substitute_env_vars_no_refs_passthrough():
    assert _substitute_env_vars("plain string with no refs") == "plain string with no refs"


def test_substitute_env_vars_multiple_in_one_line():
    with patch.dict(os.environ, {"A": "1", "B": "2"}, clear=False):
        assert _substitute_env_vars("${A}-${B}-${C:3}") == "1-2-3"


# --- load_config (YAML) -------------------------------------------------------


def test_load_config_basic_yaml(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("mode: autonomous\nrisk:\n  max_loss: 100\n")
    out = load_config(cfg)
    assert out == {"mode": "autonomous", "risk": {"max_loss": 100}}


def test_load_config_substitutes_env_vars(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("mode: ${MODE:approval}\napi_key: ${API_KEY}\n")
    with patch.dict(os.environ, {"API_KEY": "secret123"}, clear=False):
        # MODE unset -> fallback to default "approval"
        env_no_mode = {k: v for k, v in os.environ.items() if k != "MODE"}
        with patch.dict(os.environ, env_no_mode, clear=True):
            os.environ["API_KEY"] = "secret123"
            out = load_config(cfg)
    assert out["mode"] == "approval"
    assert out["api_key"] == "secret123"


def test_load_config_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_invalid_yaml_raises_config_error(tmp_path):
    cfg = tmp_path / "bad.yaml"
    # Unclosed flow mapping is unambiguous YAML error
    cfg.write_text("key: [1, 2,\n")
    with pytest.raises(ConfigError, match="parsing YAML"):
        load_config(cfg)


def test_load_config_top_level_must_be_mapping(tmp_path):
    cfg = tmp_path / "list.yaml"
    cfg.write_text("- one\n- two\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(cfg)


# --- load_strategy_config (TOML) ----------------------------------------------


def test_load_strategy_config_basic(tmp_path):
    cfg = tmp_path / "strat.toml"
    cfg.write_text(
        'timeframe = "1m"\nsymbols = ["AAPL"]\n\n[strategy]\nsma_period = 20\n'
    )
    out = load_strategy_config(cfg)
    assert out["timeframe"] == "1m"
    assert out["symbols"] == ["AAPL"]
    assert out["strategy"]["sma_period"] == 20


def test_load_strategy_config_with_env_substitution(tmp_path):
    cfg = tmp_path / "strat.toml"
    cfg.write_text('timeframe = "${TF:5m}"\n')
    # Make sure TF is unset
    env = {k: v for k, v in os.environ.items() if k != "TF"}
    with patch.dict(os.environ, env, clear=True):
        out = load_strategy_config(cfg)
    assert out["timeframe"] == "5m"


def test_load_strategy_config_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_strategy_config(tmp_path / "missing.toml")


def test_load_strategy_config_invalid_toml_raises(tmp_path):
    cfg = tmp_path / "bad.toml"
    cfg.write_text("this is = not = valid toml\n")
    with pytest.raises(ConfigError, match="parsing TOML"):
        load_strategy_config(cfg)


# --- get_credentials ----------------------------------------------------------


def test_get_credentials_pulls_from_env():
    config = {"credentials": {"api_key": "${MY_KEY}", "secret": "${MY_SEC}"}}
    with patch.dict(os.environ, {"MY_KEY": "kkk", "MY_SEC": "sss"}, clear=False):
        out = get_credentials(config)
    assert out == {"api_key": "kkk", "secret": "sss"}


def test_get_credentials_passes_through_literals():
    config = {"credentials": {"api_key": "literal_value", "port": 1234}}
    out = get_credentials(config)
    assert out == {"api_key": "literal_value", "port": "1234"}


def test_get_credentials_missing_env_var_raises():
    config = {"credentials": {"api_key": "${DEFINITELY_NOT_SET_XYZ}"}}
    env = {k: v for k, v in os.environ.items() if k != "DEFINITELY_NOT_SET_XYZ"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ConfigError, match="DEFINITELY_NOT_SET_XYZ"):
            get_credentials(config)


def test_get_credentials_no_credentials_block_returns_empty():
    assert get_credentials({}) == {}
    assert get_credentials({"other": "x"}) == {}
