"""Tests for configuration system."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from crypto.config.loader import (
    _substitute_env_vars,
    load_yaml_config,
    merge_configs,
    ConfigLoader,
)
from crypto.config.schemas import (
    DatabaseConfig,
    StrategyConfig,
    BacktestConfig,
)


class TestEnvVarSubstitution:
    """Tests for environment variable substitution."""

    def test_simple_substitution(self):
        """Test simple env var substitution."""
        os.environ["TEST_VAR"] = "test_value"
        result = _substitute_env_vars("${TEST_VAR}")
        assert result == "test_value"
        del os.environ["TEST_VAR"]

    def test_default_value(self):
        """Test default value when env var not set."""
        # Ensure var is not set
        os.environ.pop("NONEXISTENT_VAR", None)
        result = _substitute_env_vars("${NONEXISTENT_VAR:default}")
        assert result == "default"

    def test_nested_dict_substitution(self):
        """Test substitution in nested dicts."""
        os.environ["NESTED_VAR"] = "nested_value"
        data = {
            "level1": {
                "level2": "${NESTED_VAR}",
            }
        }
        result = _substitute_env_vars(data)
        assert result["level1"]["level2"] == "nested_value"
        del os.environ["NESTED_VAR"]

    def test_list_substitution(self):
        """Test substitution in lists."""
        os.environ["LIST_VAR"] = "list_value"
        data = ["${LIST_VAR}", "static"]
        result = _substitute_env_vars(data)
        assert result[0] == "list_value"
        assert result[1] == "static"
        del os.environ["LIST_VAR"]


class TestYAMLLoader:
    """Tests for YAML configuration loading."""

    def test_load_yaml_config(self):
        """Test loading a YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({"key": "value", "nested": {"inner": 123}}, f)
            f.flush()

            config = load_yaml_config(f.name)
            assert config["key"] == "value"
            assert config["nested"]["inner"] == 123

            os.unlink(f.name)

    def test_load_yaml_with_env_vars(self):
        """Test loading YAML with env var substitution."""
        os.environ["YAML_TEST"] = "from_env"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({"value": "${YAML_TEST}"}, f)
            f.flush()

            config = load_yaml_config(f.name)
            assert config["value"] == "from_env"

            os.unlink(f.name)
            del os.environ["YAML_TEST"]

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/path.yaml")


class TestMergeConfigs:
    """Tests for config merging."""

    def test_simple_merge(self):
        """Test simple config merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)

        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4

    def test_deep_merge(self):
        """Test deep config merge."""
        base = {"nested": {"a": 1, "b": 2}}
        override = {"nested": {"b": 3, "c": 4}}
        result = merge_configs(base, override)

        assert result["nested"]["a"] == 1
        assert result["nested"]["b"] == 3
        assert result["nested"]["c"] == 4


class TestConfigSchemas:
    """Tests for Pydantic config schemas."""

    def test_database_config(self):
        """Test database config schema."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            name="testdb",
            user="testuser",
            password="testpass",
        )

        assert config.host == "localhost"
        assert "postgresql" in config.async_url
        assert "testdb" in config.async_url

    def test_strategy_config(self):
        """Test strategy config schema."""
        config = StrategyConfig(
            type="sma_crossover",
            params={"fast_period": 10, "slow_period": 30},
            symbols=["BTCUSDT"],
            interval="1h",
        )

        assert config.type == "sma_crossover"
        assert config.params["fast_period"] == 10

    def test_backtest_config_date_parsing(self):
        """Test backtest config date parsing."""
        config = BacktestConfig(
            name="Test Backtest",
            strategies=["strategy1"],
            symbol="BTCUSDT",
            start="2024-01-01",
            end="2024-12-31",
        )

        assert config.start.year == 2024
        assert config.end.month == 12
