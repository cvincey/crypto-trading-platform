"""YAML configuration loader with environment variable support."""

import os
import re
from pathlib import Path
from typing import Any

import yaml


# Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in a value.
    
    Supports:
    - ${VAR_NAME} - required env var
    - ${VAR_NAME:default} - env var with default value
    
    Args:
        value: The value to process
        
    Returns:
        Value with env vars substituted
    """
    if isinstance(value, str):
        # Find all env var references
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2)
            
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default_value is not None:
                return default_value
            # Return empty string for missing vars without default
            return ""
        
        return ENV_VAR_PATTERN.sub(replacer, value)
    
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    
    return value


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """
    Load a YAML configuration file with environment variable substitution.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        return {}
    
    return _substitute_env_vars(raw_config)


def load_yaml_configs(directory: Path | str) -> dict[str, dict[str, Any]]:
    """
    Load all YAML files from a directory.
    
    Args:
        directory: Path to directory containing YAML files
        
    Returns:
        Dictionary mapping filename (without extension) to config
    """
    directory = Path(directory)
    configs = {}
    
    if not directory.exists():
        return configs
    
    for yaml_file in directory.glob("*.yaml"):
        name = yaml_file.stem
        try:
            configs[name] = load_yaml_config(yaml_file)
        except Exception as e:
            print(f"Warning: Failed to load {yaml_file}: {e}")
    
    for yml_file in directory.glob("*.yml"):
        name = yml_file.stem
        if name not in configs:  # Don't override .yaml files
            try:
                configs[name] = load_yaml_config(yml_file)
            except Exception as e:
                print(f"Warning: Failed to load {yml_file}: {e}")
    
    return configs


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries.
    
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    result: dict[str, Any] = {}
    
    for config in configs:
        result = _deep_merge(result, config)
    
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


class ConfigLoader:
    """
    Configuration loader that manages all config files.
    
    Loads and caches configuration from the config directory.
    """

    def __init__(self, config_dir: Path | str = "config"):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Path to the configuration directory
        """
        self.config_dir = Path(config_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    def load(self, name: str, *, reload: bool = False) -> dict[str, Any]:
        """
        Load a configuration file by name.
        
        Args:
            name: Name of the config file (without extension)
            reload: Force reload from disk
            
        Returns:
            Configuration dictionary
        """
        if not reload and name in self._cache:
            return self._cache[name]

        # Try .yaml first, then .yml
        yaml_path = self.config_dir / f"{name}.yaml"
        yml_path = self.config_dir / f"{name}.yml"

        if yaml_path.exists():
            config = load_yaml_config(yaml_path)
        elif yml_path.exists():
            config = load_yaml_config(yml_path)
        else:
            raise FileNotFoundError(
                f"Config '{name}' not found in {self.config_dir}"
            )

        self._cache[name] = config
        return config

    def load_all(self, *, reload: bool = False) -> dict[str, dict[str, Any]]:
        """
        Load all configuration files.
        
        Args:
            reload: Force reload from disk
            
        Returns:
            Dictionary mapping config name to config dict
        """
        if reload:
            self._cache.clear()

        configs = load_yaml_configs(self.config_dir)
        self._cache.update(configs)
        return self._cache.copy()

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
