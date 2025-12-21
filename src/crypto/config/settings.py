"""Global settings management."""

import logging
from functools import lru_cache
from pathlib import Path

from crypto.config.loader import ConfigLoader
from crypto.config.schemas import (
    BacktestsConfig,
    ExchangesConfig,
    OptimizationConfig,
    SettingsConfig,
    StrategiesConfig,
)

logger = logging.getLogger(__name__)

# Default config directory (relative to project root)
DEFAULT_CONFIG_DIR = Path("config")


class Settings:
    """
    Central settings manager for the crypto trading platform.
    
    Loads and validates all configuration from YAML files.
    Provides typed access to all configuration sections.
    """

    def __init__(self, config_dir: Path | str = DEFAULT_CONFIG_DIR):
        """
        Initialize settings from configuration directory.
        
        Args:
            config_dir: Path to configuration directory
        """
        self.config_dir = Path(config_dir)
        self._loader = ConfigLoader(self.config_dir)
        self._settings: SettingsConfig | None = None
        self._exchanges: ExchangesConfig | None = None
        self._strategies: StrategiesConfig | None = None
        self._backtests: BacktestsConfig | None = None
        self._optimization: OptimizationConfig | None = None

    def _ensure_config_dir(self) -> None:
        """Ensure config directory exists."""
        if not self.config_dir.exists():
            logger.warning(
                f"Config directory not found: {self.config_dir}. "
                "Using default settings."
            )

    @property
    def settings(self) -> SettingsConfig:
        """Get global settings."""
        if self._settings is None:
            self._ensure_config_dir()
            try:
                raw = self._loader.load("settings")
                self._settings = SettingsConfig(**raw)
            except FileNotFoundError:
                logger.info("No settings.yaml found, using defaults")
                self._settings = SettingsConfig()
        return self._settings

    @property
    def database(self):
        """Shortcut to database settings."""
        return self.settings.database

    @property
    def logging_config(self):
        """Shortcut to logging settings."""
        return self.settings.logging

    @property
    def trading(self):
        """Shortcut to trading settings."""
        return self.settings.trading

    @property
    def exchanges(self) -> ExchangesConfig:
        """Get exchanges configuration."""
        if self._exchanges is None:
            self._ensure_config_dir()
            try:
                raw = self._loader.load("exchanges")
                self._exchanges = ExchangesConfig(**raw)
            except FileNotFoundError:
                logger.info("No exchanges.yaml found, using empty config")
                self._exchanges = ExchangesConfig()
        return self._exchanges

    def get_exchange(self, name: str):
        """Get a specific exchange config by name."""
        return self.exchanges.exchanges.get(name)

    @property
    def strategies(self) -> StrategiesConfig:
        """Get strategies configuration."""
        if self._strategies is None:
            self._ensure_config_dir()
            try:
                raw = self._loader.load("strategies")
                self._strategies = StrategiesConfig(**raw)
            except FileNotFoundError:
                logger.info("No strategies.yaml found, using empty config")
                self._strategies = StrategiesConfig()
        return self._strategies

    def get_strategy(self, name: str):
        """Get a specific strategy config by name."""
        return self.strategies.get_strategy(name)

    @property
    def backtests(self) -> BacktestsConfig:
        """Get backtests configuration."""
        if self._backtests is None:
            self._ensure_config_dir()
            try:
                raw = self._loader.load("backtests")
                self._backtests = BacktestsConfig(**raw)
            except FileNotFoundError:
                logger.info("No backtests.yaml found, using empty config")
                self._backtests = BacktestsConfig()
        return self._backtests

    def get_backtest(self, name: str):
        """Get a specific backtest config by name."""
        return self.backtests.get_backtest(name)

    @property
    def optimization(self) -> OptimizationConfig:
        """Get optimization configuration."""
        if self._optimization is None:
            self._ensure_config_dir()
            try:
                raw = self._loader.load("optimization")
                self._optimization = OptimizationConfig(**raw)
            except FileNotFoundError:
                logger.info("No optimization.yaml found, using defaults")
                self._optimization = OptimizationConfig()
        return self._optimization

    def reload(self) -> None:
        """Reload all configuration from disk."""
        self._loader.clear_cache()
        self._settings = None
        self._exchanges = None
        self._strategies = None
        self._backtests = None
        self._optimization = None
        logger.info("Configuration reloaded")

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        config = self.logging_config
        logging.basicConfig(
            level=getattr(logging, config.level.upper()),
            format=config.format,
            filename=config.file,
        )


# Global settings instance
_settings: Settings | None = None


def get_settings(config_dir: Path | str | None = None) -> Settings:
    """
    Get the global settings instance.
    
    Args:
        config_dir: Optional config directory (only used on first call)
        
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings(config_dir or DEFAULT_CONFIG_DIR)
    return _settings


def reset_settings() -> None:
    """Reset global settings (mainly for testing)."""
    global _settings
    _settings = None
