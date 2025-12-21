"""Pydantic schemas for configuration validation."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Database Configuration
# =============================================================================


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = "crypto"
    user: str = "postgres"
    password: str = "crypto"

    @property
    def async_url(self) -> str:
        """Get async database URL."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def sync_url(self) -> str:
        """Get sync database URL (for migrations)."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


# =============================================================================
# Logging Configuration
# =============================================================================


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str | None = None


# =============================================================================
# Trading Configuration
# =============================================================================


class TradingConfig(BaseModel):
    """Default trading parameters."""

    default_commission: Decimal = Decimal("0.001")
    default_slippage: Decimal = Decimal("0.0005")
    max_position_size: Decimal = Decimal("0.1")
    stop_loss_pct: Decimal = Decimal("0.02")
    take_profit_pct: Decimal = Decimal("0.04")


# =============================================================================
# Settings (Root Config)
# =============================================================================


class SettingsConfig(BaseModel):
    """Root settings configuration."""

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)


# =============================================================================
# Exchange Configuration
# =============================================================================


class ExchangeConfig(BaseModel):
    """Single exchange configuration."""

    adapter: str  # Maps to registered exchange adapter
    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""
    rate_limit: int = 1200  # requests per minute
    timeout: int = 30  # seconds

    class Config:
        extra = "allow"  # Allow additional exchange-specific fields


class ExchangesConfig(BaseModel):
    """All exchanges configuration."""

    exchanges: dict[str, ExchangeConfig] = Field(default_factory=dict)


# =============================================================================
# Strategy Configuration
# =============================================================================


class StrategyConfig(BaseModel):
    """Single strategy configuration."""

    type: str  # Maps to registered strategy class
    params: dict[str, Any] = Field(default_factory=dict)
    symbols: list[str] = Field(default_factory=list)
    interval: str = "1h"
    enabled: bool = True

    class Config:
        extra = "allow"


class StrategiesConfig(BaseModel):
    """All strategies configuration."""

    strategies: dict[str, StrategyConfig] = Field(default_factory=dict)

    def get_strategy(self, name: str) -> StrategyConfig:
        """Get strategy config by name."""
        if name not in self.strategies:
            raise KeyError(f"Strategy '{name}' not found in config")
        return self.strategies[name]

    def list_enabled(self) -> list[str]:
        """List enabled strategy names."""
        return [name for name, cfg in self.strategies.items() if cfg.enabled]


# =============================================================================
# Backtest Configuration
# =============================================================================


class BacktestConfig(BaseModel):
    """Single backtest run configuration."""

    name: str
    strategies: list[str]
    symbol: str | None = None
    symbols: list[str] = Field(default_factory=list)
    start: date | datetime | str
    end: date | datetime | str
    initial_capital: Decimal = Decimal("10000")
    commission: Decimal = Decimal("0.001")
    slippage: Decimal = Decimal("0.0005")
    interval: str = "1h"

    @field_validator("start", "end", mode="before")
    @classmethod
    def parse_date(cls, v: Any) -> date:
        """Parse date from string."""
        if isinstance(v, (date, datetime)):
            return v if isinstance(v, date) else v.date()
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v

    def get_symbols(self) -> list[str]:
        """Get list of symbols to backtest."""
        if self.symbol:
            return [self.symbol]
        return self.symbols


class BacktestsConfig(BaseModel):
    """All backtests configuration."""

    backtests: dict[str, BacktestConfig] = Field(default_factory=dict)

    def get_backtest(self, name: str) -> BacktestConfig:
        """Get backtest config by name."""
        if name not in self.backtests:
            raise KeyError(f"Backtest '{name}' not found in config")
        return self.backtests[name]

    def list_all(self) -> list[str]:
        """List all backtest names."""
        return list(self.backtests.keys())


# =============================================================================
# Live Trading Configuration
# =============================================================================


class LiveTradingConfig(BaseModel):
    """Live trading session configuration."""

    name: str
    exchange: str  # Reference to exchange config
    strategy: str  # Reference to strategy config
    symbol: str
    interval: str = "1h"
    paper: bool = True  # Paper trading by default
    initial_capital: Decimal = Decimal("10000")
    max_position_size: Decimal = Decimal("0.1")
    stop_loss_pct: Decimal = Decimal("0.02")
    take_profit_pct: Decimal = Decimal("0.04")


class LiveTradingsConfig(BaseModel):
    """All live trading configurations."""

    trading: dict[str, LiveTradingConfig] = Field(default_factory=dict)
