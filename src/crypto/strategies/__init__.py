"""Trading strategies module."""

from crypto.strategies.base import Strategy
from crypto.strategies.registry import strategy_registry

# Import all strategy modules to register them
from crypto.strategies import (
    technical,
    statistical,
    momentum,
    ensemble,
    regime,
    rotation,
    multi_timeframe,
    ml,
    ml_siblings,
    ml_online,
    ml_cross_asset,
    rule_ensemble,
    rl,
    # New creative strategy modules
    cross_symbol_base,
    cross_symbol,
    alternative_data_strategies,
    calendar,
    frequency,
    meta,
    microstructure,
    # Hybrid strategies combining Phase 1 winners
    hybrid,
)

__all__ = [
    "Strategy",
    "strategy_registry",
]
