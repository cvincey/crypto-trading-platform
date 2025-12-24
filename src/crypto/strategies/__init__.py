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
    # Unexplored creative strategies - Tier 1
    volatility_trading,
    structural,
    # Hyper-creative diversification strategies - Research Note 13
    # Tier 1: OHLCV only
    information_theoretic,
    microstructure_v2,
    multi_timeframe_v2,
    volatility_v2,
    calendar_v2,
    # Tier 2: Alternative data
    sentiment,
    positioning,
    macro,
    funding_v2,
    # Hyper-creative orthogonal strategies (diversify from ratio MR)
    hyper_creative,
    hyper_creative_tier2,
)

__all__ = [
    "Strategy",
    "strategy_registry",
]
