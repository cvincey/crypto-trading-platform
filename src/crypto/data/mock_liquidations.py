"""
Mock liquidation data generator for backtesting.

Generates simulated liquidation events based on price action patterns:
- Large price drops (>3% in 1h) -> long liquidations
- Large price spikes (>3% in 1h) -> short liquidations
- ATR spikes (>95th percentile) -> increased liquidation amounts

This allows backtesting the liquidation_cascade_fade strategy without
requiring real CoinGlass/Coinalyze data.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from crypto.indicators.base import indicator_registry

logger = logging.getLogger(__name__)


def generate_mock_liquidations(
    candles: pd.DataFrame,
    atr_spike_percentile: float = 95,
    price_drop_threshold: float = 0.03,
    price_spike_threshold: float = 0.03,
    base_liquidation_amount: float = 5_000_000,
    volatility_multiplier: float = 2.0,
    atr_period: int = 14,
    random_seed: int | None = 42,
) -> pd.DataFrame:
    """
    Generate simulated liquidation events based on price action.
    
    The logic:
    1. Calculate ATR and its percentile for volatility detection
    2. Detect large price drops -> simulates long liquidations
    3. Detect large price spikes -> simulates short liquidations
    4. Scale liquidation amounts by volatility
    
    Args:
        candles: DataFrame with OHLCV data
        atr_spike_percentile: ATR percentile above which is considered a spike
        price_drop_threshold: Price drop % that triggers long liquidations
        price_spike_threshold: Price spike % that triggers short liquidations
        base_liquidation_amount: Base $ amount for liquidations
        volatility_multiplier: Scale factor for high volatility events
        atr_period: Period for ATR calculation
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns:
        - long_liquidations: $ amount of long positions liquidated
        - short_liquidations: $ amount of short positions liquidated
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Initialize result DataFrame
    result = pd.DataFrame(
        index=candles.index,
        columns=["long_liquidations", "short_liquidations"],
        dtype=float,
    )
    result.fillna(0, inplace=True)
    
    # Calculate ATR for volatility scaling
    try:
        atr = indicator_registry.compute("atr", candles, period=atr_period)
    except Exception:
        # Fallback: simple ATR calculation
        high_low = candles["high"] - candles["low"]
        high_close = abs(candles["high"] - candles["close"].shift(1))
        low_close = abs(candles["low"] - candles["close"].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(atr_period).mean()
    
    # Calculate ATR percentile for volatility spikes
    def rolling_percentile(x):
        if len(x) < 2:
            return 50
        return (x.rank().iloc[-1] / len(x)) * 100
    
    atr_percentile = atr.rolling(168, min_periods=20).apply(rolling_percentile, raw=False)
    
    # Calculate hourly price returns
    price_returns = candles["close"].pct_change()
    
    # Detect liquidation events
    for idx in candles.index:
        ret = price_returns.get(idx, 0)
        atr_pct = atr_percentile.get(idx, 50)
        
        if pd.isna(ret) or pd.isna(atr_pct):
            continue
        
        # Volatility scaling factor
        vol_scale = 1.0
        if atr_pct > atr_spike_percentile:
            vol_scale = volatility_multiplier
        
        # Add some randomness to make it more realistic
        random_factor = np.random.uniform(0.5, 1.5)
        
        # Large price DROP -> Long liquidations (longs got stopped out)
        if ret < -price_drop_threshold:
            # Scale by magnitude of drop
            magnitude = abs(ret) / price_drop_threshold
            liq_amount = base_liquidation_amount * magnitude * vol_scale * random_factor
            result.loc[idx, "long_liquidations"] = liq_amount
        
        # Large price SPIKE -> Short liquidations (shorts got squeezed)
        elif ret > price_spike_threshold:
            # Scale by magnitude of spike
            magnitude = abs(ret) / price_spike_threshold
            liq_amount = base_liquidation_amount * magnitude * vol_scale * random_factor
            result.loc[idx, "short_liquidations"] = liq_amount
    
    # Log summary
    long_events = (result["long_liquidations"] > 0).sum()
    short_events = (result["short_liquidations"] > 0).sum()
    total_long = result["long_liquidations"].sum()
    total_short = result["short_liquidations"].sum()
    
    logger.info(
        f"Generated mock liquidations: "
        f"{long_events} long events (${total_long/1e6:.1f}M), "
        f"{short_events} short events (${total_short/1e6:.1f}M)"
    )
    
    return result


def inject_liquidation_data(
    strategy: Any,
    candles: pd.DataFrame,
    config: dict | None = None,
) -> None:
    """
    Inject mock liquidation data into a strategy that requires it.
    
    Args:
        strategy: Strategy instance (e.g., LiquidationCascadeFadeStrategy)
        candles: Historical candle data
        config: Optional config dict with mock liquidation settings
    """
    # Get settings from config or use defaults
    settings = config or {}
    
    mock_data = generate_mock_liquidations(
        candles,
        atr_spike_percentile=settings.get("atr_spike_percentile", 95),
        price_drop_threshold=settings.get("price_drop_threshold", 0.03),
        price_spike_threshold=settings.get("price_spike_threshold", 0.03),
        base_liquidation_amount=settings.get("base_liquidation_amount", 5_000_000),
        volatility_multiplier=settings.get("volatility_multiplier", 2.0),
    )
    
    # Inject into strategy if it has the method
    if hasattr(strategy, "set_liquidation_data"):
        strategy.set_liquidation_data(mock_data)
        logger.info(f"Injected mock liquidation data into {strategy.name}")
    else:
        logger.warning(
            f"Strategy {type(strategy).__name__} does not have set_liquidation_data method"
        )
