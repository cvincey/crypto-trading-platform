"""
Structural trading strategies.

These strategies exploit structural market features rather than
predicting price direction. They focus on market mechanics like
basis, funding rates, and arbitrage relationships.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.data.alternative_data import FundingRateRepository
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "basis_proxy",
    description="Trade based on funding rate as proxy for spot-perp basis",
)
class BasisProxyStrategy(BaseStrategy):
    """
    Basis Proxy Strategy.
    
    Uses funding rate as a proxy for the spot-perpetual basis.
    
    Funding rate reflects the cost of holding perpetual positions:
    - Positive funding = longs pay shorts = perp trading at premium to spot
    - Negative funding = shorts pay longs = perp trading at discount to spot
    
    Strategy:
    - When funding is consistently positive and high: Market is overleveraged long
      -> Expect mean reversion DOWN or collect funding by being short perp/long spot
    - When funding is consistently negative: Market is overleveraged short
      -> Expect squeeze UP
    
    Since we can only go long spot, we use funding extremes as contrarian signals:
    - Extreme negative funding = buy (expect squeeze)
    - Extreme positive funding = avoid longs or exit
    """

    name = "basis_proxy"

    def _setup(
        self,
        funding_lookback: int = 9,  # 3 days of 8h funding = 9 periods
        entry_threshold: float = -0.0003,  # -0.03% avg funding (shorts paying)
        exit_threshold: float = 0.0003,  # +0.03% avg funding (longs paying)
        extreme_positive: float = 0.0005,  # +0.05% = very crowded long
        max_hold_hours: int = 72,
        use_funding_data: bool = True,  # If False, estimates from price action
        **kwargs,
    ) -> None:
        self.funding_lookback = funding_lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.extreme_positive = extreme_positive
        self.max_hold_hours = max_hold_hours
        self.use_funding_data = use_funding_data
        self._funding_repo = FundingRateRepository() if use_funding_data else None
        self._funding_cache: pd.Series | None = None

    def set_funding_data(self, funding_series: pd.Series) -> None:
        """Set funding rate data aligned to candle index."""
        self._funding_cache = funding_series

    def _estimate_funding_from_price(self, candles: pd.DataFrame) -> pd.Series:
        """
        Estimate funding sentiment from price action when real funding unavailable.
        
        Uses a momentum-based proxy: sustained uptrends suggest positive funding,
        sustained downtrends suggest negative funding.
        """
        close = candles["close"]
        
        # Calculate momentum indicators
        short_ma = close.rolling(24).mean()
        long_ma = close.rolling(168).mean()
        
        # Deviation from long MA as funding proxy
        # When price far above MA = likely positive funding (overleveraged long)
        # When price far below MA = likely negative funding (overleveraged short)
        deviation = (close - long_ma) / long_ma
        
        # Scale to approximate funding rate range (-0.001 to +0.001)
        funding_proxy = deviation * 0.01  # Scale factor
        
        return funding_proxy.rolling(self.funding_lookback).mean()

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Get funding data
        if self._funding_cache is not None:
            funding_avg = self._funding_cache.rolling(
                self.funding_lookback, min_periods=1
            ).mean()
        else:
            funding_avg = self._estimate_funding_from_price(candles)

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            funding = funding_avg.get(idx, 0)
            
            if pd.isna(funding):
                continue

            if not in_position:
                # Entry: Negative funding = shorts overleveraged = expect squeeze
                if funding < self.entry_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                bars_held = i - entry_bar
                
                # Exit conditions
                should_exit = False
                
                # Max hold
                if bars_held >= self.max_hold_hours:
                    should_exit = True
                
                # Funding turned positive = exit signal
                if funding > self.exit_threshold:
                    should_exit = True
                
                # Extreme positive funding = definitely exit
                if funding > self.extreme_positive:
                    should_exit = True
                
                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "multi_asset_pair_trade",
    description="Statistical arbitrage between correlated crypto assets",
)
class MultiAssetPairTradeStrategy(BaseStrategy):
    """
    Multi-Asset Pair Trading Strategy.
    
    Trades mean reversion between pairs of highly correlated crypto assets.
    When two assets typically move together diverge, bet on convergence.
    
    Example pairs:
    - SOL/AVAX (Layer 1 competitors)
    - LINK/AAVE (DeFi infrastructure)
    - LTC/BCH (Bitcoin forks)
    
    This is implemented as a single-asset strategy that uses the ratio
    with a reference asset. Use CrossSymbolBaseStrategy for full pair trading.
    
    For this implementation, we estimate pair divergence from the target
    asset's relative strength vs the market (BTC).
    """

    name = "multi_asset_pair_trade"

    def _setup(
        self,
        lookback: int = 168,  # 7 days
        entry_zscore: float = -2.0,  # Enter when underperforming by 2 std
        exit_zscore: float = -0.5,  # Exit when recovered
        max_hold_hours: int = 72,
        use_returns: bool = True,  # Compare returns vs levels
        **kwargs,
    ) -> None:
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.max_hold_hours = max_hold_hours
        self.use_returns = use_returns

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]
        
        if self.use_returns:
            # Calculate rolling returns
            returns = close.pct_change(24)  # 24h returns
            
            # Calculate z-score of recent returns vs history
            ret_mean = returns.rolling(self.lookback, min_periods=20).mean()
            ret_std = returns.rolling(self.lookback, min_periods=20).std()
            z_score = (returns - ret_mean) / ret_std
        else:
            # Calculate z-score of price level
            price_mean = close.rolling(self.lookback, min_periods=20).mean()
            price_std = close.rolling(self.lookback, min_periods=20).std()
            z_score = (close - price_mean) / price_std

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            z = z_score.get(idx, 0)
            
            if pd.isna(z):
                continue

            if not in_position:
                # Enter when significantly underperforming
                if z < self.entry_zscore:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                bars_held = i - entry_bar
                
                # Exit when recovered or max hold
                if z > self.exit_zscore or bars_held >= self.max_hold_hours:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "dxy_correlation_proxy",
    description="Trade based on inverse BTC correlation as DXY proxy",
)
class DXYCorrelationProxyStrategy(BaseStrategy):
    """
    DXY Correlation Proxy Strategy.
    
    BTC often exhibits inverse correlation with the US Dollar Index (DXY).
    When the dollar strengthens, crypto tends to weaken and vice versa.
    
    Since we don't have real-time DXY data without an API key,
    we use a proxy: BTC strength relative to its recent trend.
    
    Hypothesis:
    - Strong USD (approximated by BTC weakness) = risk-off = avoid longs
    - Weak USD (approximated by BTC strength) = risk-on = favor longs on alts
    
    This strategy uses BTC as a regime filter for alt trading.
    """

    name = "dxy_correlation_proxy"

    def _setup(
        self,
        trend_period: int = 168,  # 7-day trend
        momentum_period: int = 24,  # 24h momentum
        strength_threshold: float = 0.02,  # 2% above trend = risk-on
        weakness_threshold: float = -0.02,  # 2% below trend = risk-off
        hold_period: int = 48,
        require_positive_momentum: bool = True,
        **kwargs,
    ) -> None:
        self.trend_period = trend_period
        self.momentum_period = momentum_period
        self.strength_threshold = strength_threshold
        self.weakness_threshold = weakness_threshold
        self.hold_period = hold_period
        self.require_positive_momentum = require_positive_momentum

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]
        
        # Calculate trend line (SMA)
        trend = close.rolling(self.trend_period, min_periods=20).mean()
        
        # Calculate deviation from trend
        deviation = (close - trend) / trend
        
        # Calculate momentum
        momentum = close.pct_change(self.momentum_period)
        
        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            dev = deviation.get(idx, 0)
            mom = momentum.get(idx, 0)
            
            if pd.isna(dev) or pd.isna(mom):
                continue

            if not in_position:
                # Enter when above trend (risk-on environment)
                if dev > self.strength_threshold:
                    if not self.require_positive_momentum or mom > 0:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
            else:
                bars_held = i - entry_bar
                
                # Exit when environment turns risk-off or hold period reached
                if dev < self.weakness_threshold or bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)
