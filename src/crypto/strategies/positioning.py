"""
Position-based trading strategies.

These strategies use positioning data from exchanges:
- Long/Short ratio (trader positioning)
- Liquidation cluster magnets
"""

import logging

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register("long_short_ratio_fade")
class LongShortRatioFadeStrategy(BaseStrategy):
    """
    Fade extreme long/short positioning.
    
    Long/Short ratio indicators:
    - Ratio > 2.0 = Too many longs (crowded trade, fade)
    - Ratio < 0.5 = Too many shorts (crowded trade, fade)
    - Ratio ~1.0 = Balanced
    
    Strategy:
    - When L/S ratio reaches extreme, fade the crowd
    - Extreme longs = avoid new longs
    - Extreme shorts = go long (squeeze potential)
    
    Requires Long/Short ratio data - run ingest_tier2_data.py first.
    """
    
    name = "long_short_ratio_fade"
    
    def _setup(self, **params):
        self.extreme_long_ratio = params.get("extreme_long_ratio", 2.0)
        self.extreme_short_ratio = params.get("extreme_short_ratio", 0.5)
        self.entry_threshold = params.get("entry_threshold", 0.6)  # Enter when shorts crowded
        self.exit_threshold = params.get("exit_threshold", 1.5)  # Exit when longs crowded
        self.lookback_periods = params.get("lookback_periods", 24)
        self.hold_period = params.get("hold_period", 48)
        self._long_short_data: pd.Series | None = None
    
    def set_long_short_data(self, ls_series: pd.Series) -> None:
        """Set Long/Short ratio data aligned to candle index."""
        self._long_short_data = ls_series
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on long/short ratio extremes."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._long_short_data is None or self._long_short_data.empty:
            logger.warning(
                f"{self.name}: Long/Short ratio data not available. "
                "Run: python scripts/ingest_tier2_data.py --long-short-only"
            )
            return signals
        
        # Align to candles
        ls_ratio = self._long_short_data.reindex(candles.index, method="ffill")
        ls_ratio_ma = ls_ratio.rolling(self.lookback_periods, min_periods=1).mean()
        
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            ratio = ls_ratio_ma.get(idx)
            
            if pd.isna(ratio):
                continue
            
            if not in_position:
                # Too many shorts = potential squeeze = go long
                if ratio < self.entry_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_idx = i
            else:
                bars_held = i - entry_idx
                
                # Exit when longs become crowded or hold period reached
                if ratio > self.exit_threshold or bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)


@strategy_registry.register("long_short_momentum")
class LongShortMomentumStrategy(BaseStrategy):
    """
    Trade based on changes in Long/Short ratio.
    
    When L/S ratio is rapidly increasing = longs piling in = momentum
    When L/S ratio is rapidly decreasing = shorts piling in = momentum reversal
    
    Strategy:
    - Enter long when shorts are capitulating (ratio rising from lows)
    - Exit when ratio reaches overbought levels
    """
    
    name = "long_short_momentum"
    
    def _setup(self, **params):
        self.ratio_change_threshold = params.get("ratio_change_threshold", 0.2)  # 20% change
        self.lookback_periods = params.get("lookback_periods", 24)
        self.min_ratio = params.get("min_ratio", 0.7)  # Entry from low ratio
        self.max_ratio = params.get("max_ratio", 1.8)  # Exit at high ratio
        self.hold_period = params.get("hold_period", 72)
        self._long_short_data: pd.Series | None = None
    
    def set_long_short_data(self, ls_series: pd.Series) -> None:
        """Set Long/Short ratio data aligned to candle index."""
        self._long_short_data = ls_series
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on L/S ratio momentum."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._long_short_data is None or self._long_short_data.empty:
            logger.warning(
                f"{self.name}: Long/Short ratio data not available. "
                "Run: python scripts/ingest_tier2_data.py --long-short-only"
            )
            return signals
        
        # Align to candles
        ls_ratio = self._long_short_data.reindex(candles.index, method="ffill")
        
        # Calculate ratio change
        ratio_change = ls_ratio.pct_change(self.lookback_periods)
        
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            ratio = ls_ratio.get(idx)
            change = ratio_change.get(idx)
            
            if pd.isna(ratio) or pd.isna(change):
                continue
            
            if not in_position:
                # Shorts capitulating: ratio rising from low levels
                if ratio < self.min_ratio and change > self.ratio_change_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_idx = i
            else:
                bars_held = i - entry_idx
                
                # Exit conditions
                if ratio > self.max_ratio or bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)


@strategy_registry.register("liquidation_cluster_magnet")
class LiquidationClusterMagnetStrategy(BaseStrategy):
    """
    Trade toward large liquidation clusters.
    
    Liquidation clusters = price levels with large liquidation orders:
    - Price tends to gravitate toward clusters (liquidity vacuum)
    - Breaking through cluster = strong move
    - Bouncing off cluster = support/resistance
    
    Note: Uses mock liquidation data for backtesting.
    For live trading, integrate Coinglass or Coinalyze API.
    """
    
    name = "liquidation_cluster_magnet"
    
    def _setup(self, **params):
        self.cluster_threshold = params.get("cluster_threshold", 10_000_000)  # $10M
        self.price_distance_pct = params.get("price_distance_pct", 0.03)  # 3%
        self.hold_period = params.get("hold_period", 24)
        self._liquidation_data: pd.DataFrame | None = None
    
    def set_liquidation_data(self, liq_df: pd.DataFrame) -> None:
        """Set liquidation data with long_liquidations and short_liquidations columns."""
        self._liquidation_data = liq_df
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on liquidation clusters."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._liquidation_data is None or self._liquidation_data.empty:
            logger.warning(
                f"{self.name}: Liquidation data not available. "
                "Using mock data - inject via set_liquidation_data() or use mock_liquidations.py"
            )
            return signals
        
        # Align liquidation data to candles
        long_liqs = self._liquidation_data["long_liquidations"].reindex(
            candles.index, fill_value=0
        )
        short_liqs = self._liquidation_data["short_liquidations"].reindex(
            candles.index, fill_value=0
        )
        
        close = candles["close"]
        
        # Rolling sum to detect clusters
        long_cluster = long_liqs.rolling(6, min_periods=1).sum()
        short_cluster = short_liqs.rolling(6, min_periods=1).sum()
        
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            lc = long_cluster.get(idx, 0)
            sc = short_cluster.get(idx, 0)
            
            if not in_position:
                # Large short liquidation cluster = price just spiked
                # Shorts got liquidated = momentum up, but may be exhausted
                # Large long liquidation cluster = price just dropped
                # Longs got liquidated = oversold, potential bounce
                
                if lc > self.cluster_threshold:
                    # Long liquidation cascade just happened = oversold
                    # Wait 1 bar then fade
                    if i > 0:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_idx = i
            else:
                bars_held = i - entry_idx
                
                if bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)


@strategy_registry.register("crowded_trade_detector")
class CrowdedTradeDetectorStrategy(BaseStrategy):
    """
    Detect crowded trades using L/S ratio percentile.
    
    Uses historical percentile ranking of L/S ratio to identify
    when positioning is at extremes relative to recent history.
    """
    
    name = "crowded_trade_detector"
    
    def _setup(self, **params):
        self.lookback_days = params.get("lookback_days", 30)
        self.short_crowded_pct = params.get("short_crowded_pct", 10)  # Bottom 10%
        self.long_crowded_pct = params.get("long_crowded_pct", 90)   # Top 10%
        self.hold_period = params.get("hold_period", 72)
        self._long_short_data: pd.Series | None = None
    
    def set_long_short_data(self, ls_series: pd.Series) -> None:
        """Set Long/Short ratio data aligned to candle index."""
        self._long_short_data = ls_series
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on L/S ratio percentile."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._long_short_data is None or self._long_short_data.empty:
            logger.warning(
                f"{self.name}: Long/Short ratio data not available. "
                "Run: python scripts/ingest_tier2_data.py --long-short-only"
            )
            return signals
        
        # Align to candles
        ls_ratio = self._long_short_data.reindex(candles.index, method="ffill")
        
        # Calculate rolling percentile
        lookback_bars = self.lookback_days * 24
        
        def rolling_percentile(x):
            if len(x) < 10:
                return 50
            return (x.rank().iloc[-1] / len(x)) * 100
        
        ratio_pct = ls_ratio.rolling(lookback_bars, min_periods=48).apply(
            rolling_percentile, raw=False
        )
        
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            pct = ratio_pct.get(idx)
            
            if pd.isna(pct):
                continue
            
            if not in_position:
                # Shorts extremely crowded (low L/S ratio) = buy
                if pct < self.short_crowded_pct:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_idx = i
            else:
                bars_held = i - entry_idx
                
                # Exit when longs crowded or hold period
                if pct > self.long_crowded_pct or bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)
