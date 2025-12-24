"""
Advanced volatility regime strategies.

These strategies analyze volatility structure and second-order volatility:
- Volatility term structure (short-term vs long-term vol)
- Squeeze cascade (multiple compression indicators)
- Volatility of volatility (regime change detector)
"""

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry
from crypto.indicators.base import indicator_registry


@strategy_registry.register("vol_term_structure")
class VolatilityTermStructureStrategy(BaseStrategy):
    """
    Trade based on volatility term structure.
    
    Short-term vol / Long-term vol ratio reveals vol regime:
    - High ratio (>1.5) = short-term vol spike = reversion opportunity or breakout
    - Low ratio (<0.7) = calm before storm = position for expansion
    
    Strategy:
    - Calculate 24h ATR / 7d ATR ratio
    - High ratio: Trade with momentum (vol expansion)
    - Low ratio: Prepare for breakout (vol compression)
    """
    
    name = "vol_term_structure"
    
    def _setup(self, **params):
        self.short_atr_period = params.get("short_atr_period", 24)
        self.long_atr_period = params.get("long_atr_period", 168)
        self.high_ratio_threshold = params.get("high_ratio_threshold", 1.5)
        self.low_ratio_threshold = params.get("low_ratio_threshold", 0.7)
        self.hold_period = params.get("hold_period", 24)
        self.trade_direction = params.get("trade_direction", "with_spike")  # or fade_spike
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on volatility term structure."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.long_atr_period:
            return signals
        
        # Calculate ATR on different timeframes
        short_atr = indicator_registry.compute("atr", candles, period=self.short_atr_period)
        long_atr = indicator_registry.compute("atr", candles, period=self.long_atr_period)
        
        # Vol term structure ratio
        vol_ratio = short_atr / long_atr.replace(0, np.nan)
        
        # Calculate momentum for direction
        momentum = candles["close"].pct_change(periods=12)
        
        # Track position
        in_position = False
        entry_idx = None
        
        for i in range(self.long_atr_period, len(candles)):
            if pd.isna(vol_ratio.iloc[i]) or pd.isna(momentum.iloc[i]):
                continue
            
            ratio = vol_ratio.iloc[i]
            mom = momentum.iloc[i]
            
            if self.trade_direction == "with_spike":
                # High vol = trade with momentum
                if ratio > self.high_ratio_threshold:
                    if mom > 0.005:
                        signals.iloc[i] = Signal.BUY
                        in_position = True
                        entry_idx = i
                    elif mom < -0.005 and in_position:
                        signals.iloc[i] = Signal.SELL
                        in_position = False
            
            elif self.trade_direction == "fade_spike":
                # High vol = fade the move
                if ratio > self.high_ratio_threshold:
                    if mom > 0.01:  # Strong up, fade it
                        if in_position:
                            signals.iloc[i] = Signal.SELL
                            in_position = False
                    elif mom < -0.01:  # Strong down, buy dip
                        signals.iloc[i] = Signal.BUY
                        in_position = True
                        entry_idx = i
            
            # Low vol = prepare for breakout
            if ratio < self.low_ratio_threshold:
                # Position for expansion with momentum
                if mom > 0.002:
                    signals.iloc[i] = Signal.BUY
                    in_position = True
                    entry_idx = i
            
            # Exit after hold period
            if in_position and entry_idx is not None:
                bars_held = i - entry_idx
                if bars_held >= self.hold_period:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
        
        return signals


@strategy_registry.register("squeeze_cascade")
class SqueezeCascadeStrategy(BaseStrategy):
    """
    Trade when multiple squeeze indicators align.
    
    "Squeeze" = compression before expansion:
    - ATR at low percentile
    - Bollinger Band width at low percentile
    - Both conditions together = high probability breakout
    
    Strategy:
    - Wait for ATR < 20th percentile AND BB width < 20th percentile
    - Enter on breakout confirmation
    - Direction determined by breakout direction
    """
    
    name = "squeeze_cascade"
    
    def _setup(self, **params):
        self.atr_period = params.get("atr_period", 14)
        self.bb_period = params.get("bb_period", 20)
        self.bb_std = params.get("bb_std", 2.0)
        self.percentile_window = params.get("percentile_window", 168)
        self.squeeze_threshold = params.get("squeeze_threshold", 20)
        self.breakout_confirmation = params.get("breakout_confirmation", True)
        self.hold_period = params.get("hold_period", 48)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on squeeze cascade."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.percentile_window:
            return signals
        
        # Calculate ATR
        atr = indicator_registry.compute("atr", candles, period=self.atr_period)
        
        # Calculate Bollinger Bands
        bb_middle = candles["close"].rolling(window=self.bb_period).mean()
        bb_std_val = candles["close"].rolling(window=self.bb_period).std()
        bb_upper = bb_middle + (bb_std_val * self.bb_std)
        bb_lower = bb_middle - (bb_std_val * self.bb_std)
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Calculate percentiles
        atr_percentile = atr.rolling(window=self.percentile_window).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50,
            raw=False
        )
        
        bb_width_percentile = bb_width.rolling(window=self.percentile_window).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50,
            raw=False
        )
        
        # Squeeze condition: both indicators compressed
        in_squeeze = (
            (atr_percentile < self.squeeze_threshold) & 
            (bb_width_percentile < self.squeeze_threshold)
        )
        
        # Track position
        in_position = False
        entry_idx = None
        squeeze_active = False
        
        for i in range(self.percentile_window, len(candles)):
            if pd.isna(in_squeeze.iloc[i]):
                continue
            
            # Enter squeeze zone
            if in_squeeze.iloc[i]:
                squeeze_active = True
                continue
            
            # Breakout from squeeze
            if squeeze_active and not in_squeeze.iloc[i]:
                if self.breakout_confirmation:
                    # Confirm with price vs BB
                    if candles["close"].iloc[i] > bb_upper.iloc[i]:
                        signals.iloc[i] = Signal.BUY
                        in_position = True
                        entry_idx = i
                        squeeze_active = False
                    elif candles["close"].iloc[i] < bb_lower.iloc[i]:
                        signals.iloc[i] = Signal.SELL
                        in_position = False
                        squeeze_active = False
                else:
                    # Enter without confirmation
                    momentum = candles["close"].iloc[i] - candles["close"].iloc[i-1]
                    if momentum > 0:
                        signals.iloc[i] = Signal.BUY
                        in_position = True
                        entry_idx = i
                    squeeze_active = False
            
            # Exit after hold period
            if in_position and entry_idx is not None:
                bars_held = i - entry_idx
                if bars_held >= self.hold_period:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
        
        return signals


@strategy_registry.register("vol_of_vol_breakout")
class VolOfVolBreakoutStrategy(BaseStrategy):
    """
    Trade when volatility of volatility spikes.
    
    Vol of vol = standard deviation of ATR
    - Low vol-of-vol = stable regime
    - High vol-of-vol = regime change
    
    A spike in vol-of-vol often precedes trend changes or breakouts.
    
    Strategy:
    - Calculate rolling std dev of ATR (volatility of volatility)
    - When vol-of-vol spikes (>2 std dev), regime change likely
    - Enter with momentum direction
    """
    
    name = "vol_of_vol_breakout"
    
    def _setup(self, **params):
        self.atr_period = params.get("atr_period", 14)
        self.vol_of_vol_window = params.get("vol_of_vol_window", 24)
        self.spike_threshold = params.get("spike_threshold", 2.0)
        self.momentum_period = params.get("momentum_period", 12)
        self.hold_period = params.get("hold_period", 24)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on volatility of volatility."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.atr_period + self.vol_of_vol_window + 20:
            return signals
        
        # Calculate ATR (volatility)
        atr = indicator_registry.compute("atr", candles, period=self.atr_period)
        
        # Calculate volatility of volatility
        atr_mean = atr.rolling(window=self.vol_of_vol_window).mean()
        atr_std = atr.rolling(window=self.vol_of_vol_window).std()
        
        # Z-score of ATR (measures vol-of-vol spike)
        atr_zscore = (atr - atr_mean) / atr_std.replace(0, np.nan)
        
        # Calculate momentum for direction
        momentum = candles["close"].pct_change(periods=self.momentum_period)
        
        # Track position
        in_position = False
        entry_idx = None
        
        for i in range(self.atr_period + self.vol_of_vol_window, len(candles)):
            if pd.isna(atr_zscore.iloc[i]) or pd.isna(momentum.iloc[i]):
                continue
            
            zscore = atr_zscore.iloc[i]
            mom = momentum.iloc[i]
            
            # Vol-of-vol spike detected
            if abs(zscore) > self.spike_threshold:
                # Trade with momentum
                if mom > 0.005:
                    signals.iloc[i] = Signal.BUY
                    in_position = True
                    entry_idx = i
                elif mom < -0.005:
                    if in_position:
                        signals.iloc[i] = Signal.SELL
                        in_position = False
            
            # Exit after hold period
            if in_position and entry_idx is not None:
                bars_held = i - entry_idx
                if bars_held >= self.hold_period:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
        
        return signals
