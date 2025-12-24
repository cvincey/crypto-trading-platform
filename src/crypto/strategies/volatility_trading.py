"""
Volatility-based trading strategies.

These strategies trade volatility itself rather than price direction.
They exploit the mean-reverting nature of volatility and volatility breakouts.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "volatility_mean_reversion",
    description="Trade when volatility is extreme, expecting mean reversion",
)
class VolatilityMeanReversionStrategy(BaseStrategy):
    """
    Volatility Mean Reversion Strategy.
    
    When volatility spikes to extreme levels (measured by ATR percentile),
    expect it to revert to the mean. High volatility often precedes
    consolidation periods.
    
    Trading logic:
    - When ATR percentile > high_threshold: Expect consolidation, reduce risk
    - When ATR percentile < low_threshold: Expect breakout, increase exposure
    - Use price momentum to determine direction
    """

    name = "volatility_mean_reversion"

    def _setup(
        self,
        atr_period: int = 14,
        percentile_window: int = 168,  # 7 days
        high_threshold: float = 90,  # 90th percentile = extreme high vol
        low_threshold: float = 20,  # 20th percentile = extreme low vol
        hold_period: int = 24,
        use_momentum_direction: bool = True,
        momentum_period: int = 24,
        **kwargs,
    ) -> None:
        self.atr_period = atr_period
        self.percentile_window = percentile_window
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.hold_period = hold_period
        self.use_momentum_direction = use_momentum_direction
        self.momentum_period = momentum_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Calculate ATR
        atr = indicator_registry.compute("atr", candles, period=self.atr_period)
        
        # Calculate ATR percentile
        def rolling_percentile(x):
            if len(x) < 2:
                return 50
            return (x.rank().iloc[-1] / len(x)) * 100
        
        atr_percentile = atr.rolling(self.percentile_window, min_periods=20).apply(
            rolling_percentile, raw=False
        )
        
        # Calculate momentum for direction
        momentum = candles["close"].pct_change(self.momentum_period)
        
        # Generate signals
        in_position = False
        entry_bar = 0
        
        for i, idx in enumerate(candles.index):
            pct = atr_percentile.get(idx, 50)
            mom = momentum.get(idx, 0)
            
            if pd.isna(pct) or pd.isna(mom):
                continue
            
            if not in_position:
                # Enter when volatility is at extreme LOW (expect breakout)
                if pct < self.low_threshold:
                    # Direction based on momentum
                    if not self.use_momentum_direction or mom > 0:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
            else:
                # Exit after hold period or when vol spikes (mean reversion complete)
                bars_held = i - entry_bar
                if bars_held >= self.hold_period or pct > self.high_threshold:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "volatility_breakout",
    description="Enter on volatility expansion, exit on contraction",
)
class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Breakout Strategy.
    
    When volatility suddenly expands (ATR spike), it often signals
    the start of a new trend. Enter in the direction of the initial move
    and ride the trend until volatility contracts.
    
    Uses Bollinger Band width as a volatility measure.
    Entry: BB width expands rapidly AND price breaks band
    Exit: BB width contracts OR opposite band touched
    """

    name = "volatility_breakout"

    def _setup(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        width_expansion_threshold: float = 1.5,  # 50% expansion
        width_lookback: int = 12,
        min_hold_period: int = 4,
        max_hold_period: int = 72,
        **kwargs,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.width_expansion_threshold = width_expansion_threshold
        self.width_lookback = width_lookback
        self.min_hold_period = min_hold_period
        self.max_hold_period = max_hold_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]
        
        # Calculate Bollinger Bands
        bb_middle = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        bb_upper = bb_middle + (self.bb_std * bb_std)
        bb_lower = bb_middle - (self.bb_std * bb_std)
        
        # Calculate BB width (normalized)
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Calculate width expansion ratio
        width_avg = bb_width.rolling(self.width_lookback).mean()
        width_expansion = bb_width / width_avg
        
        # Generate signals
        in_position = False
        entry_bar = 0
        position_direction = None
        
        for i, idx in enumerate(candles.index):
            expansion = width_expansion.get(idx, 1.0)
            price = close.get(idx)
            upper = bb_upper.get(idx)
            lower = bb_lower.get(idx)
            
            if pd.isna(expansion) or pd.isna(price) or pd.isna(upper):
                continue
            
            if not in_position:
                # Enter on volatility expansion + band break
                if expansion > self.width_expansion_threshold:
                    if price > upper:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
                        position_direction = "long"
                    # Could add short logic here if shorting is enabled
            else:
                bars_held = i - entry_bar
                
                # Minimum hold check
                if bars_held < self.min_hold_period:
                    continue
                
                # Exit conditions
                should_exit = False
                
                # Max hold reached
                if bars_held >= self.max_hold_period:
                    should_exit = True
                
                # Volatility contracting (width back to normal)
                if expansion < 1.0:
                    should_exit = True
                
                # Price hit opposite band
                if position_direction == "long" and price < lower:
                    should_exit = True
                
                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
                    position_direction = None

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "regime_volatility_switch",
    description="Switch strategies based on volatility regime",
)
class RegimeVolatilitySwitchStrategy(BaseStrategy):
    """
    Regime Volatility Switch Strategy.
    
    Adapts trading behavior based on the current volatility regime:
    - Low volatility: Use mean reversion (expect range-bound market)
    - High volatility: Use momentum (trends develop in volatile markets)
    
    This meta-strategy selects the appropriate approach based on
    recent volatility characteristics.
    """

    name = "regime_volatility_switch"

    def _setup(
        self,
        atr_period: int = 14,
        regime_window: int = 168,  # 7 days
        high_vol_threshold: float = 70,  # Percentile
        low_vol_threshold: float = 30,  # Percentile
        # Mean reversion params (low vol)
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        # Momentum params (high vol)
        momentum_period: int = 24,
        momentum_threshold: float = 0.02,  # 2% move
        hold_period: int = 24,
        **kwargs,
    ) -> None:
        self.atr_period = atr_period
        self.regime_window = regime_window
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.hold_period = hold_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Calculate ATR percentile for regime detection
        atr = indicator_registry.compute("atr", candles, period=self.atr_period)
        
        def rolling_percentile(x):
            if len(x) < 2:
                return 50
            return (x.rank().iloc[-1] / len(x)) * 100
        
        atr_percentile = atr.rolling(self.regime_window, min_periods=20).apply(
            rolling_percentile, raw=False
        )
        
        # Calculate RSI for mean reversion signals
        rsi = indicator_registry.compute("rsi", candles, period=self.rsi_period)
        
        # Calculate momentum for trend signals
        momentum = candles["close"].pct_change(self.momentum_period)
        
        # Generate signals based on regime
        in_position = False
        entry_bar = 0
        
        for i, idx in enumerate(candles.index):
            vol_pct = atr_percentile.get(idx, 50)
            rsi_val = rsi.get(idx, 50)
            mom = momentum.get(idx, 0)
            
            if pd.isna(vol_pct) or pd.isna(rsi_val) or pd.isna(mom):
                continue
            
            if not in_position:
                # LOW volatility regime: Use mean reversion
                if vol_pct < self.low_vol_threshold:
                    if rsi_val < self.rsi_oversold:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
                
                # HIGH volatility regime: Use momentum
                elif vol_pct > self.high_vol_threshold:
                    if mom > self.momentum_threshold:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
            else:
                bars_held = i - entry_bar
                
                # Exit after hold period
                if bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
                
                # Exit on regime change (volatility moved to opposite extreme)
                elif (vol_pct > self.high_vol_threshold and 
                      atr_percentile.get(candles.index[entry_bar], 50) < self.low_vol_threshold):
                    signals.loc[idx] = Signal.SELL
                    in_position = False
                elif (vol_pct < self.low_vol_threshold and 
                      atr_percentile.get(candles.index[entry_bar], 50) > self.high_vol_threshold):
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)
