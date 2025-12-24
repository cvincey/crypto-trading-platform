"""
Advanced market microstructure strategies.

These strategies analyze volume dynamics and order flow patterns:
- Volume-time bars instead of clock-time bars
- Liquidity vacuums (volume spikes followed by dry-ups)
- Candle body ratios (indecision vs decision patterns)
- Taker buy/sell imbalance from OHLC data
"""

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry


@strategy_registry.register("volume_clock_momentum")
class VolumeClockMomentumStrategy(BaseStrategy):
    """
    Measure momentum in volume-time instead of clock-time.
    
    Traditional momentum uses fixed time periods (e.g., 20 hours).
    This strategy adapts to market activity by using volume as the clock:
    - 1M volume = 1 "bar"
    - Calculate momentum over N volume bars
    - Adapts to high/low liquidity periods
    
    Strategy:
    - Accumulate candles until volume_bar_size reached
    - Calculate momentum across volume bars
    - Enter on strong volume-momentum
    """
    
    name = "volume_clock_momentum"
    
    def _setup(self, **params):
        self.volume_bar_size = params.get("volume_bar_size", 1000000)
        self.momentum_bars = params.get("momentum_bars", 10)
        self.momentum_threshold = params.get("momentum_threshold", 0.02)
        self.max_clock_hours = params.get("max_clock_hours", 48)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on volume-clock momentum."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.momentum_bars + 10:
            return signals
        
        # Convert to volume bars
        volume_bars = []
        current_bar = {
            "start_idx": 0,
            "end_idx": 0,
            "volume": 0,
            "close": candles["close"].iloc[0],
            "timestamp": candles.index[0]
        }
        
        for i, idx in enumerate(candles.index):
            current_bar["volume"] += candles["volume"].iloc[i]
            current_bar["end_idx"] = i
            
            # Bar complete
            if current_bar["volume"] >= self.volume_bar_size:
                current_bar["close"] = candles["close"].iloc[i]
                current_bar["timestamp"] = idx
                volume_bars.append(current_bar.copy())
                
                # Start new bar
                current_bar = {
                    "start_idx": i + 1,
                    "end_idx": i + 1,
                    "volume": 0,
                    "close": candles["close"].iloc[i],
                    "timestamp": idx
                }
            
            # Prevent infinite accumulation
            if i - current_bar["start_idx"] > self.max_clock_hours:
                current_bar["close"] = candles["close"].iloc[i]
                current_bar["timestamp"] = idx
                volume_bars.append(current_bar.copy())
                
                current_bar = {
                    "start_idx": i + 1,
                    "end_idx": i + 1,
                    "volume": 0,
                    "close": candles["close"].iloc[i],
                    "timestamp": idx
                }
        
        if len(volume_bars) < self.momentum_bars + 1:
            return signals
        
        # Calculate momentum across volume bars
        vol_df = pd.DataFrame(volume_bars)
        vol_df["momentum"] = vol_df["close"].pct_change(periods=self.momentum_bars)
        
        # Map back to clock-time signals
        for vbar in volume_bars:
            if "momentum" not in vbar:
                continue
            
            momentum = vol_df.loc[vol_df["timestamp"] == vbar["timestamp"], "momentum"].values
            if len(momentum) == 0 or np.isnan(momentum[0]):
                continue
            
            momentum_val = momentum[0]
            
            # Generate signal at bar completion
            idx = vbar["end_idx"]
            if idx < len(signals):
                if momentum_val > self.momentum_threshold:
                    signals.iloc[idx] = Signal.BUY
                elif momentum_val < -self.momentum_threshold:
                    signals.iloc[idx] = Signal.SELL
        
        return signals


@strategy_registry.register("liquidity_vacuum")
class LiquidityVacuumStrategy(BaseStrategy):
    """
    Trade after volume spike followed by volume dry-up.
    
    Liquidity vacuum = price discovery is complete, range breakout coming:
    1. Volume spikes (2x+ average)
    2. Volume then drops significantly (to 40% of spike)
    3. Price forms tight range
    4. Breakout from range = trade direction
    
    This identifies accumulation/distribution zones.
    """
    
    name = "liquidity_vacuum"
    
    def _setup(self, **params):
        self.volume_spike_threshold = params.get("volume_spike_threshold", 2.0)
        self.volume_drop_threshold = params.get("volume_drop_threshold", 0.4)
        self.lookback = params.get("lookback", 24)
        self.breakout_period = params.get("breakout_period", 12)
        self.hold_period = params.get("hold_period", 24)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on liquidity vacuum."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.lookback + self.breakout_period:
            return signals
        
        volume = candles["volume"]
        volume_ma = volume.rolling(window=self.lookback).mean()
        
        # Detect volume spikes
        volume_spike = volume > (volume_ma * self.volume_spike_threshold)
        
        # Track state
        in_position = False
        entry_idx = None
        spike_volume = None
        
        for i in range(self.lookback, len(candles)):
            # Look for spike
            if volume_spike.iloc[i]:
                spike_volume = volume.iloc[i]
                continue
            
            # After spike, look for volume drop
            if spike_volume is not None:
                current_vol = volume.iloc[i]
                
                # Volume dried up
                if current_vol < (spike_volume * self.volume_drop_threshold):
                    # Calculate breakout range
                    range_start = max(0, i - self.breakout_period)
                    range_high = candles["high"].iloc[range_start:i].max()
                    range_low = candles["low"].iloc[range_start:i].min()
                    
                    # Upside breakout
                    if candles["close"].iloc[i] > range_high:
                        signals.iloc[i] = Signal.BUY
                        in_position = True
                        entry_idx = i
                        spike_volume = None
                    # Downside breakout
                    elif candles["close"].iloc[i] < range_low:
                        signals.iloc[i] = Signal.SELL
                        in_position = False
                        spike_volume = None
            
            # Exit after hold period
            if in_position and entry_idx is not None:
                bars_held = i - entry_idx
                if bars_held >= self.hold_period:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
        
        return signals


@strategy_registry.register("body_ratio_sequence")
class BodyRatioSequenceStrategy(BaseStrategy):
    """
    Trade indecision → decision transitions.
    
    Candlestick body ratio = abs(close - open) / (high - low)
    - Low ratio (< 0.3) = indecision, small body, long wicks
    - High ratio (> 0.7) = conviction, large body, small wicks
    
    Strategy:
    - Wait for 5+ consecutive indecision candles (compression)
    - Enter on first strong decision candle (expansion)
    - Direction = body direction (bullish/bearish)
    """
    
    name = "body_ratio_sequence"
    
    def _setup(self, **params):
        self.indecision_threshold = params.get("indecision_threshold", 0.3)
        self.decision_threshold = params.get("decision_threshold", 0.7)
        self.min_indecision_bars = params.get("min_indecision_bars", 5)
        self.hold_period = params.get("hold_period", 12)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on body ratio sequence."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.min_indecision_bars + 5:
            return signals
        
        # Calculate body ratio
        body = (candles["close"] - candles["open"]).abs()
        total_range = candles["high"] - candles["low"]
        total_range = total_range.replace(0, np.nan)  # Avoid division by zero
        body_ratio = body / total_range
        
        # Candle direction
        bullish = candles["close"] > candles["open"]
        bearish = candles["close"] < candles["open"]
        
        # Track indecision sequence
        indecision_count = 0
        in_position = False
        entry_idx = None
        
        for i in range(len(candles)):
            if pd.isna(body_ratio.iloc[i]):
                indecision_count = 0
                continue
            
            # Count indecision bars
            if body_ratio.iloc[i] < self.indecision_threshold:
                indecision_count += 1
            
            # Decision bar after indecision sequence
            elif body_ratio.iloc[i] > self.decision_threshold:
                if indecision_count >= self.min_indecision_bars:
                    # Enter in direction of decision candle
                    if bullish.iloc[i]:
                        signals.iloc[i] = Signal.BUY
                        in_position = True
                        entry_idx = i
                    elif bearish.iloc[i]:
                        signals.iloc[i] = Signal.SELL
                        in_position = False
                
                # Reset counter
                indecision_count = 0
            
            else:
                # Medium body ratio - reset
                indecision_count = 0
            
            # Exit after hold period
            if in_position and entry_idx is not None:
                bars_held = i - entry_idx
                if bars_held >= self.hold_period:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
        
        return signals


@strategy_registry.register("taker_imbalance_momentum")
class TakerImbalanceMomentumStrategy(BaseStrategy):
    """
    Estimate taker buy/sell imbalance from OHLC data.
    
    We don't have actual taker buy/sell volume, but can estimate:
    - Taker buy pressure ≈ (close - low) / (high - low)
    - Taker sell pressure ≈ (high - close) / (high - low)
    
    High buy pressure = aggressive buyers lifting offers
    High sell pressure = aggressive sellers hitting bids
    
    Strategy:
    - Calculate rolling imbalance over N bars
    - Enter long when buy pressure > 65%
    - Enter short when buy pressure < 35% (65% sell pressure)
    """
    
    name = "taker_imbalance_momentum"
    
    def _setup(self, **params):
        self.imbalance_window = params.get("imbalance_window", 12)
        self.buy_threshold = params.get("buy_threshold", 0.65)
        self.sell_threshold = params.get("sell_threshold", 0.35)
        self.smoothing_period = params.get("smoothing_period", 3)
        self.hold_period = params.get("hold_period", 24)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on taker imbalance."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.imbalance_window + self.smoothing_period:
            return signals
        
        # Calculate buy pressure for each candle
        total_range = candles["high"] - candles["low"]
        total_range = total_range.replace(0, np.nan)
        
        # Buy pressure = how close to high did it close
        buy_pressure = (candles["close"] - candles["low"]) / total_range
        
        # Smooth and average over window
        buy_pressure_smooth = buy_pressure.rolling(
            window=self.smoothing_period
        ).mean()
        
        buy_pressure_avg = buy_pressure_smooth.rolling(
            window=self.imbalance_window
        ).mean()
        
        # Track position
        in_position = False
        entry_idx = None
        
        for i in range(self.imbalance_window + self.smoothing_period, len(candles)):
            if pd.isna(buy_pressure_avg.iloc[i]):
                continue
            
            bp = buy_pressure_avg.iloc[i]
            
            # Strong buy pressure
            if bp > self.buy_threshold and not in_position:
                signals.iloc[i] = Signal.BUY
                in_position = True
                entry_idx = i
            
            # Strong sell pressure
            elif bp < self.sell_threshold:
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
