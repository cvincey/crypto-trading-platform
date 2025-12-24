"""
Calendar and structural time-based strategies.

These strategies exploit time-of-day and day-of-week patterns:
- Asian session reversals
- Weekend gap fades
- Funding hour momentum
"""

import numpy as np
import pandas as pd
from datetime import time

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry


@strategy_registry.register("asian_session_reversal")
class AsianSessionReversalStrategy(BaseStrategy):
    """
    Fade overnight Asian session moves at European open.
    
    Pattern observed:
    - Overnight (00:00-08:00 UTC) moves often reverse at European open
    - Asian session = lower liquidity, thinner markets
    - European/US sessions bring liquidity that reverses thin moves
    
    Strategy:
    - Measure overnight move (00:00-08:00 UTC)
    - If move > 1%, fade it at 08:00 UTC
    - Hold for 12 hours
    """
    
    name = "asian_session_reversal"
    
    def _setup(self, **params):
        self.asian_start_hour = params.get("asian_start_hour", 0)
        self.asian_end_hour = params.get("asian_end_hour", 8)
        self.european_open_hour = params.get("european_open_hour", 8)
        self.min_move_threshold = params.get("min_move_threshold", 0.01)
        self.hold_period = params.get("hold_period", 12)
        self.fade_mode = params.get("fade_mode", True)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on Asian session reversal."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < 24:
            return signals
        
        # Track Asian session moves
        asian_start_price = None
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            hour = idx.hour
            
            # Asian session start
            if hour == self.asian_start_hour:
                asian_start_price = candles["close"].iloc[i]
            
            # European open - check for fade opportunity
            if hour == self.european_open_hour and asian_start_price is not None:
                current_price = candles["close"].iloc[i]
                move = (current_price - asian_start_price) / asian_start_price
                
                if abs(move) > self.min_move_threshold:
                    if self.fade_mode:
                        # Fade the move
                        if move > 0:  # Up overnight, short
                            if in_position:
                                signals.iloc[i] = Signal.SELL
                                in_position = False
                        else:  # Down overnight, long
                            signals.iloc[i] = Signal.BUY
                            in_position = True
                            entry_idx = i
                    else:
                        # Trend with the move
                        if move > 0:
                            signals.iloc[i] = Signal.BUY
                            in_position = True
                            entry_idx = i
                        else:
                            if in_position:
                                signals.iloc[i] = Signal.SELL
                                in_position = False
                
                asian_start_price = None
            
            # Exit after hold period
            if in_position and entry_idx is not None:
                bars_held = i - entry_idx
                if bars_held >= self.hold_period:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
        
        return signals


@strategy_registry.register("weekend_gap_fade")
class WeekendGapFadeStrategy(BaseStrategy):
    """
    Fade weekend gaps in crypto markets.
    
    Weekend effect in crypto:
    - Friday 20:00 to Sunday 20:00 gap can form
    - Gaps tend to fill during the week
    - Lower weekend liquidity creates inefficiencies
    
    Strategy:
    - Measure gap from Friday close to Sunday open
    - If gap > 2%, expect it to fill
    - Enter after Sunday open, hold for 48h
    """
    
    name = "weekend_gap_fade"
    
    def _setup(self, **params):
        self.friday_hour = params.get("friday_hour", 20)
        self.sunday_hour = params.get("sunday_hour", 20)
        self.min_gap_threshold = params.get("min_gap_threshold", 0.02)
        self.hold_period = params.get("hold_period", 48)
        self.entry_delay = params.get("entry_delay", 4)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on weekend gap fade."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < 72:  # Need at least 3 days
            return signals
        
        friday_close = None
        sunday_open = None
        in_position = False
        entry_idx = None
        gap_direction = None
        
        for i, idx in enumerate(candles.index):
            hour = idx.hour
            day_of_week = idx.dayofweek  # 0=Monday, 4=Friday, 6=Sunday
            
            # Friday close
            if day_of_week == 4 and hour == self.friday_hour:
                friday_close = candles["close"].iloc[i]
            
            # Sunday open
            if day_of_week == 6 and hour == self.sunday_hour and friday_close is not None:
                sunday_open = candles["close"].iloc[i]
                gap = (sunday_open - friday_close) / friday_close
                
                # Significant gap detected
                if abs(gap) > self.min_gap_threshold:
                    gap_direction = "up" if gap > 0 else "down"
                    
                    # Wait for entry delay before entering
                    entry_time = i + self.entry_delay
                    if entry_time < len(candles):
                        # Fade the gap
                        if gap_direction == "up":
                            # Gap up, expect fill (short/sell)
                            if in_position:
                                signals.iloc[entry_time] = Signal.SELL
                                in_position = False
                        else:
                            # Gap down, expect fill (long/buy)
                            signals.iloc[entry_time] = Signal.BUY
                            in_position = True
                            entry_idx = entry_time
                
                friday_close = None
                sunday_open = None
            
            # Exit after hold period
            if in_position and entry_idx is not None:
                bars_held = i - entry_idx
                if bars_held >= self.hold_period:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
        
        return signals


@strategy_registry.register("funding_hour_momentum")
class FundingHourMomentumStrategy(BaseStrategy):
    """
    Trade patterns around funding time.
    
    Funding happens at 00:00, 08:00, 16:00 UTC on most exchanges.
    
    Observed patterns:
    - Pre-funding: Positions adjusted (sometimes price impact)
    - Post-funding: Relief rally or selling pressure
    
    Strategy:
    - Monitor 1h before and after funding
    - Detect momentum patterns
    - Enter on consistent pre/post-funding moves
    """
    
    name = "funding_hour_momentum"
    
    def _setup(self, **params):
        self.funding_hours = params.get("funding_hours", [0, 8, 16])
        self.pre_funding_bars = params.get("pre_funding_bars", 1)
        self.post_funding_bars = params.get("post_funding_bars", 1)
        self.momentum_threshold = params.get("momentum_threshold", 0.005)
        self.hold_period = params.get("hold_period", 8)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on funding hour momentum."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < 24:
            return signals
        
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            hour = idx.hour
            
            # Check if this is a funding hour
            if hour in self.funding_hours:
                # Pre-funding momentum (1 hour before)
                if i >= self.pre_funding_bars:
                    pre_funding_start = i - self.pre_funding_bars
                    pre_funding_move = (
                        candles["close"].iloc[i] - candles["close"].iloc[pre_funding_start]
                    ) / candles["close"].iloc[pre_funding_start]
                    
                    # Strong pre-funding move = position for post-funding continuation
                    if abs(pre_funding_move) > self.momentum_threshold:
                        # Wait for post-funding confirmation
                        if i + self.post_funding_bars < len(candles):
                            post_idx = i + self.post_funding_bars
                            post_funding_move = (
                                candles["close"].iloc[post_idx] - candles["close"].iloc[i]
                            ) / candles["close"].iloc[i]
                            
                            # Momentum continues post-funding
                            if (pre_funding_move > 0 and post_funding_move > 0 and 
                                post_funding_move > self.momentum_threshold):
                                signals.iloc[post_idx] = Signal.BUY
                                in_position = True
                                entry_idx = post_idx
                            
                            elif (pre_funding_move < 0 and post_funding_move < 0 and
                                  post_funding_move < -self.momentum_threshold):
                                if in_position:
                                    signals.iloc[post_idx] = Signal.SELL
                                    in_position = False
            
            # Exit after hold period
            if in_position and entry_idx is not None:
                bars_held = i - entry_idx
                if bars_held >= self.hold_period:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
        
        return signals
