"""
Sentiment-based trading strategies.

These strategies use external sentiment indicators:
- Fear & Greed Index (alternative.me)
- Divergence between sentiment and price
- Contrarian fades of extreme sentiment
"""

import logging

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register("fear_greed_divergence")
class FearGreedDivergenceStrategy(BaseStrategy):
    """
    Trade when Fear & Greed Index diverges from price.
    
    Divergence signals:
    - Price up but F&G down = sentiment lagging, possible reversal
    - Price down but F&G up = sentiment too optimistic, possible reversal
    
    Strategy:
    - Detect divergences over 7-day windows
    - Minimum 5% price move + 10 point F&G move
    - Trade the divergence (expect convergence)
    
    Requires Fear & Greed Index data - run ingest_tier2_data.py first.
    """
    
    name = "fear_greed_divergence"
    
    def _setup(self, **params):
        self.divergence_window = params.get("divergence_window", 7)  # days
        self.min_price_change = params.get("min_price_change", 0.05)  # 5%
        self.min_fg_change = params.get("min_fg_change", 10)  # points
        self.hold_period = params.get("hold_period", 168)  # hours
        self._fear_greed_data: pd.Series | None = None
    
    def set_fear_greed_data(self, fg_series: pd.Series) -> None:
        """Set Fear & Greed data aligned to candle index."""
        self._fear_greed_data = fg_series
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on Fear & Greed divergence."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._fear_greed_data is None or self._fear_greed_data.empty:
            logger.warning(
                f"{self.name}: Fear & Greed Index data not available. "
                "Run: python scripts/ingest_tier2_data.py --fear-greed-only"
            )
            return signals
        
        # Align fear greed to candles (forward fill daily to hourly)
        fear_greed = self._fear_greed_data.reindex(candles.index, method="ffill")
        
        close = candles["close"]
        lookback_bars = self.divergence_window * 24  # Convert days to hours
        
        in_position = False
        entry_idx = None
        
        for i in range(lookback_bars, len(candles)):
            idx = candles.index[i]
            window_start_idx = i - lookback_bars
            
            # Calculate changes over window
            price_change = (close.iloc[i] - close.iloc[window_start_idx]) / close.iloc[window_start_idx]
            
            fg_current = fear_greed.iloc[i]
            fg_start = fear_greed.iloc[window_start_idx]
            
            if pd.isna(fg_current) or pd.isna(fg_start):
                continue
            
            fg_change = fg_current - fg_start
            
            if not in_position:
                # Bearish divergence: price up, F&G down
                if price_change > self.min_price_change and fg_change < -self.min_fg_change:
                    # Sentiment not confirming rally - expect pullback
                    # For spot only, we exit longs rather than short
                    pass  # Could set a "no entry" flag
                
                # Bullish divergence: price down, F&G up or stable
                elif price_change < -self.min_price_change and fg_change > -self.min_fg_change:
                    # Sentiment holding up despite price drop - buy the dip
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_idx = i
                
                # Extreme fear + price drop = buy opportunity
                elif fg_current < 25 and price_change < -self.min_price_change / 2:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_idx = i
            else:
                bars_held = i - entry_idx
                
                # Exit conditions
                if bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
                elif fg_current > 75:  # Extreme greed - take profits
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)


@strategy_registry.register("fear_greed_extreme_fade")
class FearGreedExtremeFadeStrategy(BaseStrategy):
    """
    Classic contrarian strategy on Fear & Greed extremes.
    
    Fear & Greed Index scale:
    - 0-20: Extreme Fear
    - 20-40: Fear
    - 40-60: Neutral
    - 60-80: Greed
    - 80-100: Extreme Greed
    
    Strategy:
    - F&G < 20: Buy (extreme fear = oversold)
    - F&G > 80: Sell/exit longs (extreme greed = overbought)
    - Hold for 1 week
    
    Requires Fear & Greed Index data - run ingest_tier2_data.py first.
    """
    
    name = "fear_greed_extreme_fade"
    
    def _setup(self, **params):
        self.extreme_fear = params.get("extreme_fear", 20)
        self.extreme_greed = params.get("extreme_greed", 80)
        self.fear_entry = params.get("fear_entry", 25)  # Enter on fear
        self.hold_period = params.get("hold_period", 168)  # hours
        self.confirmation_bars = params.get("confirmation_bars", 24)  # 1 day
        self._fear_greed_data: pd.Series | None = None
    
    def set_fear_greed_data(self, fg_series: pd.Series) -> None:
        """Set Fear & Greed data aligned to candle index."""
        self._fear_greed_data = fg_series
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on extreme Fear & Greed."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._fear_greed_data is None or self._fear_greed_data.empty:
            logger.warning(
                f"{self.name}: Fear & Greed Index data not available. "
                "Run: python scripts/ingest_tier2_data.py --fear-greed-only"
            )
            return signals
        
        # Align fear greed to candles
        fear_greed = self._fear_greed_data.reindex(candles.index, method="ffill")
        
        in_position = False
        entry_idx = None
        fear_bars = 0
        
        for i, idx in enumerate(candles.index):
            fg_value = fear_greed.get(idx)
            
            if pd.isna(fg_value):
                continue
            
            if not in_position:
                # Track consecutive fear bars
                if fg_value < self.fear_entry:
                    fear_bars += 1
                else:
                    fear_bars = 0
                
                # Enter on extreme fear with confirmation
                if fg_value < self.extreme_fear and fear_bars >= self.confirmation_bars:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_idx = i
                    fear_bars = 0
            else:
                bars_held = i - entry_idx
                
                # Exit conditions
                should_exit = False
                
                # Exit on extreme greed
                if fg_value > self.extreme_greed:
                    should_exit = True
                
                # Exit after hold period
                if bars_held >= self.hold_period:
                    should_exit = True
                
                # Exit if back to neutral and profitable
                if fg_value > 50 and bars_held >= 48:  # 2 days minimum
                    should_exit = True
                
                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)


@strategy_registry.register("fear_greed_contrarian")
class FearGreedContrarianStrategy(BaseStrategy):
    """
    Simple contrarian strategy: buy fear, avoid greed.
    
    This is a simplified version that's easier to validate:
    - Buy when F&G drops below threshold
    - Sell when F&G rises above threshold
    - Uses rolling average to smooth daily noise
    """
    
    name = "fear_greed_contrarian"
    
    def _setup(self, **params):
        self.buy_threshold = params.get("buy_threshold", 30)
        self.sell_threshold = params.get("sell_threshold", 70)
        self.smoothing_period = params.get("smoothing_period", 3)  # days
        self.min_hold_hours = params.get("min_hold_hours", 48)
        self._fear_greed_data: pd.Series | None = None
    
    def set_fear_greed_data(self, fg_series: pd.Series) -> None:
        """Set Fear & Greed data aligned to candle index."""
        self._fear_greed_data = fg_series
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on Fear & Greed levels."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._fear_greed_data is None or self._fear_greed_data.empty:
            logger.warning(
                f"{self.name}: Fear & Greed Index data not available. "
                "Run: python scripts/ingest_tier2_data.py --fear-greed-only"
            )
            return signals
        
        # Align and smooth
        fear_greed = self._fear_greed_data.reindex(candles.index, method="ffill")
        fg_smooth = fear_greed.rolling(self.smoothing_period * 24, min_periods=1).mean()
        
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            fg = fg_smooth.get(idx)
            
            if pd.isna(fg):
                continue
            
            if not in_position:
                if fg < self.buy_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_idx = i
            else:
                bars_held = i - entry_idx
                
                if bars_held >= self.min_hold_hours and fg > self.sell_threshold:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)
