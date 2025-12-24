"""
Information-theoretic and statistical physics-based trading strategies.

These strategies use concepts from information theory and statistical physics
to detect regime changes and market structure:
- Entropy measures order/chaos in price returns
- Hurst exponent measures market "memory" (trending vs mean-reverting)
- Fractal dimension measures market complexity
"""

import numpy as np
import pandas as pd
from scipy import stats

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry


@strategy_registry.register("entropy_collapse")
class EntropyCollapseStrategy(BaseStrategy):
    """
    Trade when Shannon entropy of returns collapses.
    
    Low entropy = market is becoming more predictable/ordered
    This often precedes regime changes or volatility spikes.
    
    Strategy:
    - Calculate Shannon entropy of 24h returns distribution
    - When entropy drops below 10th percentile, enter with momentum
    - Exit when entropy rises above 90th percentile
    """
    
    name = "entropy_collapse"
    
    def _setup(self, **params):
        self.entropy_window = params.get("entropy_window", 24)
        self.percentile_window = params.get("percentile_window", 168)
        self.low_threshold = params.get("low_threshold", 10)
        self.high_threshold = params.get("high_threshold", 90)
        self.hold_period = params.get("hold_period", 12)
        self.use_momentum_direction = params.get("use_momentum_direction", True)
    
    def _calculate_entropy(self, returns: pd.Series) -> float:
        """Calculate Shannon entropy of returns distribution."""
        if len(returns) < 2 or returns.std() == 0:
            return np.nan
        
        # Create histogram bins
        n_bins = min(10, len(returns) // 3)
        if n_bins < 2:
            return np.nan
        
        hist, _ = np.histogram(returns, bins=n_bins)
        
        # Convert to probabilities
        probs = hist / hist.sum()
        probs = probs[probs > 0]  # Remove zeros
        
        # Shannon entropy: -sum(p * log(p))
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on entropy collapse."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.percentile_window:
            return signals
        
        # Calculate returns
        returns = candles["close"].pct_change()
        
        # Calculate rolling entropy
        entropy = returns.rolling(window=self.entropy_window).apply(
            self._calculate_entropy, raw=False
        )
        
        # Calculate percentiles
        entropy_low = entropy.rolling(window=self.percentile_window).quantile(
            self.low_threshold / 100
        )
        entropy_high = entropy.rolling(window=self.percentile_window).quantile(
            self.high_threshold / 100
        )
        
        # Entry: entropy collapse (low entropy)
        entry_condition = entropy < entropy_low
        
        # Determine direction from momentum if requested
        if self.use_momentum_direction:
            momentum = returns.rolling(window=self.hold_period).mean()
            long_direction = momentum > 0
            short_direction = momentum < 0
            
            # Long on low entropy + positive momentum
            signals.loc[entry_condition & long_direction] = Signal.BUY
            # Short would be: signals.loc[entry_condition & short_direction] = Signal.SELL
        else:
            # Default to long
            signals.loc[entry_condition] = Signal.BUY
        
        # Exit: entropy expansion (high entropy = chaos)
        exit_condition = entropy > entropy_high
        signals.loc[exit_condition] = Signal.SELL
        
        # Apply position tracking (hold for hold_period bars minimum)
        signals = self._apply_hold_period(signals, candles.index)
        
        return signals
    
    def _apply_hold_period(self, signals: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Apply minimum hold period to positions."""
        result = signals.copy()
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(index):
            if signals.iloc[i] == Signal.BUY and not in_position:
                in_position = True
                entry_idx = i
                result.iloc[i] = Signal.BUY
            elif in_position:
                # Hold position for hold_period bars
                bars_held = i - entry_idx
                if bars_held < self.hold_period:
                    result.iloc[i] = Signal.HOLD
                elif signals.iloc[i] == Signal.SELL:
                    result.iloc[i] = Signal.SELL
                    in_position = False
                else:
                    result.iloc[i] = Signal.HOLD
            else:
                result.iloc[i] = Signal.HOLD
        
        return result


@strategy_registry.register("hurst_regime")
class HurstRegimeStrategy(BaseStrategy):
    """
    Adapt strategy based on Hurst exponent.
    
    Hurst exponent measures market "memory":
    - H < 0.5: Mean-reverting (use RSI)
    - H = 0.5: Random walk
    - H > 0.5: Trending (use momentum)
    
    Strategy:
    - Calculate Hurst exponent over lookback window
    - H < 0.4: Trade mean reversion (RSI oversold/overbought)
    - H > 0.6: Trade momentum breakouts
    - 0.4 <= H <= 0.6: No trades (ambiguous regime)
    """
    
    name = "hurst_regime"
    
    def _setup(self, **params):
        self.hurst_window = params.get("hurst_window", 168)
        self.mean_revert_threshold = params.get("mean_revert_threshold", 0.4)
        self.trending_threshold = params.get("trending_threshold", 0.6)
        self.recompute_every = params.get("recompute_every", 24)
        
        # Mean reversion params
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.rsi_overbought = params.get("rsi_overbought", 70)
        
        # Momentum params
        self.momentum_period = params.get("momentum_period", 20)
        self.momentum_threshold = params.get("momentum_threshold", 0.02)
    
    def _calculate_hurst(self, prices: pd.Series) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        if len(prices) < 20:
            return 0.5  # Default to random walk
        
        try:
            # Convert to log returns
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            if len(log_returns) < 10:
                return 0.5
            
            # Calculate R/S statistic for different lags
            lags = range(2, min(len(log_returns) // 2, 20))
            rs_values = []
            
            for lag in lags:
                # Split into lag-sized chunks
                chunks = [log_returns.iloc[i:i+lag] for i in range(0, len(log_returns), lag)]
                chunks = [c for c in chunks if len(c) == lag]
                
                if len(chunks) == 0:
                    continue
                
                rs_list = []
                for chunk in chunks:
                    mean_chunk = chunk.mean()
                    std_chunk = chunk.std()
                    
                    if std_chunk == 0:
                        continue
                    
                    # Cumulative deviation from mean
                    cumdev = (chunk - mean_chunk).cumsum()
                    
                    # Range
                    R = cumdev.max() - cumdev.min()
                    
                    # R/S ratio
                    rs = R / std_chunk if std_chunk > 0 else 0
                    rs_list.append(rs)
                
                if len(rs_list) > 0:
                    rs_values.append(np.mean(rs_list))
            
            if len(rs_values) < 2:
                return 0.5
            
            # Hurst = slope of log(R/S) vs log(lag)
            log_lags = np.log(list(lags)[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Linear regression
            slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
            
            # Constrain to [0, 1]
            hurst = max(0.0, min(1.0, slope))
            return hurst
            
        except Exception:
            return 0.5  # Default to random walk on error
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on Hurst regime."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.hurst_window:
            return signals
        
        prices = candles["close"]
        
        # Calculate Hurst exponent (recompute every N bars)
        hurst_values = pd.Series(index=candles.index, dtype=float)
        
        for i in range(self.hurst_window, len(candles), self.recompute_every):
            window = prices.iloc[max(0, i - self.hurst_window):i]
            hurst = self._calculate_hurst(window)
            
            # Forward fill until next recomputation
            end_idx = min(i + self.recompute_every, len(candles))
            hurst_values.iloc[i:end_idx] = hurst
        
        # Forward fill any remaining NaNs
        hurst_values = hurst_values.fillna(method='ffill')
        
        # Calculate RSI for mean-reverting regime
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate momentum for trending regime
        momentum = prices.pct_change(periods=self.momentum_period)
        
        # Generate signals based on regime
        for i, idx in enumerate(candles.index):
            h = hurst_values.iloc[i]
            
            if pd.isna(h):
                continue
            
            # Mean-reverting regime
            if h < self.mean_revert_threshold:
                if rsi.iloc[i] < self.rsi_oversold:
                    signals.iloc[i] = Signal.BUY
                elif rsi.iloc[i] > self.rsi_overbought:
                    signals.iloc[i] = Signal.SELL
            
            # Trending regime
            elif h > self.trending_threshold:
                if momentum.iloc[i] > self.momentum_threshold:
                    signals.iloc[i] = Signal.BUY
                elif momentum.iloc[i] < -self.momentum_threshold:
                    signals.iloc[i] = Signal.SELL
        
        return signals


@strategy_registry.register("fractal_breakout")
class FractalBreakoutStrategy(BaseStrategy):
    """
    Trade when fractal dimension drops (consolidation) then breaks range.
    
    Fractal dimension measures market complexity:
    - Low dimension = simple, ranging market
    - High dimension = complex, choppy market
    
    Strategy:
    - Calculate fractal dimension using box-counting
    - When dimension drops below threshold, market is consolidating
    - Enter on breakout from consolidation range
    """
    
    name = "fractal_breakout"
    
    def _setup(self, **params):
        self.fractal_window = params.get("fractal_window", 48)
        self.dimension_threshold = params.get("dimension_threshold", 1.2)
        self.breakout_period = params.get("breakout_period", 24)
        self.min_hold_period = params.get("min_hold_period", 6)
        self.max_hold_period = params.get("max_hold_period", 48)
    
    def _calculate_fractal_dimension(self, prices: pd.Series) -> float:
        """Calculate fractal dimension using Higuchi's method."""
        if len(prices) < 10:
            return 1.5  # Default
        
        try:
            # Convert to numpy array
            x = prices.values
            n = len(x)
            
            # Higuchi's method
            k_max = min(10, n // 4)
            k_values = range(1, k_max + 1)
            lk_values = []
            
            for k in k_values:
                lm_values = []
                
                for m in range(k):
                    # Construct m-th sequence
                    indices = range(m, n, k)
                    if len(indices) < 2:
                        continue
                    
                    # Calculate length of curve
                    length = 0
                    for i in range(len(indices) - 1):
                        length += abs(x[indices[i + 1]] - x[indices[i]])
                    
                    # Normalize
                    length = length * (n - 1) / (len(indices) * k)
                    lm_values.append(length)
                
                if len(lm_values) > 0:
                    lk_values.append(np.mean(lm_values))
            
            if len(lk_values) < 2:
                return 1.5
            
            # Fractal dimension = slope of log(L(k)) vs log(1/k)
            log_k = np.log(k_values[:len(lk_values)])
            log_lk = np.log(lk_values)
            
            slope, _, _, _, _ = stats.linregress(log_k, log_lk)
            
            # Fractal dimension
            fractal_dim = -slope
            
            # Constrain to reasonable range
            return max(1.0, min(2.0, fractal_dim))
            
        except Exception:
            return 1.5
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on fractal breakout."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.fractal_window + self.breakout_period:
            return signals
        
        prices = candles["close"]
        highs = candles["high"]
        lows = candles["low"]
        
        # Calculate fractal dimension
        fractal_dim = pd.Series(index=candles.index, dtype=float)
        
        for i in range(self.fractal_window, len(candles)):
            window = prices.iloc[i - self.fractal_window:i]
            fd = self._calculate_fractal_dimension(window)
            fractal_dim.iloc[i] = fd
        
        # Detect consolidation (low fractal dimension)
        in_consolidation = fractal_dim < self.dimension_threshold
        
        # Calculate breakout levels
        rolling_high = highs.rolling(window=self.breakout_period).max()
        rolling_low = lows.rolling(window=self.breakout_period).min()
        
        # Entry conditions
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            if i < self.fractal_window:
                continue
            
            # Check for breakout from consolidation
            if in_consolidation.iloc[i]:
                # Upside breakout
                if candles["close"].iloc[i] > rolling_high.iloc[i - 1]:
                    signals.iloc[i] = Signal.BUY
                    in_position = True
                    entry_idx = i
                # Downside breakout
                elif candles["close"].iloc[i] < rolling_low.iloc[i - 1]:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
            
            # Exit after hold period
            if in_position and entry_idx is not None:
                bars_held = i - entry_idx
                if bars_held >= self.max_hold_period:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
        
        return signals
