"""Momentum and trend following trading strategies."""

import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry


@strategy_registry.register(
    "momentum_breakout",
    description="Momentum Breakout Strategy",
)
class MomentumBreakoutStrategy(BaseStrategy):
    """
    Momentum Breakout Strategy.
    
    BUY when price breaks above recent high with strong momentum.
    SELL when price breaks below recent low.
    """

    name = "momentum_breakout"

    def _setup(
        self,
        lookback: int = 20,
        threshold: float = 0.02,
        volume_confirmation: bool = True,
        **kwargs,
    ) -> None:
        self.lookback = lookback
        self.threshold = threshold
        self.volume_confirmation = volume_confirmation

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        close = candles["close"]
        high = candles["high"]
        low = candles["low"]
        volume = candles["volume"]

        # Calculate breakout levels
        highest_high = high.rolling(window=self.lookback).max()
        lowest_low = low.rolling(window=self.lookback).min()
        
        # Volume confirmation
        volume_sma = volume.rolling(window=self.lookback).mean()
        high_volume = volume > volume_sma * 1.5

        signals = self.create_signal_series(candles.index)

        # Breakout above resistance
        breakout_up = close > highest_high.shift(1) * (1 + self.threshold)
        if self.volume_confirmation:
            breakout_up = breakout_up & high_volume
        signals.loc[breakout_up] = Signal.BUY

        # Breakdown below support
        breakdown = close < lowest_low.shift(1) * (1 - self.threshold)
        if self.volume_confirmation:
            breakdown = breakdown & high_volume
        signals.loc[breakdown] = Signal.SELL

        return signals


@strategy_registry.register(
    "trend_following",
    description="Trend Following Strategy",
)
class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy.
    
    Identifies and follows established trends using multiple
    indicators for confirmation.
    """

    name = "trend_following"

    def _setup(
        self,
        fast_period: int = 10,
        slow_period: int = 50,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        **kwargs,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        close = candles["close"]
        
        fast_ma = indicator_registry.compute("ema", candles, period=self.fast_period)
        slow_ma = indicator_registry.compute("ema", candles, period=self.slow_period)
        atr = indicator_registry.compute("atr", candles, period=self.atr_period)

        # Trend direction
        uptrend = fast_ma > slow_ma
        downtrend = fast_ma < slow_ma

        # Price momentum
        momentum = indicator_registry.compute("momentum", candles, period=self.fast_period)
        
        signals = self.create_signal_series(candles.index)

        # BUY on trend confirmation with positive momentum
        uptrend_shifted = uptrend.shift(1)
        uptrend_shifted = uptrend_shifted.where(pd.notna(uptrend_shifted), False)
        buy_signals = uptrend & (momentum > 0) & ~uptrend_shifted
        signals.loc[buy_signals] = Signal.BUY

        # SELL on downtrend with negative momentum
        downtrend_shifted = downtrend.shift(1)
        downtrend_shifted = downtrend_shifted.where(pd.notna(downtrend_shifted), False)
        sell_signals = downtrend & (momentum < 0) & ~downtrend_shifted
        signals.loc[sell_signals] = Signal.SELL

        return signals


@strategy_registry.register(
    "channel_breakout",
    description="Donchian Channel Breakout Strategy",
)
class ChannelBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel Breakout Strategy (Turtle Trading inspired).
    
    BUY when price breaks above N-period high.
    SELL when price breaks below N-period low.
    """

    name = "channel_breakout"

    def _setup(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        **kwargs,
    ) -> None:
        self.entry_period = entry_period
        self.exit_period = exit_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        high = candles["high"]
        low = candles["low"]
        close = candles["close"]

        # Entry channels
        entry_high = high.rolling(window=self.entry_period).max()
        entry_low = low.rolling(window=self.entry_period).min()

        signals = self.create_signal_series(candles.index)

        # BUY on break of entry high
        buy_signals = close > entry_high.shift(1)
        signals.loc[buy_signals] = Signal.BUY

        # SELL on break of entry low
        sell_signals = close < entry_low.shift(1)
        signals.loc[sell_signals] = Signal.SELL

        return signals


@strategy_registry.register(
    "adx_trend",
    description="ADX Trend Strength Strategy",
)
class ADXTrendStrategy(BaseStrategy):
    """
    ADX Trend Strength Strategy.
    
    Uses ADX to measure trend strength and +DI/-DI for direction.
    Only trades when ADX indicates strong trend.
    """

    name = "adx_trend"

    def _setup(
        self,
        period: int = 14,
        adx_threshold: int = 25,
        **kwargs,
    ) -> None:
        self.period = period
        self.adx_threshold = adx_threshold

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        high = candles["high"]
        low = candles["low"]
        close = candles["close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smooth with EMA
        atr = tr.ewm(span=self.period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.period, adjust=False).mean() / atr)

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=self.period, adjust=False).mean()

        signals = self.create_signal_series(candles.index)

        # Strong trend condition
        strong_trend = adx > self.adx_threshold

        # BUY when +DI crosses above -DI in strong trend
        bullish = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
        buy_signals = bullish & strong_trend
        signals.loc[buy_signals] = Signal.BUY

        # SELL when -DI crosses above +DI in strong trend
        bearish = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))
        sell_signals = bearish & strong_trend
        signals.loc[sell_signals] = Signal.SELL

        return signals


@strategy_registry.register(
    "roc_momentum",
    description="Rate of Change Momentum Strategy",
)
class ROCMomentumStrategy(BaseStrategy):
    """
    Rate of Change Momentum Strategy.
    
    BUY when ROC is strongly positive.
    SELL when ROC is strongly negative.
    """

    name = "roc_momentum"

    def _setup(
        self,
        period: int = 10,
        threshold: float = 5.0,
        **kwargs,
    ) -> None:
        self.period = period
        self.threshold = threshold

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        roc = indicator_registry.compute("roc", candles, period=self.period)

        signals = self.create_signal_series(candles.index)

        # BUY on strong positive momentum
        buy_signals = (roc > self.threshold) & (roc.shift(1) <= self.threshold)
        signals.loc[buy_signals] = Signal.BUY

        # SELL on strong negative momentum
        sell_signals = (roc < -self.threshold) & (roc.shift(1) >= -self.threshold)
        signals.loc[sell_signals] = Signal.SELL

        return signals


@strategy_registry.register(
    "momentum_quality",
    description="Quality-filtered momentum: only trade high-quality momentum signals",
)
class MomentumQualityStrategy(BaseStrategy):
    """
    Momentum Quality Strategy.
    
    Not all momentum is created equal. This strategy filters momentum signals
    by quality indicators:
    
    1. Volume confirmation: Momentum must be accompanied by above-average volume
    2. Trend alignment: Short-term momentum should align with longer-term trend
    3. Volatility normalization: Momentum should be significant relative to ATR
    
    This filters out false breakouts and noise-driven moves.
    """

    name = "momentum_quality"

    def _setup(
        self,
        momentum_period: int = 12,
        trend_period: int = 48,
        atr_period: int = 14,
        volume_period: int = 24,
        min_momentum_atr: float = 1.0,  # Momentum must be > 1x ATR
        min_volume_ratio: float = 1.2,  # Volume must be > 1.2x average
        hold_period: int = 24,
        require_trend_alignment: bool = True,
        **kwargs,
    ) -> None:
        self.momentum_period = momentum_period
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.volume_period = volume_period
        self.min_momentum_atr = min_momentum_atr
        self.min_volume_ratio = min_volume_ratio
        self.hold_period = hold_period
        self.require_trend_alignment = require_trend_alignment

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]
        volume = candles["volume"]

        # Calculate momentum (price change over period)
        momentum = close.diff(self.momentum_period)
        
        # Calculate trend direction
        trend_ma = close.rolling(self.trend_period, min_periods=10).mean()
        uptrend = close > trend_ma
        
        # Calculate ATR for volatility normalization
        atr = indicator_registry.compute("atr", candles, period=self.atr_period)
        
        # Normalize momentum by ATR
        momentum_atr = momentum / atr
        
        # Volume quality
        volume_avg = volume.rolling(self.volume_period, min_periods=5).mean()
        volume_ratio = volume / volume_avg
        
        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            mom_atr = momentum_atr.get(idx, 0)
            vol_ratio = volume_ratio.get(idx, 1)
            is_uptrend = uptrend.get(idx, False)
            
            if pd.isna(mom_atr) or pd.isna(vol_ratio):
                continue

            if not in_position:
                # Quality momentum signal
                is_quality_momentum = (
                    mom_atr > self.min_momentum_atr and
                    vol_ratio > self.min_volume_ratio
                )
                
                # Trend alignment check
                trend_ok = not self.require_trend_alignment or is_uptrend
                
                if is_quality_momentum and trend_ok:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                bars_held = i - entry_bar
                
                # Exit after hold period or on quality sell signal
                if bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
                elif mom_atr < -self.min_momentum_atr and vol_ratio > self.min_volume_ratio:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "trend_strength_filter",
    description="Meta-strategy: only trade when trend is strong (ADX filter)",
)
class TrendStrengthFilterStrategy(BaseStrategy):
    """
    Trend Strength Filter Strategy.
    
    A meta-strategy that filters trading signals based on ADX trend strength.
    
    Logic:
    - When ADX > strong_threshold: Strong trend, use momentum signals
    - When ADX < weak_threshold: Ranging market, use mean reversion signals
    - When ADX between thresholds: Ambiguous, don't trade
    
    This exploits the observation that different strategies work in
    different market regimes.
    """

    name = "trend_strength_filter"

    def _setup(
        self,
        adx_period: int = 14,
        strong_threshold: int = 30,
        weak_threshold: int = 20,
        # Momentum params (for trending markets)
        momentum_lookback: int = 20,
        momentum_threshold: float = 0.02,
        # Mean reversion params (for ranging markets)
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        hold_period: int = 24,
        **kwargs,
    ) -> None:
        self.adx_period = adx_period
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.momentum_lookback = momentum_lookback
        self.momentum_threshold = momentum_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.hold_period = hold_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]
        
        # Calculate ADX
        adx = indicator_registry.compute("adx", candles, period=self.adx_period)
        
        # Calculate momentum
        momentum = close.pct_change(self.momentum_lookback)
        
        # Calculate RSI
        rsi = indicator_registry.compute("rsi", candles, period=self.rsi_period)
        
        # Generate signals based on regime
        in_position = False
        entry_bar = 0
        entry_regime = None

        for i, idx in enumerate(candles.index):
            adx_val = adx.get(idx, 25)  # Default to middle range
            mom = momentum.get(idx, 0)
            rsi_val = rsi.get(idx, 50)
            
            if pd.isna(adx_val) or pd.isna(mom) or pd.isna(rsi_val):
                continue

            if not in_position:
                # Strong trend regime: Use momentum
                if adx_val > self.strong_threshold:
                    if mom > self.momentum_threshold:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
                        entry_regime = "trend"
                
                # Weak trend regime: Use mean reversion
                elif adx_val < self.weak_threshold:
                    if rsi_val < self.rsi_oversold:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
                        entry_regime = "range"
                
                # Ambiguous regime: Don't trade
                # (no signal generated)
            
            else:
                bars_held = i - entry_bar
                should_exit = False
                
                # Exit after hold period
                if bars_held >= self.hold_period:
                    should_exit = True
                
                # Regime-specific exits
                if entry_regime == "trend":
                    # Exit if momentum reverses or regime changes
                    if mom < -self.momentum_threshold / 2:
                        should_exit = True
                    if adx_val < self.weak_threshold:
                        should_exit = True
                
                elif entry_regime == "range":
                    # Exit if RSI recovers or regime changes
                    if rsi_val > self.rsi_overbought:
                        should_exit = True
                    if adx_val > self.strong_threshold:
                        should_exit = True
                
                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
                    entry_regime = None

        return self.apply_filters(signals, candles)
