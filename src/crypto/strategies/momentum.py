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
