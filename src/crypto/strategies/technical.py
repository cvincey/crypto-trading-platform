"""Technical indicator-based trading strategies."""

import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry


@strategy_registry.register(
    "sma_crossover",
    description="Simple Moving Average Crossover Strategy",
)
class SMACrossoverStrategy(BaseStrategy):
    """
    SMA Crossover Strategy.
    
    Generates BUY signal when fast SMA crosses above slow SMA.
    Generates SELL signal when fast SMA crosses below slow SMA.
    """

    name = "sma_crossover"

    def _setup(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        **kwargs,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        # Calculate SMAs
        fast_sma = indicator_registry.compute("sma", candles, period=self.fast_period)
        slow_sma = indicator_registry.compute("sma", candles, period=self.slow_period)

        # Initialize signals
        signals = self.create_signal_series(candles.index)

        # Generate crossover signals
        fast_above = fast_sma > slow_sma
        fast_below = fast_sma < slow_sma

        # BUY when fast crosses above slow
        buy_signals = fast_above & fast_below.shift(1)
        signals.loc[buy_signals] = Signal.BUY

        # SELL when fast crosses below slow
        sell_signals = fast_below & fast_above.shift(1)
        signals.loc[sell_signals] = Signal.SELL

        return signals


@strategy_registry.register(
    "ema_crossover",
    description="Exponential Moving Average Crossover Strategy",
)
class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy.
    
    Similar to SMA crossover but uses exponential moving averages.
    """

    name = "ema_crossover"

    def _setup(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        **kwargs,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        fast_ema = indicator_registry.compute("ema", candles, period=self.fast_period)
        slow_ema = indicator_registry.compute("ema", candles, period=self.slow_period)

        signals = self.create_signal_series(candles.index)

        fast_above = fast_ema > slow_ema
        fast_below = fast_ema < slow_ema

        buy_signals = fast_above & fast_below.shift(1)
        signals.loc[buy_signals] = Signal.BUY

        sell_signals = fast_below & fast_above.shift(1)
        signals.loc[sell_signals] = Signal.SELL

        return signals


@strategy_registry.register(
    "rsi_overbought_oversold",
    description="RSI Overbought/Oversold Strategy",
)
class RSIOverboughtOversoldStrategy(BaseStrategy):
    """
    RSI Overbought/Oversold Strategy.
    
    BUY when RSI crosses below oversold threshold (e.g., 30).
    SELL when RSI crosses above overbought threshold (e.g., 70).
    """

    name = "rsi_overbought_oversold"

    def _setup(
        self,
        period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
        **kwargs,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        rsi = indicator_registry.compute("rsi", candles, period=self.period)
        signals = self.create_signal_series(candles.index)

        # BUY when RSI crosses up through oversold
        oversold_cross = (rsi > self.oversold) & (rsi.shift(1) <= self.oversold)
        signals.loc[oversold_cross] = Signal.BUY

        # SELL when RSI crosses down through overbought
        overbought_cross = (rsi < self.overbought) & (rsi.shift(1) >= self.overbought)
        signals.loc[overbought_cross] = Signal.SELL

        return signals


@strategy_registry.register(
    "macd_crossover",
    description="MACD Crossover Strategy",
)
class MACDCrossoverStrategy(BaseStrategy):
    """
    MACD Crossover Strategy.
    
    BUY when MACD line crosses above signal line.
    SELL when MACD line crosses below signal line.
    """

    name = "macd_crossover"

    def _setup(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **kwargs,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        macd_df = indicator_registry.compute(
            "macd",
            candles,
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period,
        )

        macd_line = macd_df["macd"]
        signal_line = macd_df["signal"]

        signals = self.create_signal_series(candles.index)

        macd_above = macd_line > signal_line
        macd_below = macd_line < signal_line

        # BUY on bullish crossover
        buy_signals = macd_above & macd_below.shift(1)
        signals.loc[buy_signals] = Signal.BUY

        # SELL on bearish crossover
        sell_signals = macd_below & macd_above.shift(1)
        signals.loc[sell_signals] = Signal.SELL

        return signals


@strategy_registry.register(
    "bollinger_breakout",
    description="Bollinger Bands Breakout Strategy",
)
class BollingerBreakoutStrategy(BaseStrategy):
    """
    Bollinger Bands Breakout Strategy.
    
    BUY when price breaks above upper band (momentum breakout).
    SELL when price breaks below lower band.
    
    Alternative mode: Mean reversion (buy at lower, sell at upper).
    """

    name = "bollinger_breakout"

    def _setup(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        mean_reversion: bool = False,
        **kwargs,
    ) -> None:
        self.period = period
        self.std_dev = std_dev
        self.mean_reversion = mean_reversion

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        bb = indicator_registry.compute(
            "bollinger",
            candles,
            period=self.period,
            std_dev=self.std_dev,
        )

        close = candles["close"]
        signals = self.create_signal_series(candles.index)

        if self.mean_reversion:
            # Mean reversion: buy at lower, sell at upper
            buy_signals = close < bb["lower"]
            sell_signals = close > bb["upper"]
        else:
            # Breakout: buy above upper, sell below lower
            buy_signals = (close > bb["upper"]) & (close.shift(1) <= bb["upper"].shift(1))
            sell_signals = (close < bb["lower"]) & (close.shift(1) >= bb["lower"].shift(1))

        signals.loc[buy_signals] = Signal.BUY
        signals.loc[sell_signals] = Signal.SELL

        return signals


@strategy_registry.register(
    "triple_ma",
    description="Triple Moving Average Strategy",
)
class TripleMAStrategy(BaseStrategy):
    """
    Triple Moving Average Strategy.
    
    Uses three MAs (fast, medium, slow) for trend confirmation.
    BUY when fast > medium > slow (strong uptrend).
    SELL when fast < medium < slow (strong downtrend).
    """

    name = "triple_ma"

    def _setup(
        self,
        fast_period: int = 5,
        medium_period: int = 20,
        slow_period: int = 50,
        ma_type: str = "sma",
        **kwargs,
    ) -> None:
        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period
        self.ma_type = ma_type

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        fast = indicator_registry.compute(self.ma_type, candles, period=self.fast_period)
        medium = indicator_registry.compute(self.ma_type, candles, period=self.medium_period)
        slow = indicator_registry.compute(self.ma_type, candles, period=self.slow_period)

        signals = self.create_signal_series(candles.index)

        # Strong uptrend
        uptrend = (fast > medium) & (medium > slow)
        uptrend_start = uptrend & ~uptrend.shift(1).fillna(False)
        signals.loc[uptrend_start] = Signal.BUY

        # Strong downtrend
        downtrend = (fast < medium) & (medium < slow)
        downtrend_start = downtrend & ~downtrend.shift(1).fillna(False)
        signals.loc[downtrend_start] = Signal.SELL

        return signals
