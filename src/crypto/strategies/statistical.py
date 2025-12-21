"""Statistical and mean reversion trading strategies."""

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry


@strategy_registry.register(
    "rsi_mean_reversion",
    description="RSI Mean Reversion Strategy",
)
class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.
    
    Assumes prices revert to mean after extreme RSI readings.
    BUY when RSI is oversold.
    SELL when RSI is overbought.
    """

    name = "rsi_mean_reversion"

    def _setup(
        self,
        period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
        exit_middle: int = 50,
        **kwargs,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_middle = exit_middle

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        rsi = indicator_registry.compute("rsi", candles, period=self.period)
        signals = self.create_signal_series(candles.index)

        # BUY when RSI goes below oversold
        buy_signals = rsi < self.oversold
        signals.loc[buy_signals] = Signal.BUY

        # SELL when RSI goes above overbought
        sell_signals = rsi > self.overbought
        signals.loc[sell_signals] = Signal.SELL

        # Apply config-driven filters
        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "z_score_mean_reversion",
    description="Z-Score Mean Reversion Strategy",
)
class ZScoreMeanReversionStrategy(BaseStrategy):
    """
    Z-Score Mean Reversion Strategy.
    
    Uses z-score (standard deviations from mean) to identify
    extreme price moves that are likely to revert.
    """

    name = "z_score_mean_reversion"

    def _setup(
        self,
        lookback: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        **kwargs,
    ) -> None:
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        close = candles["close"]
        
        # Calculate z-score
        rolling_mean = close.rolling(window=self.lookback).mean()
        rolling_std = close.rolling(window=self.lookback).std()
        z_score = (close - rolling_mean) / rolling_std

        signals = self.create_signal_series(candles.index)

        # BUY when z-score is very negative (price is low)
        buy_signals = z_score < -self.entry_threshold
        signals.loc[buy_signals] = Signal.BUY

        # SELL when z-score is very positive (price is high)
        sell_signals = z_score > self.entry_threshold
        signals.loc[sell_signals] = Signal.SELL

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "bollinger_mean_reversion",
    description="Bollinger Bands Mean Reversion Strategy",
)
class BollingerMeanReversionStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.
    
    BUY when price touches lower band (oversold).
    SELL when price touches upper band (overbought).
    """

    name = "bollinger_mean_reversion"

    def _setup(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        **kwargs,
    ) -> None:
        self.period = period
        self.std_dev = std_dev

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

        # BUY when price crosses below lower band
        buy_signals = (close < bb["lower"]) & (close.shift(1) >= bb["lower"].shift(1))
        signals.loc[buy_signals] = Signal.BUY

        # SELL when price crosses above upper band
        sell_signals = (close > bb["upper"]) & (close.shift(1) <= bb["upper"].shift(1))
        signals.loc[sell_signals] = Signal.SELL

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "pairs_ratio",
    description="Pairs Trading Ratio Strategy",
)
class PairsRatioStrategy(BaseStrategy):
    """
    Simple Pairs/Ratio Trading Strategy.
    
    For single-asset version: trades based on price ratio to its MA.
    When ratio is extreme, expects reversion.
    """

    name = "pairs_ratio"

    def _setup(
        self,
        lookback: int = 20,
        entry_std: float = 2.0,
        exit_std: float = 0.5,
        **kwargs,
    ) -> None:
        self.lookback = lookback
        self.entry_std = entry_std
        self.exit_std = exit_std

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        close = candles["close"]
        ma = close.rolling(window=self.lookback).mean()
        
        # Ratio of price to MA
        ratio = close / ma
        ratio_mean = ratio.rolling(window=self.lookback).mean()
        ratio_std = ratio.rolling(window=self.lookback).std()
        
        z_ratio = (ratio - ratio_mean) / ratio_std

        signals = self.create_signal_series(candles.index)

        # BUY when ratio is abnormally low
        buy_signals = z_ratio < -self.entry_std
        signals.loc[buy_signals] = Signal.BUY

        # SELL when ratio is abnormally high
        sell_signals = z_ratio > self.entry_std
        signals.loc[sell_signals] = Signal.SELL

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "keltner_mean_reversion",
    description="Keltner Channel Mean Reversion Strategy",
)
class KeltnerMeanReversionStrategy(BaseStrategy):
    """
    Keltner Channel Mean Reversion Strategy.
    
    Uses EMA and ATR to create bands.
    BUY when price is below lower band.
    SELL when price is above upper band.
    """

    name = "keltner_mean_reversion"

    def _setup(
        self,
        ema_period: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        **kwargs,
    ) -> None:
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        ema = indicator_registry.compute("ema", candles, period=self.ema_period)
        atr = indicator_registry.compute("atr", candles, period=self.atr_period)

        upper = ema + (atr * self.atr_multiplier)
        lower = ema - (atr * self.atr_multiplier)

        close = candles["close"]
        signals = self.create_signal_series(candles.index)

        # BUY when price is below lower channel
        buy_signals = close < lower
        signals.loc[buy_signals] = Signal.BUY

        # SELL when price is above upper channel
        sell_signals = close > upper
        signals.loc[sell_signals] = Signal.SELL

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "vwap_reversion",
    description="VWAP Reversion Strategy",
)
class VWAPReversionStrategy(BaseStrategy):
    """
    VWAP Reversion Strategy.
    
    Trades reversion to Volume Weighted Average Price.
    BUY when price is significantly below VWAP.
    SELL when price is significantly above VWAP.
    
    Config params:
        vwap_period: VWAP calculation period
        entry_deviation: Percentage deviation from VWAP to enter (e.g., 0.02 = 2%)
        exit_deviation: Percentage deviation from VWAP to exit
    """

    name = "vwap_reversion"

    def _setup(
        self,
        vwap_period: int = 20,
        entry_deviation: float = 0.02,
        exit_deviation: float = 0.005,
        **kwargs,
    ) -> None:
        self.vwap_period = vwap_period
        self.entry_deviation = entry_deviation
        self.exit_deviation = exit_deviation

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        vwap = indicator_registry.compute("vwap", candles, period=self.vwap_period)
        close = candles["close"]

        signals = self.create_signal_series(candles.index)

        # Calculate deviation from VWAP
        deviation = (close - vwap) / vwap

        # BUY when price is significantly below VWAP
        buy_signals = deviation < -self.entry_deviation
        signals.loc[buy_signals] = Signal.BUY

        # SELL when price is significantly above VWAP
        sell_signals = deviation > self.entry_deviation
        signals.loc[sell_signals] = Signal.SELL

        return self.apply_filters(signals, candles)
