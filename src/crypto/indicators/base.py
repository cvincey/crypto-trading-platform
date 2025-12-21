"""Technical indicator base classes and registry."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from crypto.core.registry import Registry


class Indicator(ABC):
    """
    Base class for technical indicators.
    
    Indicators take OHLCV data and compute derived values
    that can be used by trading strategies.
    """

    name: str = "base_indicator"

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Compute the indicator value(s).
        
        Args:
            df: DataFrame with OHLCV data
                Expected columns: open, high, low, close, volume
                
        Returns:
            Series or DataFrame with indicator values
        """
        pass

    def __call__(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Allow calling indicator as function."""
        return self.compute(df)


class IndicatorRegistry(Registry[Indicator]):
    """Registry for technical indicators."""

    def compute(
        self,
        name: str,
        df: pd.DataFrame,
        **params: Any,
    ) -> pd.Series | pd.DataFrame:
        """
        Compute an indicator by name.
        
        Args:
            name: Registered indicator name
            df: OHLCV DataFrame
            **params: Indicator parameters
            
        Returns:
            Indicator values
        """
        indicator = self.create(name, **params)
        return indicator.compute(df)


# Global indicator registry
indicator_registry = IndicatorRegistry("indicators")


# =============================================================================
# Common Indicator Implementations
# =============================================================================


@indicator_registry.register("sma", description="Simple Moving Average")
class SMA(Indicator):
    """Simple Moving Average indicator."""

    name = "sma"

    def __init__(self, period: int = 20, column: str = "close"):
        self.period = period
        self.column = column

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df[self.column].rolling(window=self.period).mean()


@indicator_registry.register("ema", description="Exponential Moving Average")
class EMA(Indicator):
    """Exponential Moving Average indicator."""

    name = "ema"

    def __init__(self, period: int = 20, column: str = "close"):
        self.period = period
        self.column = column

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df[self.column].ewm(span=self.period, adjust=False).mean()


@indicator_registry.register("rsi", description="Relative Strength Index")
class RSI(Indicator):
    """Relative Strength Index indicator."""

    name = "rsi"

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        delta = df["close"].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


@indicator_registry.register("macd", description="MACD Indicator")
class MACD(Indicator):
    """Moving Average Convergence Divergence indicator."""

    name = "macd"

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        fast_ema = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df["close"].ewm(span=self.slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        })


@indicator_registry.register("bollinger", description="Bollinger Bands")
class BollingerBands(Indicator):
    """Bollinger Bands indicator."""

    name = "bollinger"

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        middle = df["close"].rolling(window=self.period).mean()
        std = df["close"].rolling(window=self.period).std()
        
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        return pd.DataFrame({
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "bandwidth": (upper - lower) / middle,
        })


@indicator_registry.register("atr", description="Average True Range")
class ATR(Indicator):
    """Average True Range indicator."""

    name = "atr"

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.period).mean()
        
        return atr


@indicator_registry.register("volume_sma", description="Volume SMA")
class VolumeSMA(Indicator):
    """Volume Simple Moving Average."""

    name = "volume_sma"

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["volume"].rolling(window=self.period).mean()


@indicator_registry.register("momentum", description="Price Momentum")
class Momentum(Indicator):
    """Price momentum indicator."""

    name = "momentum"

    def __init__(self, period: int = 10):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].diff(self.period)


@indicator_registry.register("roc", description="Rate of Change")
class ROC(Indicator):
    """Rate of Change indicator."""

    name = "roc"

    def __init__(self, period: int = 10):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(self.period) * 100


@indicator_registry.register("adx", description="Average Directional Index")
class ADX(Indicator):
    """
    Average Directional Index indicator.
    
    Measures trend strength regardless of direction.
    ADX > 25 typically indicates a strong trend.
    """

    name = "adx"

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smooth with Wilder's smoothing (EMA)
        atr = tr.ewm(span=self.period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.period, adjust=False).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=self.period, adjust=False).mean()

        return adx


@indicator_registry.register("obv", description="On Balance Volume")
class OBV(Indicator):
    """
    On Balance Volume indicator.
    
    Cumulative volume that adds volume on up days and subtracts on down days.
    Used to confirm price trends with volume.
    """

    name = "obv"

    def __init__(self):
        pass

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]
        
        # Calculate direction: 1 for up, -1 for down, 0 for unchanged
        direction = np.sign(close.diff())
        
        # OBV is cumulative sum of signed volume
        obv = (direction * volume).cumsum()
        
        return obv


@indicator_registry.register("keltner", description="Keltner Channels")
class KeltnerChannels(Indicator):
    """
    Keltner Channels indicator.
    
    Uses EMA and ATR to create bands. Often used with Bollinger Bands
    to detect "squeeze" setups (BB inside KC = low volatility squeeze).
    """

    name = "keltner"

    def __init__(self, ema_period: int = 20, atr_period: int = 14, atr_multiplier: float = 2.0):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate EMA (middle line)
        middle = df["close"].ewm(span=self.ema_period, adjust=False).mean()
        
        # Calculate ATR
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        # Calculate bands
        upper = middle + (atr * self.atr_multiplier)
        lower = middle - (atr * self.atr_multiplier)
        
        return pd.DataFrame({
            "upper": upper,
            "middle": middle,
            "lower": lower,
        })


@indicator_registry.register("vwap", description="Volume Weighted Average Price")
class VWAP(Indicator):
    """
    Volume Weighted Average Price indicator.
    
    Calculates the average price weighted by volume over a period.
    Commonly used for intraday trading as support/resistance.
    """

    name = "vwap"

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Typical price
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        
        # Volume-weighted typical price
        vwtp = typical_price * df["volume"]
        
        # Rolling VWAP
        vwap = vwtp.rolling(window=self.period).sum() / df["volume"].rolling(window=self.period).sum()
        
        return vwap


@indicator_registry.register("bb_width", description="Bollinger Band Width")
class BollingerBandWidth(Indicator):
    """Bollinger Band Width indicator - measures volatility."""

    name = "bb_width"

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def compute(self, df: pd.DataFrame) -> pd.Series:
        middle = df["close"].rolling(window=self.period).mean()
        std = df["close"].rolling(window=self.period).std()
        
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        # Width as percentage of middle band
        return (upper - lower) / middle * 100


@indicator_registry.register("volume_momentum", description="Volume Momentum")
class VolumeMomentum(Indicator):
    """Volume momentum - rate of change in volume."""

    name = "volume_momentum"

    def __init__(self, period: int = 10):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df["volume"].pct_change(self.period) * 100


@indicator_registry.register("atr_ratio", description="ATR Ratio")
class ATRRatio(Indicator):
    """ATR as a ratio of close price - normalized volatility."""

    name = "atr_ratio"

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        
        return atr / close * 100


# =============================================================================
# Enhanced Volume Indicators (Based on Feature Importance Analysis)
# =============================================================================


@indicator_registry.register("volume_zscore", description="Volume Z-Score")
class VolumeZScore(Indicator):
    """
    Volume Z-Score indicator.
    
    Measures how many standard deviations the current volume is from the mean.
    Useful for detecting unusual volume spikes that may precede moves.
    
    Values:
    - > 2: Very high volume (unusual)
    - 1-2: Above average volume
    - -1 to 1: Normal volume
    - < -1: Below average volume
    """

    name = "volume_zscore"

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        volume = df["volume"]
        
        rolling_mean = volume.rolling(window=self.period).mean()
        rolling_std = volume.rolling(window=self.period).std()
        
        # Z-score: (value - mean) / std
        zscore = (volume - rolling_mean) / rolling_std
        
        return zscore


@indicator_registry.register("volume_breakout", description="Volume Breakout Flag")
class VolumeBreakout(Indicator):
    """
    Volume Breakout indicator.
    
    Binary flag indicating whether volume exceeds a multiple of the average.
    Useful for confirming breakouts with high volume.
    
    Values:
    - 1: Volume > multiplier * average (breakout)
    - 0: Normal volume
    """

    name = "volume_breakout"

    def __init__(self, period: int = 20, multiplier: float = 2.0):
        self.period = period
        self.multiplier = multiplier

    def compute(self, df: pd.DataFrame) -> pd.Series:
        volume = df["volume"]
        
        rolling_mean = volume.rolling(window=self.period).mean()
        threshold = rolling_mean * self.multiplier
        
        # Binary flag: 1 if volume > threshold, 0 otherwise
        breakout = (volume > threshold).astype(int)
        
        return breakout


@indicator_registry.register("obv_trend", description="OBV Trend")
class OBVTrend(Indicator):
    """
    OBV Trend indicator.
    
    Measures the rate of change in On-Balance Volume over multiple periods.
    Captures volume-confirmed price momentum.
    
    Positive values indicate increasing buying pressure.
    Negative values indicate increasing selling pressure.
    """

    name = "obv_trend"

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]
        
        # Calculate OBV
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        
        # Calculate OBV EMAs
        obv_fast = obv.ewm(span=self.fast_period, adjust=False).mean()
        obv_slow = obv.ewm(span=self.slow_period, adjust=False).mean()
        
        # Trend: normalized difference between fast and slow OBV
        # Normalize by slow to make it comparable across assets
        trend = (obv_fast - obv_slow) / obv_slow.abs().replace(0, 1) * 100
        
        return trend


@indicator_registry.register("obv_divergence", description="OBV Divergence")
class OBVDivergence(Indicator):
    """
    OBV Divergence indicator.
    
    Detects divergence between price and OBV trends.
    Divergence often precedes price reversals.
    
    Values:
    - Positive: Bullish divergence (price down, OBV up)
    - Negative: Bearish divergence (price up, OBV down)
    - Near zero: No divergence (price and OBV aligned)
    """

    name = "obv_divergence"

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        volume = df["volume"]
        
        # Calculate OBV
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        
        # Calculate ROC for price and OBV
        price_roc = close.pct_change(self.period) * 100
        obv_roc = obv.pct_change(self.period) * 100
        
        # Divergence: difference in normalized trends
        # Clip to avoid extreme values
        divergence = (obv_roc - price_roc).clip(-100, 100)
        
        return divergence


# =============================================================================
# Regime Detection Indicators
# =============================================================================


@indicator_registry.register("volatility_regime", description="Volatility Regime")
class VolatilityRegime(Indicator):
    """
    Volatility Regime indicator.
    
    Classifies the current volatility regime based on ATR percentile.
    
    Values:
    - 0: Low volatility (bottom 33%)
    - 1: Medium volatility (middle 33%)
    - 2: High volatility (top 33%)
    """

    name = "volatility_regime"

    def __init__(self, period: int = 14, lookback: int = 100):
        self.period = period
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        
        # Normalize ATR by price
        atr_pct = atr / close * 100
        
        # Calculate rolling percentile rank
        def percentile_rank(x):
            if len(x) < 2:
                return 50
            return (x.values < x.values[-1]).sum() / (len(x) - 1) * 100
        
        percentile = atr_pct.rolling(window=self.lookback).apply(percentile_rank, raw=False)
        
        # Classify into regimes
        regime = pd.Series(index=df.index, dtype=int)
        regime[percentile <= 33] = 0  # Low
        regime[(percentile > 33) & (percentile <= 66)] = 1  # Medium
        regime[percentile > 66] = 2  # High
        
        return regime


@indicator_registry.register("trend_regime", description="Trend Regime")
class TrendRegime(Indicator):
    """
    Trend Regime indicator.
    
    Classifies the market as trending or ranging based on ADX.
    
    Values:
    - 0: Ranging (ADX < threshold)
    - 1: Weak trend (threshold <= ADX < strong_threshold)
    - 2: Strong trend (ADX >= strong_threshold)
    """

    name = "trend_regime"

    def __init__(self, period: int = 14, threshold: int = 25, strong_threshold: int = 40):
        self.period = period
        self.threshold = threshold
        self.strong_threshold = strong_threshold

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate ADX (same as ADX indicator)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = tr.ewm(span=self.period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.period, adjust=False).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=self.period, adjust=False).mean()

        # Classify into regimes
        regime = pd.Series(index=df.index, dtype=int)
        regime[adx < self.threshold] = 0  # Ranging
        regime[(adx >= self.threshold) & (adx < self.strong_threshold)] = 1  # Weak trend
        regime[adx >= self.strong_threshold] = 2  # Strong trend
        
        return regime


@indicator_registry.register("trend_direction", description="Trend Direction")
class TrendDirection(Indicator):
    """
    Trend Direction indicator.
    
    Indicates the direction of the trend when in trending regime.
    
    Values:
    - 1: Bullish trend (+DI > -DI)
    - 0: No clear trend
    - -1: Bearish trend (-DI > +DI)
    """

    name = "trend_direction"

    def __init__(self, period: int = 14, adx_threshold: int = 25):
        self.period = period
        self.adx_threshold = adx_threshold

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate +DI and -DI
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = tr.ewm(span=self.period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.period, adjust=False).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=self.period, adjust=False).mean()

        # Determine direction only when trending
        direction = pd.Series(0, index=df.index)
        trending = adx >= self.adx_threshold
        direction[trending & (plus_di > minus_di)] = 1   # Bullish
        direction[trending & (minus_di > plus_di)] = -1  # Bearish
        
        return direction


def add_indicators(
    df: pd.DataFrame,
    indicators: list[tuple[str, dict[str, Any]]],
) -> pd.DataFrame:
    """
    Add multiple indicators to a DataFrame.
    
    Args:
        df: OHLCV DataFrame
        indicators: List of (name, params) tuples
        
    Returns:
        DataFrame with indicator columns added
    """
    result = df.copy()
    
    for name, params in indicators:
        indicator = indicator_registry.create(name, **params)
        values = indicator.compute(result)
        
        if isinstance(values, pd.Series):
            result[f"{name}_{params.get('period', '')}".rstrip("_")] = values
        else:
            for col in values.columns:
                result[f"{name}_{col}"] = values[col]
    
    return result
