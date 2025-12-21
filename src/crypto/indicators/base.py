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
