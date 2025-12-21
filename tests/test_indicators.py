"""Tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from crypto.indicators.base import (
    SMA,
    EMA,
    RSI,
    MACD,
    BollingerBands,
    ATR,
    indicator_registry,
)


@pytest.fixture
def sample_candles():
    """Create sample candle data for testing."""
    np.random.seed(42)
    n = 100
    
    # Generate random walk price data
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })
    
    return df


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_basic(self, sample_candles):
        """Test basic SMA calculation."""
        sma = SMA(period=20)
        result = sma.compute(sample_candles)

        assert len(result) == len(sample_candles)
        assert result.isna().sum() == 19  # First 19 should be NaN

    def test_sma_values(self):
        """Test SMA values are correct."""
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
        sma = SMA(period=3)
        result = sma.compute(df)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 2.0  # (1+2+3)/3
        assert result.iloc[3] == 3.0  # (2+3+4)/3
        assert result.iloc[4] == 4.0  # (3+4+5)/3


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_basic(self, sample_candles):
        """Test basic EMA calculation."""
        ema = EMA(period=20)
        result = ema.compute(sample_candles)

        assert len(result) == len(sample_candles)
        # EMA has values from the start (ewm doesn't produce NaN)

    def test_ema_faster_than_sma(self, sample_candles):
        """Test that EMA responds faster to price changes."""
        sma = SMA(period=20)
        ema = EMA(period=20)

        sma_result = sma.compute(sample_candles)
        ema_result = ema.compute(sample_candles)

        # EMA should be different from SMA
        # (they would be equal only if prices are constant)
        assert not np.allclose(
            sma_result.dropna().values,
            ema_result.iloc[19:].values,
            rtol=0.001,
        )


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_range(self, sample_candles):
        """Test RSI is between 0 and 100."""
        rsi = RSI(period=14)
        result = rsi.compute(sample_candles)

        valid = result.dropna()
        assert all(valid >= 0)
        assert all(valid <= 100)

    def test_rsi_extreme_up(self):
        """Test RSI approaches 100 in strong uptrend."""
        df = pd.DataFrame({"close": list(range(1, 102))})  # 101 values going up
        rsi = RSI(period=14)
        result = rsi.compute(df)

        # Last RSI should be close to 100
        assert result.iloc[-1] > 90


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_output_shape(self, sample_candles):
        """Test MACD outputs correct DataFrame."""
        macd = MACD()
        result = macd.compute(sample_candles)

        assert isinstance(result, pd.DataFrame)
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns

    def test_macd_histogram(self, sample_candles):
        """Test MACD histogram = macd - signal."""
        macd = MACD()
        result = macd.compute(sample_candles)

        expected_hist = result["macd"] - result["signal"]
        assert np.allclose(result["histogram"].dropna(), expected_hist.dropna())


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bollinger_output_shape(self, sample_candles):
        """Test Bollinger Bands outputs correct DataFrame."""
        bb = BollingerBands()
        result = bb.compute(sample_candles)

        assert "upper" in result.columns
        assert "middle" in result.columns
        assert "lower" in result.columns

    def test_bollinger_band_relationship(self, sample_candles):
        """Test upper > middle > lower."""
        bb = BollingerBands()
        result = bb.compute(sample_candles)

        valid = result.dropna()
        assert all(valid["upper"] >= valid["middle"])
        assert all(valid["middle"] >= valid["lower"])


class TestATR:
    """Tests for Average True Range."""

    def test_atr_positive(self, sample_candles):
        """Test ATR is always positive."""
        atr = ATR(period=14)
        result = atr.compute(sample_candles)

        valid = result.dropna()
        assert all(valid > 0)


class TestIndicatorRegistry:
    """Tests for indicator registry."""

    def test_registry_has_indicators(self):
        """Test registry has built-in indicators."""
        assert "sma" in indicator_registry
        assert "ema" in indicator_registry
        assert "rsi" in indicator_registry
        assert "macd" in indicator_registry
        assert "bollinger" in indicator_registry

    def test_registry_compute(self, sample_candles):
        """Test computing indicator through registry."""
        result = indicator_registry.compute("sma", sample_candles, period=10)
        assert len(result) == len(sample_candles)
