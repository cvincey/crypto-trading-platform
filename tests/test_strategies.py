"""Tests for trading strategies."""

import numpy as np
import pandas as pd
import pytest

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry


@pytest.fixture
def sample_candles():
    """Create sample candle data for testing."""
    np.random.seed(42)
    n = 100
    
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    
    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    }, index=dates)
    
    return df


@pytest.fixture
def trending_up_candles():
    """Create candle data with a trend change (flat then up)."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    
    # Start flat, then trend up - this creates a crossover
    close = np.concatenate([
        np.ones(30) * 100 + np.random.randn(30) * 0.5,  # Flat period
        np.linspace(100, 150, 70) + np.random.randn(70) * 0.5,  # Uptrend
    ])
    
    df = pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    }, index=dates)
    
    return df


class TestStrategyRegistry:
    """Tests for strategy registry."""

    def test_registry_has_strategies(self):
        """Test registry has built-in strategies."""
        # Import strategies to trigger registration
        from crypto.strategies import technical, statistical, momentum

        assert "sma_crossover" in strategy_registry
        assert "rsi_mean_reversion" in strategy_registry
        assert "macd_crossover" in strategy_registry

    def test_create_strategy(self):
        """Test creating strategy from registry."""
        from crypto.strategies import technical

        strategy = strategy_registry.create("sma_crossover", fast_period=5, slow_period=20)
        assert strategy.fast_period == 5
        assert strategy.slow_period == 20

    def test_list_strategies(self):
        """Test listing available strategies."""
        from crypto.strategies import technical

        strategies = strategy_registry.list()
        assert len(strategies) > 0


class TestSMACrossoverStrategy:
    """Tests for SMA Crossover strategy."""

    def test_generate_signals(self, sample_candles):
        """Test signal generation."""
        from crypto.strategies.technical import SMACrossoverStrategy

        strategy = SMACrossoverStrategy(fast_period=5, slow_period=20)
        signals = strategy.generate_signals(sample_candles)

        assert len(signals) == len(sample_candles)
        assert all(s in Signal for s in signals.unique())

    def test_signals_in_uptrend(self, trending_up_candles):
        """Test that strategy generates BUY signals in uptrend."""
        from crypto.strategies.technical import SMACrossoverStrategy

        strategy = SMACrossoverStrategy(fast_period=5, slow_period=20)
        signals = strategy.generate_signals(trending_up_candles)

        # Should have at least one BUY signal
        buy_count = (signals == Signal.BUY).sum()
        assert buy_count >= 1

    def test_get_parameters(self):
        """Test getting strategy parameters."""
        from crypto.strategies.technical import SMACrossoverStrategy

        strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)
        params = strategy.get_parameters()

        assert params["fast_period"] == 10
        assert params["slow_period"] == 30


class TestRSIMeanReversionStrategy:
    """Tests for RSI Mean Reversion strategy."""

    def test_generate_signals(self, sample_candles):
        """Test signal generation."""
        from crypto.strategies.statistical import RSIMeanReversionStrategy

        strategy = RSIMeanReversionStrategy(period=14, oversold=30, overbought=70)
        signals = strategy.generate_signals(sample_candles)

        assert len(signals) == len(sample_candles)


class TestMomentumBreakoutStrategy:
    """Tests for Momentum Breakout strategy."""

    def test_generate_signals(self, sample_candles):
        """Test signal generation."""
        from crypto.strategies.momentum import MomentumBreakoutStrategy

        strategy = MomentumBreakoutStrategy(lookback=20, threshold=0.02)
        signals = strategy.generate_signals(sample_candles)

        assert len(signals) == len(sample_candles)


class TestBaseStrategy:
    """Tests for BaseStrategy abstract class."""

    def test_validate_candles_missing_columns(self):
        """Test validation catches missing columns."""

        class TestStrategy(BaseStrategy):
            name = "test"

            def generate_signals(self, candles):
                self.validate_candles(candles)
                return pd.Series(Signal.HOLD, index=candles.index)

        strategy = TestStrategy()
        bad_df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.generate_signals(bad_df)

    def test_create_signal_series(self, sample_candles):
        """Test creating signal series."""

        class TestStrategy(BaseStrategy):
            name = "test"

            def generate_signals(self, candles):
                return self.create_signal_series(candles.index, Signal.HOLD)

        strategy = TestStrategy()
        signals = strategy.generate_signals(sample_candles)

        assert len(signals) == len(sample_candles)
        assert all(s == Signal.HOLD for s in signals)
