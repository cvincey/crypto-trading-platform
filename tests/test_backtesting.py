"""Tests for backtesting engine."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from crypto.backtesting.engine import BacktestEngine
from crypto.backtesting.metrics import calculate_metrics, PerformanceMetrics
from crypto.backtesting.portfolio import Portfolio
from crypto.core.types import OrderSide, Signal
from crypto.strategies.base import BaseStrategy


@pytest.fixture
def sample_candles():
    """Create sample candle data for backtesting."""
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


class AlwaysBuyStrategy(BaseStrategy):
    """Strategy that always buys (for testing)."""
    
    name = "always_buy"
    
    def generate_signals(self, candles):
        signals = pd.Series(Signal.HOLD, index=candles.index)
        signals.iloc[0] = Signal.BUY
        return signals


class BuySellStrategy(BaseStrategy):
    """Strategy that alternates buy/sell (for testing)."""
    
    name = "buy_sell"
    
    def generate_signals(self, candles):
        signals = pd.Series(Signal.HOLD, index=candles.index)
        signals.iloc[10] = Signal.BUY
        signals.iloc[50] = Signal.SELL
        signals.iloc[60] = Signal.BUY
        signals.iloc[90] = Signal.SELL
        return signals


class TestPortfolio:
    """Tests for Portfolio class."""

    def test_initial_state(self):
        """Test portfolio initial state."""
        portfolio = Portfolio(initial_capital=Decimal("10000"))
        
        assert portfolio.cash == Decimal("10000")
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0

    def test_execute_buy_signal(self):
        """Test executing a buy signal."""
        portfolio = Portfolio(
            initial_capital=Decimal("10000"),
            commission_rate=Decimal("0.001"),
        )
        
        from datetime import datetime
        
        trade = portfolio.execute_signal(
            signal=Signal.BUY,
            symbol="BTCUSDT",
            price=Decimal("100"),
            timestamp=datetime.now(),
        )
        
        assert trade is not None
        assert trade.side == OrderSide.BUY
        assert portfolio.cash < Decimal("10000")
        assert "BTCUSDT" in portfolio.positions

    def test_execute_sell_signal(self):
        """Test executing a sell signal."""
        portfolio = Portfolio(initial_capital=Decimal("10000"))
        
        from datetime import datetime
        now = datetime.now()
        
        # First buy
        portfolio.execute_signal(Signal.BUY, "BTCUSDT", Decimal("100"), now)
        
        # Then sell
        trade = portfolio.execute_signal(Signal.SELL, "BTCUSDT", Decimal("110"), now)
        
        assert trade is not None
        assert trade.side == OrderSide.SELL
        # Should have more cash than initial due to profit
        assert portfolio.cash > Decimal("10000") * Decimal("0.99")  # Account for commission

    def test_hold_signal_no_trade(self):
        """Test that HOLD signal produces no trade."""
        portfolio = Portfolio(initial_capital=Decimal("10000"))
        
        from datetime import datetime
        
        trade = portfolio.execute_signal(
            signal=Signal.HOLD,
            symbol="BTCUSDT",
            price=Decimal("100"),
            timestamp=datetime.now(),
        )
        
        assert trade is None

    def test_commission_applied(self):
        """Test that commission is applied to trades."""
        portfolio = Portfolio(
            initial_capital=Decimal("10000"),
            commission_rate=Decimal("0.01"),  # 1%
        )
        
        from datetime import datetime
        
        trade = portfolio.execute_signal(
            signal=Signal.BUY,
            symbol="BTCUSDT",
            price=Decimal("100"),
            timestamp=datetime.now(),
        )
        
        assert trade.commission > 0
        # Commission should be about 1% of trade value
        assert trade.commission > trade.value * Decimal("0.009")


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    def test_run_backtest(self, sample_candles):
        """Test running a basic backtest."""
        engine = BacktestEngine(
            initial_capital=Decimal("10000"),
            commission=Decimal("0.001"),
        )
        
        strategy = BuySellStrategy()
        result = engine.run(strategy, sample_candles, symbol="BTCUSDT")
        
        assert result is not None
        assert result.strategy_name == "buy_sell"
        assert result.metrics is not None

    def test_backtest_metrics(self, sample_candles):
        """Test that backtest produces metrics."""
        engine = BacktestEngine()
        strategy = BuySellStrategy()
        result = engine.run(strategy, sample_candles, symbol="BTCUSDT")
        
        metrics = result.metrics
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades >= 0

    def test_run_multiple_strategies(self, sample_candles):
        """Test running multiple strategies."""
        engine = BacktestEngine()
        strategies = [AlwaysBuyStrategy(), BuySellStrategy()]
        
        results = engine.run_multiple(strategies, sample_candles, symbol="BTCUSDT")
        
        assert len(results) == 2

    def test_compare_results(self, sample_candles):
        """Test comparing backtest results."""
        engine = BacktestEngine()
        strategies = [AlwaysBuyStrategy(), BuySellStrategy()]
        results = engine.run_multiple(strategies, sample_candles, symbol="BTCUSDT")
        
        comparison = engine.compare_results(results)
        
        assert len(comparison) == 2
        assert "total_return_pct" in comparison.columns


class TestPerformanceMetrics:
    """Tests for performance metrics calculation."""

    def test_calculate_metrics(self, sample_candles):
        """Test calculating metrics from portfolio."""
        portfolio = Portfolio(initial_capital=Decimal("10000"))
        
        from datetime import datetime
        
        # Simulate some trades
        for i, (timestamp, row) in enumerate(sample_candles.iterrows()):
            price = Decimal(str(row["close"]))
            
            if i == 10:
                portfolio.execute_signal(Signal.BUY, "BTCUSDT", price, timestamp)
            elif i == 50:
                portfolio.execute_signal(Signal.SELL, "BTCUSDT", price, timestamp)
            
            portfolio.update_equity(timestamp, {"BTCUSDT": price})
        
        metrics = calculate_metrics(portfolio)
        
        assert metrics.total_trades >= 0
        assert metrics.equity_curve is not None

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(
            total_return=1000.0,
            total_return_pct=10.0,
            sharpe_ratio=1.5,
        )
        
        data = metrics.to_dict()
        
        assert data["total_return"] == 1000.0
        assert data["sharpe_ratio"] == 1.5
