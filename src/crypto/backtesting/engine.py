"""Backtesting engine for evaluating trading strategies."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from crypto.backtesting.metrics import PerformanceMetrics, calculate_metrics, generate_report
from crypto.backtesting.portfolio import Portfolio
from crypto.core.types import Signal
from crypto.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    backtest_id: str
    strategy_name: str
    symbol: str
    interval: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    commission: Decimal

    # Results
    metrics: PerformanceMetrics
    portfolio: Portfolio

    # Signals and trades
    signals: pd.Series = field(default=None)
    
    # Metadata
    strategy_params: dict[str, Any] = field(default_factory=dict)
    run_time_seconds: float = 0.0

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the backtest result."""
        return {
            "backtest_id": self.backtest_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "interval": self.interval,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": float(self.initial_capital),
            "total_return_pct": self.metrics.total_return_pct,
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "max_drawdown": self.metrics.max_drawdown,
            "total_trades": self.metrics.total_trades,
            "win_rate": self.metrics.win_rate,
        }

    def print_report(self) -> None:
        """Print a formatted report."""
        report = generate_report(self.metrics, self.strategy_name)
        print(report)


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    
    Supports:
    - Config-driven backtesting
    - Multiple strategies
    - Performance metrics calculation
    - Trade simulation with commission and slippage
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal("10000"),
        commission: Decimal = Decimal("0.001"),
        slippage: Decimal = Decimal("0.0005"),
    ):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        strategy: Strategy,
        candles: pd.DataFrame,
        symbol: str = "UNKNOWN",
        interval: str = "1h",
    ) -> BacktestResult:
        """
        Run a backtest on a strategy with historical data.
        
        Args:
            strategy: Strategy instance to test
            candles: DataFrame with OHLCV data (index should be datetime)
            symbol: Trading pair symbol
            interval: Candle interval
            
        Returns:
            BacktestResult with metrics and trades
        """
        import time
        start_time = time.time()

        logger.info(
            f"Running backtest: {strategy.name} on {symbol} "
            f"({len(candles)} candles)"
        )

        # Validate input
        if candles.empty:
            raise ValueError("Candles DataFrame is empty")

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(candles.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission_rate=self.commission,
            slippage_rate=self.slippage,
        )

        # Generate signals
        signals = strategy.generate_signals(candles)

        # Run simulation
        for i, (timestamp, row) in enumerate(candles.iterrows()):
            price = Decimal(str(row["close"]))
            signal = signals.iloc[i] if i < len(signals) else Signal.HOLD

            # Execute signal
            portfolio.execute_signal(
                signal=signal,
                symbol=symbol,
                price=price,
                timestamp=timestamp,
            )

            # Update equity curve
            portfolio.update_equity(
                timestamp=timestamp,
                prices={symbol: price},
            )

        # Calculate metrics
        # Determine periods per year based on interval
        periods_per_year = self._get_periods_per_year(interval)
        metrics = calculate_metrics(portfolio, periods_per_year=periods_per_year)

        run_time = time.time() - start_time

        result = BacktestResult(
            backtest_id=str(uuid.uuid4())[:8],
            strategy_name=strategy.name,
            symbol=symbol,
            interval=interval,
            start_date=candles.index[0],
            end_date=candles.index[-1],
            initial_capital=self.initial_capital,
            commission=self.commission,
            metrics=metrics,
            portfolio=portfolio,
            signals=signals,
            strategy_params=strategy.get_parameters(),
            run_time_seconds=run_time,
        )

        logger.info(
            f"Backtest complete: {strategy.name} - "
            f"Return: {metrics.total_return_pct:.2f}%, "
            f"Sharpe: {metrics.sharpe_ratio:.2f}, "
            f"Trades: {metrics.total_trades}"
        )

        return result

    def run_multiple(
        self,
        strategies: list[Strategy],
        candles: pd.DataFrame,
        symbol: str = "UNKNOWN",
        interval: str = "1h",
    ) -> list[BacktestResult]:
        """
        Run backtests on multiple strategies.
        
        Args:
            strategies: List of strategies to test
            candles: OHLCV DataFrame
            symbol: Trading pair
            interval: Candle interval
            
        Returns:
            List of BacktestResult objects
        """
        results = []
        for strategy in strategies:
            try:
                result = self.run(strategy, candles, symbol, interval)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run backtest for {strategy.name}: {e}")

        return results

    def compare_results(
        self,
        results: list[BacktestResult],
    ) -> pd.DataFrame:
        """
        Compare multiple backtest results.
        
        Args:
            results: List of backtest results
            
        Returns:
            DataFrame with comparison
        """
        rows = []
        for result in results:
            row = {
                "strategy": result.strategy_name,
                "symbol": result.symbol,
                "total_return_pct": result.metrics.total_return_pct,
                "annualized_return": result.metrics.annualized_return,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "sortino_ratio": result.metrics.sortino_ratio,
                "max_drawdown": result.metrics.max_drawdown,
                "win_rate": result.metrics.win_rate,
                "profit_factor": result.metrics.profit_factor,
                "total_trades": result.metrics.total_trades,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.sort_values("sharpe_ratio", ascending=False, inplace=True)
        return df

    @staticmethod
    def _get_periods_per_year(interval: str) -> int:
        """Get number of periods per year for an interval."""
        intervals = {
            "1m": 525600,    # 365 * 24 * 60
            "5m": 105120,    # 365 * 24 * 12
            "15m": 35040,    # 365 * 24 * 4
            "30m": 17520,    # 365 * 24 * 2
            "1h": 8760,      # 365 * 24
            "2h": 4380,
            "4h": 2190,
            "6h": 1460,
            "8h": 1095,
            "12h": 730,
            "1d": 365,
            "3d": 122,
            "1w": 52,
            "1M": 12,
        }
        return intervals.get(interval, 365)


async def run_backtest_from_db(
    strategy: Strategy,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    initial_capital: Decimal = Decimal("10000"),
    commission: Decimal = Decimal("0.001"),
) -> BacktestResult:
    """
    Run a backtest using data from the database.
    
    Args:
        strategy: Strategy to test
        symbol: Trading pair
        interval: Candle interval
        start: Start date
        end: End date
        initial_capital: Starting capital
        commission: Commission rate
        
    Returns:
        BacktestResult
    """
    from crypto.data.repository import CandleRepository

    repository = CandleRepository()
    candles = await repository.get_candles_df(symbol, interval, start, end)

    if candles.empty:
        raise ValueError(
            f"No data found for {symbol} {interval} from {start} to {end}"
        )

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
    )

    return engine.run(strategy, candles, symbol, interval)
