"""Batch backtest runner from configuration."""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from crypto.backtesting.engine import BacktestEngine, BacktestResult
from crypto.backtesting.metrics import compare_strategies, generate_report
from crypto.config.schemas import BacktestConfig
from crypto.config.settings import get_settings
from crypto.data.repository import CandleRepository
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


# Default reference symbols for cross-symbol strategies
DEFAULT_REFERENCE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


class BacktestRunner:
    """
    Run backtests from YAML configuration.
    
    Loads backtest configurations from config/backtests.yaml
    and runs them with the specified strategies.
    """

    def __init__(
        self,
        repository: CandleRepository | None = None,
        reference_symbols: list[str] | None = None,
    ):
        """
        Initialize the backtest runner.
        
        Args:
            repository: Optional candle repository
            reference_symbols: Symbols to load for cross-symbol strategies
        """
        self._repository = repository or CandleRepository()
        self._settings = get_settings()
        self._reference_symbols = reference_symbols or DEFAULT_REFERENCE_SYMBOLS
        self._reference_data_cache: dict[str, pd.DataFrame] = {}

    async def run(self, backtest_name: str) -> list[BacktestResult]:
        """
        Run a backtest from configuration.
        
        Args:
            backtest_name: Name of the backtest in config/backtests.yaml
            
        Returns:
            List of BacktestResult for each strategy
        """
        config = self._settings.get_backtest(backtest_name)
        return await self._run_backtest(config)

    async def run_all(self) -> dict[str, list[BacktestResult]]:
        """
        Run all configured backtests.
        
        Returns:
            Dict mapping backtest name to list of results
        """
        all_results = {}
        backtest_names = self._settings.backtests.list_all()

        for name in backtest_names:
            try:
                results = await self.run(name)
                all_results[name] = results
                logger.info(f"Completed backtest: {name}")
            except Exception as e:
                logger.error(f"Failed to run backtest {name}: {e}")

        return all_results

    async def _run_backtest(
        self,
        config: BacktestConfig,
    ) -> list[BacktestResult]:
        """Run a backtest from config."""
        results = []
        symbols = config.get_symbols()

        # Create engine with config parameters
        engine = BacktestEngine(
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage,
        )

        # Preload reference data for cross-symbol strategies
        start_dt = datetime.combine(config.start, datetime.min.time())
        end_dt = datetime.combine(config.end, datetime.max.time())
        await self._load_reference_data(config.interval, start_dt, end_dt)

        for symbol in symbols:
            # Fetch data
            candles = await self._repository.get_candles_df(
                symbol=symbol,
                interval=config.interval,
                start=start_dt,
                end=end_dt,
            )

            if candles.empty:
                logger.warning(
                    f"No data for {symbol} {config.interval} "
                    f"from {config.start} to {config.end}"
                )
                continue

            logger.info(
                f"Running {config.name} on {symbol} with {len(candles)} candles"
            )

            # Run each strategy
            for strategy_name in config.strategies:
                try:
                    # Create strategy from config
                    strategy = strategy_registry.create_from_config(strategy_name)

                    # Set reference data for cross-symbol strategies
                    self._setup_cross_symbol_strategy(strategy)

                    # Run backtest
                    result = engine.run(
                        strategy=strategy,
                        candles=candles,
                        symbol=symbol,
                        interval=config.interval,
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(
                        f"Failed to run strategy {strategy_name} "
                        f"on {symbol}: {e}"
                    )

        return results

    async def _load_reference_data(
        self,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> None:
        """
        Load reference data for cross-symbol strategies.
        
        Args:
            interval: Candle interval
            start: Start datetime
            end: End datetime
        """
        for symbol in self._reference_symbols:
            if symbol in self._reference_data_cache:
                # Check if cached data covers the requested range
                cached = self._reference_data_cache[symbol]
                if not cached.empty:
                    cached_start = cached.index.min()
                    cached_end = cached.index.max()
                    if cached_start <= start and cached_end >= end:
                        continue  # Cache is valid

            # Load data
            candles = await self._repository.get_candles_df(
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
            )
            
            if not candles.empty:
                self._reference_data_cache[symbol] = candles
                logger.debug(f"Loaded {len(candles)} reference candles for {symbol}")
            else:
                logger.warning(f"No reference data available for {symbol}")

    def _setup_cross_symbol_strategy(self, strategy: Any) -> None:
        """
        Set up reference data for cross-symbol strategies.
        
        Args:
            strategy: Strategy instance to set up
        """
        # Check if strategy is a cross-symbol strategy
        from crypto.strategies.cross_symbol_base import CrossSymbolBaseStrategy
        
        if isinstance(strategy, CrossSymbolBaseStrategy):
            # Get required reference symbols
            ref_symbols = strategy.get_reference_symbols()
            
            # If no specific symbols requested, use all available
            if not ref_symbols:
                ref_symbols = list(self._reference_data_cache.keys())
            
            # Set reference data
            for symbol in ref_symbols:
                if symbol in self._reference_data_cache:
                    strategy.set_reference_data(symbol, self._reference_data_cache[symbol])
                else:
                    logger.warning(
                        f"Reference symbol {symbol} requested by {strategy.name} "
                        f"but not available in cache"
                    )

    def clear_reference_cache(self) -> None:
        """Clear the reference data cache."""
        self._reference_data_cache.clear()

    def compare(
        self,
        results: list[BacktestResult],
    ) -> pd.DataFrame:
        """
        Compare backtest results.
        
        Args:
            results: List of backtest results
            
        Returns:
            DataFrame with comparison
        """
        engine = BacktestEngine()
        return engine.compare_results(results)

    def generate_report(
        self,
        results: list[BacktestResult],
        output_format: str = "text",
    ) -> str:
        """
        Generate a report from backtest results.
        
        Args:
            results: List of backtest results
            output_format: "text" or "html"
            
        Returns:
            Report string
        """
        if output_format == "text":
            lines = []
            for result in results:
                lines.append(generate_report(result.metrics, result.strategy_name))
                lines.append("")
            return "\n".join(lines)
        else:
            # HTML report
            return self._generate_html_report(results)

    def _generate_html_report(
        self,
        results: list[BacktestResult],
    ) -> str:
        """Generate an HTML report."""
        comparison_df = self.compare(results)

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Backtest Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            ".positive { color: green; }",
            ".negative { color: red; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Backtest Report</h1>",
            f"<p>Generated: {datetime.now().isoformat()}</p>",
            "<h2>Strategy Comparison</h2>",
            comparison_df.to_html(classes="comparison"),
            "<h2>Individual Results</h2>",
        ]

        for result in results:
            html_parts.append(f"<h3>{result.strategy_name} - {result.symbol}</h3>")
            html_parts.append("<table>")
            metrics = result.metrics.to_dict()
            for key, value in metrics.items():
                if isinstance(value, float):
                    formatted = f"{value:.2f}"
                else:
                    formatted = str(value)
                html_parts.append(f"<tr><td>{key}</td><td>{formatted}</td></tr>")
            html_parts.append("</table>")

        html_parts.extend([
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)


async def run_quick_backtest(
    strategy_name: str,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 30,
    initial_capital: Decimal = Decimal("10000"),
) -> BacktestResult:
    """
    Quick backtest for a single strategy.
    
    Args:
        strategy_name: Strategy name in config
        symbol: Trading pair
        interval: Candle interval
        days: Number of days to backtest
        initial_capital: Starting capital
        
    Returns:
        BacktestResult
    """
    from datetime import timedelta

    repository = CandleRepository()
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    candles = await repository.get_candles_df(symbol, interval, start, end)

    if candles.empty:
        raise ValueError(f"No data for {symbol} {interval}")

    strategy = strategy_registry.create_from_config(strategy_name)
    engine = BacktestEngine(initial_capital=initial_capital)

    return engine.run(strategy, candles, symbol, interval)
