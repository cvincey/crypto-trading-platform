#!/usr/bin/env python3
"""
Backtest all strategies on the top 20 tickers in the database.

Finds tickers by volume/data availability and runs all registered strategies.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(__file__).replace("/scripts/backtest_top20.py", "/src"))

from crypto.backtesting.engine import BacktestEngine, BacktestResult
from crypto.config.settings import get_settings
from crypto.data.database import get_async_session
from crypto.data.models import CandleModel
from crypto.data.repository import CandleRepository
from crypto.strategies.registry import strategy_registry

# Import strategies to trigger registration
from crypto.strategies import technical, statistical, momentum, ml
from crypto.strategies import ensemble, regime, multi_timeframe, rotation
# RL strategies are optional (require extra dependencies)
try:
    from crypto.strategies import rl
except ImportError:
    pass

from sqlalchemy import select, func, distinct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

console = Console()


async def get_top_tickers(n: int = 20, interval: str = "1h") -> list[dict[str, Any]]:
    """
    Get the top N tickers from the database by data availability and volume.
    
    Args:
        n: Number of top tickers to return
        interval: Candle interval to consider
        
    Returns:
        List of dicts with symbol info
    """
    async with get_async_session() as session:
        # Get symbols with most data points and highest total volume
        stmt = (
            select(
                CandleModel.symbol,
                func.count(CandleModel.id).label("candle_count"),
                func.sum(CandleModel.volume).label("total_volume"),
                func.min(CandleModel.open_time).label("first_candle"),
                func.max(CandleModel.open_time).label("last_candle"),
            )
            .where(CandleModel.interval == interval)
            .group_by(CandleModel.symbol)
            .order_by(func.sum(CandleModel.volume).desc())
            .limit(n)
        )
        
        result = await session.execute(stmt)
        rows = result.all()
        
        return [
            {
                "symbol": row.symbol,
                "candle_count": row.candle_count,
                "total_volume": float(row.total_volume) if row.total_volume else 0,
                "first_candle": row.first_candle,
                "last_candle": row.last_candle,
            }
            for row in rows
        ]


def get_all_strategy_names() -> list[str]:
    """Get all enabled strategy names from config."""
    settings = get_settings()
    return settings.strategies.list_enabled()


async def run_backtest_for_ticker(
    symbol: str,
    strategy_name: str,
    interval: str,
    start: datetime,
    end: datetime,
    initial_capital: Decimal = Decimal("10000"),
) -> BacktestResult | None:
    """Run a single backtest for a ticker/strategy combination."""
    try:
        repository = CandleRepository()
        candles = await repository.get_candles_df(symbol, interval, start, end)
        
        if candles.empty or len(candles) < 50:
            logger.warning(f"Insufficient data for {symbol} ({len(candles)} candles)")
            return None
        
        strategy = strategy_registry.create_from_config(strategy_name)
        engine = BacktestEngine(initial_capital=initial_capital)
        
        return engine.run(strategy, candles, symbol, interval)
        
    except Exception as e:
        logger.error(f"Failed backtest {strategy_name} on {symbol}: {e}")
        return None


async def run_all_backtests(
    tickers: list[dict],
    strategies: list[str],
    interval: str = "1h",
    days: int = 90,
) -> list[BacktestResult]:
    """Run all backtests for all ticker/strategy combinations."""
    
    all_results: list[BacktestResult] = []
    total_combinations = len(tickers) * len(strategies)
    
    console.print(f"\n[bold]Running {total_combinations} backtests[/bold]")
    console.print(f"  Tickers:    {len(tickers)}")
    console.print(f"  Strategies: {len(strategies)}")
    console.print(f"  Interval:   {interval}")
    console.print(f"  Period:     {days} days\n")
    
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Backtesting...", total=total_combinations)
        
        for ticker in tickers:
            symbol = ticker["symbol"]
            
            # Use ticker's actual data range if available
            ticker_start = ticker.get("first_candle", start)
            ticker_end = ticker.get("last_candle", end)
            
            # Ensure we have enough history
            actual_start = max(start, ticker_start) if ticker_start else start
            actual_end = min(end, ticker_end) if ticker_end else end
            
            for strategy_name in strategies:
                progress.update(
                    task,
                    description=f"[cyan]{symbol}[/cyan] / [yellow]{strategy_name}[/yellow]"
                )
                
                result = await run_backtest_for_ticker(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    interval=interval,
                    start=actual_start,
                    end=actual_end,
                )
                
                if result:
                    all_results.append(result)
                
                progress.advance(task)
    
    return all_results


def display_results(results: list[BacktestResult]) -> None:
    """Display backtest results in a nice table."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return
    
    # Sort by Sharpe ratio
    results_sorted = sorted(results, key=lambda r: r.metrics.sharpe_ratio, reverse=True)
    
    # Summary table
    table = Table(title="Backtest Results - All Strategies on Top 20 Tickers")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Symbol", style="cyan")
    table.add_column("Strategy", style="yellow")
    table.add_column("Return %", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD %", justify="right")
    table.add_column("Win Rate %", justify="right")
    table.add_column("Trades", justify="right")
    
    for i, result in enumerate(results_sorted[:50], 1):  # Top 50 results
        m = result.metrics
        
        # Color return based on positive/negative
        return_style = "green" if m.total_return_pct > 0 else "red"
        sharpe_style = "green" if m.sharpe_ratio > 1 else ("yellow" if m.sharpe_ratio > 0 else "red")
        
        table.add_row(
            str(i),
            result.symbol,
            result.strategy_name,
            f"[{return_style}]{m.total_return_pct:+.2f}[/{return_style}]",
            f"[{sharpe_style}]{m.sharpe_ratio:.2f}[/{sharpe_style}]",
            f"{m.max_drawdown:.2f}",
            f"{m.win_rate:.1f}",
            str(m.total_trades),
        )
    
    console.print(table)
    
    # Strategy summary
    console.print("\n[bold]Strategy Summary (Average Across All Tickers)[/bold]")
    
    strategy_table = Table()
    strategy_table.add_column("Strategy", style="yellow")
    strategy_table.add_column("Avg Return %", justify="right")
    strategy_table.add_column("Avg Sharpe", justify="right")
    strategy_table.add_column("Best Ticker", style="cyan")
    strategy_table.add_column("Worst Ticker", style="red")
    
    # Group by strategy
    from collections import defaultdict
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r.strategy_name].append(r)
    
    for strategy_name, strategy_results in sorted(by_strategy.items()):
        avg_return = sum(r.metrics.total_return_pct for r in strategy_results) / len(strategy_results)
        avg_sharpe = sum(r.metrics.sharpe_ratio for r in strategy_results) / len(strategy_results)
        
        best = max(strategy_results, key=lambda r: r.metrics.total_return_pct)
        worst = min(strategy_results, key=lambda r: r.metrics.total_return_pct)
        
        return_style = "green" if avg_return > 0 else "red"
        sharpe_style = "green" if avg_sharpe > 1 else ("yellow" if avg_sharpe > 0 else "red")
        
        strategy_table.add_row(
            strategy_name,
            f"[{return_style}]{avg_return:+.2f}[/{return_style}]",
            f"[{sharpe_style}]{avg_sharpe:.2f}[/{sharpe_style}]",
            f"{best.symbol} ({best.metrics.total_return_pct:+.2f}%)",
            f"{worst.symbol} ({worst.metrics.total_return_pct:+.2f}%)",
        )
    
    console.print(strategy_table)
    
    # Ticker summary
    console.print("\n[bold]Ticker Summary (Average Across All Strategies)[/bold]")
    
    ticker_table = Table()
    ticker_table.add_column("Symbol", style="cyan")
    ticker_table.add_column("Avg Return %", justify="right")
    ticker_table.add_column("Avg Sharpe", justify="right")
    ticker_table.add_column("Best Strategy", style="yellow")
    
    # Group by ticker
    by_ticker = defaultdict(list)
    for r in results:
        by_ticker[r.symbol].append(r)
    
    ticker_summaries = []
    for symbol, ticker_results in by_ticker.items():
        avg_return = sum(r.metrics.total_return_pct for r in ticker_results) / len(ticker_results)
        avg_sharpe = sum(r.metrics.sharpe_ratio for r in ticker_results) / len(ticker_results)
        best = max(ticker_results, key=lambda r: r.metrics.sharpe_ratio)
        ticker_summaries.append((symbol, avg_return, avg_sharpe, best))
    
    # Sort by avg sharpe
    for symbol, avg_return, avg_sharpe, best in sorted(ticker_summaries, key=lambda x: x[2], reverse=True):
        return_style = "green" if avg_return > 0 else "red"
        sharpe_style = "green" if avg_sharpe > 1 else ("yellow" if avg_sharpe > 0 else "red")
        
        ticker_table.add_row(
            symbol,
            f"[{return_style}]{avg_return:+.2f}[/{return_style}]",
            f"[{sharpe_style}]{avg_sharpe:.2f}[/{sharpe_style}]",
            f"{best.strategy_name} (Sharpe: {best.metrics.sharpe_ratio:.2f})",
        )
    
    console.print(ticker_table)


async def main():
    """Main entry point."""
    console.print("[bold blue]═══ Crypto Backtest: All Strategies on Top 20 Tickers ═══[/bold blue]\n")
    
    # Configuration
    interval = "1h"
    days = 90
    top_n = 20
    
    # Get top tickers from database
    console.print(f"[bold]Fetching top {top_n} tickers from database...[/bold]")
    tickers = await get_top_tickers(n=top_n, interval=interval)
    
    if not tickers:
        console.print("[red]No tickers found in database![/red]")
        console.print("Run data ingestion first: crypto ingest fetch --symbol BTCUSDT --days 90")
        return
    
    # Display found tickers
    console.print(f"\n[green]Found {len(tickers)} tickers:[/green]")
    ticker_list = Table(show_header=True)
    ticker_list.add_column("Symbol")
    ticker_list.add_column("Candles", justify="right")
    ticker_list.add_column("First Date")
    ticker_list.add_column("Last Date")
    
    for t in tickers:
        ticker_list.add_row(
            t["symbol"],
            str(t["candle_count"]),
            t["first_candle"].strftime("%Y-%m-%d") if t["first_candle"] else "N/A",
            t["last_candle"].strftime("%Y-%m-%d") if t["last_candle"] else "N/A",
        )
    console.print(ticker_list)
    
    # Get all strategies
    strategies = get_all_strategy_names()
    console.print(f"\n[green]Found {len(strategies)} strategies:[/green] {', '.join(strategies)}")
    
    if not strategies:
        console.print("[red]No strategies found in config![/red]")
        return
    
    # Run all backtests
    results = await run_all_backtests(
        tickers=tickers,
        strategies=strategies,
        interval=interval,
        days=days,
    )
    
    # Display results
    console.print(f"\n[bold green]Completed {len(results)} backtests[/bold green]\n")
    display_results(results)


if __name__ == "__main__":
    asyncio.run(main())
