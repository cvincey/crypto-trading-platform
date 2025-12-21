#!/usr/bin/env python3
"""
Grid search for optimal stop-loss and take-profit values.

Tests all combinations of SL and TP across multiple strategies and tickers.
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from itertools import product

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(__file__).replace("/scripts/grid_search_risk.py", "/src"))

from crypto.backtesting.engine import BacktestEngine
from crypto.data.database import get_async_session
from crypto.data.models import CandleModel
from crypto.data.repository import CandleRepository
from crypto.strategies.registry import strategy_registry

# Import strategies to trigger registration
from crypto.strategies import technical, statistical, momentum, ml
from crypto.strategies import ml_siblings  # Import ML sibling strategies

from sqlalchemy import select, func

console = Console()


def load_config_from_yaml() -> dict:
    """Load grid search config from optimization.yaml if available."""
    try:
        from crypto.config.settings import get_settings
        settings = get_settings()
        risk_config = settings.optimization.optimization.risk_params
        return {
            "stop_loss_values": risk_config.stop_loss_pct,
            "take_profit_values": risk_config.take_profit_pct,
            "strategies": risk_config.strategies,
            "symbols": risk_config.symbols,
            "interval": risk_config.interval,
            "days": risk_config.days,
        }
    except Exception:
        return {}


# Try to load from config, fall back to defaults
_config = load_config_from_yaml()

# Grid search parameters (from config or defaults)
STOP_LOSS_VALUES = _config.get("stop_loss_values", [0.02, 0.03, 0.04, 0.05, 0.06])
TAKE_PROFIT_VALUES = _config.get("take_profit_values", [0.06, 0.08, 0.10, 0.12, 0.15])

# Strategies to test (focus on best ML performers)
STRATEGIES = _config.get("strategies", [
    "ml_classifier_xgb",
    "ml_classifier_hybrid",
    "ml_ensemble_voting",
])

# Test on top tickers
TOP_N_TICKERS = 10
INTERVAL = _config.get("interval", "1h")
DAYS = _config.get("days", 90)


async def get_top_tickers(n: int = 10, interval: str = "1h") -> list[str]:
    """Get top N tickers by volume."""
    async with get_async_session() as session:
        stmt = (
            select(CandleModel.symbol)
            .where(CandleModel.interval == interval)
            .group_by(CandleModel.symbol)
            .order_by(func.sum(CandleModel.volume).desc())
            .limit(n)
        )
        result = await session.execute(stmt)
        return [row[0] for row in result.all()]


async def run_single_backtest(
    symbol: str,
    strategy_name: str,
    stop_loss: float,
    take_profit: float,
    candles_cache: dict,
) -> dict:
    """Run a single backtest with given SL/TP values."""
    try:
        candles = candles_cache.get(symbol)
        if candles is None or candles.empty:
            return None

        strategy = strategy_registry.create(strategy_name)
        engine = BacktestEngine(
            stop_loss_pct=Decimal(str(stop_loss)),
            take_profit_pct=Decimal(str(take_profit)),
        )

        result = engine.run(strategy, candles, symbol, INTERVAL)

        return {
            "symbol": symbol,
            "strategy": strategy_name,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "return_pct": result.metrics.total_return_pct,
            "sharpe": result.metrics.sharpe_ratio,
            "max_dd": result.metrics.max_drawdown,
            "win_rate": result.metrics.win_rate,
            "trades": result.metrics.total_trades,
        }
    except Exception as e:
        return None


async def main():
    console.print("[bold blue]═══ Grid Search: Stop-Loss / Take-Profit Optimization ═══[/bold blue]\n")

    # Get tickers
    console.print(f"Fetching top {TOP_N_TICKERS} tickers...")
    tickers = await get_top_tickers(TOP_N_TICKERS, INTERVAL)
    
    if not tickers:
        console.print("[red]No tickers found![/red]")
        return

    console.print(f"[green]Tickers: {', '.join(tickers)}[/green]\n")

    # Pre-fetch all candle data
    console.print("Loading candle data...")
    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)
    
    candles_cache = {}
    for symbol in tickers:
        candles = await repository.get_candles_df(symbol, INTERVAL, start, end)
        if not candles.empty:
            candles_cache[symbol] = candles
            console.print(f"  {symbol}: {len(candles)} candles")

    # Generate all combinations
    combinations = list(product(STOP_LOSS_VALUES, TAKE_PROFIT_VALUES))
    total_tests = len(combinations) * len(tickers) * len(STRATEGIES)
    
    console.print(f"\n[bold]Running {total_tests} backtests ({len(combinations)} SL/TP combinations × {len(tickers)} tickers × {len(STRATEGIES)} strategies)[/bold]\n")

    # Run grid search
    all_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Grid search...", total=total_tests)

        for sl, tp in combinations:
            for symbol in tickers:
                for strategy_name in STRATEGIES:
                    progress.update(task, description=f"SL={sl:.0%} TP={tp:.0%} | {symbol}")
                    
                    result = await run_single_backtest(
                        symbol, strategy_name, sl, tp, candles_cache
                    )
                    
                    if result:
                        all_results.append(result)
                    
                    progress.advance(task)

    # Aggregate results by SL/TP combination
    console.print(f"\n[bold green]Completed {len(all_results)} backtests[/bold green]\n")

    # Group by (SL, TP) and calculate averages
    from collections import defaultdict
    
    combo_stats = defaultdict(lambda: {
        "returns": [], "sharpes": [], "max_dds": [], "win_rates": [], "trades": []
    })
    
    for r in all_results:
        key = (r["stop_loss"], r["take_profit"])
        combo_stats[key]["returns"].append(r["return_pct"])
        combo_stats[key]["sharpes"].append(r["sharpe"])
        combo_stats[key]["max_dds"].append(r["max_dd"])
        combo_stats[key]["win_rates"].append(r["win_rate"])
        combo_stats[key]["trades"].append(r["trades"])

    # Calculate averages
    combo_averages = []
    for (sl, tp), stats in combo_stats.items():
        n = len(stats["returns"])
        avg_return = sum(stats["returns"]) / n
        avg_sharpe = sum(stats["sharpes"]) / n
        avg_max_dd = sum(stats["max_dds"]) / n
        avg_win_rate = sum(stats["win_rates"]) / n
        avg_trades = sum(stats["trades"]) / n
        
        combo_averages.append({
            "sl": sl,
            "tp": tp,
            "avg_return": avg_return,
            "avg_sharpe": avg_sharpe,
            "avg_max_dd": avg_max_dd,
            "avg_win_rate": avg_win_rate,
            "avg_trades": avg_trades,
            "n_tests": n,
        })

    # Sort by Sharpe ratio
    combo_averages.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    # Display results table
    table = Table(title="Grid Search Results - Sorted by Avg Sharpe Ratio")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Stop Loss", justify="center")
    table.add_column("Take Profit", justify="center")
    table.add_column("Avg Return %", justify="right")
    table.add_column("Avg Sharpe", justify="right")
    table.add_column("Avg Max DD %", justify="right")
    table.add_column("Avg Win Rate %", justify="right")
    table.add_column("Avg Trades", justify="right")

    for i, combo in enumerate(combo_averages, 1):
        return_style = "green" if combo["avg_return"] > 0 else "red"
        sharpe_style = "green" if combo["avg_sharpe"] > 1 else ("yellow" if combo["avg_sharpe"] > 0 else "red")
        
        # Highlight best row
        row_style = "bold" if i == 1 else ""
        
        table.add_row(
            str(i),
            f"{combo['sl']:.0%}",
            f"{combo['tp']:.0%}",
            f"[{return_style}]{combo['avg_return']:+.2f}[/{return_style}]",
            f"[{sharpe_style}]{combo['avg_sharpe']:.2f}[/{sharpe_style}]",
            f"{combo['avg_max_dd']:.2f}",
            f"{combo['avg_win_rate']:.1f}",
            f"{combo['avg_trades']:.0f}",
            style=row_style,
        )

    console.print(table)

    # Display heatmap-style matrix
    console.print("\n[bold]Sharpe Ratio Matrix (SL × TP)[/bold]")
    
    matrix_table = Table(show_header=True, header_style="bold")
    matrix_table.add_column("SL \\ TP", style="bold")
    for tp in TAKE_PROFIT_VALUES:
        matrix_table.add_column(f"{tp:.0%}", justify="center")

    # Build matrix
    sharpe_matrix = {}
    for combo in combo_averages:
        sharpe_matrix[(combo["sl"], combo["tp"])] = combo["avg_sharpe"]

    best_sharpe = max(c["avg_sharpe"] for c in combo_averages)
    
    for sl in STOP_LOSS_VALUES:
        row = [f"{sl:.0%}"]
        for tp in TAKE_PROFIT_VALUES:
            sharpe = sharpe_matrix.get((sl, tp), 0)
            if sharpe == best_sharpe:
                row.append(f"[bold green]{sharpe:.2f}[/bold green]")
            elif sharpe > best_sharpe * 0.9:
                row.append(f"[green]{sharpe:.2f}[/green]")
            elif sharpe > 0:
                row.append(f"[yellow]{sharpe:.2f}[/yellow]")
            else:
                row.append(f"[red]{sharpe:.2f}[/red]")
        matrix_table.add_row(*row)

    console.print(matrix_table)

    # Display Return matrix
    console.print("\n[bold]Average Return % Matrix (SL × TP)[/bold]")
    
    return_matrix_table = Table(show_header=True, header_style="bold")
    return_matrix_table.add_column("SL \\ TP", style="bold")
    for tp in TAKE_PROFIT_VALUES:
        return_matrix_table.add_column(f"{tp:.0%}", justify="center")

    return_matrix = {}
    for combo in combo_averages:
        return_matrix[(combo["sl"], combo["tp"])] = combo["avg_return"]

    best_return = max(c["avg_return"] for c in combo_averages)
    
    for sl in STOP_LOSS_VALUES:
        row = [f"{sl:.0%}"]
        for tp in TAKE_PROFIT_VALUES:
            ret = return_matrix.get((sl, tp), 0)
            if ret == best_return:
                row.append(f"[bold green]{ret:+.0f}[/bold green]")
            elif ret > best_return * 0.8:
                row.append(f"[green]{ret:+.0f}[/green]")
            elif ret > 0:
                row.append(f"[yellow]{ret:+.0f}[/yellow]")
            else:
                row.append(f"[red]{ret:+.0f}[/red]")
        return_matrix_table.add_row(*row)

    console.print(return_matrix_table)

    # Best combination
    best = combo_averages[0]
    console.print(f"\n[bold green]═══ BEST COMBINATION ═══[/bold green]")
    console.print(f"[bold]Stop Loss:    {best['sl']:.0%}[/bold]")
    console.print(f"[bold]Take Profit:  {best['tp']:.0%}[/bold]")
    console.print(f"[bold]Avg Return:   {best['avg_return']:+.2f}%[/bold]")
    console.print(f"[bold]Avg Sharpe:   {best['avg_sharpe']:.2f}[/bold]")
    console.print(f"[bold]Avg Max DD:   {best['avg_max_dd']:.2f}%[/bold]")
    console.print(f"[bold]Avg Win Rate: {best['avg_win_rate']:.1f}%[/bold]")


if __name__ == "__main__":
    asyncio.run(main())
