#!/usr/bin/env python3
"""
Out-of-sample testing for ML strategies.

Tests strategies on holdout data (symbols and time periods) not used during development.
Compares performance on development vs holdout data to detect overfitting.

Configuration: config/optimization.yaml
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.backtesting.engine import BacktestEngine, BacktestResult
from crypto.config.settings import get_settings
from crypto.data.repository import CandleRepository
from crypto.strategies.registry import strategy_registry

# Import strategies to trigger registration
from crypto.strategies import ml
from crypto.strategies import ml_siblings

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


async def run_out_of_sample_test() -> dict[str, Any]:
    """Run out-of-sample testing from config."""
    settings = get_settings()
    config = settings.optimization.optimization.out_of_sample

    if config is None:
        console.print("[yellow]Out-of-sample testing not configured[/yellow]")
        return {}

    console.print(Panel(
        "[bold blue]Out-of-Sample Testing[/bold blue]\n"
        f"Holdout symbols: {', '.join(config.holdout_symbols)}\n"
        f"Test period: {config.test_period.start} to {config.test_period.end}",
        expand=False,
    ))

    repository = CandleRepository()

    # Convert dates to datetime
    test_start = datetime.combine(config.test_period.start, datetime.min.time())
    test_end = datetime.combine(config.test_period.end, datetime.max.time())

    # Also get earlier data for comparison (development period)
    dev_end = test_start - timedelta(days=1)
    dev_start = dev_end - timedelta(days=90)

    console.print(f"\n[bold]Development period: {dev_start.date()} to {dev_end.date()}[/bold]")
    console.print(f"[bold]Test period: {test_start.date()} to {test_end.date()}[/bold]")

    # Load data for holdout symbols
    console.print(f"\n[bold]Loading holdout symbol data...[/bold]")
    holdout_data: dict[str, tuple] = {}  # symbol -> (dev_candles, test_candles)

    for symbol in config.holdout_symbols:
        dev_candles = await repository.get_candles_df(
            symbol, config.interval, dev_start, dev_end
        )
        test_candles = await repository.get_candles_df(
            symbol, config.interval, test_start, test_end
        )

        if not dev_candles.empty and not test_candles.empty:
            holdout_data[symbol] = (dev_candles, test_candles)
            console.print(
                f"  {symbol}: dev={len(dev_candles)}, test={len(test_candles)} candles"
            )
        else:
            console.print(f"  {symbol}: [red]Insufficient data[/red]")

    # Load data for development symbols (for comparison)
    console.print(f"\n[bold]Loading development symbol data...[/bold]")
    dev_symbol_data: dict[str, tuple] = {}

    for symbol in config.dev_symbols:
        dev_candles = await repository.get_candles_df(
            symbol, config.interval, dev_start, dev_end
        )
        test_candles = await repository.get_candles_df(
            symbol, config.interval, test_start, test_end
        )

        if not dev_candles.empty and not test_candles.empty:
            dev_symbol_data[symbol] = (dev_candles, test_candles)
            console.print(
                f"  {symbol}: dev={len(dev_candles)}, test={len(test_candles)} candles"
            )

    # Verify strategies
    console.print(f"\n[bold]Strategies to test:[/bold]")
    available_strategies = []
    for strategy_name in config.strategies:
        try:
            strategy_registry.create(strategy_name)
            console.print(f"  ✓ {strategy_name}")
            available_strategies.append(strategy_name)
        except Exception as e:
            console.print(f"  ✗ {strategy_name}: [red]{e}[/red]")

    if not available_strategies:
        console.print("[red]No strategies available![/red]")
        return {}

    results = {
        "holdout_results": [],
        "dev_results": [],
        "comparison": {},
    }

    total_tests = len(available_strategies) * (len(holdout_data) + len(dev_symbol_data)) * 2
    engine = BacktestEngine(initial_capital=Decimal("10000"))

    console.print(f"\n[bold]Running {total_tests} backtests...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Out-of-sample testing...", total=total_tests)

        # Test on holdout symbols
        for strategy_name in available_strategies:
            for symbol, (dev_candles, test_candles) in holdout_data.items():
                # Development period backtest
                progress.update(
                    task,
                    description=f"[cyan]{symbol}[/cyan] (holdout) / [yellow]{strategy_name}[/yellow] / dev",
                )
                try:
                    strategy = strategy_registry.create(strategy_name)
                    dev_result = engine.run(
                        strategy, dev_candles, symbol, config.interval
                    )
                    results["holdout_results"].append({
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "type": "holdout",
                        "period": "development",
                        "return_pct": dev_result.metrics.total_return_pct,
                        "sharpe": dev_result.metrics.sharpe_ratio,
                        "win_rate": dev_result.metrics.win_rate,
                        "trades": dev_result.metrics.total_trades,
                    })
                except Exception as e:
                    logger.error(f"Failed: {strategy_name} on {symbol} (dev): {e}")
                progress.advance(task)

                # Test period backtest
                progress.update(
                    task,
                    description=f"[cyan]{symbol}[/cyan] (holdout) / [yellow]{strategy_name}[/yellow] / test",
                )
                try:
                    strategy = strategy_registry.create(strategy_name)
                    test_result = engine.run(
                        strategy, test_candles, symbol, config.interval
                    )
                    results["holdout_results"].append({
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "type": "holdout",
                        "period": "test",
                        "return_pct": test_result.metrics.total_return_pct,
                        "sharpe": test_result.metrics.sharpe_ratio,
                        "win_rate": test_result.metrics.win_rate,
                        "trades": test_result.metrics.total_trades,
                    })
                except Exception as e:
                    logger.error(f"Failed: {strategy_name} on {symbol} (test): {e}")
                progress.advance(task)

        # Test on development symbols (for comparison)
        for strategy_name in available_strategies:
            for symbol, (dev_candles, test_candles) in dev_symbol_data.items():
                # Development period
                progress.update(
                    task,
                    description=f"[cyan]{symbol}[/cyan] (dev) / [yellow]{strategy_name}[/yellow] / dev",
                )
                try:
                    strategy = strategy_registry.create(strategy_name)
                    dev_result = engine.run(
                        strategy, dev_candles, symbol, config.interval
                    )
                    results["dev_results"].append({
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "type": "development",
                        "period": "development",
                        "return_pct": dev_result.metrics.total_return_pct,
                        "sharpe": dev_result.metrics.sharpe_ratio,
                        "win_rate": dev_result.metrics.win_rate,
                        "trades": dev_result.metrics.total_trades,
                    })
                except Exception as e:
                    logger.error(f"Failed: {strategy_name} on {symbol}: {e}")
                progress.advance(task)

                # Test period
                progress.update(
                    task,
                    description=f"[cyan]{symbol}[/cyan] (dev) / [yellow]{strategy_name}[/yellow] / test",
                )
                try:
                    strategy = strategy_registry.create(strategy_name)
                    test_result = engine.run(
                        strategy, test_candles, symbol, config.interval
                    )
                    results["dev_results"].append({
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "type": "development",
                        "period": "test",
                        "return_pct": test_result.metrics.total_return_pct,
                        "sharpe": test_result.metrics.sharpe_ratio,
                        "win_rate": test_result.metrics.win_rate,
                        "trades": test_result.metrics.total_trades,
                    })
                except Exception as e:
                    logger.error(f"Failed: {strategy_name} on {symbol}: {e}")
                progress.advance(task)

    # Compute comparison stats
    for strategy_name in available_strategies:
        holdout_test = [
            r for r in results["holdout_results"]
            if r["strategy"] == strategy_name and r["period"] == "test"
        ]
        dev_test = [
            r for r in results["dev_results"]
            if r["strategy"] == strategy_name and r["period"] == "test"
        ]

        if holdout_test and dev_test:
            results["comparison"][strategy_name] = {
                "holdout_avg_sharpe": sum(r["sharpe"] for r in holdout_test) / len(holdout_test),
                "holdout_avg_return": sum(r["return_pct"] for r in holdout_test) / len(holdout_test),
                "dev_avg_sharpe": sum(r["sharpe"] for r in dev_test) / len(dev_test),
                "dev_avg_return": sum(r["return_pct"] for r in dev_test) / len(dev_test),
            }

    return results


def display_results(results: dict) -> None:
    """Display out-of-sample test results."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    console.print("\n")
    console.print(Panel("[bold blue]Out-of-Sample Test Results[/bold blue]", expand=False))

    # Holdout symbols performance
    console.print("\n[bold]Holdout Symbols (Test Period)[/bold]")

    holdout_test = [r for r in results.get("holdout_results", []) if r["period"] == "test"]

    if holdout_test:
        table = Table()
        table.add_column("Strategy", style="yellow")
        table.add_column("Symbol", style="cyan")
        table.add_column("Return %", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Win Rate %", justify="right")
        table.add_column("Trades", justify="right")

        for r in sorted(holdout_test, key=lambda x: x["sharpe"], reverse=True):
            sharpe_style = "green" if r["sharpe"] > 1 else ("yellow" if r["sharpe"] > 0 else "red")
            return_style = "green" if r["return_pct"] > 0 else "red"

            table.add_row(
                r["strategy"],
                r["symbol"],
                f"[{return_style}]{r['return_pct']:+.2f}[/{return_style}]",
                f"[{sharpe_style}]{r['sharpe']:.2f}[/{sharpe_style}]",
                f"{r['win_rate']:.1f}",
                str(r["trades"]),
            )

        console.print(table)

    # Comparison: Holdout vs Development
    comparison = results.get("comparison", {})
    if comparison:
        console.print("\n[bold]Comparison: Holdout vs Development Symbols[/bold]")

        cmp_table = Table()
        cmp_table.add_column("Strategy", style="yellow")
        cmp_table.add_column("Holdout Sharpe", justify="right")
        cmp_table.add_column("Dev Sharpe", justify="right")
        cmp_table.add_column("Difference", justify="right")
        cmp_table.add_column("Generalization", justify="center")

        for strategy_name, stats in comparison.items():
            holdout_sharpe = stats["holdout_avg_sharpe"]
            dev_sharpe = stats["dev_avg_sharpe"]
            diff = holdout_sharpe - dev_sharpe

            # Generalization assessment
            if holdout_sharpe >= dev_sharpe * 0.8:
                gen = "[green]GOOD[/green]"
            elif holdout_sharpe >= dev_sharpe * 0.5:
                gen = "[yellow]FAIR[/yellow]"
            else:
                gen = "[red]POOR[/red]"

            diff_style = "green" if diff >= 0 else "red"

            cmp_table.add_row(
                strategy_name,
                f"{holdout_sharpe:.2f}",
                f"{dev_sharpe:.2f}",
                f"[{diff_style}]{diff:+.2f}[/{diff_style}]",
                gen,
            )

        console.print(cmp_table)

    # Recommendations
    if comparison:
        best_generalizer = max(
            comparison.items(),
            key=lambda x: x[1]["holdout_avg_sharpe"] / max(x[1]["dev_avg_sharpe"], 0.01),
        )

        console.print("\n")
        if best_generalizer[1]["holdout_avg_sharpe"] > 0:
            console.print(Panel(
                f"✅ [bold green]{best_generalizer[0]}[/bold green] shows best generalization "
                f"to holdout symbols (Sharpe {best_generalizer[1]['holdout_avg_sharpe']:.2f})",
                title="[bold]Recommendation[/bold]",
                border_style="green",
            ))
        else:
            console.print(Panel(
                "⚠️ All strategies show poor performance on holdout symbols. "
                "Consider reducing model complexity or adding more diverse training data.",
                title="[bold]Warning[/bold]",
                border_style="yellow",
            ))


def save_results(results: dict, output_dir: Path) -> None:
    """Save out-of-sample test results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "out_of_sample.json", "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")


async def main():
    """Main entry point."""
    console.print("[bold blue]═══ Out-of-Sample Testing ═══[/bold blue]\n")

    results = await run_out_of_sample_test()

    if results:
        display_results(results)

        settings = get_settings()
        output_dir = Path(settings.optimization.optimization.output.results_dir)
        save_results(results, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
