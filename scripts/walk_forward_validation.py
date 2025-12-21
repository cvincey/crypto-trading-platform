#!/usr/bin/env python3
"""
Walk-forward validation for ML strategies.

Runs proper out-of-sample testing with rolling train/test windows
to detect overfitting and get realistic performance estimates.

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

from crypto.backtesting.walk_forward import (
    WalkForwardEngine,
    WalkForwardResult,
    compare_walk_forward_results,
)
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


async def run_walk_forward_validation() -> list[WalkForwardResult]:
    """Run walk-forward validation from config."""
    settings = get_settings()
    config = settings.optimization.optimization.walk_forward

    if not config.enabled:
        console.print("[yellow]Walk-forward validation is disabled in config[/yellow]")
        return []

    console.print(Panel(
        f"[bold blue]Walk-Forward Validation[/bold blue]\n"
        f"Train: {config.train_window} bars, Test: {config.test_window} bars, "
        f"Step: {config.step_size} bars",
        expand=False,
    ))

    # Load data
    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=config.days)

    console.print(f"\n[bold]Loading data for {len(config.symbols)} symbols...[/bold]")
    
    candles_cache: dict[str, Any] = {}
    for symbol in config.symbols:
        candles = await repository.get_candles_df(
            symbol, config.interval, start, end
        )
        if not candles.empty:
            candles_cache[symbol] = candles
            console.print(f"  {symbol}: {len(candles)} candles")
        else:
            console.print(f"  {symbol}: [red]No data[/red]")

    if not candles_cache:
        console.print("[red]No data available![/red]")
        return []

    # Verify strategies
    console.print(f"\n[bold]Strategies to validate:[/bold]")
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
        return []

    # Create engine
    engine = WalkForwardEngine(
        train_window=config.train_window,
        test_window=config.test_window,
        step_size=config.step_size,
        min_train_samples=config.min_train_samples,
    )

    # Run walk-forward
    total_runs = len(available_strategies) * len(candles_cache)
    all_results: list[WalkForwardResult] = []

    console.print(f"\n[bold]Running {total_runs} walk-forward validations...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Walk-forward...", total=total_runs)

        for strategy_name in available_strategies:
            for symbol, candles in candles_cache.items():
                progress.update(
                    task,
                    description=f"[cyan]{symbol}[/cyan] / [yellow]{strategy_name}[/yellow]",
                )

                try:
                    result = engine.run(
                        strategy_name=strategy_name,
                        candles=candles,
                        symbol=symbol,
                        interval=config.interval,
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Failed: {strategy_name} on {symbol}: {e}")

                progress.advance(task)

    return all_results


def display_results(results: list[WalkForwardResult]) -> None:
    """Display walk-forward results."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    console.print("\n")
    console.print(Panel("[bold blue]Walk-Forward Validation Results[/bold blue]", expand=False))

    # Comparison table
    df = compare_walk_forward_results(results)

    table = Table(title="Strategy Performance (Out-of-Sample)")
    table.add_column("Strategy", style="yellow")
    table.add_column("Symbol", style="cyan")
    table.add_column("OOS Return %", justify="right")
    table.add_column("OOS Sharpe", justify="right")
    table.add_column("IS Sharpe", justify="right")
    table.add_column("Degradation %", justify="right")
    table.add_column("OOS Trades", justify="right")
    table.add_column("Folds", justify="right")

    for _, row in df.iterrows():
        oos_style = "green" if row["oos_sharpe"] > 1 else ("yellow" if row["oos_sharpe"] > 0 else "red")
        deg_style = "green" if row["sharpe_degradation"] < 30 else ("yellow" if row["sharpe_degradation"] < 50 else "red")

        table.add_row(
            row["strategy"],
            row["symbol"],
            f"{row['oos_return_pct']:+.2f}",
            f"[{oos_style}]{row['oos_sharpe']:.2f}[/{oos_style}]",
            f"{row['is_sharpe']:.2f}",
            f"[{deg_style}]{row['sharpe_degradation']:.1f}[/{deg_style}]",
            str(row["oos_trades"]),
            str(row["folds"]),
        )

    console.print(table)

    # Strategy summary
    console.print("\n[bold]Strategy Summary (Average Across Symbols)[/bold]")

    from collections import defaultdict
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r.strategy_name].append(r)

    summary_table = Table()
    summary_table.add_column("Strategy", style="yellow")
    summary_table.add_column("Avg OOS Return %", justify="right")
    summary_table.add_column("Avg OOS Sharpe", justify="right")
    summary_table.add_column("Avg IS Sharpe", justify="right")
    summary_table.add_column("Avg Degradation %", justify="right")
    summary_table.add_column("Overfitting Risk", justify="center")

    strategy_summaries = []
    for strategy_name, strategy_results in by_strategy.items():
        avg_oos_return = sum(r.oos_return_pct for r in strategy_results) / len(strategy_results)
        avg_oos_sharpe = sum(r.oos_sharpe for r in strategy_results) / len(strategy_results)
        avg_is_sharpe = sum(r.is_sharpe for r in strategy_results) / len(strategy_results)
        avg_degradation = sum(r.sharpe_degradation for r in strategy_results) / len(strategy_results)

        strategy_summaries.append({
            "strategy": strategy_name,
            "avg_oos_return": avg_oos_return,
            "avg_oos_sharpe": avg_oos_sharpe,
            "avg_is_sharpe": avg_is_sharpe,
            "avg_degradation": avg_degradation,
        })

    # Sort by OOS Sharpe
    strategy_summaries.sort(key=lambda x: x["avg_oos_sharpe"], reverse=True)

    for s in strategy_summaries:
        oos_style = "green" if s["avg_oos_sharpe"] > 1 else ("yellow" if s["avg_oos_sharpe"] > 0 else "red")

        # Overfitting risk assessment
        if s["avg_degradation"] < 20:
            risk = "[green]LOW[/green]"
        elif s["avg_degradation"] < 40:
            risk = "[yellow]MEDIUM[/yellow]"
        else:
            risk = "[red]HIGH[/red]"

        summary_table.add_row(
            s["strategy"],
            f"{s['avg_oos_return']:+.2f}",
            f"[{oos_style}]{s['avg_oos_sharpe']:.2f}[/{oos_style}]",
            f"{s['avg_is_sharpe']:.2f}",
            f"{s['avg_degradation']:.1f}",
            risk,
        )

    console.print(summary_table)

    # Recommendations
    best = strategy_summaries[0] if strategy_summaries else None
    if best:
        console.print("\n")
        if best["avg_degradation"] < 30 and best["avg_oos_sharpe"] > 1:
            console.print(Panel(
                f"✅ [bold green]{best['strategy']}[/bold green] shows strong OOS performance "
                f"(Sharpe {best['avg_oos_sharpe']:.2f}) with low overfitting risk "
                f"({best['avg_degradation']:.1f}% degradation)",
                title="[bold]Recommendation[/bold]",
                border_style="green",
            ))
        elif best["avg_oos_sharpe"] > 0:
            console.print(Panel(
                f"⚠️ [bold yellow]{best['strategy']}[/bold yellow] is the best performer "
                f"but shows {best['avg_degradation']:.1f}% Sharpe degradation OOS. "
                f"Consider parameter reduction or regularization.",
                title="[bold]Recommendation[/bold]",
                border_style="yellow",
            ))
        else:
            console.print(Panel(
                f"❌ All strategies show negative OOS Sharpe. "
                f"Results may be overfit. Consider simpler features.",
                title="[bold]Warning[/bold]",
                border_style="red",
            ))


def save_results(results: list[WalkForwardResult], output_dir: Path) -> None:
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = [r.to_dict() for r in results]
    
    with open(output_dir / "walk_forward_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Save summary
    from collections import defaultdict
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r.strategy_name].append(r)

    summary = {}
    for strategy_name, strategy_results in by_strategy.items():
        summary[strategy_name] = {
            "avg_oos_return": sum(r.oos_return_pct for r in strategy_results) / len(strategy_results),
            "avg_oos_sharpe": sum(r.oos_sharpe for r in strategy_results) / len(strategy_results),
            "avg_is_sharpe": sum(r.is_sharpe for r in strategy_results) / len(strategy_results),
            "avg_degradation": sum(r.sharpe_degradation for r in strategy_results) / len(strategy_results),
            "symbols_tested": len(strategy_results),
        }

    with open(output_dir / "walk_forward_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")


async def main():
    """Main entry point."""
    console.print("[bold blue]═══ Walk-Forward Validation ═══[/bold blue]\n")

    results = await run_walk_forward_validation()

    if results:
        console.print(f"\n[bold green]Completed {len(results)} validations[/bold green]")
        display_results(results)

        # Save results
        settings = get_settings()
        output_dir = Path(settings.optimization.optimization.output.results_dir)
        save_results(results, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
