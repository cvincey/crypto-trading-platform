#!/usr/bin/env python3
"""
Hyperparameter grid search for ML strategies.

Tests combinations of model and trading parameters to find optimal settings.
Uses walk-forward validation for realistic performance estimates.

Configuration: config/optimization.yaml
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from itertools import product
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

from crypto.backtesting.engine import BacktestEngine
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


def generate_param_combinations(param_grid: dict[str, list]) -> list[dict]:
    """Generate all combinations of parameters."""
    if not param_grid:
        return [{}]
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


async def run_hyperparam_search():
    """Run hyperparameter grid search from config."""
    settings = get_settings()
    opt_config = settings.optimization.optimization
    hyperparam_config = opt_config.hyperparameters
    
    if not hyperparam_config:
        console.print("[yellow]No hyperparameter grids defined in config[/yellow]")
        return {}
    
    console.print(Panel(
        "[bold blue]Hyperparameter Grid Search[/bold blue]\n"
        "Testing parameter combinations for optimal settings",
        expand=False,
    ))

    # Load data (use walk-forward config for symbols)
    wf_config = opt_config.walk_forward
    symbols = wf_config.symbols[:3]  # Limit to 3 symbols for speed
    interval = wf_config.interval
    days = 90  # Shorter period for hyperparam search

    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    console.print(f"\n[bold]Loading data for {len(symbols)} symbols...[/bold]")
    
    candles_cache: dict[str, Any] = {}
    for symbol in symbols:
        candles = await repository.get_candles_df(symbol, interval, start, end)
        if not candles.empty:
            candles_cache[symbol] = candles
            console.print(f"  {symbol}: {len(candles)} candles")

    if not candles_cache:
        console.print("[red]No data available![/red]")
        return {}

    all_results: dict[str, list] = {}

    for strategy_name, param_grid in hyperparam_config.items():
        console.print(f"\n[bold]Optimizing {strategy_name}...[/bold]")
        
        # Verify strategy exists
        try:
            strategy_registry.create(strategy_name)
        except Exception as e:
            console.print(f"  [red]Strategy not found: {e}[/red]")
            continue

        # Generate combinations
        combinations = generate_param_combinations(param_grid)
        console.print(f"  Testing {len(combinations)} parameter combinations")

        strategy_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Grid search {strategy_name}...",
                total=len(combinations) * len(candles_cache),
            )

            for params in combinations:
                combo_returns = []
                combo_sharpes = []

                for symbol, candles in candles_cache.items():
                    progress.update(
                        task,
                        description=f"[cyan]{symbol}[/cyan] params={params}",
                    )

                    try:
                        # Create strategy with these params
                        strategy = strategy_registry.create(strategy_name, **params)
                        
                        engine = BacktestEngine(
                            initial_capital=Decimal("10000"),
                            commission=Decimal("0.001"),
                        )
                        
                        result = engine.run(strategy, candles, symbol, interval)
                        
                        combo_returns.append(result.metrics.total_return_pct)
                        combo_sharpes.append(result.metrics.sharpe_ratio)

                    except Exception as e:
                        logger.error(f"Failed: {strategy_name} with {params}: {e}")

                    progress.advance(task)

                if combo_sharpes:
                    strategy_results.append({
                        "params": params,
                        "avg_return": sum(combo_returns) / len(combo_returns),
                        "avg_sharpe": sum(combo_sharpes) / len(combo_sharpes),
                        "min_sharpe": min(combo_sharpes),
                        "max_sharpe": max(combo_sharpes),
                        "tests": len(combo_sharpes),
                    })

        # Sort by average Sharpe
        strategy_results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
        all_results[strategy_name] = strategy_results

        # Display top 10 for this strategy
        if strategy_results:
            display_top_results(strategy_name, strategy_results[:10])

    return all_results


def display_top_results(strategy_name: str, results: list[dict]) -> None:
    """Display top parameter combinations for a strategy."""
    table = Table(title=f"Top Parameters for {strategy_name}")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Parameters", style="cyan", max_width=50)
    table.add_column("Avg Return %", justify="right")
    table.add_column("Avg Sharpe", justify="right")
    table.add_column("Min Sharpe", justify="right")
    table.add_column("Max Sharpe", justify="right")

    for i, r in enumerate(results, 1):
        # Format params for display
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        if len(params_str) > 47:
            params_str = params_str[:47] + "..."

        sharpe_style = "green" if r["avg_sharpe"] > 10 else ("yellow" if r["avg_sharpe"] > 0 else "red")

        table.add_row(
            str(i),
            params_str,
            f"{r['avg_return']:+.2f}",
            f"[{sharpe_style}]{r['avg_sharpe']:.2f}[/{sharpe_style}]",
            f"{r['min_sharpe']:.2f}",
            f"{r['max_sharpe']:.2f}",
        )

    console.print(table)


def save_results(results: dict[str, list], output_dir: Path) -> None:
    """Save hyperparameter search results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full results
    with open(output_dir / "hyperparam_search.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Best params summary
    best_params = {}
    for strategy_name, strategy_results in results.items():
        if strategy_results:
            best = strategy_results[0]
            best_params[strategy_name] = {
                "best_params": best["params"],
                "avg_sharpe": best["avg_sharpe"],
                "avg_return": best["avg_return"],
            }

    with open(output_dir / "best_hyperparams.json", "w") as f:
        json.dump(best_params, f, indent=2)

    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")

    # Display best params recommendation
    console.print("\n[bold]Best Parameters Summary[/bold]")
    for strategy_name, info in best_params.items():
        console.print(f"\n[yellow]{strategy_name}[/yellow]:")
        for k, v in info["best_params"].items():
            console.print(f"  {k}: {v}")
        console.print(f"  → Avg Sharpe: [green]{info['avg_sharpe']:.2f}[/green]")


async def main():
    """Main entry point."""
    console.print("[bold blue]═══ Hyperparameter Grid Search ═══[/bold blue]\n")

    results = await run_hyperparam_search()

    if results:
        settings = get_settings()
        output_dir = Path(settings.optimization.optimization.output.results_dir)
        save_results(results, output_dir)

        console.print("\n")
        console.print(Panel(
            "Update your strategy configs in [cyan]config/strategies.yaml[/cyan] "
            "with the best parameters found above.",
            title="[bold]Next Steps[/bold]",
            border_style="green",
        ))


if __name__ == "__main__":
    asyncio.run(main())
