#!/usr/bin/env python3
"""
Grid Search for Orthogonal Strategies.

Parallel grid search to optimize:
- funding_term_structure
- btc_dominance_rotation
- gamma_mimic_breakout (fix zero-trade issue)
- liquidity_vacuum_detector (fix zero-trade issue)
- correlation_regime_switch (fix zero-trade issue)

Uses ProcessPoolExecutor for parallel computation.

Usage:
    python scripts/grid_search_orthogonal.py              # All strategies
    python scripts/grid_search_orthogonal.py --strategy funding_term_structure
    python scripts/grid_search_orthogonal.py --workers 8  # Use 8 parallel workers
"""

import argparse
import asyncio
import itertools
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

OUTPUT_DIR = Path("notes/orthogonal_results/grid_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PARAMETER GRIDS
# =============================================================================

PARAM_GRIDS = {
    "funding_term_structure": {
        "lookback_periods": [2, 3, 4, 5],
        "extreme_funding": [0.0003, 0.0004, 0.0005, 0.0006],
        "hold_period": [12, 24, 48],
    },
    "btc_dominance_rotation": {
        "risk_on_threshold": [-0.02, -0.015, -0.01, -0.005],
        "risk_off_threshold": [0.005, 0.01, 0.015, 0.02],
        "hold_period": [48, 72, 96],
    },
    "gamma_mimic_breakout": {
        "min_compression_bars": [48, 72, 96, 120],
        "compression_percentile": [15, 20, 25, 30],
        "breakout_atr_mult": [2.0, 2.5, 3.0],
        "hold_period": [48, 72, 96],
    },
    "liquidity_vacuum_detector": {
        "range_percentile": [80, 85, 90],
        "volume_percentile": [30, 40, 50],
        "stall_threshold": [0.4, 0.5, 0.6],
        "hold_period": [6, 12, 18],
    },
    "correlation_regime_switch": {
        "high_correlation": [0.65, 0.70, 0.75, 0.80],
        "low_correlation": [0.40, 0.50, 0.60],
        "stability_threshold": [0.08, 0.10, 0.12],
        "min_hold_period": [24, 48, 72],
    },
}

# Symbols and config
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DAYS = 180
TRAIN_WINDOW = 1440
TEST_WINDOW = 336
STEP_SIZE = 336


@dataclass
class GridSearchResult:
    """Result from a single grid search evaluation."""
    strategy: str
    params: dict
    avg_oos_sharpe: float
    avg_is_sharpe: float
    total_trades: int
    pass_rate: float
    symbols_tested: int


def generate_param_combinations(strategy: str) -> list[dict]:
    """Generate all parameter combinations for a strategy."""
    grid = PARAM_GRIDS.get(strategy, {})
    if not grid:
        return []
    
    keys = list(grid.keys())
    values = list(grid.values())
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def run_single_backtest(args: tuple) -> dict | None:
    """
    Run backtest for a single parameter combination.
    
    This function runs in a separate process.
    """
    strategy_name, params, symbol, candles_dict, reference_dict = args
    
    try:
        # Import inside function for multiprocessing
        import pandas as pd
        from crypto.strategies import hyper_creative, hyper_creative_tier2
        from crypto.backtesting.walk_forward import WalkForwardEngine
        
        # Reconstruct DataFrames from dict
        candles = pd.DataFrame(candles_dict)
        candles.index = pd.to_datetime(candles.index)
        
        reference_data = {}
        for sym, ref_dict in reference_dict.items():
            ref_df = pd.DataFrame(ref_dict)
            ref_df.index = pd.to_datetime(ref_df.index)
            reference_data[sym] = ref_df
        
        # Run walk-forward using strategy_name API
        engine = WalkForwardEngine(
            train_window=TRAIN_WINDOW,
            test_window=TEST_WINDOW,
            step_size=STEP_SIZE,
        )
        
        result = engine.run(
            strategy_name=strategy_name,
            candles=candles,
            symbol=symbol,
            interval="1h",
            strategy_params=params,
            reference_data=reference_data,
        )
        
        return {
            "symbol": symbol,
            "oos_sharpe": result.oos_sharpe,
            "is_sharpe": result.is_sharpe,
            "total_trades": result.oos_total_trades or 0,
            "passed": result.oos_sharpe is not None and result.oos_sharpe > 0,
        }
        
    except Exception as e:
        logger.error(f"Error in backtest {strategy_name} with {params}: {e}")
        return None


async def load_data() -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Load candle data and prepare for multiprocessing."""
    from crypto.data.repository import CandleRepository
    
    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS + 30)
    
    console.print("\n[bold]Loading data...[/bold]")
    data = {}
    
    for symbol in SYMBOLS:
        try:
            candles = await repository.get_candles_df(symbol, "1h", start, end)
            if not candles.empty:
                data[symbol] = candles
                console.print(f"  ✓ {symbol}: {len(candles):,} candles")
        except Exception as e:
            console.print(f"  [red]✗ {symbol}: {e}[/red]")
    
    # Convert to dict format for multiprocessing
    data_dicts = {}
    for symbol, df in data.items():
        df_copy = df.copy()
        df_copy.index = df_copy.index.astype(str)
        data_dicts[symbol] = df_copy.to_dict()
    
    return data, data_dicts


def run_grid_search_parallel(
    strategy: str,
    data: dict[str, pd.DataFrame],
    data_dicts: dict[str, dict],
    max_workers: int = 4,
) -> list[GridSearchResult]:
    """Run grid search with parallel processing."""
    param_combos = generate_param_combinations(strategy)
    
    if not param_combos:
        console.print(f"[yellow]No parameter grid defined for {strategy}[/yellow]")
        return []
    
    console.print(f"\n[bold]Grid search: {strategy}[/bold]")
    console.print(f"  Parameter combinations: {len(param_combos)}")
    console.print(f"  Symbols: {len(data)}")
    console.print(f"  Total tests: {len(param_combos) * len(data)}")
    console.print(f"  Workers: {max_workers}")
    
    # Prepare reference data dict
    reference_dicts = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        if sym in data_dicts:
            reference_dicts[sym] = data_dicts[sym]
    
    # Build task list
    tasks = []
    for params in param_combos:
        for symbol in data.keys():
            tasks.append((
                strategy,
                params,
                symbol,
                data_dicts[symbol],
                reference_dicts,
            ))
    
    # Run parallel
    results_by_params = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Testing {strategy}...", total=len(tasks))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_backtest, t): t for t in tasks}
            
            for future in as_completed(futures):
                task_args = futures[future]
                params = task_args[1]
                params_key = json.dumps(params, sort_keys=True)
                
                try:
                    result = future.result()
                    
                    if params_key not in results_by_params:
                        results_by_params[params_key] = {
                            "params": params,
                            "results": [],
                        }
                    
                    if result:
                        results_by_params[params_key]["results"].append(result)
                        
                except Exception as e:
                    logger.error(f"Future error: {e}")
                
                progress.advance(task)
    
    # Aggregate results
    grid_results = []
    
    for params_key, data_item in results_by_params.items():
        params = data_item["params"]
        results = data_item["results"]
        
        if not results:
            continue
        
        oos_sharpes = [r["oos_sharpe"] for r in results if r["oos_sharpe"] is not None]
        is_sharpes = [r["is_sharpe"] for r in results if r["is_sharpe"] is not None]
        total_trades = sum(r["total_trades"] for r in results)
        passed = sum(1 for r in results if r["passed"])
        
        grid_results.append(GridSearchResult(
            strategy=strategy,
            params=params,
            avg_oos_sharpe=np.mean(oos_sharpes) if oos_sharpes else 0.0,
            avg_is_sharpe=np.mean(is_sharpes) if is_sharpes else 0.0,
            total_trades=total_trades,
            pass_rate=passed / len(results) if results else 0.0,
            symbols_tested=len(results),
        ))
    
    # Sort by OOS Sharpe
    grid_results.sort(key=lambda x: x.avg_oos_sharpe, reverse=True)
    
    return grid_results


def display_results(results: list[GridSearchResult], strategy: str) -> None:
    """Display top results in a table."""
    if not results:
        console.print(f"[yellow]No results for {strategy}[/yellow]")
        return
    
    table = Table(title=f"Top 10 Results: {strategy}")
    table.add_column("Rank", justify="right", style="cyan", width=4)
    table.add_column("OOS Sharpe", justify="right", style="green")
    table.add_column("Trades", justify="right")
    table.add_column("Pass%", justify="right")
    table.add_column("Parameters", style="dim")
    
    for i, result in enumerate(results[:10], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in result.params.items())
        
        table.add_row(
            str(i),
            f"{result.avg_oos_sharpe:.3f}",
            str(result.total_trades),
            f"{result.pass_rate * 100:.0f}%",
            params_str[:60] + ("..." if len(params_str) > 60 else ""),
        )
    
    console.print(table)


def save_results(all_results: dict[str, list[GridSearchResult]]) -> None:
    """Save all grid search results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {}
    for strategy, results in all_results.items():
        output[strategy] = [
            {
                "params": r.params,
                "avg_oos_sharpe": r.avg_oos_sharpe,
                "avg_is_sharpe": r.avg_is_sharpe,
                "total_trades": r.total_trades,
                "pass_rate": r.pass_rate,
                "symbols_tested": r.symbols_tested,
            }
            for r in results
        ]
    
    output_file = OUTPUT_DIR / f"grid_search_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    console.print(f"\n[bold]Results saved:[/bold] {output_file}")
    
    # Also save best params
    best_params = {}
    for strategy, results in all_results.items():
        if results:
            best = results[0]
            best_params[strategy] = {
                "params": best.params,
                "avg_oos_sharpe": best.avg_oos_sharpe,
                "total_trades": best.total_trades,
            }
    
    best_file = OUTPUT_DIR / f"best_params_{timestamp}.json"
    with open(best_file, "w") as f:
        json.dump(best_params, f, indent=2)
    
    console.print(f"[bold]Best params saved:[/bold] {best_file}")


async def main(args: argparse.Namespace):
    """Main entry point."""
    
    # Determine strategies to optimize
    if args.strategy:
        strategies = [args.strategy]
    else:
        strategies = list(PARAM_GRIDS.keys())
    
    console.print(Panel.fit(
        f"[bold blue]Orthogonal Strategy Grid Search[/bold blue]\n\n"
        f"Strategies: {len(strategies)}\n"
        f"Symbols: {len(SYMBOLS)}\n"
        f"Days: {DAYS}\n"
        f"Workers: {args.workers}\n"
        f"Total param combos: {sum(len(generate_param_combinations(s)) for s in strategies)}"
    ))
    
    # Load data
    data, data_dicts = await load_data()
    
    if not data:
        console.print("[red]No data loaded![/red]")
        return
    
    # Run grid search for each strategy
    all_results = {}
    
    for strategy in strategies:
        results = run_grid_search_parallel(
            strategy=strategy,
            data=data,
            data_dicts=data_dicts,
            max_workers=args.workers,
        )
        
        all_results[strategy] = results
        display_results(results, strategy)
        
        # Show best params
        if results:
            best = results[0]
            console.print(f"\n[bold green]Best for {strategy}:[/bold green]")
            console.print(f"  OOS Sharpe: {best.avg_oos_sharpe:.3f}")
            console.print(f"  Trades: {best.total_trades}")
            console.print(f"  Params: {best.params}")
    
    # Save all results
    save_results(all_results)
    
    # Summary
    console.print("\n" + "=" * 80)
    console.print("[bold]GRID SEARCH SUMMARY[/bold]")
    console.print("=" * 80)
    
    for strategy, results in all_results.items():
        if results:
            best = results[0]
            improvement = "✓" if best.avg_oos_sharpe > 0.3 else "○"
            console.print(f"  {improvement} {strategy}: Sharpe {best.avg_oos_sharpe:.3f}, Trades {best.total_trades}")
        else:
            console.print(f"  ✗ {strategy}: No valid results")
    
    console.print("\n[bold green]Grid search complete![/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search orthogonal strategies")
    parser.add_argument("--strategy", type=str, help="Specific strategy to optimize")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    asyncio.run(main(args))
