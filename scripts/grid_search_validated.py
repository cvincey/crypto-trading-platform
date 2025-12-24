#!/usr/bin/env python3
"""
Grid search optimization for validated strategies.

Performs hyperparameter grid search on the 5 winning strategies from Notes 08-09:
- eth_btc_ratio_reversion
- eth_btc_ratio_confirmed  
- volume_divergence
- signal_confirmation_delay
- basis_proxy

Uses walk-forward validation for realistic out-of-sample scoring.
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

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.backtesting.engine import BacktestEngine
from crypto.backtesting.walk_forward import WalkForwardEngine
from crypto.config.settings import get_settings
from crypto.data.repository import CandleRepository

# Import strategies to trigger registration
from crypto.strategies import (
    cross_symbol,
    frequency,
    hybrid,
    microstructure,
    structural,
)
from crypto.strategies.registry import strategy_registry

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()

# Validated strategies to optimize
VALIDATED_STRATEGIES = [
    "eth_btc_ratio_reversion",
    "eth_btc_ratio_confirmed",
    "volume_divergence",
    "signal_confirmation_delay",
    "basis_proxy",
]

# Test configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "1h"
DAYS = 180

# Walk-forward configuration (faster than full validation)
TRAIN_WINDOW = 720   # 30 days
TEST_WINDOW = 168    # 7 days  
STEP_SIZE = 168      # 7 days


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


async def load_data(
    symbols: list[str],
    interval: str,
    days: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load candle data for all symbols."""
    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    candles_cache: dict[str, pd.DataFrame] = {}
    reference_data: dict[str, pd.DataFrame] = {}
    
    for symbol in symbols:
        candles = await repository.get_candles_df(symbol, interval, start, end)
        if not candles.empty:
            candles_cache[symbol] = candles
            reference_data[symbol] = candles
            console.print(f"  {symbol}: {len(candles)} candles")
    
    return candles_cache, reference_data


def run_quick_backtest(
    strategy_name: str,
    params: dict[str, Any],
    candles: pd.DataFrame,
    symbol: str,
    interval: str,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    reference_data: dict[str, pd.DataFrame] | None = None,
) -> float:
    """
    Run quick backtest and return Sharpe ratio.
    
    For initial grid sweep - faster than walk-forward.
    """
    try:
        # Separate risk params from strategy params
        strategy_params = {k: v for k, v in params.items() 
                         if k not in ('stop_loss_pct', 'take_profit_pct')}
        
        strategy = strategy_registry.create(strategy_name, **strategy_params)
        
        # Set reference data for cross-symbol strategies
        if hasattr(strategy, "set_reference_data") and reference_data:
            for ref_symbol, ref_candles in reference_data.items():
                strategy.set_reference_data(ref_symbol, ref_candles)
        
        engine = BacktestEngine(
            initial_capital=Decimal("10000"),
            commission=Decimal("0.001"),
            slippage=Decimal("0.0005"),
        )
        
        result = engine.run(strategy, candles, symbol, interval)
        return result.metrics.sharpe_ratio
        
    except Exception as e:
        logger.warning(f"Backtest failed for {strategy_name} with {params}: {e}")
        return -999.0


def run_walk_forward(
    strategy_name: str,
    params: dict[str, Any],
    candles: pd.DataFrame,
    symbol: str,
    interval: str,
    reference_data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, float]:
    """
    Run walk-forward validation and return detailed metrics.
    
    For final validation of top candidates.
    """
    try:
        # Separate risk params from strategy params
        strategy_params = {k: v for k, v in params.items() 
                         if k not in ('stop_loss_pct', 'take_profit_pct')}
        
        engine = WalkForwardEngine(
            train_window=TRAIN_WINDOW,
            test_window=TEST_WINDOW,
            step_size=STEP_SIZE,
            initial_capital=Decimal("10000"),
            commission=Decimal("0.001"),
            slippage=Decimal("0.0005"),
        )
        
        result = engine.run(
            strategy_name=strategy_name,
            candles=candles,
            symbol=symbol,
            interval=interval,
            strategy_params=strategy_params,
            reference_data=reference_data,
        )
        
        return {
            "oos_sharpe": result.oos_sharpe,
            "oos_return_pct": result.oos_return_pct,
            "oos_trades": result.oos_total_trades,
            "oos_win_rate": result.oos_win_rate,
            "sharpe_degradation": result.sharpe_degradation,
        }
        
    except Exception as e:
        logger.warning(f"Walk-forward failed for {strategy_name}: {e}")
        return {
            "oos_sharpe": -999.0,
            "oos_return_pct": 0.0,
            "oos_trades": 0,
            "oos_win_rate": 0.0,
            "sharpe_degradation": 100.0,
        }


async def grid_search_strategy(
    strategy_name: str,
    param_grid: dict[str, list],
    candles_cache: dict[str, pd.DataFrame],
    reference_data: dict[str, pd.DataFrame],
    top_n: int = 20,
) -> list[dict]:
    """
    Run grid search for a single strategy.
    
    Phase 1: Quick backtest sweep to find top candidates
    Phase 2: Walk-forward validation on top candidates
    """
    combinations = generate_param_combinations(param_grid)
    console.print(f"  Testing {len(combinations)} parameter combinations")
    
    # Phase 1: Quick backtest sweep
    quick_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Phase 1: Quick sweep",
            total=len(combinations) * len(candles_cache),
        )
        
        for params in combinations:
            sharpes = []
            
            for symbol, candles in candles_cache.items():
                progress.update(task, description=f"[cyan]{symbol}[/cyan]")
                
                sharpe = run_quick_backtest(
                    strategy_name,
                    params,
                    candles,
                    symbol,
                    INTERVAL,
                    reference_data=reference_data,
                )
                sharpes.append(sharpe)
                progress.advance(task)
            
            if sharpes and all(s > -900 for s in sharpes):
                quick_results.append({
                    "params": params,
                    "avg_sharpe": sum(sharpes) / len(sharpes),
                    "min_sharpe": min(sharpes),
                    "max_sharpe": max(sharpes),
                    "sharpe_spread": max(sharpes) - min(sharpes),
                })
    
    # Sort by average Sharpe and take top N
    quick_results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    top_candidates = quick_results[:top_n]
    
    if not top_candidates:
        console.print(f"  [red]No valid results for {strategy_name}[/red]")
        return []
    
    console.print(f"  Phase 1 complete. Top {len(top_candidates)} candidates selected.")
    
    # Phase 2: Walk-forward validation on top candidates
    final_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Phase 2: Walk-forward",
            total=len(top_candidates) * len(candles_cache),
        )
        
        for candidate in top_candidates:
            params = candidate["params"]
            wf_results = []
            
            for symbol, candles in candles_cache.items():
                progress.update(task, description=f"[cyan]{symbol}[/cyan]")
                
                metrics = run_walk_forward(
                    strategy_name,
                    params,
                    candles,
                    symbol,
                    INTERVAL,
                    reference_data=reference_data,
                )
                wf_results.append(metrics)
                progress.advance(task)
            
            if wf_results and all(r["oos_sharpe"] > -900 for r in wf_results):
                avg_oos_sharpe = sum(r["oos_sharpe"] for r in wf_results) / len(wf_results)
                avg_oos_return = sum(r["oos_return_pct"] for r in wf_results) / len(wf_results)
                total_trades = sum(r["oos_trades"] for r in wf_results)
                avg_degradation = sum(r["sharpe_degradation"] for r in wf_results) / len(wf_results)
                
                final_results.append({
                    "params": params,
                    "quick_sharpe": candidate["avg_sharpe"],
                    "oos_sharpe": avg_oos_sharpe,
                    "oos_return_pct": avg_oos_return,
                    "total_trades": total_trades,
                    "sharpe_degradation": avg_degradation,
                    "sharpe_spread": candidate["sharpe_spread"],
                })
    
    # Sort by OOS Sharpe
    final_results.sort(key=lambda x: x["oos_sharpe"], reverse=True)
    
    return final_results


def display_results(strategy_name: str, results: list[dict], top_n: int = 10) -> None:
    """Display top results for a strategy."""
    if not results:
        console.print(f"[yellow]No results for {strategy_name}[/yellow]")
        return
    
    table = Table(title=f"Top Parameters for {strategy_name}")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Parameters", style="cyan", max_width=60)
    table.add_column("Quick", justify="right", width=8)
    table.add_column("OOS", justify="right", width=8)
    table.add_column("Return%", justify="right", width=8)
    table.add_column("Trades", justify="right", width=7)
    table.add_column("Degrad%", justify="right", width=8)
    
    for i, r in enumerate(results[:top_n], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        if len(params_str) > 57:
            params_str = params_str[:57] + "..."
        
        oos_style = "green" if r["oos_sharpe"] > 0.5 else ("yellow" if r["oos_sharpe"] > 0 else "red")
        degrad_style = "green" if r["sharpe_degradation"] < 30 else ("yellow" if r["sharpe_degradation"] < 50 else "red")
        
        table.add_row(
            str(i),
            params_str,
            f"{r['quick_sharpe']:.2f}",
            f"[{oos_style}]{r['oos_sharpe']:.2f}[/{oos_style}]",
            f"{r['oos_return_pct']:+.1f}",
            str(r["total_trades"]),
            f"[{degrad_style}]{r['sharpe_degradation']:.0f}[/{degrad_style}]",
        )
    
    console.print(table)


def save_results(all_results: dict[str, list], output_dir: Path) -> None:
    """Save grid search results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full results
    with open(output_dir / "grid_search_validated.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Best params summary
    best_params = {}
    for strategy_name, results in all_results.items():
        if results:
            best = results[0]
            best_params[strategy_name] = {
                "best_params": best["params"],
                "oos_sharpe": best["oos_sharpe"],
                "oos_return_pct": best["oos_return_pct"],
                "quick_sharpe": best["quick_sharpe"],
                "sharpe_degradation": best["sharpe_degradation"],
            }
    
    with open(output_dir / "best_params_validated.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")


async def main():
    """Run grid search on all validated strategies."""
    console.print(Panel(
        "[bold blue]Grid Optimization for Validated Strategies[/bold blue]\n"
        "Testing hyperparameters on 5 winning strategies from Notes 08-09",
        expand=False,
    ))
    
    # Load hyperparameter grids from config
    settings = get_settings()
    all_grids = settings.optimization.optimization.hyperparameters
    
    # Filter to validated strategies only
    strategy_grids = {}
    for strategy_name in VALIDATED_STRATEGIES:
        if strategy_name in all_grids:
            strategy_grids[strategy_name] = all_grids[strategy_name]
        else:
            console.print(f"[yellow]Warning: No grid defined for {strategy_name}[/yellow]")
    
    if not strategy_grids:
        console.print("[red]No strategy grids found in config![/red]")
        return
    
    console.print(f"\nStrategies to optimize: {len(strategy_grids)}")
    for name in strategy_grids:
        combos = len(generate_param_combinations(strategy_grids[name]))
        console.print(f"  • {name}: {combos} combinations")
    
    # Load data
    console.print(f"\n[bold]Loading data ({DAYS} days)...[/bold]")
    candles_cache, reference_data = await load_data(SYMBOLS, INTERVAL, DAYS)
    
    if not candles_cache:
        console.print("[red]No data available![/red]")
        return
    
    # Run grid search for each strategy
    all_results = {}
    
    for strategy_name, param_grid in strategy_grids.items():
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold]Optimizing: {strategy_name}[/bold]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")
        
        results = await grid_search_strategy(
            strategy_name,
            param_grid,
            candles_cache,
            reference_data,
            top_n=20,
        )
        
        all_results[strategy_name] = results
        
        if results:
            display_results(strategy_name, results)
    
    # Save results
    output_dir = Path("notes/optimization_results")
    save_results(all_results, output_dir)
    
    # Summary
    console.print("\n")
    console.print(Panel(
        "[bold green]Grid Search Complete[/bold green]\n\n"
        "Best parameters saved to:\n"
        f"  • {output_dir}/grid_search_validated.json\n"
        f"  • {output_dir}/best_params_validated.json\n\n"
        "Next: Update strategy configs with optimized parameters",
        expand=False,
    ))


if __name__ == "__main__":
    asyncio.run(main())
