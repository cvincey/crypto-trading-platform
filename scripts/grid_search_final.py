#!/usr/bin/env python3
"""
Comprehensive grid search for the 5 final active strategies.
Uses parallel processing for speed.

Strategies:
1. eth_btc_ratio_optimized (eth_btc_ratio_reversion on ETHUSDT)
2. sol_btc_ratio (ratio on SOLUSDT)
3. ltc_btc_ratio (ratio on LTCUSDT)
4. eth_btc_ratio_confirmed_optimized (confirmed on ETHUSDT)
5. basis_proxy (on BTC/ETH/SOL)
"""

import asyncio
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Number of parallel workers
MAX_WORKERS = min(cpu_count(), 8)

from crypto.backtesting.walk_forward import WalkForwardEngine
from crypto.data.repository import CandleRepository
from crypto.strategies import cross_symbol, hybrid, structural
from crypto.strategies.registry import strategy_registry

console = Console()

# Configuration
DAYS = 365  # Full year for comprehensive test
INTERVAL = "1h"

# Walk-forward settings
TRAIN_WINDOW = 720   # 30 days
TEST_WINDOW = 168    # 7 days
STEP_SIZE = 168      # 7 days

# Strategy configurations with parameter grids (reduced for faster execution)
STRATEGIES = {
    "eth_btc_ratio_optimized": {
        "type": "eth_btc_ratio_reversion",
        "symbol": "ETHUSDT",
        "grid": {
            "lookback": [72, 96, 120],
            "entry_threshold": [-1.5, -1.2],
            "exit_threshold": [-0.5, -0.7],
            "max_hold_hours": [48, 72],
            "stop_loss_pct": [0.03, 0.04],
            "take_profit_pct": [0.08, 0.10],
        },
    },
    "sol_btc_ratio": {
        "type": "eth_btc_ratio_reversion",
        "symbol": "SOLUSDT",
        "grid": {
            "lookback": [72, 96, 120],
            "entry_threshold": [-1.5, -1.2],
            "exit_threshold": [-0.5, -0.7],
            "max_hold_hours": [48, 72],
            "stop_loss_pct": [0.03, 0.04],
            "take_profit_pct": [0.08, 0.10],
        },
    },
    "ltc_btc_ratio": {
        "type": "eth_btc_ratio_reversion",
        "symbol": "LTCUSDT",
        "grid": {
            "lookback": [72, 96, 120],
            "entry_threshold": [-1.5, -1.2],
            "exit_threshold": [-0.5, -0.7],
            "max_hold_hours": [48, 72],
            "stop_loss_pct": [0.03, 0.04],
            "take_profit_pct": [0.08, 0.10],
        },
    },
    "eth_btc_ratio_confirmed": {
        "type": "eth_btc_ratio_confirmed",
        "symbol": "ETHUSDT",
        "grid": {
            "lookback": [72, 96, 120],
            "entry_threshold": [-1.5, -1.2],
            "exit_threshold": [-0.5, -0.7],
            "max_hold_hours": [48, 72],
            "confirmation_delay": [2, 3],
            "stop_loss_pct": [0.03, 0.04],
            "take_profit_pct": [0.08, 0.10],
        },
    },
    "basis_proxy": {
        "type": "basis_proxy",
        "symbol": "BTCUSDT",
        "grid": {
            "funding_lookback": [4, 6, 9],
            "entry_threshold": [-0.0004, -0.0003],
            "exit_threshold": [0.0002, 0.0003],
            "max_hold_hours": [48, 72],
            "stop_loss_pct": [0.02, 0.03],
            "take_profit_pct": [0.06, 0.08],
        },
    },
}


def generate_combinations(grid: dict) -> list[dict]:
    """Generate all parameter combinations."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


async def load_data() -> tuple[dict, dict]:
    """Load all required candle data."""
    repo = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LTCUSDT"]
    candles = {}
    
    console.print(f"\n[bold]Loading {DAYS} days of data...[/bold]")
    for symbol in symbols:
        df = await repo.get_candles_df(symbol, INTERVAL, start, end)
        if not df.empty:
            candles[symbol] = df
            console.print(f"  {symbol}: {len(df)} candles")
    
    reference_data = {"BTCUSDT": candles["BTCUSDT"], "ETHUSDT": candles["ETHUSDT"]}
    return candles, reference_data


def run_single_test(args: tuple) -> dict:
    """
    Run a single walk-forward test. Designed for parallel execution.
    
    Args is a tuple: (strategy_type, params, candles_dict, symbol, reference_dict)
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from decimal import Decimal
    from crypto.backtesting.walk_forward import WalkForwardEngine
    from crypto.strategies import cross_symbol, hybrid, structural
    
    strategy_type, params, candles_dict, symbol, reference_dict = args
    
    try:
        # Reconstruct DataFrames from dict
        import pandas as pd
        candles = pd.DataFrame(candles_dict)
        candles.index = pd.to_datetime(candles.index)
        
        reference_data = {}
        for ref_sym, ref_dict in reference_dict.items():
            ref_df = pd.DataFrame(ref_dict)
            ref_df.index = pd.to_datetime(ref_df.index)
            reference_data[ref_sym] = ref_df
        
        # Separate strategy params from risk params
        strategy_params = {k: v for k, v in params.items() 
                         if k not in ("stop_loss_pct", "take_profit_pct")}
        
        engine = WalkForwardEngine(
            train_window=720,   # 30 days
            test_window=168,    # 7 days
            step_size=168,      # 7 days
            initial_capital=Decimal("10000"),
            commission=Decimal("0.001"),
            slippage=Decimal("0.0005"),
        )
        
        result = engine.run(
            strategy_name=strategy_type,
            candles=candles,
            symbol=symbol,
            interval="1h",
            strategy_params=strategy_params,
            reference_data=reference_data,
        )
        
        return {
            "params": params,
            "oos_sharpe": result.oos_sharpe,
            "oos_return": result.oos_return_pct,
            "oos_trades": result.oos_total_trades,
            "oos_win_rate": result.oos_win_rate,
            "degradation": result.sharpe_degradation,
            "is_sharpe": result.is_sharpe,
        }
    except Exception as e:
        return {"params": params, "error": str(e), "oos_sharpe": -999}


def grid_search_strategy_parallel(
    name: str,
    config: dict,
    candles: dict,
    reference_data: dict,
) -> list[dict]:
    """Run grid search for one strategy using parallel processing."""
    strategy_type = config["type"]
    symbol = config["symbol"]
    grid = config["grid"]
    
    combinations = generate_combinations(grid)
    console.print(f"\n  Testing {len(combinations)} combinations with {MAX_WORKERS} workers...")
    
    # Convert DataFrames to dicts for pickling
    symbol_candles_dict = candles[symbol].to_dict()
    reference_dict = {sym: df.to_dict() for sym, df in reference_data.items()}
    
    # Prepare arguments for parallel execution
    args_list = [
        (strategy_type, params, symbol_candles_dict, symbol, reference_dict)
        for params in combinations
    ]
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]{name}[/cyan]", total=len(combinations))
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(run_single_test, args): args for args in args_list}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result.get("oos_sharpe", -999) > -900:
                        results.append(result)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                
                progress.advance(task)
    
    # Sort by OOS Sharpe
    results.sort(key=lambda x: x["oos_sharpe"], reverse=True)
    return results


def display_top_results(name: str, results: list[dict], top_n: int = 5):
    """Display top results for a strategy."""
    if not results:
        console.print(f"[red]No results for {name}[/red]")
        return
    
    table = Table(title=f"Top {top_n} for {name}")
    table.add_column("#", width=3)
    table.add_column("Key Params", max_width=40)
    table.add_column("OOS Sharpe", justify="right")
    table.add_column("OOS Return%", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Degrad%", justify="right")
    
    for i, r in enumerate(results[:top_n], 1):
        # Show key params only
        key_params = {k: v for k, v in r["params"].items() 
                     if k not in ("stop_loss_pct", "take_profit_pct")}
        params_str = ", ".join(f"{k}={v}" for k, v in key_params.items())
        if len(params_str) > 37:
            params_str = params_str[:37] + "..."
        
        sharpe_style = "green" if r["oos_sharpe"] > 1 else ("yellow" if r["oos_sharpe"] > 0 else "red")
        
        table.add_row(
            str(i),
            params_str,
            f"[{sharpe_style}]{r['oos_sharpe']:.2f}[/{sharpe_style}]",
            f"{r['oos_return']:+.1f}",
            str(r["oos_trades"]),
            f"{r['degradation']:.0f}",
        )
    
    console.print(table)


async def main():
    console.print(Panel(
        "[bold blue]Comprehensive Grid Search - 5 Final Strategies[/bold blue]\n"
        f"Period: {DAYS} days | Walk-forward: {TRAIN_WINDOW}h train / {TEST_WINDOW}h test\n"
        f"Parallel workers: {MAX_WORKERS}",
        expand=False,
    ))
    
    # Load data
    candles, reference_data = await load_data()
    
    if len(candles) < 4:
        console.print("[red]Missing required data![/red]")
        return
    
    # Run grid search for each strategy (sequentially, but each uses parallel internally)
    all_results = {}
    best_params = {}
    
    for name, config in STRATEGIES.items():
        console.print(f"\n{'='*60}")
        console.print(f"[bold yellow]{name}[/bold yellow]")
        console.print(f"{'='*60}")
        
        results = grid_search_strategy_parallel(name, config, candles, reference_data)
        all_results[name] = results
        
        if results:
            display_top_results(name, results)
            best = results[0]
            best_params[name] = {
                "params": best["params"],
                "oos_sharpe": best["oos_sharpe"],
                "oos_return": best["oos_return"],
                "oos_trades": best["oos_trades"],
                "symbol": config["symbol"],
            }
    
    # Save results
    output_dir = Path("notes/optimization_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "grid_search_final_5.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    with open(output_dir / "best_params_final_5.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    
    # Summary
    console.print("\n")
    console.print("=" * 60)
    console.print("[bold green]FINAL SUMMARY[/bold green]")
    console.print("=" * 60)
    
    summary_table = Table(title="Optimal Parameters for 5 Strategies")
    summary_table.add_column("Strategy", style="cyan")
    summary_table.add_column("Symbol")
    summary_table.add_column("OOS Sharpe", justify="right")
    summary_table.add_column("OOS Return%", justify="right")
    summary_table.add_column("Trades", justify="right")
    
    for name, info in best_params.items():
        sharpe_style = "green" if info["oos_sharpe"] > 1 else "yellow"
        summary_table.add_row(
            name,
            info["symbol"],
            f"[{sharpe_style}]{info['oos_sharpe']:.2f}[/{sharpe_style}]",
            f"{info['oos_return']:+.1f}",
            str(info["oos_trades"]),
        )
    
    console.print(summary_table)
    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
