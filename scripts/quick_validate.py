#!/usr/bin/env python3
"""
Quick validation script - fast, focused testing.

Instead of 133 validations (7 strategies × 19 symbols), this runs:
- 3 representative symbols (BTC, ETH, SOL)
- 4 strategies (2 new robust + 2 legacy for comparison)
- Fewer folds (monthly step instead of weekly)

Total: 12 validations instead of 133 (~10x faster)

Features:
- Buy-and-hold comparison for each symbol
- Trades per month metric
- Reference data loading for cross-symbol strategies

Usage:
    python scripts/quick_validate.py
"""

import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.backtesting.walk_forward import (
    AcceptanceGate,
    WalkForwardEngine,
    WalkForwardResult,
)
from crypto.data.repository import CandleRepository
from crypto.strategies.registry import strategy_registry

# Import all strategies
from crypto.strategies import (
    ml, ml_siblings, rule_ensemble, ml_online, ml_cross_asset,
    cross_symbol, calendar, frequency, meta, microstructure,
    alternative_data_strategies,
)

logging.basicConfig(level=logging.WARNING)
console = Console()


def calculate_buy_and_hold_metrics(candles: pd.DataFrame) -> dict:
    """
    Calculate buy-and-hold metrics for comparison.
    
    Args:
        candles: OHLCV DataFrame
        
    Returns:
        Dict with Sharpe, return, max_dd
    """
    if candles.empty or len(candles) < 2:
        return {"sharpe": 0, "return_pct": 0, "max_dd": 0}
    
    # Calculate returns
    close = candles["close"].astype(float)
    returns = close.pct_change().dropna()
    
    if len(returns) < 2:
        return {"sharpe": 0, "return_pct": 0, "max_dd": 0}
    
    # Total return
    total_return = (close.iloc[-1] / close.iloc[0]) - 1
    
    # Sharpe ratio (annualized, hourly data)
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return > 0:
        sharpe = (mean_return / std_return) * np.sqrt(24 * 365)  # Annualized for hourly
    else:
        sharpe = 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_dd = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0
    
    return {
        "sharpe": float(sharpe),
        "return_pct": float(total_return * 100),
        "max_dd": float(max_dd),
    }

# =============================================================================
# QUICK VALIDATION CONFIG - Edit these for faster iteration
# =============================================================================

QUICK_CONFIG = {
    # Representative symbols (high volume, different characteristics)
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    
    # Focus strategies (new robust + legacy comparison)
    "strategies": [
        "ml_classifier_v5",      # New: simplified, should be robust
        "rule_ensemble",         # New: no ML, cannot overfit
        "ml_classifier_xgb",     # Legacy: best performer (but overfit)
        "ml_ensemble_voting",    # Legacy: most consistent
    ],
    
    # Walk-forward settings (faster)
    "train_window": 1440,   # 60 days (was 90)
    "test_window": 336,     # 14 days
    "step_size": 336,       # Bi-weekly steps (was weekly) = half the folds
    "min_train_samples": 1000,
    
    # Data
    "days": 180,            # 6 months (was 12)
    "interval": "1h",
}

# Acceptance gates
GATES = [
    AcceptanceGate("positive_oos", "oos_sharpe", "gt", 0),
    AcceptanceGate("low_degradation", "sharpe_degradation", "lt", 60),
    AcceptanceGate("enough_trades", "oos_total_trades", "gt", 10),
]


async def run_quick_validation():
    """Run quick validation."""
    config = QUICK_CONFIG
    
    console.print(Panel(
        f"[bold blue]Quick Validation[/bold blue]\n"
        f"Symbols: {len(config['symbols'])} | Strategies: {len(config['strategies'])}\n"
        f"Train: {config['train_window']} bars | Test: {config['test_window']} bars\n"
        f"Step: {config['step_size']} bars | Data: {config['days']} days",
        expand=False,
    ))
    
    # Load data
    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=config["days"])
    
    console.print("\n[bold]Loading data...[/bold]")
    candles_cache = {}
    for symbol in config["symbols"]:
        candles = await repository.get_candles_df(symbol, config["interval"], start, end)
        if not candles.empty:
            candles_cache[symbol] = candles
            console.print(f"  ✓ {symbol}: {len(candles)} candles")
    
    # Verify strategies
    console.print("\n[bold]Strategies:[/bold]")
    available = []
    for name in config["strategies"]:
        try:
            strategy_registry.create(name)
            console.print(f"  ✓ {name}")
            available.append(name)
        except Exception as e:
            console.print(f"  ✗ {name}: {e}")
    
    # Calculate expected runs
    total = len(available) * len(candles_cache)
    folds_per_run = (config["days"] * 24 - config["train_window"]) // config["step_size"]
    
    console.print(f"\n[bold]Running {total} validations (~{folds_per_run} folds each)...[/bold]\n")
    
    # Create engine
    engine = WalkForwardEngine(
        train_window=config["train_window"],
        test_window=config["test_window"],
        step_size=config["step_size"],
        min_train_samples=config["min_train_samples"],
    )
    
    # Calculate buy-and-hold metrics for each symbol
    console.print("\n[bold]Calculating buy-and-hold baselines...[/bold]")
    bh_metrics = {}
    for symbol, candles in candles_cache.items():
        bh_metrics[symbol] = calculate_buy_and_hold_metrics(candles)
        console.print(f"  {symbol}: Sharpe {bh_metrics[symbol]['sharpe']:.2f}, Return {bh_metrics[symbol]['return_pct']:.1f}%")
    
    console.print()
    
    # Run validations
    results = []
    for i, strategy_name in enumerate(available):
        for j, (symbol, candles) in enumerate(candles_cache.items()):
            progress = (i * len(candles_cache) + j + 1) / total * 100
            console.print(f"  [{progress:5.1f}%] {strategy_name} on {symbol}...", end="")
            
            try:
                start_time = datetime.now()
                result = engine.run(
                    strategy_name=strategy_name,
                    candles=candles,
                    symbol=symbol,
                    interval=config["interval"],
                )
                result.check_acceptance(GATES)
                results.append(result)
                
                elapsed = (datetime.now() - start_time).total_seconds()
                status = "[green]✓[/green]" if result.passed_validation else "[red]✗[/red]"
                
                # Show comparison to buy-and-hold
                bh_sharpe = bh_metrics[symbol]["sharpe"]
                vs_bh = result.oos_sharpe - bh_sharpe
                vs_bh_str = f"vs B&H: {vs_bh:+.2f}"
                
                console.print(f" {status} ({elapsed:.1f}s) OOS Sharpe: {result.oos_sharpe:.2f} ({vs_bh_str})")
                
            except Exception as e:
                console.print(f" [red]ERROR: {e}[/red]")
    
    return results, bh_metrics


def display_results(results: list[WalkForwardResult], bh_metrics: dict[str, dict]):
    """Display results with buy-and-hold comparison."""
    if not results:
        return
    
    console.print("\n")
    
    # Buy-and-hold baseline table
    console.print("[bold]Buy-and-Hold Baseline (for comparison):[/bold]")
    bh_table = Table()
    bh_table.add_column("Symbol", style="cyan")
    bh_table.add_column("Sharpe", justify="right")
    bh_table.add_column("Return %", justify="right")
    bh_table.add_column("Max DD %", justify="right")
    
    for symbol, metrics in bh_metrics.items():
        ret_style = "green" if metrics["return_pct"] > 0 else "red"
        bh_table.add_row(
            symbol,
            f"{metrics['sharpe']:.2f}",
            f"[{ret_style}]{metrics['return_pct']:.1f}%[/{ret_style}]",
            f"{metrics['max_dd']:.1f}%",
        )
    console.print(bh_table)
    console.print()
    
    # Summary table
    table = Table(title="Quick Validation Results")
    table.add_column("Strategy", style="yellow")
    table.add_column("Symbol", style="cyan")
    table.add_column("OOS Sharpe", justify="right")
    table.add_column("vs B&H", justify="right")  # New column
    table.add_column("IS Sharpe", justify="right")
    table.add_column("Degradation", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Trades/Mo", justify="right")  # New column
    table.add_column("Status", justify="center")
    
    for r in sorted(results, key=lambda x: x.oos_sharpe, reverse=True):
        oos_style = "green" if r.oos_sharpe > 0.5 else ("yellow" if r.oos_sharpe > 0 else "red")
        deg_style = "green" if r.sharpe_degradation < 30 else ("yellow" if r.sharpe_degradation < 60 else "red")
        status = "[green]PASS[/green]" if r.passed_validation else "[red]FAIL[/red]"
        
        # Calculate vs buy-and-hold
        bh_sharpe = bh_metrics.get(r.symbol, {}).get("sharpe", 0)
        vs_bh = r.oos_sharpe - bh_sharpe
        vs_bh_style = "green" if vs_bh > 0 else "red"
        
        # Calculate trades per month (assuming 6 months / 180 days)
        trades_per_month = r.oos_total_trades / 6 if r.oos_total_trades else 0
        trades_style = "green" if trades_per_month < 20 else ("yellow" if trades_per_month < 50 else "red")
        
        table.add_row(
            r.strategy_name,
            r.symbol,
            f"[{oos_style}]{r.oos_sharpe:.2f}[/{oos_style}]",
            f"[{vs_bh_style}]{vs_bh:+.2f}[/{vs_bh_style}]",
            f"{r.is_sharpe:.2f}",
            f"[{deg_style}]{r.sharpe_degradation:.1f}%[/{deg_style}]",
            str(r.oos_total_trades),
            f"[{trades_style}]{trades_per_month:.1f}[/{trades_style}]",
            status,
        )
    
    console.print(table)
    
    # Strategy summary
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r.strategy_name].append(r)
    
    console.print("\n[bold]Strategy Summary:[/bold]")
    for name, strat_results in by_strategy.items():
        avg_oos = sum(r.oos_sharpe for r in strat_results) / len(strat_results)
        avg_deg = sum(r.sharpe_degradation for r in strat_results) / len(strat_results)
        avg_trades = sum(r.oos_total_trades for r in strat_results) / len(strat_results)
        trades_per_month = avg_trades / 6
        passed = sum(1 for r in strat_results if r.passed_validation)
        
        # Check vs buy-and-hold
        beats_bh = sum(1 for r in strat_results if r.oos_sharpe > bh_metrics.get(r.symbol, {}).get("sharpe", 0))
        
        if avg_oos > 0.5 and passed == len(strat_results) and beats_bh == len(strat_results):
            status = "[bold green]★ PROMISING[/bold green]"
        elif avg_oos > 0 and beats_bh > 0:
            status = "[yellow]◐ NEEDS WORK[/yellow]"
        else:
            status = "[red]✗ FAILS B&H[/red]"
        
        console.print(
            f"  {name}: OOS Sharpe {avg_oos:.2f}, "
            f"Trades/Mo {trades_per_month:.1f}, "
            f"Beats B&H {beats_bh}/{len(strat_results)}, "
            f"Pass {passed}/{len(strat_results)} {status}"
        )
    
    # Save results
    output_dir = Path("notes/quick_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-safe format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, bool):
            return str(obj).lower()
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        return obj
    
    with open(output_dir / "results.json", "w") as f:
        json.dump([make_serializable(r.to_dict()) for r in results], f, indent=2)
    
    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")


async def main():
    console.print("[bold blue]═══ Quick Strategy Validation ═══[/bold blue]\n")
    
    start = datetime.now()
    results, bh_metrics = await run_quick_validation()
    elapsed = (datetime.now() - start).total_seconds()
    
    console.print(f"\n[bold green]Completed in {elapsed:.1f}s[/bold green]")
    display_results(results, bh_metrics)


if __name__ == "__main__":
    asyncio.run(main())
