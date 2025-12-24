#!/usr/bin/env python3
"""
Orthogonal Strategy Testing Script.

Tests strategies designed to be orthogonal to ratio mean reversion strategies.
Focus: Diversification via different state variables (crashes, vol regimes, breadth, etc.)

Usage:
    python scripts/run_orthogonal_testing.py              # Tier 1, quick (3 symbols, 6mo)
    python scripts/run_orthogonal_testing.py --full       # Tier 1, full (8 symbols, 12mo)
    python scripts/run_orthogonal_testing.py --tier 2     # Include Tier 2 strategies
"""

import argparse
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.backtesting.walk_forward import (
    AcceptanceGate,
    WalkForwardEngine,
    WalkForwardResult,
)
from crypto.data.repository import CandleRepository
from crypto.strategies.registry import strategy_registry

# Import strategies
from crypto.strategies import hyper_creative, hyper_creative_tier2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

OUTPUT_DIR = Path("notes/hypercreative_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tier 1 Strategies (OHLCV only)
TIER1_STRATEGIES = [
    "crash_only_trend_filter",
    "gamma_mimic_breakout",
    "volatility_targeting_overlay",
    "market_breadth_alt_participation",
    "cross_sectional_momentum",
    "gap_reversion",
    "liquidity_vacuum_detector",
    "correlation_regime_switch",
]

# Tier 2 Strategies (external data)
TIER2_STRATEGIES = [
    "funding_term_structure",
    "funding_vol_interaction",
    "liquidation_cluster_fade",
    "stablecoin_liquidity_pulse",
    "btc_dominance_rotation",
]

# Quick config
QUICK_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
QUICK_DAYS = 180
QUICK_CONFIG = {"train_window": 1440, "test_window": 336, "step_size": 336}

# Full config
FULL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT",
    "LINKUSDT", "NEARUSDT", "DOTUSDT", "APTUSDT",
]
FULL_DAYS = 365
FULL_CONFIG = {"train_window": 2160, "test_window": 336, "step_size": 168}


async def load_data(symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
    """Load candle data."""
    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 30)
    
    console.print("\n[bold]Loading data...[/bold]")
    data = {}
    
    for symbol in symbols:
        try:
            candles = await repository.get_candles_df(symbol, "1h", start, end)
            if not candles.empty:
                data[symbol] = candles
                console.print(f"  ✓ {symbol}: {len(candles):,} candles")
        except Exception as e:
            console.print(f"  [red]✗ {symbol}: {e}[/red]")
    
    return data


async def run_test(
    strategy_name: str,
    symbol: str,
    candles: pd.DataFrame,
    reference_data: dict,
    config: dict,
    gates: list,
) -> WalkForwardResult | None:
    """Run walk-forward test."""
    try:
        # Load strategy config from YAML
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "research" / "creative_testing.yaml"
        
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        strat_configs = cfg.get("creative_testing", {}).get("orthogonal_strategies", {})
        
        # Find strategy config (check tier1 and tier2)
        tier1 = strat_configs.get("tier1", {})
        tier2 = strat_configs.get("tier2", {})
        
        if strategy_name in tier1:
            params = tier1[strategy_name].get("params", {})
        elif strategy_name in tier2:
            params = tier2[strategy_name].get("params", {})
        else:
            logger.warning(f"Strategy {strategy_name} not in config")
            return None
        
        # Run walk-forward
        engine = WalkForwardEngine(
            train_window=config["train_window"],
            test_window=config["test_window"],
            step_size=config["step_size"],
        )
        
        result = engine.run(
            strategy_name=strategy_name,
            candles=candles,
            symbol=symbol,
            interval="1h",
            strategy_params=params,
            reference_data=reference_data,
        )
        
        # Check gates
        result.check_acceptance(gates)
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing {strategy_name} on {symbol}: {e}", exc_info=True)
        return None


async def main(args):
    """Main entry point."""
    
    # Configuration
    if args.full:
        symbols = FULL_SYMBOLS
        days = FULL_DAYS
        config = FULL_CONFIG
        mode = "Full"
    else:
        symbols = QUICK_SYMBOLS
        days = QUICK_DAYS
        config = QUICK_CONFIG
        mode = "Quick"
    
    strategies = TIER1_STRATEGIES.copy()
    if args.tier == 2:
        strategies += TIER2_STRATEGIES
        tier_name = "Tier 1 + 2"
    else:
        tier_name = "Tier 1 Only"
    
    # Gates (lower bar for orthogonal strategies)
    gates = [
        AcceptanceGate("positive_sharpe", "oos_sharpe", "gt", 0.3),
        AcceptanceGate("enough_trades", "oos_total_trades", "gt", 5),
    ]
    
    # Display
    console.print(Panel.fit(
        f"[bold blue]Orthogonal Strategy Testing[/bold blue]\n\n"
        f"Mode: {mode}\n"
        f"Tier: {tier_name} ({len(strategies)} strategies)\n"
        f"Symbols: {len(symbols)}\n"
        f"Days: {days}\n"
        f"Train: {config['train_window']}h | Test: {config['test_window']}h | Step: {config['step_size']}h"
    ))
    
    # Load data
    data = await load_data(symbols, days)
    
    if not data:
        console.print("[red]No data loaded![/red]")
        return
    
    # Reference data
    reference_data = {sym: df for sym, df in data.items() if sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}
    console.print(f"\n[bold]Reference data:[/bold] {list(reference_data.keys())}")
    
    # Run tests
    results = {}
    total = len(strategies) * len(data)
    console.print(f"\n[bold]Running {total} tests...[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Testing...", total=total)
        
        for strategy_name in strategies:
            results[strategy_name] = {}
            
            for symbol, candles in data.items():
                progress.update(task, description=f"{strategy_name} on {symbol}...")
                
                result = await run_test(
                    strategy_name,
                    symbol,
                    candles,
                    reference_data,
                    config,
                    gates,
                )
                
                if result:
                    results[strategy_name][symbol] = result
                
                progress.update(task, advance=1)
    
    # Analyze
    console.print("\n" + "=" * 80)
    console.print("[bold]RESULTS SUMMARY[/bold]")
    console.print("=" * 80 + "\n")
    
    summary = []
    for strategy_name in strategies:
        if not results.get(strategy_name):
            continue
        
        symbol_results = results[strategy_name]
        oos_sharpes = []
        is_sharpes = []
        trades = []
        passed = 0
        
        for result in symbol_results.values():
            if result.oos_sharpe is not None:
                oos_sharpes.append(result.oos_sharpe)
            if result.is_sharpe is not None:
                is_sharpes.append(result.is_sharpe)
            if result.oos_total_trades:
                trades.append(result.oos_total_trades)
            if result.passed_validation:
                passed += 1
        
        avg_oos = np.mean(oos_sharpes) if oos_sharpes else 0.0
        avg_is = np.mean(is_sharpes) if is_sharpes else 0.0
        avg_trades = np.mean(trades) if trades else 0.0
        pass_rate = passed / len(symbol_results) if symbol_results else 0.0
        
        degrad = ((avg_oos - avg_is) / abs(avg_is) * 100) if avg_is != 0 else 0.0
        
        summary.append({
            "strategy": strategy_name,
            "avg_oos_sharpe": avg_oos,
            "avg_is_sharpe": avg_is,
            "degradation_pct": degrad,
            "avg_trades": avg_trades,
            "pass_rate": pass_rate,
            "n_symbols": len(symbol_results),
            "n_passed": passed,
        })
    
    summary.sort(key=lambda x: x["avg_oos_sharpe"], reverse=True)
    
    # Table
    table = Table(title="Orthogonal Strategy Rankings")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Strategy", style="magenta")
    table.add_column("OOS Sharpe", justify="right", style="green")
    table.add_column("Degrad%", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Pass%", justify="right")
    table.add_column("Tests", justify="right")
    
    for i, s in enumerate(summary, 1):
        deg_color = "green" if s["degradation_pct"] < 0 else "yellow" if s["degradation_pct"] < 60 else "red"
        pass_color = "green" if s["pass_rate"] >= 0.5 else "yellow" if s["pass_rate"] >= 0.3 else "red"
        
        table.add_row(
            str(i),
            s["strategy"],
            f"{s['avg_oos_sharpe']:.2f}",
            f"[{deg_color}]{s['degradation_pct']:.0f}%[/{deg_color}]",
            f"{s['avg_trades']:.0f}",
            f"[{pass_color}]{s['pass_rate']*100:.0f}%[/{pass_color}]",
            f"{s['n_passed']}/{s['n_symbols']}",
        )
    
    console.print(table)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f"orthogonal_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "mode": mode,
            "tier": tier_name,
            "symbols": symbols,
            "config": config,
            "summary": summary,
        }, f, indent=2, default=str)
    
    console.print(f"\n[bold]Results saved:[/bold] {results_file}")
    
    # Winners
    winners = [s for s in summary if s["avg_oos_sharpe"] > 0.3 and s["pass_rate"] >= 0.3]
    console.print(f"\n[bold green]Promising Strategies ({len(winners)}):[/bold green]")
    for w in winners:
        console.print(f"  • {w['strategy']}: Sharpe {w['avg_oos_sharpe']:.2f}, Pass {w['pass_rate']*100:.0f}%")
    
    if not winners:
        console.print("  [yellow]No strategies passed acceptance gates.[/yellow]")
    
    console.print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Full test (8 symbols, 12 months)")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2], help="Tier level")
    
    args = parser.parse_args()
    asyncio.run(main(args))
