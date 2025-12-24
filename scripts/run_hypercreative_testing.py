#!/usr/bin/env python3
"""
Hyper-Creative Strategy Testing Script.

Tests all 24 hyper-creative diversification strategies from Research Note 13:
- Tier 1 (16 strategies): OHLCV only, no external data
- Tier 2 (8 strategies): Alternative data (Fear&Greed, BTC Dominance, L/S Ratio)

Focus: Diversification from ratio strategies with orthogonal signal sources.

Usage:
    python scripts/run_hypercreative_testing.py                    # Tier 1 only
    python scripts/run_hypercreative_testing.py --tier 2           # Include Tier 2
    python scripts/run_hypercreative_testing.py --full             # All 50 symbols
    python scripts/run_hypercreative_testing.py --quick            # Quick 3-symbol test
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
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

from crypto.backtesting.walk_forward import (
    AcceptanceGate,
    WalkForwardEngine,
    WalkForwardResult,
    check_acceptance_gates,
)
from crypto.data.repository import CandleRepository
from crypto.strategies.registry import strategy_registry

# Import all strategies to register them
from crypto.strategies import (
    # Core
    technical, statistical, momentum, ml,
    # Existing creative
    cross_symbol, calendar, frequency, meta, microstructure,
    # Hyper-creative Tier 1
    information_theoretic, microstructure_v2, multi_timeframe_v2,
    volatility_v2, calendar_v2,
    # Hyper-creative Tier 2
    sentiment, positioning, macro, funding_v2,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

# Output directory
OUTPUT_DIR = Path("notes/hypercreative_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TIER 1 STRATEGIES - OHLCV only (16 strategies)
# =============================================================================

TIER1_STRATEGIES = [
    # Information-Theoretic
    "entropy_collapse",
    "hurst_regime",
    "fractal_breakout",
    # Microstructure
    "volume_clock_momentum",
    "liquidity_vacuum",
    "body_ratio_sequence",
    "taker_imbalance_momentum",
    # Multi-Timeframe
    "mtf_momentum_divergence",
    "leader_follower_rotation",
    "sector_dispersion_trade",
    # Volatility
    "vol_term_structure",
    "squeeze_cascade",
    "vol_of_vol_breakout",
    # Calendar
    "asian_session_reversal",
    "weekend_gap_fade",
    "funding_hour_momentum",
]

# =============================================================================
# TIER 2 STRATEGIES - Alternative data (8 strategies)
# =============================================================================

TIER2_STRATEGIES = [
    # Sentiment
    "fear_greed_divergence",
    "fear_greed_extreme_fade",
    "long_short_ratio_fade",
    # Market Structure
    "btc_dominance_rotation",
    "dominance_momentum",
    "liquidation_cluster_magnet",
    # Funding
    "funding_arbitrage_proxy",
    "funding_momentum",
]

# Baseline ratio strategies for correlation analysis
RATIO_STRATEGIES = [
    "eth_btc_ratio_reversion",
    "sol_btc_ratio",
    "ltc_btc_ratio",
]

# =============================================================================
# CONFIGURATION
# =============================================================================

# Quick test: 3 symbols, 6 months
QUICK_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
QUICK_DAYS = 180
QUICK_TRAIN = 1440  # 60 days
QUICK_TEST = 336    # 14 days
QUICK_STEP = 336    # Bi-weekly

# Full test: All 50 symbols, 12 months
FULL_SYMBOLS = [
    # Top 20
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT",
    "DOTUSDT", "XLMUSDT", "SUIUSDT", "LTCUSDT", "BCHUSDT",
    "NEARUSDT", "APTUSDT", "UNIUSDT", "ICPUSDT", "ETCUSDT",
    # Top 21-50
    "RENDERUSDT", "AAVEUSDT", "FILUSDT", "ATOMUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "IMXUSDT", "VETUSDT", "GRTUSDT",
    "ALGOUSDT", "FTMUSDT", "RUNEUSDT", "SEIUSDT", "TIAUSDT",
    "LDOUSDT", "MKRUSDT", "THETAUSDT", "FLOKIUSDT", "BONKUSDT",
    "JUPUSDT", "SANDUSDT", "AXSUSDT", "WLDUSDT", "SNXUSDT",
    "FLOWUSDT", "EGLDUSDT", "CHZUSDT", "APEUSDT", "MANAUSDT",
]
FULL_DAYS = 365
FULL_TRAIN = 2160   # 90 days
FULL_TEST = 336     # 14 days
FULL_STEP = 168     # Weekly

# Acceptance gates
GATES = [
    AcceptanceGate("positive_sharpe", "oos_sharpe", "gt", 0.5),
    AcceptanceGate("enough_trades", "oos_total_trades", "gt", 10),
    AcceptanceGate("not_overfit", "degradation_pct", "lt", 50),
]


async def load_data(
    symbols: list[str],
    days: int,
    interval: str = "1h",
) -> dict[str, pd.DataFrame]:
    """Load candle data for all symbols."""
    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 30)  # Extra buffer

    console.print("\n[bold]Loading data...[/bold]")
    data = {}
    
    for symbol in symbols:
        try:
            candles = await repository.get_candles_df(symbol, interval, start, end)
            if not candles.empty:
                data[symbol] = candles
                console.print(f"  ✓ {symbol}: {len(candles)} candles")
            else:
                console.print(f"  [yellow]✗ {symbol}: No data[/yellow]")
        except Exception as e:
            console.print(f"  [red]✗ {symbol}: Error - {e}[/red]")

    return data


async def run_strategy_test(
    strategy_name: str,
    symbol: str,
    candles: pd.DataFrame,
    reference_data: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> WalkForwardResult | None:
    """Run walk-forward test for one strategy on one symbol."""
    try:
        # Get strategy from config
        from crypto.core.config import load_config
        
        # Load strategy params from config
        config_obj = load_config()
        strategy_configs = config_obj.get("creative_testing", {}).get("hypercreative_strategies", {})
        
        if strategy_name not in strategy_configs:
            logger.warning(f"Strategy {strategy_name} not found in config")
            return None
        
        strategy_config = strategy_configs[strategy_name]
        params = strategy_config.get("params", {})
        
        # Create strategy instance
        strategy = strategy_registry.create(strategy_name, **params)
        
        # Setup engine
        engine = WalkForwardEngine(
            train_window=config["train_window"],
            test_window=config["test_window"],
            step_size=config["step_size"],
        )
        
        # Run walk-forward
        result = await engine.run_walk_forward(
            strategy=strategy,
            candles=candles,
            reference_data=reference_data,
            gates=GATES,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing {strategy_name} on {symbol}: {e}")
        return None


async def main(args: argparse.Namespace):
    """Main entry point."""
    
    # Determine test configuration
    if args.quick:
        symbols = QUICK_SYMBOLS
        days = QUICK_DAYS
        config = {
            "train_window": QUICK_TRAIN,
            "test_window": QUICK_TEST,
            "step_size": QUICK_STEP,
        }
        mode = "Quick"
    else:
        symbols = FULL_SYMBOLS if args.full else QUICK_SYMBOLS
        days = FULL_DAYS if args.full else QUICK_DAYS
        config = {
            "train_window": FULL_TRAIN if args.full else QUICK_TRAIN,
            "test_window": FULL_TEST if args.full else QUICK_TEST,
            "step_size": FULL_STEP if args.full else QUICK_STEP,
        }
        mode = "Full" if args.full else "Standard"
    
    # Determine which strategies to test
    if args.tier == 2:
        strategies = TIER1_STRATEGIES + TIER2_STRATEGIES
        tier_name = "Tier 1 + 2"
    else:
        strategies = TIER1_STRATEGIES
        tier_name = "Tier 1 Only"
    
    # Display configuration
    console.print(Panel.fit(
        f"[bold blue]Hyper-Creative Strategy Testing[/bold blue]\n\n"
        f"Mode: {mode}\n"
        f"Strategies: {tier_name} ({len(strategies)})\n"
        f"Symbols: {len(symbols)}\n"
        f"Days: {days}\n"
        f"Train: {config['train_window']}h, Test: {config['test_window']}h, Step: {config['step_size']}h"
    ))
    
    # Load data
    data = await load_data(symbols, days)
    
    if not data:
        console.print("[red]No data available![/red]")
        return
    
    # Load reference data for cross-asset strategies
    reference_data = {}
    for ref_symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        if ref_symbol in data:
            reference_data[ref_symbol] = data[ref_symbol]
    
    console.print(f"\n[bold]Reference data loaded:[/bold] {list(reference_data.keys())}")
    
    # Run tests
    results = {}
    total_tests = len(strategies) * len(data)
    completed = 0
    
    console.print(f"\n[bold]Running {total_tests} tests...[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Testing...", total=total_tests)
        
        for strategy_name in strategies:
            results[strategy_name] = {}
            
            for symbol, candles in data.items():
                progress.update(
                    task,
                    description=f"Testing {strategy_name} on {symbol}...",
                    advance=0,
                )
                
                result = await run_strategy_test(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    candles=candles,
                    reference_data=reference_data,
                    config=config,
                )
                
                if result:
                    results[strategy_name][symbol] = result
                
                completed += 1
                progress.update(task, advance=1)
    
    # Analyze results
    console.print("\n" + "=" * 80)
    console.print("[bold]RESULTS SUMMARY[/bold]")
    console.print("=" * 80 + "\n")
    
    # Aggregate by strategy
    strategy_summary = []
    
    for strategy_name in strategies:
        if strategy_name not in results or not results[strategy_name]:
            continue
        
        symbol_results = results[strategy_name]
        
        # Aggregate metrics
        oos_sharpes = []
        is_sharpes = []
        total_trades = []
        passed = 0
        
        for symbol, result in symbol_results.items():
            if result.oos_sharpe is not None:
                oos_sharpes.append(result.oos_sharpe)
            if result.is_sharpe is not None:
                is_sharpes.append(result.is_sharpe)
            if result.oos_total_trades is not None:
                total_trades.append(result.oos_total_trades)
            if result.passed:
                passed += 1
        
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0
        avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
        avg_trades = np.mean(total_trades) if total_trades else 0.0
        pass_rate = passed / len(symbol_results) if symbol_results else 0.0
        
        # Degradation
        if avg_is_sharpe != 0:
            degradation = ((avg_oos_sharpe - avg_is_sharpe) / abs(avg_is_sharpe)) * 100
        else:
            degradation = 0.0
        
        strategy_summary.append({
            "strategy": strategy_name,
            "avg_oos_sharpe": avg_oos_sharpe,
            "avg_is_sharpe": avg_is_sharpe,
            "degradation_pct": degradation,
            "avg_trades": avg_trades,
            "pass_rate": pass_rate,
            "n_symbols": len(symbol_results),
            "n_passed": passed,
        })
    
    # Sort by OOS Sharpe
    strategy_summary.sort(key=lambda x: x["avg_oos_sharpe"], reverse=True)
    
    # Display table
    table = Table(title="Strategy Performance Rankings")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Strategy", style="magenta")
    table.add_column("OOS Sharpe", justify="right", style="green")
    table.add_column("Degrad%", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Pass%", justify="right")
    table.add_column("Tests", justify="right")
    
    for i, summary in enumerate(strategy_summary, 1):
        degrad_color = "green" if summary["degradation_pct"] < 0 else "yellow" if summary["degradation_pct"] < 50 else "red"
        pass_color = "green" if summary["pass_rate"] >= 0.6 else "yellow" if summary["pass_rate"] >= 0.4 else "red"
        
        table.add_row(
            str(i),
            summary["strategy"],
            f"{summary['avg_oos_sharpe']:.2f}",
            f"[{degrad_color}]{summary['degradation_pct']:.0f}%[/{degrad_color}]",
            f"{summary['avg_trades']:.0f}",
            f"[{pass_color}]{summary['pass_rate']*100:.0f}%[/{pass_color}]",
            f"{summary['n_passed']}/{summary['n_symbols']}",
        )
    
    console.print(table)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full results
    results_file = OUTPUT_DIR / f"hypercreative_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({
            "config": config,
            "mode": mode,
            "tier": tier_name,
            "symbols": symbols,
            "summary": strategy_summary,
        }, f, indent=2, default=str)
    
    console.print(f"\n[bold]Results saved to:[/bold] {results_file}")
    
    # Winners
    winners = [s for s in strategy_summary if s["avg_oos_sharpe"] > 0.5 and s["pass_rate"] >= 0.4]
    
    console.print(f"\n[bold green]Promising Strategies ({len(winners)}):[/bold green]")
    for winner in winners:
        console.print(f"  • {winner['strategy']}: Sharpe {winner['avg_oos_sharpe']:.2f}")
    
    if not winners:
        console.print("  [yellow]No strategies passed acceptance gates.[/yellow]")
    
    console.print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hyper-creative strategies")
    parser.add_argument("--quick", action="store_true", help="Quick test (3 symbols, 6 months)")
    parser.add_argument("--full", action="store_true", help="Full test (50 symbols, 12 months)")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2], help="Tier level (1 or 2)")
    
    args = parser.parse_args()
    
    asyncio.run(main(args))
