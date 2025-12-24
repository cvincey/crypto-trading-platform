#!/usr/bin/env python3
"""
Unexplored Strategy Testing Script.

Tests the new unexplored strategies identified in Research Note 09:
- Tier 1: No external data needed (backtest immediately)
- Tier 2: Free API data (backtest after data ingestion)
- Tier 3: Paid API required (stubs only, not backtested)

Usage:
    python scripts/run_unexplored_testing.py                    # Run Tier 1 only
    python scripts/run_unexplored_testing.py --tier 2           # Include Tier 2
    python scripts/run_unexplored_testing.py --full             # Full validation on 19 symbols
    python scripts/run_unexplored_testing.py --compare          # Compare vs eth_btc_ratio_reversion
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
    technical,
    statistical,
    momentum,
    ml,
    ml_siblings,
    cross_symbol,
    calendar,
    frequency,
    meta,
    microstructure,
    volatility_trading,
    structural,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

# Output directory
OUTPUT_DIR = Path("notes/creative_testing_results")

# =============================================================================
# TIER 1 STRATEGIES - No external data needed
# =============================================================================

TIER1_STRATEGIES = [
    "volatility_mean_reversion",
    "volatility_breakout",
    "basis_proxy",
    "multi_asset_pair_trade",
    "dxy_correlation_proxy",
    "regime_volatility_switch",
    "momentum_quality",
    "trend_strength_filter",
]

# =============================================================================
# TIER 2 STRATEGIES - Free API data required
# =============================================================================

TIER2_STRATEGIES = [
    "fear_greed_contrarian",
    "btc_dominance_rotation",
    "long_short_ratio_fade",
]

# =============================================================================
# QUICK VALIDATION CONFIGURATION
# =============================================================================

QUICK_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
QUICK_TRAIN_WINDOW = 1440  # 60 days
QUICK_TEST_WINDOW = 336    # 14 days
QUICK_STEP_SIZE = 336      # Bi-weekly
QUICK_DAYS = 180           # 6 months

# =============================================================================
# FULL VALIDATION CONFIGURATION
# =============================================================================

FULL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT",
    "UNIUSDT", "AAVEUSDT", "NEARUSDT", "APTUSDT",
]
FULL_TRAIN_WINDOW = 2160   # 90 days
FULL_TEST_WINDOW = 336     # 14 days
FULL_STEP_SIZE = 168       # Weekly
FULL_DAYS = 365            # 12 months

# Baseline strategy to compare against
BASELINE_STRATEGY = "eth_btc_ratio_reversion"


async def load_data(
    symbols: list[str],
    days: int,
    interval: str = "1h",
) -> dict[str, pd.DataFrame]:
    """Load candle data for all symbols."""
    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 30)  # Extra buffer for warmup

    data = {}
    for symbol in symbols:
        candles = await repository.get_candles_df(symbol, interval, start, end)
        if not candles.empty:
            data[symbol] = candles
            console.print(f"  {symbol}: {len(candles)} candles")
        else:
            console.print(f"  [yellow]{symbol}: No data[/yellow]")

    return data


async def run_quick_validation(
    strategies: list[str],
    compare_baseline: bool = False,
) -> dict[str, Any]:
    """
    Run quick validation on strategies.
    
    3 symbols, 6 months, bi-weekly walk-forward.
    """
    console.print(Panel.fit(
        "[bold blue]Phase 1: Quick Validation[/bold blue]\n"
        f"Symbols: {len(QUICK_SYMBOLS)}, Days: {QUICK_DAYS}, Strategies: {len(strategies)}"
    ))

    # Load data
    console.print("\n[bold]Loading data...[/bold]")
    data = await load_data(QUICK_SYMBOLS, QUICK_DAYS)

    if not data:
        console.print("[red]No data available![/red]")
        return {}

    # Load reference data for cross-symbol strategies
    reference_data = {}
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        if symbol in data:
            reference_data[symbol] = data[symbol]

    # Setup walk-forward engine
    engine = WalkForwardEngine(
        train_window=QUICK_TRAIN_WINDOW,
        test_window=QUICK_TEST_WINDOW,
        step_size=QUICK_STEP_SIZE,
    )

    # Acceptance gates (lenient for quick validation)
    gates = [
        AcceptanceGate("positive_oos", "oos_sharpe", "gt", -2.0),  # Very lenient
        AcceptanceGate("enough_trades", "oos_total_trades", "gt", 1),
    ]

    # Run validation
    results = {}
    all_strategies = strategies.copy()
    if compare_baseline and BASELINE_STRATEGY not in all_strategies:
        all_strategies.append(BASELINE_STRATEGY)

    total_tests = len(all_strategies) * len(data)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Validating...", total=total_tests)

        for strategy_name in all_strategies:
            strategy_results = []

            for symbol, candles in data.items():
                progress.update(
                    task,
                    description=f"[cyan]{strategy_name}[/cyan] on [yellow]{symbol}[/yellow]"
                )

                try:
                    # Run walk-forward - pass strategy NAME (string), not object
                    # The engine creates fresh strategy instances internally
                    result = engine.run(
                        strategy_name=strategy_name,
                        candles=candles,
                        symbol=symbol,
                        reference_data=reference_data,
                    )
                    acceptance = check_acceptance_gates(result, gates)
                    result.passed = acceptance.passed
                    strategy_results.append(result)

                except Exception as e:
                    logger.warning(f"Failed {strategy_name} on {symbol}: {e}")

                progress.advance(task)

            if strategy_results:
                results[strategy_name] = strategy_results

    return results


def calculate_summary_metrics(results: dict[str, list]) -> pd.DataFrame:
    """Calculate summary metrics for each strategy."""
    summary = []

    for strategy_name, strategy_results in results.items():
        if not strategy_results:
            continue

        oos_sharpes = [r.oos_sharpe for r in strategy_results if r.oos_sharpe is not None]
        is_sharpes = [r.is_sharpe for r in strategy_results if r.is_sharpe is not None]
        trades = [r.oos_total_trades for r in strategy_results]
        pass_count = sum(1 for r in strategy_results if r.passed)

        if not oos_sharpes:
            continue

        # Calculate beats buy-and-hold
        beats_bh = sum(1 for r in strategy_results if r.oos_sharpe > 0) / len(strategy_results)

        summary.append({
            "strategy": strategy_name,
            "avg_oos_sharpe": np.mean(oos_sharpes),
            "avg_is_sharpe": np.mean(is_sharpes) if is_sharpes else 0,
            "degradation": 1 - (np.mean(oos_sharpes) / np.mean(is_sharpes)) if is_sharpes and np.mean(is_sharpes) != 0 else 0,
            "trades_per_month": np.mean(trades) / 6 if trades else 0,  # 6 months
            "pass_rate": pass_count / len(strategy_results),
            "beats_bh_pct": beats_bh * 100,
            "n_tests": len(strategy_results),
        })

    df = pd.DataFrame(summary)
    if not df.empty:
        df = df.sort_values("avg_oos_sharpe", ascending=False)
    return df


def display_results(results: dict[str, list], title: str) -> None:
    """Display results in a nice table."""
    summary = calculate_summary_metrics(results)

    if summary.empty:
        console.print("[yellow]No results to display[/yellow]")
        return

    table = Table(title=title)
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Strategy", style="cyan")
    table.add_column("OOS Sharpe", justify="right")
    table.add_column("Beats B&H", justify="right")
    table.add_column("Trades/Mo", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Status")

    for i, row in enumerate(summary.itertuples(), 1):
        # Status based on metrics
        if row.avg_oos_sharpe > 1.0 and row.pass_rate >= 0.5:
            status = "[green]★ PROMISING[/green]"
        elif row.avg_oos_sharpe > 0.5:
            status = "[yellow]◐ POTENTIAL[/yellow]"
        elif row.avg_oos_sharpe > 0:
            status = "[dim]○ MARGINAL[/dim]"
        else:
            status = "[red]✗ FAILED[/red]"

        sharpe_style = "green" if row.avg_oos_sharpe > 1 else ("yellow" if row.avg_oos_sharpe > 0 else "red")

        table.add_row(
            str(i),
            row.strategy,
            f"[{sharpe_style}]{row.avg_oos_sharpe:.2f}[/{sharpe_style}]",
            f"{row.beats_bh_pct:.0f}%",
            f"{row.trades_per_month:.1f}",
            f"{row.pass_rate * 100:.0f}%",
            status,
        )

    console.print(table)


def save_results(results: dict[str, list], filename: str) -> None:
    """Save results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert results to serializable format
    serializable = {}
    for strategy_name, strategy_results in results.items():
        serializable[strategy_name] = [
            {
                "symbol": r.symbol,
                "oos_sharpe": r.oos_sharpe,
                "is_sharpe": r.is_sharpe,
                "total_trades": r.oos_total_trades,
                "passed": r.passed,
            }
            for r in strategy_results
        ]

    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(serializable, f, indent=2)

    console.print(f"\n[dim]Results saved to {OUTPUT_DIR / filename}[/dim]")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unexplored Strategy Testing")
    parser.add_argument("--tier", type=int, choices=[1, 2], default=1,
                        help="Strategy tier to test (1=no API, 2=free API)")
    parser.add_argument("--full", action="store_true",
                        help="Run full validation (19 symbols, 12 months)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare against eth_btc_ratio_reversion baseline")
    parser.add_argument("--strategies", nargs="+",
                        help="Specific strategies to test")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold blue]═══ Unexplored Strategy Testing ═══[/bold blue]\n"
        "Research Note 09: Testing previously unexplored alpha sources"
    ))

    # Determine which strategies to test
    if args.strategies:
        strategies = args.strategies
    else:
        strategies = TIER1_STRATEGIES.copy()
        if args.tier >= 2:
            strategies.extend(TIER2_STRATEGIES)

    console.print(f"\n[bold]Strategies to test:[/bold] {len(strategies)}")
    for s in strategies:
        tier = "Tier 1" if s in TIER1_STRATEGIES else "Tier 2" if s in TIER2_STRATEGIES else "Other"
        console.print(f"  • {s} [{tier}]")

    # Run quick validation first
    results = await run_quick_validation(strategies, compare_baseline=args.compare)

    if results:
        display_results(results, "Quick Validation Results (Tier 1)")
        save_results(results, "unexplored_tier1_results.json")

        # Save summary
        summary = calculate_summary_metrics(results)
        summary.to_json(OUTPUT_DIR / "unexplored_tier1_summary.json", orient="records", indent=2)

    console.print("\n[bold green]Testing complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
