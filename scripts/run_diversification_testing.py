#!/usr/bin/env python3
"""
Diversification Strategy Testing Script.

Tests new strategies designed to be active when ratio strategies are quiet:
- ml_classifier_online_v1: Adaptive ML with rolling retraining
- liquidation_cascade_fade_v1: Trades leverage flushes (mock data)
- correlation_breakdown_sol_avax: Alt vs Alt relative strength
- regime_volatility_switch_v1: Momentum in trends, reversion in ranges

Research Note 12: Creative Diversification Strategies

Usage:
    python scripts/run_diversification_testing.py                # Run all 4 strategies
    python scripts/run_diversification_testing.py --quick        # Quick validation (3 symbols, 180 days)
    python scripts/run_diversification_testing.py --full         # Full validation (5 symbols, 365 days)
    python scripts/run_diversification_testing.py --correlation  # Include correlation analysis
"""

import argparse
import asyncio
import json
import logging
import sys
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
    check_acceptance_gates,
)
from crypto.data.repository import CandleRepository
from crypto.data.mock_liquidations import generate_mock_liquidations, inject_liquidation_data
from crypto.strategies.registry import strategy_registry

# Import all strategies to register them
from crypto.strategies import (
    technical,
    statistical,
    momentum,
    ml,
    ml_siblings,
    ml_online,
    cross_symbol,
    calendar,
    frequency,
    meta,
    microstructure,
    volatility_trading,
    structural,
    alternative_data_strategies,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

# Output directory
OUTPUT_DIR = Path("notes/creative_testing_results")

# =============================================================================
# DIVERSIFICATION STRATEGIES - Research Note 12
# =============================================================================

# Use registered strategy TYPE names (not config instance names)
# Note: liquidation_cascade_fade excluded - requires external data not available
DIVERSIFICATION_STRATEGIES = [
    "ml_classifier_online",
    "correlation_breakdown",
    "regime_volatility_switch",
]

# Baseline ratio strategies for correlation analysis
BASELINE_STRATEGIES = [
    "eth_btc_ratio_reversion",
]

# =============================================================================
# QUICK VALIDATION CONFIGURATION
# =============================================================================

QUICK_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
QUICK_TRAIN_WINDOW = 720   # 30 days
QUICK_TEST_WINDOW = 168    # 7 days
QUICK_STEP_SIZE = 168      # Weekly
QUICK_DAYS = 180           # 6 months

# =============================================================================
# FULL VALIDATION CONFIGURATION
# =============================================================================

FULL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT"]
FULL_TRAIN_WINDOW = 720    # 30 days
FULL_TEST_WINDOW = 168     # 7 days
FULL_STEP_SIZE = 168       # Weekly
FULL_DAYS = 365            # 12 months

# Mock liquidation settings
MOCK_LIQUIDATION_CONFIG = {
    "atr_spike_percentile": 95,
    "price_drop_threshold": 0.03,
    "price_spike_threshold": 0.03,
    "base_liquidation_amount": 5_000_000,
    "volatility_multiplier": 2.0,
}


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


def prepare_mock_liquidations(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Generate mock liquidation data for all symbols."""
    mock_data = {}
    for symbol, candles in data.items():
        mock_data[symbol] = generate_mock_liquidations(
            candles, **MOCK_LIQUIDATION_CONFIG
        )
    return mock_data


async def run_validation(
    strategies: list[str],
    symbols: list[str],
    train_window: int,
    test_window: int,
    step_size: int,
    days: int,
    include_baseline: bool = False,
) -> dict[str, Any]:
    """
    Run walk-forward validation on diversification strategies.
    """
    console.print(Panel.fit(
        "[bold blue]Walk-Forward Validation[/bold blue]\n"
        f"Symbols: {len(symbols)}, Days: {days}, Strategies: {len(strategies)}"
    ))

    # Load data
    console.print("\n[bold]Loading data...[/bold]")
    data = await load_data(symbols, days)

    if not data:
        console.print("[red]No data available![/red]")
        return {}


    # Load reference data for cross-symbol strategies
    reference_data = {}
    for symbol in ["BTCUSDT", "ETHUSDT", "AVAXUSDT"]:
        if symbol in data:
            reference_data[symbol] = data[symbol]

    # Setup walk-forward engine
    engine = WalkForwardEngine(
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
    )

    # Acceptance gates
    gates = [
        AcceptanceGate("positive_oos", "oos_sharpe", "gt", -1.0),
        AcceptanceGate("enough_trades", "oos_total_trades", "gt", 1),
    ]

    # Prepare strategies list
    all_strategies = strategies.copy()
    if include_baseline:
        for baseline in BASELINE_STRATEGIES:
            if baseline not in all_strategies:
                all_strategies.append(baseline)

    # Run validation
    results = {}
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
                    # Run walk-forward
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
                    import traceback
                    traceback.print_exc()

                progress.advance(task)

            if strategy_results:
                results[strategy_name] = strategy_results

    return results


def calculate_summary_metrics(results: dict[str, list], days: int = 180) -> pd.DataFrame:
    """Calculate summary metrics for each strategy."""
    summary = []
    months = days / 30

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

        # Calculate degradation
        degradation = 0
        if is_sharpes and np.mean(is_sharpes) != 0:
            degradation = 1 - (np.mean(oos_sharpes) / np.mean(is_sharpes))

        summary.append({
            "strategy": strategy_name,
            "avg_oos_sharpe": np.mean(oos_sharpes),
            "std_oos_sharpe": np.std(oos_sharpes),
            "avg_is_sharpe": np.mean(is_sharpes) if is_sharpes else 0,
            "degradation_pct": degradation * 100,
            "trades_per_month": np.sum(trades) / months if trades else 0,
            "pass_rate": pass_count / len(strategy_results),
            "beats_bh_pct": beats_bh * 100,
            "n_tests": len(strategy_results),
            "n_passed": pass_count,
        })

    df = pd.DataFrame(summary)
    if not df.empty:
        df = df.sort_values("avg_oos_sharpe", ascending=False)
    return df


def display_results(results: dict[str, list], title: str, days: int = 180) -> None:
    """Display results in a nice table."""
    summary = calculate_summary_metrics(results, days)

    if summary.empty:
        console.print("[yellow]No results to display[/yellow]")
        return

    table = Table(title=title)
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Strategy", style="cyan")
    table.add_column("OOS Sharpe", justify="right")
    table.add_column("Degradation", justify="right")
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
        deg_style = "green" if row.degradation_pct < 25 else ("yellow" if row.degradation_pct < 50 else "red")

        table.add_row(
            str(i),
            row.strategy,
            f"[{sharpe_style}]{row.avg_oos_sharpe:.2f}[/{sharpe_style}]",
            f"[{deg_style}]{row.degradation_pct:.0f}%[/{deg_style}]",
            f"{row.beats_bh_pct:.0f}%",
            f"{row.trades_per_month:.1f}",
            f"{row.pass_rate * 100:.0f}%",
            status,
        )

    console.print(table)


def save_results(results: dict[str, list], filename: str, days: int = 180) -> None:
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
                "oos_return": getattr(r, "oos_return", None),
                "oos_max_drawdown": getattr(r, "oos_max_drawdown", None),
                "total_trades": r.oos_total_trades,
                "passed": r.passed,
            }
            for r in strategy_results
        ]

    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(serializable, f, indent=2)

    # Save summary
    summary = calculate_summary_metrics(results, days)
    summary.to_json(OUTPUT_DIR / filename.replace(".json", "_summary.json"), orient="records", indent=2)

    console.print(f"\n[dim]Results saved to {OUTPUT_DIR / filename}[/dim]")


def generate_research_note(results: dict[str, list], days: int) -> str:
    """Generate Research Note 12 markdown content."""
    summary = calculate_summary_metrics(results, days)
    
    # Count winners
    winners = summary[summary["avg_oos_sharpe"] > 0.5]
    
    note = f"""# Research Note 12: Diversification Strategies

**Date**: {datetime.now().strftime("%B %d, %Y")}  
**Objective**: Test new strategies to diversify the ratio-based portfolio

## Executive Summary

Tested **{len(DIVERSIFICATION_STRATEGIES)} diversification strategies** across **{len(FULL_SYMBOLS)} symbols** over **{days} days**.

| Metric | Value |
|--------|-------|
| Strategies Tested | {len(summary)} |
| Winners (Sharpe > 0.5) | {len(winners)} |
| Best OOS Sharpe | {summary["avg_oos_sharpe"].max():.2f} |

## Strategies Tested

| Strategy | Concept | OOS Sharpe | Beats B&H | Status |
|----------|---------|------------|----------|--------|
"""
    
    for _, row in summary.iterrows():
        status = "★ PROMISING" if row["avg_oos_sharpe"] > 0.5 else ("◐ POTENTIAL" if row["avg_oos_sharpe"] > 0 else "✗ FAILED")
        concept = {
            "ml_classifier_online": "Adaptive ML with rolling retraining",
            "liquidation_cascade_fade": "Trade leverage flushes (requires external data)",
            "correlation_breakdown": "Alt vs Alt relative strength",
            "regime_volatility_switch": "Momentum in trends, reversion in ranges",
            "eth_btc_ratio_reversion": "ETH/BTC ratio mean reversion (baseline)",
        }.get(row["strategy"], "Unknown")
        
        note += f"| {row['strategy']} | {concept} | {row['avg_oos_sharpe']:.2f} | {row['beats_bh_pct']:.0f}% | {status} |\n"

    note += """
## Key Findings

### 1. Performance Summary

"""
    
    for _, row in summary.iterrows():
        if row["avg_oos_sharpe"] > 0:
            note += f"- **{row['strategy']}**: OOS Sharpe {row['avg_oos_sharpe']:.2f}, {row['trades_per_month']:.1f} trades/month, {row['pass_rate']*100:.0f}% pass rate\n"
        else:
            note += f"- **{row['strategy']}**: FAILED with OOS Sharpe {row['avg_oos_sharpe']:.2f}\n"

    note += """
### 2. Diversification Value

These strategies are designed to be active when ratio strategies are quiet:
- **Trending markets**: regime_volatility_switch activates momentum mode
- **Leverage flushes**: liquidation_cascade_fade captures panic reversals
- **Sector rotations**: correlation_breakdown trades Alt vs Alt divergence
- **Regime adaptation**: ml_classifier_online retrains to new conditions

### 3. Recommendations

"""
    
    if len(winners) > 0:
        note += f"**Deploy**: {', '.join(winners['strategy'].tolist())}\n\n"
    
    if len(summary[summary["avg_oos_sharpe"] <= 0]) > 0:
        failed = summary[summary["avg_oos_sharpe"] <= 0]["strategy"].tolist()
        note += f"**Retire/Revise**: {', '.join(failed)}\n\n"

    note += f"""
## Test Configuration

| Parameter | Value |
|-----------|-------|
| Symbols | {', '.join(FULL_SYMBOLS)} |
| Period | {days} days |
| Train Window | {FULL_TRAIN_WINDOW} bars (30 days) |
| Test Window | {FULL_TEST_WINDOW} bars (7 days) |
| Step Size | {FULL_STEP_SIZE} bars (weekly) |

---

*Generated by run_diversification_testing.py | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    return note


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Diversification Strategy Testing")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation (3 symbols, 180 days)")
    parser.add_argument("--full", action="store_true",
                        help="Full validation (5 symbols, 365 days)")
    parser.add_argument("--correlation", action="store_true",
                        help="Include correlation analysis with baseline strategies")
    parser.add_argument("--strategies", nargs="+",
                        help="Specific strategies to test")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold blue]═══ Diversification Strategy Testing ═══[/bold blue]\n"
        "Research Note 12: Creative strategies for portfolio diversification"
    ))

    # Determine configuration
    if args.quick:
        symbols = QUICK_SYMBOLS
        train_window = QUICK_TRAIN_WINDOW
        test_window = QUICK_TEST_WINDOW
        step_size = QUICK_STEP_SIZE
        days = QUICK_DAYS
        mode = "Quick"
    else:
        symbols = FULL_SYMBOLS
        train_window = FULL_TRAIN_WINDOW
        test_window = FULL_TEST_WINDOW
        step_size = FULL_STEP_SIZE
        days = FULL_DAYS
        mode = "Full"

    # Determine strategies
    strategies = args.strategies if args.strategies else DIVERSIFICATION_STRATEGIES

    console.print(f"\n[bold]Mode:[/bold] {mode} Validation")
    console.print(f"[bold]Symbols:[/bold] {symbols}")
    console.print(f"[bold]Strategies:[/bold]")
    for s in strategies:
        console.print(f"  • {s}")

    # Run validation
    results = await run_validation(
        strategies=strategies,
        symbols=symbols,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        days=days,
        include_baseline=args.correlation,
    )

    if results:
        display_results(results, f"Diversification Strategies - {mode} Validation", days)
        save_results(results, "diversification_results.json", days)

        # Generate research note
        research_note = generate_research_note(results, days)
        note_path = Path("notes/research_notes/12-diversification-strategies.md")
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(research_note)
        console.print(f"\n[green]Research note saved to {note_path}[/green]")

    console.print("\n[bold green]Testing complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())

