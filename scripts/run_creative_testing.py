#!/usr/bin/env python3
"""
Creative Strategy Testing Script.

Runs the full creative strategy testing pipeline:
- Phase 1: Quick validation (3 symbols, 6 months)
- Phase 2: Deep validation (19 symbols, 12 months)
- Phase 3: Robustness testing (cost sensitivity, time periods)

Outputs results to notes/creative_testing_results/

Usage:
    python scripts/run_creative_testing.py                    # Run all phases
    python scripts/run_creative_testing.py --phase 1          # Quick validation only
    python scripts/run_creative_testing.py --phase 2          # Deep validation only
    python scripts/run_creative_testing.py --strategies btc_lead_alt_follow weekly_momentum
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
)
from crypto.data.repository import CandleRepository
from crypto.data.alternative_data import FundingRateRepository
from crypto.strategies.registry import strategy_registry
from crypto.strategies.cross_symbol_base import CrossSymbolBaseStrategy
from crypto.strategies.alternative_data_strategies import AlternativeDataBaseStrategy

# Import all strategies
from crypto.strategies import (
    ml, ml_siblings, rule_ensemble, ml_online, ml_cross_asset,
    cross_symbol, calendar, frequency, meta, microstructure,
    alternative_data_strategies,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

# Output directory
OUTPUT_DIR = Path("notes/creative_testing_results")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Creative strategies to test
CREATIVE_STRATEGIES = {
    "cross_symbol": [
        "btc_lead_alt_follow",
        "eth_btc_ratio_reversion",
        "correlation_breakdown",
        "sector_momentum_rotation",
        "btc_volatility_filter",
    ],
    "alternative_data": [
        "funding_rate_fade",
        "funding_rate_carry",
        # "open_interest_divergence",  # Binance OI history API restricted
        # "liquidation_cascade_fade",  # Requires external data
    ],
    "calendar": [
        "weekend_effect",
        "hour_of_day_filter",
        "month_end_rebalancing",
    ],
    "frequency_reduction": [
        "weekly_momentum",
        "signal_confirmation_delay",
        "signal_strength_filter",
    ],
    "meta": [
        "regime_gate",
        "drawdown_pause",
        "strategy_momentum",
    ],
    "microstructure": [
        "volume_breakout_confirmation",
        "volume_divergence",
        "buy_sell_imbalance",
    ],
}

# Phase 1: Quick validation
PHASE1_CONFIG = {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "train_window": 1440,   # 60 days
    "test_window": 336,     # 14 days
    "step_size": 336,       # Bi-weekly
    "days": 180,            # 6 months
    "interval": "1h",
    "pass_threshold": -2.0,  # Lenient for quick filter
    "top_n_to_advance": 8,
}

# Phase 2: Deep validation
PHASE2_CONFIG = {
    "symbols": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        "LTCUSDT", "NEARUSDT", "APTUSDT", "ATOMUSDT", "UNIUSDT",
        "AAVEUSDT", "XLMUSDT", "MATICUSDT", "SHIBUSDT",
    ],
    "train_window": 2160,   # 90 days
    "test_window": 336,     # 14 days
    "step_size": 168,       # Weekly
    "days": 365,            # Full year
    "interval": "1h",
}

# Acceptance gates
ACCEPTANCE_GATES = [
    AcceptanceGate("positive_oos", "oos_sharpe", "gt", 0.5),
    AcceptanceGate("low_degradation", "sharpe_degradation", "lt", 50),
    AcceptanceGate("enough_trades", "oos_total_trades", "gt", 10),
    AcceptanceGate("not_too_many_trades", "oos_total_trades", "lt", 600),  # ~20/month for 6mo
]

# Reference symbols for cross-symbol strategies
REFERENCE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def calculate_buy_and_hold_metrics(candles: pd.DataFrame) -> dict:
    """Calculate buy-and-hold metrics for comparison."""
    if candles.empty or len(candles) < 2:
        return {"sharpe": 0, "return_pct": 0, "max_dd": 0}
    
    close = candles["close"].astype(float)
    returns = close.pct_change().dropna()
    
    if len(returns) < 2:
        return {"sharpe": 0, "return_pct": 0, "max_dd": 0}
    
    total_return = (close.iloc[-1] / close.iloc[0]) - 1
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(24 * 365) if std_return > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_dd = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0
    
    return {
        "sharpe": float(sharpe),
        "return_pct": float(total_return * 100),
        "max_dd": float(max_dd),
    }


def get_all_strategies() -> list[str]:
    """Get all creative strategy names."""
    strategies = []
    for category_strategies in CREATIVE_STRATEGIES.values():
        strategies.extend(category_strategies)
    return strategies


async def load_data(
    repository: CandleRepository,
    symbols: list[str],
    interval: str,
    days: int,
) -> dict[str, pd.DataFrame]:
    """Load candle data for all symbols."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    candles_cache = {}
    for symbol in symbols:
        candles = await repository.get_candles_df(symbol, interval, start, end)
        if not candles.empty:
            candles_cache[symbol] = candles
    
    return candles_cache


async def load_reference_data(
    repository: CandleRepository,
    interval: str,
    days: int,
) -> dict[str, pd.DataFrame]:
    """Load reference data for cross-symbol strategies."""
    return await load_data(repository, REFERENCE_SYMBOLS, interval, days)


async def load_funding_data(
    symbols: list[str],
    days: int,
) -> dict[str, pd.Series]:
    """Load funding rate data for alternative data strategies."""
    from datetime import timezone
    
    funding_repo = FundingRateRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    funding_cache = {}
    for symbol in symbols:
        try:
            df = await funding_repo.get_funding_rates_df(symbol, start, end)
            if not df.empty:
                funding_cache[symbol] = df["funding_rate"]
        except Exception as e:
            logger.warning(f"Could not load funding data for {symbol}: {e}")
    
    return funding_cache


def setup_cross_symbol_strategy(strategy: Any, reference_data: dict[str, pd.DataFrame]) -> None:
    """Set up reference data for cross-symbol strategies."""
    if isinstance(strategy, CrossSymbolBaseStrategy):
        for symbol, candles in reference_data.items():
            strategy.set_reference_data(symbol, candles)


async def run_phase1(strategies: list[str] | None = None) -> tuple[list[WalkForwardResult], dict]:
    """
    Phase 1: Quick Validation.
    
    Fast filtering to identify promising candidates.
    """
    console.print(Panel(
        "[bold blue]Phase 1: Quick Validation[/bold blue]\n\n"
        f"Symbols: {len(PHASE1_CONFIG['symbols'])}\n"
        f"Days: {PHASE1_CONFIG['days']}\n"
        f"Train: {PHASE1_CONFIG['train_window']} bars\n"
        f"Test: {PHASE1_CONFIG['test_window']} bars",
        expand=False,
    ))
    
    config = PHASE1_CONFIG
    strategies_to_test = strategies or get_all_strategies()
    
    # Load data
    repository = CandleRepository()
    console.print("\n[bold]Loading data...[/bold]")
    candles_cache = await load_data(
        repository, config["symbols"], config["interval"], config["days"]
    )
    reference_data = await load_reference_data(
        repository, config["interval"], config["days"]
    )
    
    for symbol, candles in candles_cache.items():
        console.print(f"  ✓ {symbol}: {len(candles)} candles")
    
    console.print("\n[bold]Reference data for cross-symbol strategies:[/bold]")
    for symbol, candles in reference_data.items():
        console.print(f"  ✓ {symbol}: {len(candles)} candles")
    
    # Load funding data for alternative data strategies
    console.print("\n[bold]Loading funding data...[/bold]")
    funding_data = await load_funding_data(config["symbols"], config["days"])
    for symbol, rates in funding_data.items():
        console.print(f"  ✓ {symbol}: {len(rates)} funding rates")
    
    # Calculate buy-and-hold baselines
    console.print("\n[bold]Buy-and-hold baselines:[/bold]")
    bh_metrics = {}
    for symbol, candles in candles_cache.items():
        bh_metrics[symbol] = calculate_buy_and_hold_metrics(candles)
        console.print(
            f"  {symbol}: Sharpe {bh_metrics[symbol]['sharpe']:.2f}, "
            f"Return {bh_metrics[symbol]['return_pct']:.1f}%"
        )
    
    # Verify strategies
    console.print("\n[bold]Strategies to test:[/bold]")
    available = []
    for name in strategies_to_test:
        try:
            strategy_registry.create(name)
            console.print(f"  ✓ {name}")
            available.append(name)
        except Exception as e:
            console.print(f"  ✗ {name}: {e}")
    
    # Create engine
    engine = WalkForwardEngine(
        train_window=config["train_window"],
        test_window=config["test_window"],
        step_size=config["step_size"],
        min_train_samples=1000,
    )
    
    # Run validations
    total = len(available) * len(candles_cache)
    console.print(f"\n[bold]Running {total} validations...[/bold]\n")
    
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Validating...", total=total)
        
        for strategy_name in available:
            for symbol, candles in candles_cache.items():
                progress.update(task, description=f"{strategy_name} on {symbol}")
                
                try:
                    result = engine.run(
                        strategy_name=strategy_name,
                        candles=candles,
                        symbol=symbol,
                        interval=config["interval"],
                        reference_data=reference_data,
                        funding_data=funding_data,
                    )
                    result.check_acceptance(ACCEPTANCE_GATES)
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error running {strategy_name} on {symbol}: {e}")
                
                progress.advance(task)
    
    return results, bh_metrics


async def run_phase2(strategies: list[str]) -> tuple[list[WalkForwardResult], dict]:
    """
    Phase 2: Deep Validation.
    
    Full walk-forward on promising candidates.
    """
    console.print(Panel(
        "[bold blue]Phase 2: Deep Validation[/bold blue]\n\n"
        f"Symbols: {len(PHASE2_CONFIG['symbols'])}\n"
        f"Days: {PHASE2_CONFIG['days']}\n"
        f"Train: {PHASE2_CONFIG['train_window']} bars\n"
        f"Test: {PHASE2_CONFIG['test_window']} bars",
        expand=False,
    ))
    
    config = PHASE2_CONFIG
    
    # Load data
    repository = CandleRepository()
    console.print("\n[bold]Loading data (this may take a while)...[/bold]")
    candles_cache = await load_data(
        repository, config["symbols"], config["interval"], config["days"]
    )
    reference_data = await load_reference_data(
        repository, config["interval"], config["days"]
    )
    
    console.print(f"  Loaded {len(candles_cache)} symbols")
    
    # Load funding data
    funding_data = await load_funding_data(config["symbols"], config["days"])
    console.print(f"  Loaded funding data for {len(funding_data)} symbols")
    
    # Calculate buy-and-hold baselines
    bh_metrics = {}
    for symbol, candles in candles_cache.items():
        bh_metrics[symbol] = calculate_buy_and_hold_metrics(candles)
    
    # Create engine
    engine = WalkForwardEngine(
        train_window=config["train_window"],
        test_window=config["test_window"],
        step_size=config["step_size"],
        min_train_samples=1500,
    )
    
    # Run validations
    total = len(strategies) * len(candles_cache)
    console.print(f"\n[bold]Running {total} validations...[/bold]\n")
    
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Validating...", total=total)
        
        for strategy_name in strategies:
            for symbol, candles in candles_cache.items():
                progress.update(task, description=f"{strategy_name} on {symbol}")
                
                try:
                    strategy = strategy_registry.create(strategy_name)
                    result = engine.run(
                        strategy_name=strategy_name,
                        candles=candles,
                        symbol=symbol,
                        interval=config["interval"],
                        reference_data=reference_data,
                        funding_data=funding_data,
                    )
                    result.check_acceptance(ACCEPTANCE_GATES)
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error running {strategy_name} on {symbol}: {e}")
                
                progress.advance(task)
    
    return results, bh_metrics


def display_results(results: list[WalkForwardResult], bh_metrics: dict, phase: str) -> dict:
    """Display and analyze results."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return {}
    
    console.print(f"\n[bold]═══ {phase} Results ═══[/bold]\n")
    
    # Summary by strategy
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r.strategy_name].append(r)
    
    # Summary table
    table = Table(title=f"{phase} Strategy Summary")
    table.add_column("Strategy", style="yellow")
    table.add_column("Avg OOS Sharpe", justify="right")
    table.add_column("Beats B&H", justify="right")
    table.add_column("Trades/Mo", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Status", justify="center")
    
    summary = {}
    for name, strat_results in sorted(by_strategy.items(), key=lambda x: -sum(r.oos_sharpe for r in x[1]) / len(x[1])):
        avg_oos = sum(r.oos_sharpe for r in strat_results) / len(strat_results)
        avg_trades = sum(r.oos_total_trades for r in strat_results) / len(strat_results)
        trades_per_month = avg_trades / 6  # Assuming 6 months
        passed = sum(1 for r in strat_results if r.passed_validation)
        
        beats_bh = sum(
            1 for r in strat_results
            if r.oos_sharpe > bh_metrics.get(r.symbol, {}).get("sharpe", 0)
        )
        
        pass_rate = passed / len(strat_results) * 100
        bh_rate = beats_bh / len(strat_results) * 100
        
        if avg_oos > 0.5 and pass_rate > 60 and bh_rate > 50:
            status = "[bold green]★ PROMISING[/bold green]"
        elif avg_oos > 0 and bh_rate > 30:
            status = "[yellow]◐ POTENTIAL[/yellow]"
        else:
            status = "[red]✗ FAILS[/red]"
        
        oos_style = "green" if avg_oos > 0.5 else ("yellow" if avg_oos > 0 else "red")
        trades_style = "green" if trades_per_month < 20 else ("yellow" if trades_per_month < 50 else "red")
        
        table.add_row(
            name,
            f"[{oos_style}]{avg_oos:.2f}[/{oos_style}]",
            f"{beats_bh}/{len(strat_results)} ({bh_rate:.0f}%)",
            f"[{trades_style}]{trades_per_month:.1f}[/{trades_style}]",
            f"{passed}/{len(strat_results)} ({pass_rate:.0f}%)",
            status,
        )
        
        summary[name] = {
            "avg_oos_sharpe": avg_oos,
            "beats_bh_rate": bh_rate,
            "trades_per_month": trades_per_month,
            "pass_rate": pass_rate,
            "status": "promising" if avg_oos > 0.5 and pass_rate > 60 and bh_rate > 50 else "potential" if avg_oos > 0 else "fails",
        }
    
    console.print(table)
    
    return summary


def save_results(results: list[WalkForwardResult], bh_metrics: dict, summary: dict, phase: str) -> None:
    """Save results to files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert results to JSON-safe format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        return obj
    
    # Save detailed results
    results_file = OUTPUT_DIR / f"{phase.lower().replace(' ', '_')}_results.json"
    with open(results_file, "w") as f:
        json.dump([make_serializable(r.to_dict()) for r in results], f, indent=2, default=str)
    
    # Save summary
    summary_file = OUTPUT_DIR / f"{phase.lower().replace(' ', '_')}_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "bh_metrics": bh_metrics,
            "strategy_summary": summary,
        }, f, indent=2)
    
    console.print(f"\n[dim]Results saved to {OUTPUT_DIR}[/dim]")


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    console.print("[bold blue]═══ Creative Strategy Testing ═══[/bold blue]\n")
    
    start_time = datetime.now()
    strategies_to_test = args.strategies if args.strategies else None
    
    if args.phase in [None, 1]:
        # Phase 1: Quick Validation
        results, bh_metrics = await run_phase1(strategies_to_test)
        summary = display_results(results, bh_metrics, "Phase 1: Quick Validation")
        save_results(results, bh_metrics, summary, "phase1_quick")
        
        # Select top performers for Phase 2
        promising = [
            name for name, data in summary.items()
            if data["status"] in ["promising", "potential"]
        ][:PHASE1_CONFIG["top_n_to_advance"]]
        
        console.print(f"\n[bold]Strategies advancing to Phase 2:[/bold] {promising}")
    else:
        promising = strategies_to_test or get_all_strategies()[:8]
    
    if args.phase in [None, 2] and promising:
        # Phase 2: Deep Validation
        console.print("\n")
        results, bh_metrics = await run_phase2(promising)
        summary = display_results(results, bh_metrics, "Phase 2: Deep Validation")
        save_results(results, bh_metrics, summary, "phase2_deep")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(f"\n[bold green]Completed in {elapsed:.1f}s[/bold green]")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run creative strategy testing")
    
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Run only specific phase (1=quick, 2=deep, 3=robustness)",
    )
    
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="Specific strategies to test",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
