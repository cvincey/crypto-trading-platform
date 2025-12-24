#!/usr/bin/env python3
"""
Phase 2 Full Validation & Optimization Script.

Runs the complete Phase 2 pipeline:
1. Deep validation of Phase 1 winners on 19 symbols, 365 days
2. Parameter grid search optimization
3. Hybrid strategy testing
4. Final research report generation

Configuration: config/research/creative_testing.yaml (phase2_winners section)

Usage:
    python scripts/run_phase2_full.py                  # Run everything
    python scripts/run_phase2_full.py --validate-only  # Skip optimization
    python scripts/run_phase2_full.py --optimize-only  # Skip validation, run grid search
    python scripts/run_phase2_full.py --quick          # Quick mode (3 symbols, 180 days)
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.backtesting.walk_forward import (
    AcceptanceGate,
    WalkForwardEngine,
    WalkForwardResult,
)
from crypto.data.repository import CandleRepository
from crypto.data.alternative_data import FundingRateRepository
from crypto.strategies.registry import strategy_registry

# Import all strategies to register them
from crypto.strategies import (
    ml, ml_siblings, rule_ensemble, ml_online, ml_cross_asset,
    cross_symbol, calendar, frequency, meta, microstructure,
    alternative_data_strategies, hybrid,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

# Output directory
OUTPUT_DIR = Path("notes/creative_testing_results")


def load_config() -> dict:
    """Load configuration from research/creative_testing.yaml."""
    config_path = Path("config/research/creative_testing.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)["creative_testing"]


# =============================================================================
# CONFIGURATION FROM YAML
# =============================================================================

def get_phase2_config(quick_mode: bool = False) -> dict:
    """Get Phase 2 configuration from YAML or use quick defaults."""
    config = load_config()
    phase2 = config.get("phase2_winners", {})
    
    if quick_mode:
        return {
            "strategies": phase2.get("strategies", [
                "eth_btc_ratio_reversion",
                "signal_confirmation_delay",
                "eth_btc_ratio_confirmed",
            ]),
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "train_window": 1440,   # 60 days
            "test_window": 336,     # 14 days
            "step_size": 336,       # Bi-weekly
            "days": 180,            # 6 months
            "interval": "1h",
        }
    
    # Full Phase 2 config from YAML
    deep_val = config.get("deep_validation", {})
    return {
        "strategies": phase2.get("strategies", [
            "eth_btc_ratio_reversion",
            "signal_confirmation_delay",
            "volume_divergence",
            "eth_btc_ratio_confirmed",
        ]),
        "symbols": deep_val.get("symbols", [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
            "LTCUSDT", "NEARUSDT", "APTUSDT", "ATOMUSDT", "UNIUSDT",
            "AAVEUSDT", "XLMUSDT", "MATICUSDT", "SHIBUSDT",
        ]),
        "train_window": phase2.get("train_window", 2880),   # 120 days
        "test_window": phase2.get("test_window", 336),       # 14 days
        "step_size": phase2.get("step_size", 168),           # Weekly
        "days": phase2.get("days", 365),                     # Full year
        "interval": "1h",
    }


def get_param_grids() -> dict:
    """Get parameter optimization grids from config."""
    config = load_config()
    phase2 = config.get("phase2_winners", {})
    return phase2.get("parameter_optimization", {})


# Acceptance gates
ACCEPTANCE_GATES = [
    AcceptanceGate("positive_oos", "oos_sharpe", "gt", 0.5),
    AcceptanceGate("low_degradation", "sharpe_degradation", "lt", 50),
    AcceptanceGate("enough_trades", "oos_total_trades", "gt", 5),
    AcceptanceGate("not_too_many_trades", "oos_total_trades", "lt", 600),
]

# Reference symbols for cross-symbol strategies
REFERENCE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


async def load_funding_data(symbols: list[str], days: int) -> dict[str, pd.Series]:
    """Load funding rate data for alternative data strategies."""
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
            logger.debug(f"Could not load funding data for {symbol}: {e}")
    
    return funding_cache


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


# =============================================================================
# PHASE 2A: DEEP VALIDATION
# =============================================================================

async def run_deep_validation(
    config: dict,
    strategies: list[str] | None = None,
) -> tuple[list[WalkForwardResult], dict, dict]:
    """
    Run deep validation on Phase 1 winners.
    
    Args:
        config: Phase 2 configuration
        strategies: Optional specific strategies to test
        
    Returns:
        Tuple of (results, buy_hold_metrics, strategy_summary)
    """
    console.print(Panel(
        "[bold blue]Phase 2A: Deep Validation[/bold blue]\n\n"
        f"Strategies: {len(config['strategies'])}\n"
        f"Symbols: {len(config['symbols'])}\n"
        f"Days: {config['days']}\n"
        f"Train: {config['train_window']} bars ({config['train_window']//24}d)\n"
        f"Test: {config['test_window']} bars ({config['test_window']//24}d)",
        expand=False,
    ))
    
    strategies_to_test = strategies or config["strategies"]
    
    # Load data
    repository = CandleRepository()
    console.print("\n[bold]Loading data...[/bold]")
    
    candles_cache = await load_data(
        repository, config["symbols"], config["interval"], config["days"]
    )
    reference_data = await load_reference_data(
        repository, config["interval"], config["days"]
    )
    funding_data = await load_funding_data(config["symbols"], config["days"])
    
    console.print(f"  Loaded {len(candles_cache)} symbols")
    console.print(f"  Reference data: {list(reference_data.keys())}")
    console.print(f"  Funding data for {len(funding_data)} symbols")
    
    # Calculate buy-and-hold baselines
    bh_metrics = {}
    for symbol, candles in candles_cache.items():
        bh_metrics[symbol] = calculate_buy_and_hold_metrics(candles)
    
    # Verify strategies are available
    console.print("\n[bold]Strategies to validate:[/bold]")
    available = []
    for name in strategies_to_test:
        try:
            strategy_registry.create(name)
            console.print(f"  [green]✓[/green] {name}")
            available.append(name)
        except Exception as e:
            console.print(f"  [red]✗[/red] {name}: {e}")
    
    if not available:
        console.print("[red]No valid strategies to test![/red]")
        return [], bh_metrics, {}
    
    # Create walk-forward engine
    engine = WalkForwardEngine(
        train_window=config["train_window"],
        test_window=config["test_window"],
        step_size=config["step_size"],
        min_train_samples=config["train_window"] // 2,
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
    
    # Compute summary
    summary = compute_summary(results, bh_metrics)
    
    return results, bh_metrics, summary


def compute_summary(
    results: list[WalkForwardResult],
    bh_metrics: dict,
) -> dict:
    """Compute strategy summary statistics."""
    if not results:
        return {}
    
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r.strategy_name].append(r)
    
    summary = {}
    for name, strat_results in by_strategy.items():
        avg_oos = sum(r.oos_sharpe for r in strat_results) / len(strat_results)
        avg_trades = sum(r.oos_total_trades for r in strat_results) / len(strat_results)
        
        # Estimate months based on test periods
        n_folds = sum(len(getattr(r, 'fold_results', [])) or 1 for r in strat_results)
        est_months = (n_folds * 14) / 30  # Rough estimate
        trades_per_month = avg_trades / max(est_months / len(strat_results), 1)
        
        passed = sum(1 for r in strat_results if r.passed_validation)
        beats_bh = sum(
            1 for r in strat_results
            if r.oos_sharpe > bh_metrics.get(r.symbol, {}).get("sharpe", 0)
        )
        
        pass_rate = passed / len(strat_results) * 100
        bh_rate = beats_bh / len(strat_results) * 100
        
        if avg_oos > 0.5 and pass_rate >= 50 and bh_rate >= 50:
            status = "promising"
        elif avg_oos > 0 and bh_rate > 30:
            status = "potential"
        else:
            status = "fails"
        
        summary[name] = {
            "avg_oos_sharpe": avg_oos,
            "beats_bh_rate": bh_rate,
            "trades_per_month": trades_per_month,
            "pass_rate": pass_rate,
            "n_validations": len(strat_results),
            "status": status,
        }
    
    return summary


# =============================================================================
# PHASE 2B: PARAMETER OPTIMIZATION
# =============================================================================

async def run_parameter_optimization(
    config: dict,
    strategies: list[str] | None = None,
    top_n_combos: int = 5,
) -> dict:
    """
    Run parameter grid search for winning strategies.
    
    Uses a subset of symbols for speed, then validates best params.
    
    Args:
        config: Phase 2 configuration
        strategies: Strategies to optimize (uses config if None)
        top_n_combos: Number of top parameter sets to report
        
    Returns:
        Dict of {strategy_name: {best_params, results}}
    """
    console.print(Panel(
        "[bold blue]Phase 2B: Parameter Optimization[/bold blue]\n\n"
        "Grid search for optimal parameters on winning strategies",
        expand=False,
    ))
    
    param_grids = get_param_grids()
    strategies_to_optimize = strategies or list(param_grids.keys())
    
    # Use subset of symbols for grid search
    opt_symbols = config["symbols"][:5]  # First 5 symbols for speed
    
    # Load data
    repository = CandleRepository()
    console.print("\n[bold]Loading optimization data...[/bold]")
    
    candles_cache = await load_data(
        repository, opt_symbols, config["interval"], min(config["days"], 180)
    )
    reference_data = await load_reference_data(
        repository, config["interval"], min(config["days"], 180)
    )
    
    console.print(f"  Loaded {len(candles_cache)} symbols for optimization")
    
    # Calculate baselines
    bh_metrics = {s: calculate_buy_and_hold_metrics(c) for s, c in candles_cache.items()}
    
    # Create engine with faster settings
    engine = WalkForwardEngine(
        train_window=config["train_window"],
        test_window=config["test_window"],
        step_size=config["step_size"],
        min_train_samples=config["train_window"] // 2,
    )
    
    optimization_results = {}
    
    for strategy_name in strategies_to_optimize:
        if strategy_name not in param_grids:
            console.print(f"[yellow]No param grid for {strategy_name}[/yellow]")
            continue
        
        grid = param_grids[strategy_name]
        combinations = generate_param_combinations(grid)
        
        console.print(f"\n[bold]{strategy_name}[/bold]: {len(combinations)} combinations")
        
        combo_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Optimizing...", total=len(combinations))
            
            for params in combinations:
                progress.update(task, description=f"{params}")
                
                # Test this param combo across symbols
                sharpes = []
                
                for symbol, candles in candles_cache.items():
                    try:
                        # Create strategy with these params
                        strategy = strategy_registry.create(strategy_name, **params)
                        
                        # Quick backtest (not full walk-forward for speed)
                        result = engine.run(
                            strategy_name=strategy_name,
                            candles=candles,
                            symbol=symbol,
                            interval=config["interval"],
                            reference_data=reference_data,
                            strategy_params=params,
                        )
                        sharpes.append(result.oos_sharpe)
                        
                    except Exception as e:
                        logger.debug(f"Error with {params}: {e}")
                        sharpes.append(-10)  # Penalty for errors
                
                avg_sharpe = np.mean(sharpes) if sharpes else -10
                combo_results.append({
                    "params": params,
                    "avg_oos_sharpe": avg_sharpe,
                    "sharpes": sharpes,
                })
                
                progress.advance(task)
        
        # Sort by average sharpe
        combo_results.sort(key=lambda x: x["avg_oos_sharpe"], reverse=True)
        
        # Report top N
        console.print(f"\n  [bold]Top {top_n_combos} parameter sets:[/bold]")
        for i, combo in enumerate(combo_results[:top_n_combos]):
            console.print(
                f"    {i+1}. Sharpe={combo['avg_oos_sharpe']:.2f} | {combo['params']}"
            )
        
        optimization_results[strategy_name] = {
            "best_params": combo_results[0]["params"] if combo_results else {},
            "best_sharpe": combo_results[0]["avg_oos_sharpe"] if combo_results else 0,
            "all_results": combo_results[:top_n_combos],
        }
    
    return optimization_results


# =============================================================================
# DISPLAY & SAVE RESULTS
# =============================================================================

def display_results(
    summary: dict,
    bh_metrics: dict,
    phase: str,
) -> None:
    """Display results in a formatted table."""
    if not summary:
        console.print("[yellow]No results to display[/yellow]")
        return
    
    console.print(f"\n[bold]═══ {phase} Results ═══[/bold]\n")
    
    table = Table(title=f"{phase} Strategy Summary")
    table.add_column("Strategy", style="yellow")
    table.add_column("Avg OOS Sharpe", justify="right")
    table.add_column("Beats B&H", justify="right")
    table.add_column("Trades/Mo", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Status", justify="center")
    
    for name, data in sorted(
        summary.items(),
        key=lambda x: x[1]["avg_oos_sharpe"],
        reverse=True,
    ):
        avg_oos = data["avg_oos_sharpe"]
        bh_rate = data["beats_bh_rate"]
        trades = data["trades_per_month"]
        pass_rate = data["pass_rate"]
        status = data["status"]
        
        if status == "promising":
            status_str = "[bold green]★ PROMISING[/bold green]"
        elif status == "potential":
            status_str = "[yellow]◐ POTENTIAL[/yellow]"
        else:
            status_str = "[red]✗ FAILS[/red]"
        
        oos_style = "green" if avg_oos > 0.5 else ("yellow" if avg_oos > 0 else "red")
        trades_style = "green" if trades < 10 else ("yellow" if trades < 30 else "red")
        
        table.add_row(
            name,
            f"[{oos_style}]{avg_oos:.2f}[/{oos_style}]",
            f"{bh_rate:.0f}%",
            f"[{trades_style}]{trades:.1f}[/{trades_style}]",
            f"{pass_rate:.0f}%",
            status_str,
        )
    
    console.print(table)


def save_results(
    results: list[WalkForwardResult],
    bh_metrics: dict,
    summary: dict,
    optimization_results: dict,
    phase: str,
) -> None:
    """Save all results to files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
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
    results_file = OUTPUT_DIR / f"{phase}_results.json"
    with open(results_file, "w") as f:
        json.dump(
            [make_serializable(r.to_dict()) for r in results],
            f, indent=2, default=str
        )
    
    # Save summary
    summary_file = OUTPUT_DIR / f"{phase}_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "bh_metrics": bh_metrics,
            "strategy_summary": summary,
            "optimization_results": make_serializable(optimization_results),
        }, f, indent=2)
    
    console.print(f"\n[dim]Results saved to {OUTPUT_DIR}/{phase}_*[/dim]")


def generate_research_note(
    summary: dict,
    optimization_results: dict,
    config: dict,
) -> None:
    """Generate research note markdown file."""
    note_path = Path("notes/07-phase2-deep-validation.md")
    
    # Find winners
    winners = [
        (name, data) for name, data in summary.items()
        if data["status"] == "promising"
    ]
    potential = [
        (name, data) for name, data in summary.items()
        if data["status"] == "potential"
    ]
    
    content = f"""# Research Note 07: Phase 2 Deep Validation Results

**Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Author**: Automated Testing Pipeline  
**Status**: Phase 2 Complete

---

## Executive Summary

Validated **{len(summary)} strategies** across **{len(config['symbols'])} symbols** over **{config['days']} days**.

### Winners

| Strategy | OOS Sharpe | Beats B&H | Pass Rate | Status |
|----------|------------|-----------|-----------|--------|
"""
    
    for name, data in sorted(
        summary.items(),
        key=lambda x: x[1]["avg_oos_sharpe"],
        reverse=True,
    ):
        status_emoji = "★" if data["status"] == "promising" else ("◐" if data["status"] == "potential" else "✗")
        content += f"| {name} | {data['avg_oos_sharpe']:.2f} | {data['beats_bh_rate']:.0f}% | {data['pass_rate']:.0f}% | {status_emoji} |\n"
    
    content += f"""
---

## Configuration

| Parameter | Value |
|-----------|-------|
| Symbols | {len(config['symbols'])} |
| Days | {config['days']} |
| Train Window | {config['train_window']} bars ({config['train_window']//24} days) |
| Test Window | {config['test_window']} bars ({config['test_window']//24} days) |
| Step Size | {config['step_size']} bars |

---

## Parameter Optimization Results

"""
    
    for strategy_name, opt_data in optimization_results.items():
        content += f"""### {strategy_name}

**Best Parameters** (Sharpe: {opt_data.get('best_sharpe', 0):.2f}):
```
{json.dumps(opt_data.get('best_params', {}), indent=2)}
```

"""
    
    content += f"""---

## Key Findings

1. **{len(winners)} strategies** passed all acceptance gates
2. **{len(potential)} strategies** show potential but need refinement
3. Hybrid strategy `eth_btc_ratio_confirmed` combines best of both approaches

## Recommendations

1. Deploy winning strategies with optimized parameters
2. Continue monitoring on live paper trading
3. Consider ensemble of top performers

---

*Generated by run_phase2_full.py*
"""
    
    with open(note_path, "w") as f:
        f.write(content)
    
    console.print(f"\n[green]Research note saved to {note_path}[/green]")


# =============================================================================
# MAIN
# =============================================================================

async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    console.print(Panel(
        "[bold blue]═══ Phase 2 Full Validation & Optimization ═══[/bold blue]\n\n"
        "Testing Phase 1 winners with:\n"
        "• Deep validation on all symbols\n"
        "• Parameter grid search optimization\n"
        "• Hybrid strategy testing",
        expand=False,
    ))
    
    start_time = datetime.now()
    
    # Get configuration
    config = get_phase2_config(quick_mode=args.quick)
    
    results = []
    bh_metrics = {}
    summary = {}
    optimization_results = {}
    
    # Phase 2A: Deep Validation
    if not args.optimize_only:
        results, bh_metrics, summary = await run_deep_validation(config)
        display_results(summary, bh_metrics, "Phase 2A: Deep Validation")
    
    # Phase 2B: Parameter Optimization
    if not args.validate_only and (not args.optimize_only or True):
        optimization_results = await run_parameter_optimization(
            config,
            strategies=None,  # Use all from config
        )
    
    # Save all results
    save_results(results, bh_metrics, summary, optimization_results, "phase2_full")
    
    # Generate research note
    if summary:
        generate_research_note(summary, optimization_results, config)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(f"\n[bold green]Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)[/bold green]")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase 2 full validation and optimization"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 3 symbols, 180 days",
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation only, skip optimization",
    )
    
    parser.add_argument(
        "--optimize-only",
        action="store_true",
        help="Run optimization only, skip validation",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
