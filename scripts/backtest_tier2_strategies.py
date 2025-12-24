#!/usr/bin/env python3
"""
Backtest Tier 2 strategies that require alternative data.

Now that we have:
- Fear & Greed Index: 5+ years
- Long/Short Ratio: 32 days
- Macro Indicators (DXY, VIX, SPX, GOLD): 1 year
- BTC Dominance: 1 year

Usage:
    python scripts/backtest_tier2_strategies.py
    python scripts/backtest_tier2_strategies.py --category sentiment
    python scripts/backtest_tier2_strategies.py --category macro
    python scripts/backtest_tier2_strategies.py --category positioning
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from crypto.data.repository import CandleRepository
from crypto.data.alternative_data import (
    FearGreedRepository,
    LongShortRatioRepository,
    BTCDominanceRepository,
    MacroIndicatorRepository,
)
from crypto.strategies.registry import strategy_registry
from crypto.strategies import sentiment, positioning, macro
from crypto.backtesting.walk_forward import WalkForwardEngine

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

# Strategy configurations by category
STRATEGIES = {
    "sentiment": [
        {
            "name": "fear_greed_divergence",
            "params": {
                "fg_lookback": 168,  # 7 days
                "price_lookback": 168,
                "divergence_threshold": 10,
                "hold_period": 48,
            },
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data_needs": ["fear_greed"],
        },
        {
            "name": "fear_greed_extreme_fade",
            "params": {
                "extreme_fear": 25,
                "extreme_greed": 75,
                "hold_period": 72,
            },
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data_needs": ["fear_greed"],
        },
        {
            "name": "fear_greed_contrarian",
            "params": {
                "buy_threshold": 30,
                "sell_threshold": 70,
                "smoothing_period": 3,
                "hold_period": 48,
            },
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data_needs": ["fear_greed"],
        },
    ],
    "positioning": [
        {
            "name": "long_short_ratio_fade",
            "params": {
                "extreme_long": 0.6,
                "extreme_short": 0.4,
                "lookback": 48,
                "hold_period": 24,
            },
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "data_needs": ["long_short"],
        },
        {
            "name": "long_short_momentum",
            "params": {
                "momentum_period": 24,
                "momentum_threshold": 0.05,
                "hold_period": 24,
            },
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "data_needs": ["long_short"],
        },
    ],
    "macro": [
        {
            "name": "macro_correlation",
            "params": {
                "dxy_lookback": 120,
                "vix_threshold": 20,
                "dxy_change_threshold": 0.01,
                "hold_period": 48,
            },
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data_needs": ["dxy", "vix"],
        },
        {
            "name": "dxy_inverse",
            "params": {
                "dxy_ma_period": 120,
                "dxy_momentum_period": 24,
                "hold_period": 48,
            },
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data_needs": ["dxy"],
        },
        {
            "name": "dominance_momentum",
            "params": {
                "momentum_period": 168,
                "momentum_threshold": 0.02,
                "hold_period": 72,
            },
            "symbols": ["ETHUSDT", "SOLUSDT", "AVAXUSDT"],
            "data_needs": ["btc_dominance"],
        },
    ],
}


async def load_alternative_data() -> dict[str, Any]:
    """Load all alternative data from database."""
    console.print("[dim]Loading alternative data...[/dim]")
    
    from datetime import timedelta
    
    data = {}
    end = datetime.now(tz=timezone.utc)
    
    # Fear & Greed (5 years)
    try:
        repo = FearGreedRepository()
        start = end - timedelta(days=365 * 5)
        fg_data = await repo.get_fear_greed(start=start, end=end)
        if fg_data:
            df = pd.DataFrame([
                {"timestamp": d.timestamp, "value": d.value}
                for d in fg_data
            ])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            data["fear_greed"] = df["value"]
            console.print(f"  ✓ Fear & Greed: {len(df)} records")
    except Exception as e:
        console.print(f"  ✗ Fear & Greed: {e}")
    
    # Long/Short Ratio (90 days)
    try:
        repo = LongShortRatioRepository()
        start = end - timedelta(days=90)
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            ls_data = await repo.get_long_short_ratio(symbol, start=start, end=end)
            if ls_data:
                df = pd.DataFrame([
                    {"timestamp": d.timestamp, "ratio": float(d.long_short_ratio)}
                    for d in ls_data if d.ratio_type == "accounts"
                ])
                if not df.empty:
                    df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)
                    data[f"long_short_{symbol}"] = df["ratio"]
                    console.print(f"  ✓ L/S Ratio {symbol}: {len(df)} records")
    except Exception as e:
        console.print(f"  ✗ Long/Short Ratio: {e}")
    
    # BTC Dominance (1 year)
    try:
        repo = BTCDominanceRepository()
        start = end - timedelta(days=365)
        dom_data = await repo.get_btc_dominance(start=start, end=end)
        if dom_data:
            df = pd.DataFrame([
                {"timestamp": d.timestamp, "dominance": float(d.btc_dominance)}
                for d in dom_data
            ])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            data["btc_dominance"] = df["dominance"]
            console.print(f"  ✓ BTC Dominance: {len(df)} records")
    except Exception as e:
        console.print(f"  ✗ BTC Dominance: {e}")
    
    # Macro Indicators (1 year)
    try:
        repo = MacroIndicatorRepository()
        start = end - timedelta(days=365)
        for indicator in ["DXY", "VIX", "SPX", "GOLD"]:
            macro_data = await repo.get_macro_indicator(indicator, start=start, end=end)
            if macro_data:
                df = pd.DataFrame([
                    {"timestamp": d.timestamp, "value": float(d.value)}
                    for d in macro_data
                ])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                data[indicator.lower()] = df["value"]
                console.print(f"  ✓ {indicator}: {len(df)} records")
    except Exception as e:
        console.print(f"  ✗ Macro Indicators: {e}")
    
    return data


async def run_backtest(
    strategy_config: dict,
    alt_data: dict,
    days: int = 180,
) -> dict[str, Any]:
    """Run walk-forward backtest for a strategy."""
    from datetime import timedelta
    
    name = strategy_config["name"]
    params = strategy_config["params"]
    symbols = strategy_config["symbols"]
    
    results = []
    candle_repo = CandleRepository()
    
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    
    for symbol in symbols:
        try:
            # Load candles
            candles = await candle_repo.get_candles_df(
                symbol=symbol,
                interval="1h",
                start=start,
                end=end,
            )
            
            if candles is None or candles.empty or len(candles) < 500:
                continue
            
            # Create engine
            engine = WalkForwardEngine(
                train_window=720,   # 30 days
                test_window=168,    # 7 days
                step_size=168,      # Weekly
            )
            
            # Prepare reference data with alternative data
            reference_data = {}
            
            # Add fear greed data
            if "fear_greed" in strategy_config["data_needs"]:
                if "fear_greed" in alt_data:
                    reference_data["fear_greed"] = alt_data["fear_greed"]
            
            # Add long/short data
            if "long_short" in strategy_config["data_needs"]:
                key = f"long_short_{symbol}"
                if key in alt_data:
                    reference_data["long_short"] = alt_data[key]
            
            # Add macro data
            if "dxy" in strategy_config["data_needs"]:
                if "dxy" in alt_data:
                    reference_data["dxy"] = alt_data["dxy"]
            if "vix" in strategy_config["data_needs"]:
                if "vix" in alt_data:
                    reference_data["vix"] = alt_data["vix"]
            if "btc_dominance" in strategy_config["data_needs"]:
                if "btc_dominance" in alt_data:
                    reference_data["btc_dominance"] = alt_data["btc_dominance"]
            
            # Run walk-forward
            result = engine.run(
                strategy_name=name,
                candles=candles,
                symbol=symbol,
                interval="1h",
                strategy_params=params,
                reference_data=reference_data,
            )
            
            results.append({
                "symbol": symbol,
                "oos_sharpe": result.oos_sharpe,
                "is_sharpe": result.is_sharpe,
                "trades": result.oos_total_trades or 0,
                "passed": result.passed_validation,
            })
            
        except Exception as e:
            logger.warning(f"Error backtesting {name} on {symbol}: {e}")
            results.append({
                "symbol": symbol,
                "oos_sharpe": None,
                "is_sharpe": None,
                "trades": 0,
                "passed": False,
                "error": str(e),
            })
    
    # Aggregate results
    valid_results = [r for r in results if r["oos_sharpe"] is not None]
    
    if valid_results:
        avg_sharpe = sum(r["oos_sharpe"] for r in valid_results) / len(valid_results)
        total_trades = sum(r["trades"] for r in valid_results)
        pass_rate = sum(1 for r in valid_results if r["passed"]) / len(valid_results)
    else:
        avg_sharpe = 0
        total_trades = 0
        pass_rate = 0
    
    return {
        "name": name,
        "avg_oos_sharpe": avg_sharpe,
        "total_trades": total_trades,
        "pass_rate": pass_rate,
        "results": results,
    }


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    categories = [args.category] if args.category else list(STRATEGIES.keys())
    days = args.days
    
    console.print(Panel(
        f"[bold blue]Tier 2 Strategy Backtesting[/bold blue]\n\n"
        f"Categories: {', '.join(categories)}\n"
        f"Days: {days}",
        expand=False,
    ))
    
    # Load alternative data
    alt_data = await load_alternative_data()
    
    if not alt_data:
        console.print("[red]No alternative data available. Run ingestion first.[/red]")
        return
    
    # Run backtests
    all_results = []
    
    for category in categories:
        console.print(f"\n[bold]Testing {category.upper()} strategies...[/bold]")
        
        strategies = STRATEGIES.get(category, [])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for config in strategies:
                task = progress.add_task(f"  {config['name']}...", total=1)
                
                result = await run_backtest(config, alt_data, days)
                result["category"] = category
                all_results.append(result)
                
                progress.update(task, advance=1, completed=1)
    
    # Display results
    console.print("\n" + "=" * 70)
    console.print("[bold]BACKTEST RESULTS[/bold]")
    console.print("=" * 70 + "\n")
    
    table = Table(title="Tier 2 Strategy Performance")
    table.add_column("Category", style="dim")
    table.add_column("Strategy")
    table.add_column("OOS Sharpe", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Status")
    
    for result in sorted(all_results, key=lambda x: x["avg_oos_sharpe"], reverse=True):
        sharpe = result["avg_oos_sharpe"]
        trades = result["total_trades"]
        pass_rate = result["pass_rate"]
        
        if sharpe > 0.5 and trades > 5:
            status = "[green]✓ VIABLE[/green]"
        elif sharpe > 0 and trades > 0:
            status = "[yellow]○ MARGINAL[/yellow]"
        else:
            status = "[red]✗ FAILED[/red]"
        
        table.add_row(
            result["category"],
            result["name"],
            f"{sharpe:.2f}" if sharpe else "-",
            str(trades),
            f"{pass_rate:.0%}",
            status,
        )
    
    console.print(table)
    
    # Summary
    viable = [r for r in all_results if r["avg_oos_sharpe"] > 0.5 and r["total_trades"] > 5]
    
    if viable:
        console.print(f"\n[green]Viable strategies ({len(viable)}):[/green]")
        for r in viable:
            console.print(f"  • {r['name']}: Sharpe {r['avg_oos_sharpe']:.2f}, {r['total_trades']} trades")
    else:
        console.print("\n[yellow]No strategies met viability threshold (Sharpe > 0.5, trades > 5)[/yellow]")
    
    console.print("\n[bold]Backtesting complete![/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Tier 2 strategies")
    parser.add_argument(
        "--category",
        choices=["sentiment", "positioning", "macro"],
        help="Strategy category to test",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Days of history to use (default: 180)",
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))
