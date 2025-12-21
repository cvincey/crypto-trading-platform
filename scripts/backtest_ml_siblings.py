#!/usr/bin/env python3
"""
Backtest ML classifier siblings against the original ml_classifier.

Compares all ML sibling strategies across multiple tickers to identify
the most promising improvements for further development.
"""

import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.backtesting.engine import BacktestEngine, BacktestResult
from crypto.config.settings import get_settings
from crypto.data.database import get_async_session
from crypto.data.models import CandleModel
from crypto.data.repository import CandleRepository
from crypto.strategies.registry import strategy_registry

# Import strategies to trigger registration
from crypto.strategies import ml
from crypto.strategies import ml_siblings  # Import the new sibling strategies

from sqlalchemy import func, select

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()

# ML strategies to compare
ML_STRATEGIES = [
    "ml_classifier",           # Original baseline
    "ml_classifier_v2",        # Extended features + probability thresholds (fixed)
    "ml_ensemble_voting",      # Multiple models combined
    "ml_classifier_v3",        # Longer horizon + feature engineering
    "ml_classifier_xgb",       # XGBoost with class weights (best performer)
    "ml_classifier_v4",        # Adaptive thresholds
    "ml_classifier_hybrid",    # NEW: Best of XGB + Ensemble + Adaptive
    "ml_classifier_conservative",  # NEW: Low drawdown focus
]


async def get_top_tickers(n: int = 10, interval: str = "1h") -> list[dict[str, Any]]:
    """Get top N tickers by volume from database."""
    async with get_async_session() as session:
        stmt = (
            select(
                CandleModel.symbol,
                func.count(CandleModel.id).label("candle_count"),
                func.sum(CandleModel.volume).label("total_volume"),
                func.min(CandleModel.open_time).label("first_candle"),
                func.max(CandleModel.open_time).label("last_candle"),
            )
            .where(CandleModel.interval == interval)
            .group_by(CandleModel.symbol)
            .order_by(func.sum(CandleModel.volume).desc())
            .limit(n)
        )

        result = await session.execute(stmt)
        rows = result.all()

        return [
            {
                "symbol": row.symbol,
                "candle_count": row.candle_count,
                "total_volume": float(row.total_volume) if row.total_volume else 0,
                "first_candle": row.first_candle,
                "last_candle": row.last_candle,
            }
            for row in rows
        ]


async def run_single_backtest(
    symbol: str,
    strategy_name: str,
    interval: str,
    start: datetime,
    end: datetime,
    initial_capital: Decimal = Decimal("10000"),
) -> BacktestResult | None:
    """Run a single backtest."""
    try:
        repository = CandleRepository()
        candles = await repository.get_candles_df(symbol, interval, start, end)

        if candles.empty or len(candles) < 100:
            logger.warning(f"Insufficient data for {symbol} ({len(candles)} candles)")
            return None

        # Create strategy with default params
        strategy = strategy_registry.create(strategy_name)
        engine = BacktestEngine(initial_capital=initial_capital)

        return engine.run(strategy, candles, symbol, interval)

    except Exception as e:
        logger.error(f"Failed backtest {strategy_name} on {symbol}: {e}")
        return None


async def run_comparison(
    tickers: list[dict],
    strategies: list[str],
    interval: str = "1h",
    days: int = 90,
) -> list[BacktestResult]:
    """Run all backtests for comparison."""
    all_results: list[BacktestResult] = []
    total_combinations = len(tickers) * len(strategies)

    console.print(f"\n[bold]Running {total_combinations} backtests[/bold]")
    console.print(f"  Tickers:    {len(tickers)}")
    console.print(f"  Strategies: {len(strategies)}")
    console.print(f"  Interval:   {interval}")
    console.print(f"  Period:     {days} days\n")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Backtesting ML strategies...", total=total_combinations)

        for ticker in tickers:
            symbol = ticker["symbol"]
            ticker_start = ticker.get("first_candle", start)
            ticker_end = ticker.get("last_candle", end)
            actual_start = max(start, ticker_start) if ticker_start else start
            actual_end = min(end, ticker_end) if ticker_end else end

            for strategy_name in strategies:
                progress.update(
                    task,
                    description=f"[cyan]{symbol}[/cyan] / [yellow]{strategy_name}[/yellow]",
                )

                result = await run_single_backtest(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    interval=interval,
                    start=actual_start,
                    end=actual_end,
                )

                if result:
                    all_results.append(result)

                progress.advance(task)

    return all_results


def analyze_results(results: list[BacktestResult]) -> dict:
    """Analyze results and compute statistics."""
    by_strategy = defaultdict(list)
    by_ticker = defaultdict(list)

    for r in results:
        by_strategy[r.strategy_name].append(r)
        by_ticker[r.symbol].append(r)

    analysis = {
        "strategy_stats": {},
        "ticker_stats": {},
        "head_to_head": {},
        "rankings": {},
    }

    # Strategy-level statistics
    for strategy_name, strategy_results in by_strategy.items():
        returns = [r.metrics.total_return_pct for r in strategy_results]
        sharpes = [r.metrics.sharpe_ratio for r in strategy_results]
        win_rates = [r.metrics.win_rate for r in strategy_results]
        drawdowns = [r.metrics.max_drawdown for r in strategy_results]

        analysis["strategy_stats"][strategy_name] = {
            "avg_return": sum(returns) / len(returns),
            "avg_sharpe": sum(sharpes) / len(sharpes),
            "avg_win_rate": sum(win_rates) / len(win_rates),
            "avg_max_dd": sum(drawdowns) / len(drawdowns),
            "best_return": max(returns),
            "worst_return": min(returns),
            "positive_pct": sum(1 for r in returns if r > 0) / len(returns) * 100,
            "num_tests": len(strategy_results),
        }

    # Head-to-head vs original
    original_results = {r.symbol: r for r in by_strategy.get("ml_classifier", [])}
    
    for strategy_name, strategy_results in by_strategy.items():
        if strategy_name == "ml_classifier":
            continue
            
        wins = 0
        losses = 0
        ties = 0
        improvements = []
        
        for r in strategy_results:
            orig = original_results.get(r.symbol)
            if orig:
                if r.metrics.sharpe_ratio > orig.metrics.sharpe_ratio * 1.05:
                    wins += 1
                    improvements.append(
                        (r.symbol, r.metrics.sharpe_ratio - orig.metrics.sharpe_ratio)
                    )
                elif r.metrics.sharpe_ratio < orig.metrics.sharpe_ratio * 0.95:
                    losses += 1
                else:
                    ties += 1
        
        analysis["head_to_head"][strategy_name] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate_vs_original": wins / (wins + losses + ties) * 100 if (wins + losses + ties) > 0 else 0,
            "top_improvements": sorted(improvements, key=lambda x: x[1], reverse=True)[:3],
        }

    # Overall ranking by average Sharpe
    ranking = sorted(
        analysis["strategy_stats"].items(),
        key=lambda x: x[1]["avg_sharpe"],
        reverse=True,
    )
    analysis["rankings"]["by_sharpe"] = [(name, stats["avg_sharpe"]) for name, stats in ranking]

    # Ranking by consistency (% positive)
    ranking = sorted(
        analysis["strategy_stats"].items(),
        key=lambda x: x[1]["positive_pct"],
        reverse=True,
    )
    analysis["rankings"]["by_consistency"] = [(name, stats["positive_pct"]) for name, stats in ranking]

    return analysis


def display_results(results: list[BacktestResult], analysis: dict) -> None:
    """Display formatted results."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Strategy comparison table
    console.print("\n")
    console.print(Panel("[bold blue]ML Strategy Comparison Results[/bold blue]", expand=False))

    table = Table(title="Strategy Performance Summary")
    table.add_column("Strategy", style="yellow")
    table.add_column("Avg Return %", justify="right")
    table.add_column("Avg Sharpe", justify="right")
    table.add_column("Avg Win Rate %", justify="right")
    table.add_column("Avg Max DD %", justify="right")
    table.add_column("Positive %", justify="right")
    table.add_column("vs Original", justify="center")

    stats = analysis["strategy_stats"]
    h2h = analysis["head_to_head"]

    for strategy_name in ML_STRATEGIES:
        if strategy_name not in stats:
            continue
            
        s = stats[strategy_name]
        return_style = "green" if s["avg_return"] > 0 else "red"
        sharpe_style = "green" if s["avg_sharpe"] > 1 else ("yellow" if s["avg_sharpe"] > 0 else "red")

        # Head-to-head indicator
        if strategy_name == "ml_classifier":
            vs_orig = "[dim]baseline[/dim]"
        elif strategy_name in h2h:
            h = h2h[strategy_name]
            if h["wins"] > h["losses"]:
                vs_orig = f"[green]↑ {h['wins']}W/{h['losses']}L[/green]"
            elif h["losses"] > h["wins"]:
                vs_orig = f"[red]↓ {h['wins']}W/{h['losses']}L[/red]"
            else:
                vs_orig = f"[yellow]= {h['wins']}W/{h['losses']}L[/yellow]"
        else:
            vs_orig = "N/A"

        table.add_row(
            strategy_name,
            f"[{return_style}]{s['avg_return']:+.2f}[/{return_style}]",
            f"[{sharpe_style}]{s['avg_sharpe']:.2f}[/{sharpe_style}]",
            f"{s['avg_win_rate']:.1f}",
            f"{s['avg_max_dd']:.2f}",
            f"{s['positive_pct']:.0f}%",
            vs_orig,
        )

    console.print(table)

    # Rankings
    console.print("\n[bold]Rankings[/bold]")
    
    rank_table = Table(show_header=True)
    rank_table.add_column("Rank", style="dim", width=4)
    rank_table.add_column("By Sharpe Ratio", style="cyan")
    rank_table.add_column("By Consistency", style="magenta")

    by_sharpe = analysis["rankings"]["by_sharpe"]
    by_consistency = analysis["rankings"]["by_consistency"]

    for i in range(len(by_sharpe)):
        rank_table.add_row(
            f"#{i+1}",
            f"{by_sharpe[i][0]} ({by_sharpe[i][1]:.2f})",
            f"{by_consistency[i][0]} ({by_consistency[i][1]:.0f}%)",
        )

    console.print(rank_table)

    # Top individual results
    console.print("\n[bold]Top 10 Individual Results[/bold]")
    
    top_results = sorted(results, key=lambda r: r.metrics.sharpe_ratio, reverse=True)[:10]
    
    top_table = Table()
    top_table.add_column("Rank", style="dim", width=4)
    top_table.add_column("Symbol", style="cyan")
    top_table.add_column("Strategy", style="yellow")
    top_table.add_column("Return %", justify="right")
    top_table.add_column("Sharpe", justify="right")
    top_table.add_column("Win Rate %", justify="right")
    top_table.add_column("Trades", justify="right")

    for i, r in enumerate(top_results, 1):
        m = r.metrics
        return_style = "green" if m.total_return_pct > 0 else "red"
        
        # Highlight if it's a sibling strategy
        strategy_style = "bold yellow" if r.strategy_name != "ml_classifier" else "yellow"
        
        top_table.add_row(
            str(i),
            r.symbol,
            f"[{strategy_style}]{r.strategy_name}[/{strategy_style}]",
            f"[{return_style}]{m.total_return_pct:+.2f}[/{return_style}]",
            f"{m.sharpe_ratio:.2f}",
            f"{m.win_rate:.1f}",
            str(m.total_trades),
        )

    console.print(top_table)

    # Head-to-head details
    console.print("\n[bold]Head-to-Head vs Original ml_classifier[/bold]")
    
    h2h_table = Table()
    h2h_table.add_column("Strategy", style="yellow")
    h2h_table.add_column("Wins", justify="right", style="green")
    h2h_table.add_column("Losses", justify="right", style="red")
    h2h_table.add_column("Ties", justify="right")
    h2h_table.add_column("Win Rate %", justify="right")
    h2h_table.add_column("Best Improvements", style="dim")

    for strategy_name in ML_STRATEGIES:
        if strategy_name == "ml_classifier" or strategy_name not in h2h:
            continue
            
        h = h2h[strategy_name]
        improvements = ", ".join([f"{sym} (+{imp:.2f})" for sym, imp in h["top_improvements"]])
        
        h2h_table.add_row(
            strategy_name,
            str(h["wins"]),
            str(h["losses"]),
            str(h["ties"]),
            f"{h['win_rate_vs_original']:.1f}",
            improvements or "None",
        )

    console.print(h2h_table)

    # Recommendations
    console.print("\n")
    
    # Find best performers
    best_sharpe = analysis["rankings"]["by_sharpe"][0]
    best_consistency = analysis["rankings"]["by_consistency"][0]
    
    # Find best sibling (not original)
    best_sibling = None
    for name, sharpe in analysis["rankings"]["by_sharpe"]:
        if name != "ml_classifier":
            best_sibling = (name, sharpe)
            break

    recommendations = []
    
    if best_sharpe[0] != "ml_classifier":
        recommendations.append(
            f"✅ [bold green]{best_sharpe[0]}[/bold green] outperforms the original with "
            f"Sharpe {best_sharpe[1]:.2f}"
        )
    else:
        recommendations.append(
            f"⚠️  Original [bold]ml_classifier[/bold] still leads with Sharpe {best_sharpe[1]:.2f}"
        )

    if best_sibling:
        h = h2h.get(best_sibling[0], {})
        if h.get("wins", 0) > h.get("losses", 0):
            recommendations.append(
                f"🚀 [bold cyan]{best_sibling[0]}[/bold cyan] is the most promising sibling - "
                f"beats original on {h['wins']}/{h['wins'] + h['losses'] + h['ties']} tickers"
            )

    # Check for any strategy with high consistency
    for name, pct in analysis["rankings"]["by_consistency"]:
        if pct >= 80 and name != "ml_classifier":
            recommendations.append(
                f"📊 [bold magenta]{name}[/bold magenta] shows {pct:.0f}% positive rate - very consistent"
            )
            break

    console.print(Panel(
        "\n".join(recommendations) if recommendations else "No clear recommendations",
        title="[bold]Recommendations for Further Development[/bold]",
        border_style="green",
    ))


def save_results(results: list[BacktestResult], analysis: dict, output_dir: Path) -> None:
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_data = [
        {
            "symbol": r.symbol,
            "strategy": r.strategy_name,
            "total_return_pct": r.metrics.total_return_pct,
            "sharpe_ratio": r.metrics.sharpe_ratio,
            "max_drawdown": r.metrics.max_drawdown,
            "win_rate": r.metrics.win_rate,
            "total_trades": r.metrics.total_trades,
        }
        for r in results
    ]
    
    with open(output_dir / "ml_siblings_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Save analysis
    # Convert tuples to lists for JSON serialization
    analysis_serializable = {
        "strategy_stats": analysis["strategy_stats"],
        "head_to_head": {
            k: {**v, "top_improvements": [[sym, imp] for sym, imp in v["top_improvements"]]}
            for k, v in analysis["head_to_head"].items()
        },
        "rankings": {
            k: [[name, val] for name, val in v]
            for k, v in analysis["rankings"].items()
        },
    }
    
    with open(output_dir / "ml_siblings_analysis.json", "w") as f:
        json.dump(analysis_serializable, f, indent=2)
    
    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")


async def main():
    """Main entry point."""
    console.print(
        Panel(
            "[bold blue]ML Classifier Siblings Comparison[/bold blue]\n"
            "Comparing improved ML strategies against the original ml_classifier",
            expand=False,
        )
    )

    # Configuration
    interval = "1h"
    days = 90
    top_n = 10  # Use top 10 tickers for faster iteration

    # Get tickers
    console.print(f"\n[bold]Fetching top {top_n} tickers from database...[/bold]")
    tickers = await get_top_tickers(n=top_n, interval=interval)

    if not tickers:
        console.print("[red]No tickers found in database![/red]")
        console.print("Run data ingestion first: crypto ingest fetch --symbol BTCUSDT --days 90")
        return

    # Display tickers
    console.print(f"\n[green]Found {len(tickers)} tickers:[/green]")
    for t in tickers:
        console.print(f"  • {t['symbol']} ({t['candle_count']} candles)")

    # Verify strategies are registered
    console.print(f"\n[bold]ML Strategies to compare:[/bold]")
    available = []
    for strategy_name in ML_STRATEGIES:
        try:
            strategy_registry.create(strategy_name)
            console.print(f"  ✓ {strategy_name}")
            available.append(strategy_name)
        except Exception as e:
            console.print(f"  ✗ {strategy_name} - [red]{e}[/red]")

    if len(available) < 2:
        console.print("[red]Not enough strategies available for comparison![/red]")
        return

    # Run backtests
    results = await run_comparison(
        tickers=tickers,
        strategies=available,
        interval=interval,
        days=days,
    )

    if not results:
        console.print("[red]No backtest results![/red]")
        return

    console.print(f"\n[bold green]Completed {len(results)} backtests[/bold green]")

    # Analyze and display
    analysis = analyze_results(results)
    display_results(results, analysis)

    # Save results
    output_dir = Path(__file__).parent.parent / "notes" / "backtest_results"
    save_results(results, analysis, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
