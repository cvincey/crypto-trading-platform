#!/usr/bin/env python3
"""
Strategy validation pipeline with acceptance gates.

Runs walk-forward validation on all configured strategies and applies
acceptance gates to determine which strategies are production-ready.

Configuration: config/optimization.yaml

Output:
- notes/validation_results/validation_report.json - Full results
- notes/validation_results/validation_summary.md - Human-readable summary
"""

import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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

from crypto.backtesting.walk_forward import (
    AcceptanceGate,
    WalkForwardEngine,
    WalkForwardResult,
    compare_walk_forward_results,
)
from crypto.config.settings import get_settings
from crypto.data.repository import CandleRepository
from crypto.strategies.registry import strategy_registry

# Import strategies to trigger registration
from crypto.strategies import ml
from crypto.strategies import ml_siblings
from crypto.strategies import rule_ensemble
from crypto.strategies import ml_online
from crypto.strategies import ml_cross_asset

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


def build_acceptance_gates(settings) -> list[AcceptanceGate]:
    """Build acceptance gates from config."""
    gates_config = settings.optimization.optimization.validation_gates
    
    if not gates_config.enabled:
        return []
    
    gates = []
    for criterion in gates_config.criteria:
        gates.append(AcceptanceGate.from_config(criterion))
    
    return gates


async def run_validation() -> tuple[list[WalkForwardResult], list[AcceptanceGate]]:
    """Run walk-forward validation with acceptance gates."""
    settings = get_settings()
    wf_config = settings.optimization.optimization.walk_forward
    gates_config = settings.optimization.optimization.validation_gates

    if not wf_config.enabled:
        console.print("[yellow]Walk-forward validation is disabled in config[/yellow]")
        return [], []

    console.print(Panel(
        f"[bold blue]Strategy Validation Pipeline[/bold blue]\n"
        f"Train: {wf_config.train_window} bars ({wf_config.train_window // 24} days), "
        f"Test: {wf_config.test_window} bars ({wf_config.test_window // 24} days)\n"
        f"Data: {wf_config.days} days, Symbols: {len(wf_config.symbols)}",
        expand=False,
    ))

    # Build acceptance gates
    gates = build_acceptance_gates(settings)
    if gates:
        console.print(f"\n[bold]Acceptance Gates ({len(gates)}):[/bold]")
        for gate in gates:
            console.print(f"  • {gate.name}: {gate.metric} {gate.operator} {gate.threshold}")
    else:
        console.print("\n[yellow]No acceptance gates configured[/yellow]")

    # Load data
    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=wf_config.days)

    console.print(f"\n[bold]Loading data for {len(wf_config.symbols)} symbols...[/bold]")
    
    candles_cache: dict[str, Any] = {}
    for symbol in wf_config.symbols:
        candles = await repository.get_candles_df(
            symbol, wf_config.interval, start, end
        )
        if not candles.empty:
            candles_cache[symbol] = candles
            console.print(f"  ✓ {symbol}: {len(candles)} candles")
        else:
            console.print(f"  ✗ {symbol}: [red]No data[/red]")

    if not candles_cache:
        console.print("[red]No data available![/red]")
        return [], gates

    # Verify strategies
    console.print(f"\n[bold]Strategies to validate:[/bold]")
    available_strategies = []
    for strategy_name in wf_config.strategies:
        try:
            strategy_registry.create(strategy_name)
            console.print(f"  ✓ {strategy_name}")
            available_strategies.append(strategy_name)
        except Exception as e:
            console.print(f"  ✗ {strategy_name}: [red]{e}[/red]")

    if not available_strategies:
        console.print("[red]No strategies available![/red]")
        return [], gates

    # Create engine
    engine = WalkForwardEngine(
        train_window=wf_config.train_window,
        test_window=wf_config.test_window,
        step_size=wf_config.step_size,
        min_train_samples=wf_config.min_train_samples,
    )

    # Run walk-forward
    total_runs = len(available_strategies) * len(candles_cache)
    all_results: list[WalkForwardResult] = []

    console.print(f"\n[bold]Running {total_runs} validations...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Validating...", total=total_runs)

        for strategy_name in available_strategies:
            for symbol, candles in candles_cache.items():
                progress.update(
                    task,
                    description=f"[cyan]{symbol}[/cyan] / [yellow]{strategy_name}[/yellow]",
                )

                try:
                    result = engine.run(
                        strategy_name=strategy_name,
                        candles=candles,
                        symbol=symbol,
                        interval=wf_config.interval,
                    )
                    
                    # Apply acceptance gates
                    if gates:
                        result.check_acceptance(
                            gates,
                            reject_all_negative_folds=gates_config.reject_all_negative_folds,
                        )
                    
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Failed: {strategy_name} on {symbol}: {e}")

                progress.advance(task)

    return all_results, gates


def display_validation_results(
    results: list[WalkForwardResult],
    gates: list[AcceptanceGate],
) -> dict[str, Any]:
    """Display validation results and return summary."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return {}

    console.print("\n")
    console.print(Panel("[bold blue]Validation Results[/bold blue]", expand=False))

    # Group by strategy
    by_strategy: dict[str, list[WalkForwardResult]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy_name].append(r)

    # Strategy summary table
    table = Table(title="Strategy Validation Summary")
    table.add_column("Strategy", style="yellow")
    table.add_column("Symbols Tested", justify="center")
    table.add_column("Passed", justify="center")
    table.add_column("Failed", justify="center")
    table.add_column("Pass Rate", justify="center")
    table.add_column("Avg OOS Sharpe", justify="right")
    table.add_column("Avg Degradation", justify="right")
    table.add_column("Status", justify="center")

    strategy_summaries = []
    
    for strategy_name, strategy_results in by_strategy.items():
        passed = sum(1 for r in strategy_results if r.passed_validation)
        failed = len(strategy_results) - passed
        pass_rate = passed / len(strategy_results) * 100 if strategy_results else 0
        
        avg_oos_sharpe = sum(r.oos_sharpe for r in strategy_results) / len(strategy_results)
        avg_degradation = sum(r.sharpe_degradation for r in strategy_results) / len(strategy_results)
        
        # Determine status
        if pass_rate >= 80 and avg_oos_sharpe > 0.5:
            status = "[bold green]READY[/bold green]"
            status_code = "ready"
        elif pass_rate >= 50 and avg_oos_sharpe > 0:
            status = "[yellow]REVIEW[/yellow]"
            status_code = "review"
        else:
            status = "[red]FAIL[/red]"
            status_code = "fail"
        
        # Style based on values
        sharpe_style = "green" if avg_oos_sharpe > 0.5 else ("yellow" if avg_oos_sharpe > 0 else "red")
        deg_style = "green" if avg_degradation < 30 else ("yellow" if avg_degradation < 50 else "red")
        pass_style = "green" if pass_rate >= 80 else ("yellow" if pass_rate >= 50 else "red")
        
        table.add_row(
            strategy_name,
            str(len(strategy_results)),
            f"[green]{passed}[/green]",
            f"[red]{failed}[/red]" if failed > 0 else str(failed),
            f"[{pass_style}]{pass_rate:.0f}%[/{pass_style}]",
            f"[{sharpe_style}]{avg_oos_sharpe:.2f}[/{sharpe_style}]",
            f"[{deg_style}]{avg_degradation:.1f}%[/{deg_style}]",
            status,
        )
        
        strategy_summaries.append({
            "strategy": strategy_name,
            "symbols_tested": len(strategy_results),
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
            "avg_oos_sharpe": avg_oos_sharpe,
            "avg_degradation": avg_degradation,
            "status": status_code,
        })

    console.print(table)

    # Detailed failures
    failed_results = [r for r in results if not r.passed_validation]
    if failed_results:
        console.print(f"\n[bold red]Failed Validations ({len(failed_results)}):[/bold red]")
        
        fail_table = Table(show_header=True)
        fail_table.add_column("Strategy")
        fail_table.add_column("Symbol")
        fail_table.add_column("Failed Gates")
        fail_table.add_column("OOS Sharpe")
        
        for r in failed_results[:10]:  # Show first 10
            if r.acceptance_result:
                failed_gates = ", ".join(r.acceptance_result.gates_failed)
            else:
                failed_gates = "N/A"
            
            fail_table.add_row(
                r.strategy_name,
                r.symbol,
                failed_gates,
                f"{r.oos_sharpe:.2f}",
            )
        
        if len(failed_results) > 10:
            console.print(f"[dim]... and {len(failed_results) - 10} more[/dim]")
        
        console.print(fail_table)

    # Production-ready strategies
    ready_strategies = [s for s in strategy_summaries if s["status"] == "ready"]
    if ready_strategies:
        console.print("\n")
        console.print(Panel(
            "[bold green]Production-Ready Strategies[/bold green]\n" +
            "\n".join(f"  • {s['strategy']} (Sharpe: {s['avg_oos_sharpe']:.2f}, Pass: {s['pass_rate']:.0f}%)" 
                     for s in ready_strategies),
            border_style="green",
        ))
    else:
        console.print("\n")
        console.print(Panel(
            "[bold red]No Production-Ready Strategies[/bold red]\n"
            "All strategies failed validation. Consider:\n"
            "  • Simpler features\n"
            "  • Longer prediction horizons\n"
            "  • Rule-based approaches",
            border_style="red",
        ))

    return {
        "total_validations": len(results),
        "total_passed": sum(1 for r in results if r.passed_validation),
        "total_failed": sum(1 for r in results if not r.passed_validation),
        "strategies": strategy_summaries,
        "gates_used": [{"name": g.name, "metric": g.metric, "operator": g.operator, "threshold": g.threshold} for g in gates],
    }


def save_results(
    results: list[WalkForwardResult],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save validation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    results_data = [r.to_dict() for r in results]
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Save summary as JSON
    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate markdown summary
    md_content = generate_markdown_summary(results, summary)
    with open(output_dir / "validation_summary.md", "w") as f:
        f.write(md_content)

    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")


def generate_markdown_summary(
    results: list[WalkForwardResult],
    summary: dict[str, Any],
) -> str:
    """Generate markdown summary of validation results."""
    lines = [
        "# Strategy Validation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Total Validations:** {summary.get('total_validations', 0)}",
        f"- **Passed:** {summary.get('total_passed', 0)}",
        f"- **Failed:** {summary.get('total_failed', 0)}",
        "",
        "---",
        "",
        "## Strategy Results",
        "",
        "| Strategy | Pass Rate | Avg OOS Sharpe | Avg Degradation | Status |",
        "|----------|-----------|----------------|-----------------|--------|",
    ]

    for s in summary.get("strategies", []):
        status_emoji = {"ready": "✅", "review": "⚠️", "fail": "❌"}.get(s["status"], "❓")
        lines.append(
            f"| {s['strategy']} | {s['pass_rate']:.0f}% | {s['avg_oos_sharpe']:.2f} | "
            f"{s['avg_degradation']:.1f}% | {status_emoji} {s['status'].upper()} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Acceptance Gates Used",
        "",
    ])

    for gate in summary.get("gates_used", []):
        lines.append(f"- **{gate['name']}:** {gate['metric']} {gate['operator']} {gate['threshold']}")

    lines.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
    ])

    ready = [s for s in summary.get("strategies", []) if s["status"] == "ready"]
    review = [s for s in summary.get("strategies", []) if s["status"] == "review"]
    
    if ready:
        lines.append("### Production Ready")
        for s in ready:
            lines.append(f"- **{s['strategy']}** - Deploy with monitoring")
    
    if review:
        lines.append("")
        lines.append("### Needs Review")
        for s in review:
            lines.append(f"- **{s['strategy']}** - Consider parameter tuning")
    
    if not ready and not review:
        lines.extend([
            "### No Strategies Ready",
            "",
            "All strategies failed validation. Consider:",
            "- Using simpler features (volume_momentum, adx, rsi_14)",
            "- Longer prediction horizons (10+ bars)",
            "- Rule-based approaches (rule_ensemble)",
            "- Online learning with regular retraining",
        ])

    return "\n".join(lines)


async def main():
    """Main entry point."""
    console.print("[bold blue]═══ Strategy Validation Pipeline ═══[/bold blue]\n")

    results, gates = await run_validation()

    if results:
        console.print(f"\n[bold green]Completed {len(results)} validations[/bold green]")
        summary = display_validation_results(results, gates)

        # Save results
        settings = get_settings()
        output_dir = Path(settings.optimization.optimization.output.results_dir).parent / "validation_results"
        save_results(results, summary, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
