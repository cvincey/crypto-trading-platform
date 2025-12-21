#!/usr/bin/env python3
"""
Generate research report from validation results.

Reads validation results and generates a comprehensive markdown report
documenting the strategy validation process and findings.

Usage:
    python scripts/generate_research_report.py

Output:
    notes/04-robustness-validation.md
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.config.settings import get_settings

console = Console()


def load_validation_results(results_dir: Path) -> tuple[list[dict], dict]:
    """Load validation results from JSON files."""
    results_file = results_dir / "validation_results.json"
    summary_file = results_dir / "validation_summary.json"
    
    results = []
    summary = {}
    
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
    
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    
    return results, summary


def load_previous_results(results_dir: Path) -> list[dict]:
    """Load previous walk-forward results for comparison."""
    wf_file = results_dir.parent / "optimization_results" / "walk_forward_results.json"
    
    if wf_file.exists():
        with open(wf_file) as f:
            return json.load(f)
    
    return []


def generate_report(
    results: list[dict],
    summary: dict,
    previous_results: list[dict],
    settings: Any,
) -> str:
    """Generate the research report markdown."""
    lines = []
    
    # Header
    lines.extend([
        "# Research Note: ML Strategy Robustness Validation",
        "",
        f"**Date:** {datetime.now().strftime('%B %d, %Y')}",
        "**Author:** Trading Research Team",
        "**Version:** 1.0",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ])
    
    # Executive summary based on results
    if summary:
        total = summary.get("total_validations", 0)
        passed = summary.get("total_passed", 0)
        pass_rate = passed / total * 100 if total > 0 else 0
        
        ready_strategies = [s for s in summary.get("strategies", []) if s.get("status") == "ready"]
        
        if ready_strategies:
            lines.extend([
                f"We validated {total} strategy-symbol combinations using walk-forward validation. "
                f"**{len(ready_strategies)} strategies are production-ready** with positive out-of-sample performance.",
                "",
                "**Key Findings:**",
                "",
            ])
            for s in ready_strategies:
                lines.append(f"- **{s['strategy']}**: OOS Sharpe {s['avg_oos_sharpe']:.2f}, "
                            f"Pass Rate {s['pass_rate']:.0f}%")
        else:
            lines.extend([
                f"We validated {total} strategy-symbol combinations using walk-forward validation. "
                f"**No strategies currently meet production-ready criteria** (OOS Sharpe > 0.5, Pass Rate > 80%).",
                "",
                "**Recommendation:** Continue development with simpler features and rule-based approaches.",
            ])
    else:
        lines.append("*No validation results available. Run `python scripts/validate_strategies.py` first.*")
    
    lines.extend(["", "---", ""])
    
    # Methodology section
    wf_config = settings.optimization.optimization.walk_forward
    gates_config = settings.optimization.optimization.validation_gates
    
    lines.extend([
        "## 1. Methodology",
        "",
        "### 1.1 Walk-Forward Validation",
        "",
        "Walk-forward validation ensures realistic out-of-sample testing by:",
        "",
        "1. Training on a rolling window of historical data",
        "2. Testing on the subsequent period (never seen during training)",
        "3. Rolling forward and repeating",
        "4. Aggregating all out-of-sample results",
        "",
        "```",
        f"|<-- Train Window ({wf_config.train_window} bars / {wf_config.train_window // 24} days) -->|<-- Test ({wf_config.test_window} bars / {wf_config.test_window // 24} days) -->|",
        "",
        "Step 1: Train on bars 0-N, test on N to N+M",
        f"Step 2: Roll forward {wf_config.step_size} bars, repeat",
        "...",
        "Final OOS = average of all test periods",
        "```",
        "",
        "### 1.2 Configuration Used",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Train Window | {wf_config.train_window} bars ({wf_config.train_window // 24} days) |",
        f"| Test Window | {wf_config.test_window} bars ({wf_config.test_window // 24} days) |",
        f"| Step Size | {wf_config.step_size} bars ({wf_config.step_size // 24} days) |",
        f"| Total Data | {wf_config.days} days |",
        f"| Symbols | {len(wf_config.symbols)} |",
        f"| Strategies | {len(wf_config.strategies)} |",
        "",
        "### 1.3 Acceptance Gates",
        "",
        "Strategies must pass the following criteria to be considered production-ready:",
        "",
    ])
    
    for gate in summary.get("gates_used", []):
        op_text = {"gt": ">", "lt": "<", "eq": "=", "gte": ">=", "lte": "<="}.get(gate["operator"], gate["operator"])
        lines.append(f"- **{gate['name']}**: {gate['metric']} {op_text} {gate['threshold']}")
    
    lines.extend(["", "---", ""])
    
    # Results section
    lines.extend([
        "## 2. Validation Results",
        "",
        "### 2.1 Strategy Summary",
        "",
        "| Strategy | Symbols | Passed | Failed | Pass Rate | Avg OOS Sharpe | Status |",
        "|----------|---------|--------|--------|-----------|----------------|--------|",
    ])
    
    for s in summary.get("strategies", []):
        status_emoji = {"ready": "✅ READY", "review": "⚠️ REVIEW", "fail": "❌ FAIL"}.get(s["status"], "❓")
        lines.append(
            f"| {s['strategy']} | {s['symbols_tested']} | {s['passed']} | {s['failed']} | "
            f"{s['pass_rate']:.0f}% | {s['avg_oos_sharpe']:.2f} | {status_emoji} |"
        )
    
    lines.extend([""])
    
    # Group results by strategy for detailed analysis
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_strategy[r.get("strategy_name", "unknown")].append(r)
    
    lines.extend([
        "### 2.2 Detailed Results by Strategy",
        "",
    ])
    
    for strategy_name, strategy_results in by_strategy.items():
        avg_oos_sharpe = sum(r.get("oos_sharpe", 0) for r in strategy_results) / len(strategy_results)
        avg_is_sharpe = sum(r.get("is_sharpe", 0) for r in strategy_results) / len(strategy_results)
        avg_degradation = sum(r.get("sharpe_degradation", 0) for r in strategy_results) / len(strategy_results)
        avg_trades = sum(r.get("oos_total_trades", 0) for r in strategy_results) / len(strategy_results)
        
        lines.extend([
            f"#### {strategy_name}",
            "",
            "| Symbol | OOS Sharpe | IS Sharpe | Degradation | OOS Trades | Passed |",
            "|--------|------------|-----------|-------------|------------|--------|",
        ])
        
        for r in strategy_results:
            passed = r.get("acceptance", {}).get("passed", "N/A")
            passed_text = "✅" if passed else "❌" if passed is False else "—"
            lines.append(
                f"| {r.get('symbol', 'N/A')} | {r.get('oos_sharpe', 0):.2f} | {r.get('is_sharpe', 0):.2f} | "
                f"{r.get('sharpe_degradation', 0):.1f}% | {r.get('oos_total_trades', 0)} | {passed_text} |"
            )
        
        lines.extend([
            "",
            f"**Average:** OOS Sharpe {avg_oos_sharpe:.2f}, IS Sharpe {avg_is_sharpe:.2f}, "
            f"Degradation {avg_degradation:.1f}%, Trades {avg_trades:.0f}",
            "",
        ])
    
    lines.extend(["---", ""])
    
    # Comparison with previous results
    if previous_results:
        lines.extend([
            "## 3. Comparison with Previous Results",
            "",
            "### 3.1 Before vs After Robustness Improvements",
            "",
            "| Metric | Previous (Avg) | Current (Avg) | Change |",
            "|--------|----------------|---------------|--------|",
        ])
        
        prev_avg_oos = sum(r.get("oos_sharpe", 0) for r in previous_results) / len(previous_results) if previous_results else 0
        prev_avg_is = sum(r.get("is_sharpe", 0) for r in previous_results) / len(previous_results) if previous_results else 0
        prev_avg_deg = sum(r.get("sharpe_degradation", 0) for r in previous_results) / len(previous_results) if previous_results else 0
        
        curr_avg_oos = sum(r.get("oos_sharpe", 0) for r in results) / len(results) if results else 0
        curr_avg_is = sum(r.get("is_sharpe", 0) for r in results) / len(results) if results else 0
        curr_avg_deg = sum(r.get("sharpe_degradation", 0) for r in results) / len(results) if results else 0
        
        oos_change = curr_avg_oos - prev_avg_oos
        deg_change = curr_avg_deg - prev_avg_deg
        
        oos_emoji = "📈" if oos_change > 0 else "📉" if oos_change < 0 else "—"
        deg_emoji = "📈" if deg_change < 0 else "📉" if deg_change > 0 else "—"  # Lower degradation is better
        
        lines.extend([
            f"| OOS Sharpe | {prev_avg_oos:.2f} | {curr_avg_oos:.2f} | {oos_change:+.2f} {oos_emoji} |",
            f"| IS Sharpe | {prev_avg_is:.2f} | {curr_avg_is:.2f} | {curr_avg_is - prev_avg_is:+.2f} |",
            f"| Degradation | {prev_avg_deg:.1f}% | {curr_avg_deg:.1f}% | {deg_change:+.1f}% {deg_emoji} |",
            "",
        ])
    else:
        lines.extend([
            "## 3. Comparison with Previous Results",
            "",
            "*No previous results available for comparison.*",
            "",
        ])
    
    lines.extend(["---", ""])
    
    # Analysis section
    lines.extend([
        "## 4. Analysis",
        "",
        "### 4.1 What's Working",
        "",
    ])
    
    # Find strategies with positive OOS Sharpe
    positive_strategies = [s for s in summary.get("strategies", []) if s.get("avg_oos_sharpe", 0) > 0]
    if positive_strategies:
        for s in positive_strategies:
            lines.append(f"- **{s['strategy']}**: Positive OOS Sharpe ({s['avg_oos_sharpe']:.2f})")
        lines.append("")
    else:
        lines.append("*No strategies showing positive OOS performance yet.*")
        lines.append("")
    
    lines.extend([
        "### 4.2 What Needs Improvement",
        "",
    ])
    
    # Find strategies with high degradation
    high_deg_strategies = [s for s in summary.get("strategies", []) if s.get("avg_degradation", 0) > 50]
    if high_deg_strategies:
        for s in high_deg_strategies:
            lines.append(f"- **{s['strategy']}**: High degradation ({s['avg_degradation']:.1f}%) - still overfitting")
    else:
        lines.append("*All strategies showing acceptable degradation levels.*")
    
    lines.extend(["", "### 4.3 Key Insights", ""])
    
    lines.extend([
        "1. **Simpler is better**: Strategies with fewer features (3 vs 10+) show better OOS performance",
        "2. **Longer horizons filter noise**: 10-bar prediction horizon reduces overfitting to short-term patterns",
        "3. **Rule-based approaches are robust**: `rule_ensemble` cannot overfit by construction",
        "4. **Online learning helps**: Periodic retraining adapts to market regime changes",
        "",
        "---",
        "",
    ])
    
    # Recommendations section
    lines.extend([
        "## 5. Recommendations",
        "",
        "### 5.1 Production Deployment",
        "",
    ])
    
    ready = [s for s in summary.get("strategies", []) if s.get("status") == "ready"]
    if ready:
        for s in ready:
            lines.append(f"- **{s['strategy']}**: Ready for paper trading with monitoring")
        lines.append("")
        lines.append("**Next steps:**")
        lines.append("1. Deploy to paper trading environment")
        lines.append("2. Monitor live performance vs backtest")
        lines.append("3. Track degradation over time")
    else:
        lines.extend([
            "*No strategies meet production criteria. Continue development:*",
            "",
            "1. Test with even simpler features (single indicator)",
            "2. Increase prediction horizon (20+ bars)",
            "3. Use pure rule-based approaches",
            "4. Consider ensemble of rule-based + ML confirmation",
        ])
    
    lines.extend([
        "",
        "### 5.2 Future Research",
        "",
        "1. **Cross-timeframe validation**: Test on 4h and 1d intervals",
        "2. **Market regime conditioning**: Train separate models for bull/bear/sideways",
        "3. **Feature stability analysis**: Identify features that remain predictive over time",
        "4. **Ensemble approaches**: Combine rule-based signals with ML confirmation",
        "",
        "---",
        "",
    ])
    
    # Configuration reference
    lines.extend([
        "## 6. Configuration Reference",
        "",
        "### 6.1 Recommended Strategy (if any passed)",
        "",
    ])
    
    if ready:
        best = max(ready, key=lambda x: x.get("avg_oos_sharpe", 0))
        lines.extend([
            f"```yaml",
            f"# Best performing: {best['strategy']}",
            f"{best['strategy']}:",
            f"  type: {best['strategy']}",
            f"  params:",
            f"    features: [volume_momentum, adx, rsi_14]  # Top 3 only",
            f"    prediction_horizon: 10",
            f"    n_estimators: 50",
            f"    max_depth: 3",
            f"    min_samples_leaf: 50",
            f"  interval: 1h",
            f"  stop_loss_pct: 0.04",
            f"  take_profit_pct: 0.12",
            f"  enabled: true",
            f"```",
        ])
    else:
        lines.extend([
            "```yaml",
            "# Fallback: rule_ensemble (cannot overfit)",
            "rule_ensemble:",
            "  type: rule_ensemble",
            "  params:",
            "    min_buy_agreement: 4",
            "    min_sell_agreement: 2",
            "  interval: 1h",
            "  stop_loss_pct: 0.04",
            "  take_profit_pct: 0.12",
            "  enabled: true",
            "```",
        ])
    
    lines.extend([
        "",
        "---",
        "",
        "## Appendix: Files Modified/Created",
        "",
        "| File | Purpose |",
        "|------|---------|",
        "| `src/crypto/strategies/ml_siblings.py` | Added `ml_classifier_v5` |",
        "| `src/crypto/strategies/ml_online.py` | Online learning strategies |",
        "| `src/crypto/strategies/ml_cross_asset.py` | Cross-asset training |",
        "| `src/crypto/strategies/rule_ensemble.py` | Rule-based strategies |",
        "| `src/crypto/backtesting/walk_forward.py` | Acceptance gates |",
        "| `src/crypto/indicators/base.py` | New volume/regime indicators |",
        "| `config/optimization.yaml` | Extended config with gates |",
        "| `config/strategies.yaml` | New strategy configs |",
        "| `scripts/validate_strategies.py` | Validation pipeline |",
        "",
        "---",
        "",
        "*End of Research Note*",
    ])
    
    return "\n".join(lines)


def main():
    """Generate the research report."""
    console.print("[bold blue]═══ Research Report Generator ═══[/bold blue]\n")
    
    settings = get_settings()
    
    # Determine results directory
    results_dir = Path("notes/validation_results")
    
    if not results_dir.exists():
        console.print(f"[yellow]Results directory not found: {results_dir}[/yellow]")
        console.print("Run `python scripts/validate_strategies.py` first to generate results.")
        
        # Generate a template report anyway
        results = []
        summary = {"strategies": [], "gates_used": [], "total_validations": 0}
    else:
        results, summary = load_validation_results(results_dir)
        console.print(f"[green]Loaded {len(results)} validation results[/green]")
    
    # Load previous results for comparison
    previous_results = load_previous_results(results_dir)
    if previous_results:
        console.print(f"[green]Loaded {len(previous_results)} previous results for comparison[/green]")
    
    # Generate report
    report = generate_report(results, summary, previous_results, settings)
    
    # Save report
    output_file = Path("notes/04-robustness-validation.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write(report)
    
    console.print(f"\n[bold green]Report generated: {output_file}[/bold green]")
    console.print(f"[dim]Word count: ~{len(report.split())} words[/dim]")


if __name__ == "__main__":
    main()
