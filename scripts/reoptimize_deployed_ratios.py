#!/usr/bin/env python3
"""
Re-optimize Deployed Ratio Strategies

Tests the new parameter pattern (lb=48, entry=-1.8, exit=-0.9) discovered from 
the ratio universe expansion (Research Note 16) on existing deployed pairs.

Compares:
- Current deployed parameters
- New optimized parameters
- Full grid search for best params
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sqlalchemy import create_engine, text

console = Console()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:crypto@localhost:5433/crypto")


def get_candles(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Get candles from database."""
    engine = create_engine(DATABASE_URL, pool_size=2, max_overflow=5)
    
    query = text("""
        SELECT open_time, open, high, low, close, volume
        FROM candles
        WHERE symbol = :symbol AND interval = '1h'
        AND open_time >= :start AND open_time <= :end
        ORDER BY open_time
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol": symbol, "start": start, "end": end})
            rows = result.fetchall()
    finally:
        engine.dispose()
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.set_index("open_time")
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def run_backtest(
    target_symbol: str,
    reference_symbol: str,
    params: dict,
    start_date: datetime,
    end_date: datetime,
) -> dict:
    """Run a ratio strategy backtest."""
    target_data = get_candles(target_symbol, start_date, end_date)
    ref_data = get_candles(reference_symbol, start_date, end_date)
    
    if target_data.empty or ref_data.empty:
        return {"error": "No data"}
    
    common_idx = target_data.index.intersection(ref_data.index)
    if len(common_idx) < 500:
        return {"error": f"Insufficient data ({len(common_idx)})"}
    
    target_data = target_data.loc[common_idx]
    ref_data = ref_data.loc[common_idx]
    
    ratio = target_data["close"] / ref_data["close"]
    
    lookback = params["lookback"]
    entry_threshold = params["entry_threshold"]
    exit_threshold = params["exit_threshold"]
    max_hold = params["max_hold_hours"]
    stop_loss = params.get("stop_loss_pct", 0.03)
    take_profit = params.get("take_profit_pct", 0.08)
    
    ratio_mean = ratio.rolling(lookback, min_periods=lookback // 2).mean()
    ratio_std = ratio.rolling(lookback, min_periods=lookback // 2).std()
    z_score = (ratio - ratio_mean) / ratio_std
    
    trades = []
    position = None
    entry_price = 0
    entry_idx = 0
    
    for i, idx in enumerate(target_data.index):
        if i < lookback:
            continue
        
        z = z_score.loc[idx]
        price = target_data.loc[idx, "close"]
        
        if pd.isna(z):
            continue
        
        if position is None:
            if z < entry_threshold:
                position = "long"
                entry_price = price
                entry_idx = i
        else:
            bars_held = i - entry_idx
            pnl_pct = (price / entry_price - 1)
            
            exit_reason = None
            
            if pnl_pct <= -stop_loss:
                exit_reason = "stop_loss"
            elif pnl_pct >= take_profit:
                exit_reason = "take_profit"
            elif z > exit_threshold:
                exit_reason = "z_exit"
            elif bars_held >= max_hold:
                exit_reason = "max_hold"
            
            if exit_reason:
                trades.append({
                    "pnl_pct": pnl_pct,
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                })
                position = None
    
    if not trades:
        return {"error": "No trades", "data_points": len(common_idx)}
    
    trades_df = pd.DataFrame(trades)
    pnl = trades_df["pnl_pct"]
    
    commission = 0.002
    net_pnl = pnl - commission
    
    total_return = (1 + net_pnl).prod() - 1
    avg_return = net_pnl.mean()
    win_rate = (net_pnl > 0).mean()
    
    if net_pnl.std() > 0:
        avg_bars = trades_df["bars_held"].mean()
        trades_per_year = 24 * 365 / avg_bars
        sharpe = avg_return / net_pnl.std() * np.sqrt(trades_per_year)
    else:
        sharpe = 0
    
    cumulative = (1 + net_pnl).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    days = (end_date - start_date).days
    trades_per_month = len(trades) / (days / 30)
    
    return {
        "sharpe": round(sharpe, 2),
        "return_pct": round(total_return * 100, 1),
        "win_rate": round(win_rate * 100, 1),
        "max_dd": round(max_dd * 100, 1),
        "trades": len(trades),
        "trades_per_month": round(trades_per_month, 1),
    }


def compare_params(pair_name: str, target: str, reference: str, 
                   current_params: dict, new_params: dict, days: int = 365):
    """Compare current vs new parameters."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days + 60)
    
    # Run both
    current_result = run_backtest(target, reference, current_params, start_date, end_date)
    new_result = run_backtest(target, reference, new_params, start_date, end_date)
    
    return {
        "pair": pair_name,
        "current": {**current_params, **current_result},
        "new": {**new_params, **new_result},
    }


def grid_search(target: str, reference: str, days: int = 365) -> dict:
    """Find optimal parameters via grid search."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days + 60)
    
    lookbacks = [48, 72, 96, 120]
    entries = [-1.0, -1.2, -1.5, -1.8, -2.0]
    exits = [-0.5, -0.7, -0.9]
    holds = [24, 48, 72]
    
    best_sharpe = -999
    best_params = None
    best_result = None
    
    for lb in lookbacks:
        for entry in entries:
            for exit_t in exits:
                for hold in holds:
                    params = {
                        "lookback": lb,
                        "entry_threshold": entry,
                        "exit_threshold": exit_t,
                        "max_hold_hours": hold,
                        "stop_loss_pct": 0.03,
                        "take_profit_pct": 0.08,
                    }
                    
                    result = run_backtest(target, reference, params, start_date, end_date)
                    
                    if "error" in result:
                        continue
                    
                    sharpe = result.get("sharpe", -999)
                    trades = result.get("trades_per_month", 0)
                    
                    if trades >= 5 and sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params.copy()
                        best_result = result
    
    return {
        "best_params": best_params,
        "best_result": best_result,
    }


def main():
    console.print(Panel.fit(
        "[bold blue]Re-optimizing Deployed Ratio Strategies[/bold blue]\n\n"
        "Testing new parameter pattern from Research Note 16:\n"
        "  lookback=48, entry=-1.8, exit=-0.9",
        title="🔧 Parameter Re-optimization"
    ))
    
    # Define deployed pairs and their current params
    deployed_pairs = {
        "eth_btc": {
            "target": "ETHUSDT",
            "reference": "BTCUSDT",
            "current_params": {
                "lookback": 72,
                "entry_threshold": -1.2,
                "exit_threshold": -0.7,
                "max_hold_hours": 72,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.08,
            }
        },
        "sol_btc": {
            "target": "SOLUSDT",
            "reference": "BTCUSDT",
            "current_params": {
                "lookback": 72,
                "entry_threshold": -1.2,
                "exit_threshold": -0.7,
                "max_hold_hours": 48,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.08,
            }
        },
        "ltc_btc": {
            "target": "LTCUSDT",
            "reference": "BTCUSDT",
            "current_params": {
                "lookback": 120,
                "entry_threshold": -1.2,
                "exit_threshold": -0.7,
                "max_hold_hours": 48,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.08,
            }
        },
    }
    
    # New optimized params pattern from Note 16
    new_params_template = {
        "lookback": 48,
        "entry_threshold": -1.8,
        "exit_threshold": -0.9,
        "max_hold_hours": 48,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.08,
    }
    
    results = []
    
    console.print("\n[bold]1. Comparing Current vs New Parameters (365 days)[/bold]\n")
    
    for pair_name, config in deployed_pairs.items():
        console.print(f"  Testing {pair_name}...", end=" ")
        result = compare_params(
            pair_name,
            config["target"],
            config["reference"],
            config["current_params"],
            new_params_template,
            days=365
        )
        results.append(result)
        
        current_sharpe = result["current"].get("sharpe", "N/A")
        new_sharpe = result["new"].get("sharpe", "N/A")
        improvement = ""
        if isinstance(current_sharpe, (int, float)) and isinstance(new_sharpe, (int, float)):
            if new_sharpe > current_sharpe:
                improvement = f"[green]+{new_sharpe - current_sharpe:.1f}[/green]"
            else:
                improvement = f"[red]{new_sharpe - current_sharpe:.1f}[/red]"
        
        console.print(f"Current: {current_sharpe}, New: {new_sharpe} {improvement}")
    
    # Print comparison table
    console.print("\n[bold]Comparison Results:[/bold]\n")
    
    table = Table(title="Current vs New Parameters")
    table.add_column("Pair", style="cyan")
    table.add_column("Params", style="dim")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Trades/Mo", justify="right")
    
    for r in results:
        # Current row
        curr = r["current"]
        table.add_row(
            f"{r['pair']} (current)",
            f"lb={curr['lookback']}, e={curr['entry_threshold']}, x={curr['exit_threshold']}",
            f"{curr.get('sharpe', 'N/A')}",
            f"{curr.get('return_pct', 'N/A')}%",
            f"{curr.get('win_rate', 'N/A')}%",
            f"{curr.get('trades_per_month', 'N/A')}",
        )
        
        # New row
        new = r["new"]
        sharpe_style = "green" if new.get('sharpe', 0) > curr.get('sharpe', 0) else "red"
        table.add_row(
            f"[{sharpe_style}]{r['pair']} (new)[/{sharpe_style}]",
            f"lb={new['lookback']}, e={new['entry_threshold']}, x={new['exit_threshold']}",
            f"[{sharpe_style}]{new.get('sharpe', 'N/A')}[/{sharpe_style}]",
            f"{new.get('return_pct', 'N/A')}%",
            f"{new.get('win_rate', 'N/A')}%",
            f"{new.get('trades_per_month', 'N/A')}",
        )
        table.add_row("", "", "", "", "", "")
    
    console.print(table)
    
    # Grid search for each
    console.print("\n[bold]2. Full Grid Search (finding absolute best params)[/bold]\n")
    
    grid_results = {}
    for pair_name, config in deployed_pairs.items():
        console.print(f"  Grid searching {pair_name}...", end=" ")
        grid_result = grid_search(config["target"], config["reference"], days=365)
        grid_results[pair_name] = grid_result
        
        if grid_result["best_params"]:
            bp = grid_result["best_params"]
            br = grid_result["best_result"]
            console.print(f"Best Sharpe: {br['sharpe']} (lb={bp['lookback']}, e={bp['entry_threshold']}, x={bp['exit_threshold']})")
        else:
            console.print("No valid params found")
    
    # Summary table
    console.print("\n[bold]3. Summary: Recommended Parameters[/bold]\n")
    
    summary_table = Table(title="Recommended Parameters")
    summary_table.add_column("Pair", style="cyan")
    summary_table.add_column("Current Sharpe")
    summary_table.add_column("New Pattern Sharpe")
    summary_table.add_column("Grid Best Sharpe")
    summary_table.add_column("Recommended Params")
    summary_table.add_column("Action")
    
    recommendations = {}
    
    for r in results:
        pair = r["pair"]
        current_sharpe = r["current"].get("sharpe", 0)
        new_sharpe = r["new"].get("sharpe", 0)
        
        grid = grid_results.get(pair, {})
        grid_sharpe = grid.get("best_result", {}).get("sharpe", 0) if grid.get("best_result") else 0
        grid_params = grid.get("best_params", {})
        
        # Determine best option
        if grid_sharpe >= new_sharpe and grid_sharpe >= current_sharpe:
            best = "grid"
            best_sharpe = grid_sharpe
            best_params = grid_params
        elif new_sharpe >= current_sharpe:
            best = "new"
            best_sharpe = new_sharpe
            best_params = new_params_template
        else:
            best = "current"
            best_sharpe = current_sharpe
            best_params = r["current"]
        
        if best == "current":
            action = "Keep current"
        elif best_sharpe > current_sharpe * 1.1:  # 10% improvement threshold
            action = "✅ UPDATE"
        else:
            action = "Optional update"
        
        param_str = f"lb={best_params['lookback']}, e={best_params['entry_threshold']}, x={best_params['exit_threshold']}"
        
        summary_table.add_row(
            pair,
            str(current_sharpe),
            str(new_sharpe),
            str(grid_sharpe),
            param_str,
            action,
        )
        
        recommendations[pair] = {
            "current_sharpe": current_sharpe,
            "best_sharpe": best_sharpe,
            "best_params": best_params,
            "action": action,
        }
    
    console.print(summary_table)
    
    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comparisons": results,
        "grid_search": {k: {"best_params": v["best_params"], "best_result": v["best_result"]} 
                        for k, v in grid_results.items()},
        "recommendations": recommendations,
    }
    
    output_path = Path("notes/optimization_results/reoptimization_deployed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    console.print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
