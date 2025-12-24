#!/usr/bin/env python3
"""
Backtest All Ratio Universe Pairs

Runs backtests on all candidate ratio pairs, identifies promising ones,
optimizes parameters, and generates a comprehensive report.

Usage:
    python scripts/backtest_ratio_universe.py
    python scripts/backtest_ratio_universe.py --optimize
    python scripts/backtest_ratio_universe.py --pair bnb_btc
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Note: We use direct SQL queries for performance in multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_ratio_universe() -> dict:
    """Load ratio universe configuration."""
    config_path = Path("config/research/ratio_universe.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)["ratio_universe"]


# Database URL for sync connections
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:crypto@localhost:5433/crypto")


def get_candles_sync(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Synchronous wrapper for getting candles - works in multiprocessing."""
    from sqlalchemy import create_engine, text
    
    # Create engine per process for multiprocessing safety
    engine = create_engine(DATABASE_URL, pool_size=2, max_overflow=5)
    
    query = text("""
        SELECT open_time, open, high, low, close, volume
        FROM candles
        WHERE symbol = :symbol AND interval = :interval
        AND open_time >= :start AND open_time <= :end
        ORDER BY open_time
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end
            })
            rows = result.fetchall()
    finally:
        engine.dispose()
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.set_index("open_time")
    
    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def run_ratio_backtest(
    target_symbol: str,
    reference_symbol: str,
    params: dict,
    start_date: datetime,
    end_date: datetime,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.08,
) -> dict:
    """
    Run a single ratio strategy backtest.
    
    Returns dict with performance metrics.
    """
    # Get data
    target_data = get_candles_sync(target_symbol, "1h", start_date, end_date)
    ref_data = get_candles_sync(reference_symbol, "1h", start_date, end_date)
    
    if target_data.empty or ref_data.empty:
        return {"error": "No data available"}
    
    # Align data
    common_idx = target_data.index.intersection(ref_data.index)
    if len(common_idx) < 500:
        return {"error": f"Insufficient data ({len(common_idx)} bars)"}
    
    target_data = target_data.loc[common_idx]
    ref_data = ref_data.loc[common_idx]
    
    # Calculate ratio
    ratio = target_data["close"] / ref_data["close"]
    
    # Calculate z-score
    lookback = params.get("lookback", 72)
    entry_threshold = params.get("entry_threshold", -1.2)
    exit_threshold = params.get("exit_threshold", -0.7)
    max_hold = params.get("max_hold_hours", 48)
    
    ratio_mean = ratio.rolling(lookback, min_periods=lookback // 2).mean()
    ratio_std = ratio.rolling(lookback, min_periods=lookback // 2).std()
    z_score = (ratio - ratio_mean) / ratio_std
    
    # Generate signals and simulate trades
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
            # Check for entry
            if z < entry_threshold:
                position = "long"
                entry_price = price
                entry_idx = i
        else:
            # Check for exit
            bars_held = i - entry_idx
            pnl_pct = (price / entry_price - 1)
            
            exit_reason = None
            
            # Exit conditions
            if pnl_pct <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif pnl_pct >= take_profit_pct:
                exit_reason = "take_profit"
            elif z > exit_threshold:
                exit_reason = "z_exit"
            elif bars_held >= max_hold:
                exit_reason = "max_hold"
            
            if exit_reason:
                trades.append({
                    "entry_time": target_data.index[entry_idx],
                    "exit_time": idx,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                })
                position = None
    
    if not trades:
        return {
            "error": "No trades",
            "data_points": len(common_idx),
        }
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    pnl = trades_df["pnl_pct"]
    
    # Apply commission (0.1% each way)
    commission = 0.002
    net_pnl = pnl - commission
    
    total_return = (1 + net_pnl).prod() - 1
    avg_return = net_pnl.mean()
    win_rate = (net_pnl > 0).mean()
    
    # Calculate Sharpe (annualized)
    if net_pnl.std() > 0:
        # Approximate: assume average trade duration
        avg_bars = trades_df["bars_held"].mean()
        trades_per_year = 24 * 365 / avg_bars
        sharpe = avg_return / net_pnl.std() * np.sqrt(trades_per_year)
    else:
        sharpe = 0
    
    # Max drawdown
    cumulative = (1 + net_pnl).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    # Calculate days
    days = (end_date - start_date).days
    trades_per_month = len(trades) / (days / 30)
    
    return {
        "total_return": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "win_rate": round(win_rate * 100, 1),
        "max_drawdown": round(max_dd * 100, 2),
        "num_trades": len(trades),
        "trades_per_month": round(trades_per_month, 1),
        "avg_bars_held": round(trades_df["bars_held"].mean(), 1),
        "avg_return_per_trade": round(avg_return * 100, 3),
        "data_points": len(common_idx),
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict(),
    }


def backtest_pair(pair_name: str, pair_config: dict, days: int = 365) -> dict:
    """Backtest a single pair with walk-forward split."""
    target = pair_config["target"]
    reference = pair_config["reference"]
    params = pair_config.get("params", {})
    stop_loss = params.get("stop_loss_pct", 0.03)
    take_profit = params.get("take_profit_pct", 0.08)
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days + 60)  # Extra for warmup
    
    # Split: 70% in-sample, 30% out-of-sample
    split_date = end_date - timedelta(days=int(days * 0.3))
    
    logger.info(f"Backtesting {pair_name}: {target}/{reference}")
    
    try:
        # In-sample
        is_result = run_ratio_backtest(
            target, reference, params, start_date, split_date,
            stop_loss, take_profit
        )
        
        # Out-of-sample
        oos_result = run_ratio_backtest(
            target, reference, params, split_date, end_date,
            stop_loss, take_profit
        )
        
        return {
            "pair_name": pair_name,
            "pair": f"{target[:-4]}/{reference[:-4]}",
            "target": target,
            "reference": reference,
            "params": params,
            "in_sample": is_result,
            "out_of_sample": oos_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error backtesting {pair_name}: {e}")
        return {
            "pair_name": pair_name,
            "error": str(e),
        }


def optimize_pair(pair_name: str, pair_config: dict, days: int = 365) -> dict:
    """Optimize parameters for a single pair using grid search."""
    target = pair_config["target"]
    reference = pair_config["reference"]
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    # Parameter grid
    lookbacks = [48, 72, 96, 120]
    entry_thresholds = [-1.0, -1.2, -1.5, -1.8]
    exit_thresholds = [-0.5, -0.7, -0.9]
    max_holds = [24, 48, 72]
    
    best_result = None
    best_sharpe = -999
    best_params = None
    
    logger.info(f"Optimizing {pair_name}...")
    
    for lookback in lookbacks:
        for entry in entry_thresholds:
            for exit_thresh in exit_thresholds:
                for max_hold in max_holds:
                    params = {
                        "lookback": lookback,
                        "entry_threshold": entry,
                        "exit_threshold": exit_thresh,
                        "max_hold_hours": max_hold,
                    }
                    
                    result = run_ratio_backtest(
                        target, reference, params, start_date, end_date,
                        stop_loss_pct=0.03, take_profit_pct=0.08
                    )
                    
                    if "error" in result:
                        continue
                    
                    sharpe = result.get("sharpe_ratio", -999)
                    trades = result.get("trades_per_month", 0)
                    
                    # Must have reasonable trade frequency
                    if trades >= 5 and sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_result = result
                        best_params = params.copy()
    
    return {
        "pair_name": pair_name,
        "pair": f"{target[:-4]}/{reference[:-4]}",
        "best_params": best_params,
        "best_result": best_result,
    }


def run_all_backtests(days: int = 365, status_filter: str | None = None) -> list[dict]:
    """Run backtests on all pairs from config."""
    config = load_ratio_universe()
    pairs = config.get("pairs", {})
    
    results = []
    
    for pair_name, pair_config in pairs.items():
        status = pair_config.get("status", "candidate")
        
        if status_filter and status != status_filter:
            continue
        
        result = backtest_pair(pair_name, pair_config, days)
        results.append(result)
    
    return results


def run_parallel_backtests(days: int = 365, status_filter: str | None = None, max_workers: int = 4) -> list[dict]:
    """Run backtests in parallel."""
    config = load_ratio_universe()
    pairs = config.get("pairs", {})
    
    # Filter pairs
    pairs_to_test = []
    for pair_name, pair_config in pairs.items():
        status = pair_config.get("status", "candidate")
        if status_filter and status != status_filter:
            continue
        pairs_to_test.append((pair_name, pair_config))
    
    logger.info(f"Running {len(pairs_to_test)} backtests with {max_workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(backtest_pair, name, config, days): name
            for name, config in pairs_to_test
        }
        
        for future in as_completed(futures):
            pair_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                oos = result.get("out_of_sample", {})
                sharpe = oos.get("sharpe_ratio", "N/A")
                logger.info(f"  ✓ {pair_name}: Sharpe = {sharpe}")
            except Exception as e:
                logger.error(f"  ✗ {pair_name}: {e}")
                results.append({"pair_name": pair_name, "error": str(e)})
    
    return results


def run_parallel_optimization(pairs_to_optimize: list[tuple], days: int = 365, max_workers: int = 4) -> list[dict]:
    """Run parameter optimization in parallel."""
    logger.info(f"Optimizing {len(pairs_to_optimize)} pairs with {max_workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(optimize_pair, name, config, days): name
            for name, config in pairs_to_optimize
        }
        
        for future in as_completed(futures):
            pair_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                best_sharpe = result.get("best_result", {}).get("sharpe_ratio", "N/A")
                logger.info(f"  ✓ {pair_name}: Best Sharpe = {best_sharpe}")
            except Exception as e:
                logger.error(f"  ✗ {pair_name}: {e}")
    
    return results


def print_results_table(results: list[dict]) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print(" RATIO UNIVERSE BACKTEST RESULTS")
    print("=" * 100)
    
    # Sort by OOS Sharpe
    def get_oos_sharpe(r):
        return r.get("out_of_sample", {}).get("sharpe_ratio", -999)
    
    results.sort(key=get_oos_sharpe, reverse=True)
    
    print(f"\n{'Pair':<15} {'OOS Sharpe':>10} {'OOS Return':>10} {'Win Rate':>8} {'Trades/Mo':>10} {'Status':>12}")
    print("-" * 100)
    
    for r in results:
        if "error" in r and "out_of_sample" not in r:
            print(f"{r['pair_name']:<15} {'ERROR':>10} {r.get('error', '')[:40]}")
            continue
        
        oos = r.get("out_of_sample", {})
        if "error" in oos:
            print(f"{r['pair_name']:<15} {'NO TRADES':>10}")
            continue
        
        sharpe = oos.get("sharpe_ratio", 0)
        ret = oos.get("total_return", 0)
        wr = oos.get("win_rate", 0)
        tpm = oos.get("trades_per_month", 0)
        
        # Determine status recommendation
        if sharpe >= 2.0 and tpm >= 10:
            status = "🟢 DEPLOY"
        elif sharpe >= 1.0 and tpm >= 5:
            status = "🟡 VALIDATE"
        elif sharpe >= 0.5:
            status = "🟠 OPTIMIZE"
        else:
            status = "🔴 RETIRE"
        
        print(f"{r.get('pair', r['pair_name']):<15} {sharpe:>10.2f} {ret:>9.1f}% {wr:>7.1f}% {tpm:>10.1f} {status:>12}")
    
    print("=" * 100)


def classify_results(results: list[dict]) -> dict:
    """Classify results into categories."""
    deploy = []
    validate = []
    optimize = []
    retire = []
    
    for r in results:
        if "error" in r and "out_of_sample" not in r:
            retire.append(r)
            continue
        
        oos = r.get("out_of_sample", {})
        if "error" in oos:
            retire.append(r)
            continue
        
        sharpe = oos.get("sharpe_ratio", 0)
        tpm = oos.get("trades_per_month", 0)
        
        if sharpe >= 2.0 and tpm >= 10:
            deploy.append(r)
        elif sharpe >= 1.0 and tpm >= 5:
            validate.append(r)
        elif sharpe >= 0.5:
            optimize.append(r)
        else:
            retire.append(r)
    
    return {
        "deploy": deploy,
        "validate": validate,
        "optimize": optimize,
        "retire": retire,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest ratio universe")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--pair", type=str, help="Backtest specific pair")
    parser.add_argument("--optimize", action="store_true", help="Run optimization on promising pairs")
    parser.add_argument("--all", action="store_true", help="Include already deployed pairs")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Single pair
    if args.pair:
        config = load_ratio_universe()
        pair_config = config["pairs"].get(args.pair)
        if not pair_config:
            print(f"Pair not found: {args.pair}")
            return
        result = backtest_pair(args.pair, pair_config, args.days)
        print(json.dumps(result, indent=2, default=str))
        return
    
    # Determine filter
    status_filter = None if args.all else "candidate"
    
    # Run backtests
    print(f"\n🚀 Running backtests on {'all' if args.all else 'candidate'} pairs...")
    results = run_parallel_backtests(args.days, status_filter, args.workers)
    
    # Print results
    print_results_table(results)
    
    # Classify
    classified = classify_results(results)
    
    print(f"\n📊 Summary:")
    print(f"  🟢 Deploy:   {len(classified['deploy'])} pairs")
    print(f"  🟡 Validate: {len(classified['validate'])} pairs")
    print(f"  🟠 Optimize: {len(classified['optimize'])} pairs")
    print(f"  🔴 Retire:   {len(classified['retire'])} pairs")
    
    # Run optimization on promising pairs
    if args.optimize and (classified['validate'] or classified['optimize']):
        pairs_to_optimize = []
        config = load_ratio_universe()
        
        for r in classified['validate'] + classified['optimize']:
            pair_name = r['pair_name']
            if pair_name in config['pairs']:
                pairs_to_optimize.append((pair_name, config['pairs'][pair_name]))
        
        if pairs_to_optimize:
            print(f"\n🔧 Optimizing {len(pairs_to_optimize)} promising pairs...")
            opt_results = run_parallel_optimization(pairs_to_optimize, args.days, args.workers)
            
            print("\n📈 Optimization Results:")
            print("-" * 80)
            for opt in opt_results:
                if opt.get("best_result"):
                    best = opt["best_result"]
                    params = opt["best_params"]
                    print(f"\n{opt['pair']}:")
                    print(f"  Best Sharpe: {best['sharpe_ratio']:.2f}")
                    print(f"  Params: lookback={params['lookback']}, entry={params['entry_threshold']}, "
                          f"exit={params['exit_threshold']}, max_hold={params['max_hold_hours']}")
            
            # Add optimization results to output
            results.append({"optimization_results": opt_results})
    
    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "days_analyzed": args.days,
            "results": results,
            "classification": {
                "deploy": [r["pair_name"] for r in classified["deploy"]],
                "validate": [r["pair_name"] for r in classified["validate"]],
                "optimize": [r["pair_name"] for r in classified["optimize"]],
                "retire": [r["pair_name"] for r in classified["retire"]],
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()

