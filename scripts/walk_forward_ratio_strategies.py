#!/usr/bin/env python3
"""
Walk-Forward Validation for Ratio Strategies.

Runs proper out-of-sample testing with rolling train/test windows
to validate ratio mean reversion strategies.

Usage:
    python scripts/walk_forward_ratio_strategies.py
    python scripts/walk_forward_ratio_strategies.py --pair bnb_btc
    python scripts/walk_forward_ratio_strategies.py --all --output notes/walk_forward_ratio_results.json
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database URL
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:crypto@localhost:5433/crypto")


@dataclass
class WalkForwardFold:
    """Result from a single walk-forward fold."""
    fold_num: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    test_return: float
    test_sharpe: float
    test_trades: int
    test_win_rate: float


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation result."""
    pair_name: str
    pair: str
    params: dict
    folds: list[WalkForwardFold]
    avg_oos_sharpe: float
    avg_oos_return: float
    avg_oos_trades: float
    avg_win_rate: float
    sharpe_std: float
    positive_folds: int
    total_folds: int
    passed: bool
    

def get_candles_sync(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Get candles from database synchronously."""
    from sqlalchemy import create_engine, text
    
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
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def run_ratio_backtest_on_window(
    target_data: pd.DataFrame,
    ref_data: pd.DataFrame,
    params: dict,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.08,
) -> dict:
    """Run ratio backtest on a specific data window."""
    if target_data.empty or ref_data.empty:
        return {"error": "No data"}
    
    common_idx = target_data.index.intersection(ref_data.index)
    if len(common_idx) < 100:
        return {"error": f"Insufficient data ({len(common_idx)} bars)"}
    
    target_data = target_data.loc[common_idx]
    ref_data = ref_data.loc[common_idx]
    
    # Calculate ratio
    ratio = target_data["close"] / ref_data["close"]
    
    # Z-score calculation
    lookback = params.get("lookback", 48)
    entry_threshold = params.get("entry_threshold", -1.8)
    exit_threshold = params.get("exit_threshold", -0.9)
    max_hold = params.get("max_hold_hours", 48)
    
    ratio_mean = ratio.rolling(lookback, min_periods=lookback // 2).mean()
    ratio_std = ratio.rolling(lookback, min_periods=lookback // 2).std()
    z_score = (ratio - ratio_mean) / ratio_std
    
    # Simulate trades
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
                    "pnl_pct": pnl_pct,
                    "bars_held": bars_held,
                })
                position = None
    
    if not trades:
        return {"error": "No trades", "num_trades": 0}
    
    trades_df = pd.DataFrame(trades)
    pnl = trades_df["pnl_pct"]
    
    # Apply commission
    commission = 0.002
    net_pnl = pnl - commission
    
    total_return = (1 + net_pnl).prod() - 1
    avg_return = net_pnl.mean()
    win_rate = (net_pnl > 0).mean()
    
    if net_pnl.std() > 0:
        avg_bars = trades_df["bars_held"].mean()
        trades_per_year = 24 * 365 / avg_bars if avg_bars > 0 else 0
        sharpe = avg_return / net_pnl.std() * np.sqrt(trades_per_year) if trades_per_year > 0 else 0
    else:
        sharpe = 0
    
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "num_trades": len(trades),
    }


def run_walk_forward(
    pair_name: str,
    target_symbol: str,
    reference_symbol: str,
    params: dict,
    total_days: int = 730,  # 2 years
    train_days: int = 180,  # 6 months training
    test_days: int = 30,    # 1 month testing
    step_days: int = 30,    # Step forward 1 month
) -> WalkForwardResult:
    """
    Run walk-forward validation on a ratio pair.
    
    Creates rolling windows where we train on historical data
    and test on the immediately following period.
    """
    logger.info(f"Running walk-forward on {pair_name}...")
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=total_days)
    
    # Load all data upfront
    target_data = get_candles_sync(target_symbol, "1h", start_date, end_date)
    ref_data = get_candles_sync(reference_symbol, "1h", start_date, end_date)
    
    if target_data.empty or ref_data.empty:
        return WalkForwardResult(
            pair_name=pair_name,
            pair=f"{target_symbol[:-4]}/{reference_symbol[:-4]}",
            params=params,
            folds=[],
            avg_oos_sharpe=0,
            avg_oos_return=0,
            avg_oos_trades=0,
            avg_win_rate=0,
            sharpe_std=0,
            positive_folds=0,
            total_folds=0,
            passed=False,
        )
    
    # Generate fold windows
    folds = []
    fold_num = 0
    
    current_train_start = start_date
    
    while True:
        train_end = current_train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        
        if test_end > end_date:
            break
        
        fold_num += 1
        
        # Get data for this fold
        train_target = target_data.loc[
            (target_data.index >= current_train_start) & 
            (target_data.index < train_end)
        ]
        train_ref = ref_data.loc[
            (ref_data.index >= current_train_start) & 
            (ref_data.index < train_end)
        ]
        
        test_target = target_data.loc[
            (target_data.index >= test_start) & 
            (target_data.index < test_end)
        ]
        test_ref = ref_data.loc[
            (ref_data.index >= test_start) & 
            (ref_data.index < test_end)
        ]
        
        # Run backtest on test window only (using fixed params - no optimization)
        result = run_ratio_backtest_on_window(
            test_target, test_ref, params,
            stop_loss_pct=params.get("stop_loss_pct", 0.03),
            take_profit_pct=params.get("take_profit_pct", 0.08),
        )
        
        if "error" not in result or result.get("num_trades", 0) > 0:
            folds.append(WalkForwardFold(
                fold_num=fold_num,
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                test_return=result.get("total_return", 0) * 100,
                test_sharpe=result.get("sharpe_ratio", 0),
                test_trades=result.get("num_trades", 0),
                test_win_rate=result.get("win_rate", 0) * 100,
            ))
        
        # Step forward
        current_train_start += timedelta(days=step_days)
    
    if not folds:
        return WalkForwardResult(
            pair_name=pair_name,
            pair=f"{target_symbol[:-4]}/{reference_symbol[:-4]}",
            params=params,
            folds=[],
            avg_oos_sharpe=0,
            avg_oos_return=0,
            avg_oos_trades=0,
            avg_win_rate=0,
            sharpe_std=0,
            positive_folds=0,
            total_folds=0,
            passed=False,
        )
    
    # Aggregate results
    sharpes = [f.test_sharpe for f in folds]
    returns = [f.test_return for f in folds]
    trades = [f.test_trades for f in folds]
    win_rates = [f.test_win_rate for f in folds]
    
    avg_sharpe = np.mean(sharpes)
    sharpe_std = np.std(sharpes)
    positive_folds = sum(1 for s in sharpes if s > 0)
    
    # Acceptance criteria: avg Sharpe > 0.5, >50% positive folds
    passed = avg_sharpe > 0.5 and positive_folds / len(folds) > 0.5
    
    return WalkForwardResult(
        pair_name=pair_name,
        pair=f"{target_symbol[:-4]}/{reference_symbol[:-4]}",
        params=params,
        folds=folds,
        avg_oos_sharpe=round(avg_sharpe, 2),
        avg_oos_return=round(np.mean(returns), 2),
        avg_oos_trades=round(np.mean(trades), 1),
        avg_win_rate=round(np.mean(win_rates), 1),
        sharpe_std=round(sharpe_std, 2),
        positive_folds=positive_folds,
        total_folds=len(folds),
        passed=passed,
    )


def run_walk_forward_worker(args: tuple) -> WalkForwardResult:
    """Worker function for parallel execution."""
    pair_name, pair_config = args
    target = pair_config["target"]
    reference = pair_config["reference"]
    params = pair_config.get("params", {})
    
    return run_walk_forward(pair_name, target, reference, params)


def load_ratio_universe() -> dict:
    """Load ratio universe configuration."""
    config_path = Path("config/research/ratio_universe.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)["ratio_universe"]


def print_results_table(results: list[WalkForwardResult]) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 110)
    print(" WALK-FORWARD VALIDATION RESULTS")
    print("=" * 110)
    
    # Sort by avg OOS Sharpe
    results.sort(key=lambda x: x.avg_oos_sharpe, reverse=True)
    
    print(f"\n{'Pair':<12} {'Avg Sharpe':>10} {'Std':>6} {'Avg Return':>10} {'Win Rate':>8} "
          f"{'Trades/Fold':>11} {'Pos Folds':>10} {'Status':>10}")
    print("-" * 110)
    
    for r in results:
        if r.total_folds == 0:
            print(f"{r.pair:<12} {'NO DATA':>10}")
            continue
        
        status = "✅ PASS" if r.passed else "❌ FAIL"
        pos_folds = f"{r.positive_folds}/{r.total_folds}"
        
        print(f"{r.pair:<12} {r.avg_oos_sharpe:>10.2f} {r.sharpe_std:>6.2f} {r.avg_oos_return:>9.1f}% "
              f"{r.avg_win_rate:>7.1f}% {r.avg_oos_trades:>11.1f} {pos_folds:>10} {status:>10}")
    
    print("=" * 110)
    
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed and r.total_folds > 0]
    
    print(f"\n📊 Summary: {len(passed)} passed, {len(failed)} failed")
    print(f"   Acceptance criteria: Avg OOS Sharpe > 0.5, >50% positive folds")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation for ratio strategies")
    parser.add_argument("--pair", type=str, help="Test specific pair")
    parser.add_argument("--all", action="store_true", help="Test all pairs (deployed + validated)")
    parser.add_argument("--deployed", action="store_true", help="Test deployed pairs only")
    parser.add_argument("--validated", action="store_true", help="Test validated pairs only")
    parser.add_argument("--days", type=int, default=730, help="Total days of history")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    config = load_ratio_universe()
    pairs = config.get("pairs", {})
    
    # Filter pairs
    pairs_to_test = []
    for pair_name, pair_config in pairs.items():
        status = pair_config.get("status", "candidate")
        
        if args.pair:
            if pair_name == args.pair:
                pairs_to_test.append((pair_name, pair_config))
        elif args.all:
            if status in ["deployed", "validated"]:
                pairs_to_test.append((pair_name, pair_config))
        elif args.deployed:
            if status == "deployed":
                pairs_to_test.append((pair_name, pair_config))
        elif args.validated:
            if status == "validated":
                pairs_to_test.append((pair_name, pair_config))
        else:
            # Default: deployed + validated
            if status in ["deployed", "validated"]:
                pairs_to_test.append((pair_name, pair_config))
    
    if not pairs_to_test:
        print("No pairs to test. Use --all, --deployed, --validated, or --pair <name>")
        return
    
    print(f"\n🚀 Running walk-forward validation on {len(pairs_to_test)} pairs...")
    print(f"   Config: {args.days} days, train=180d, test=30d, step=30d\n")
    
    # Run in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_walk_forward_worker, pair): pair[0]
            for pair in pairs_to_test
        }
        
        for future in as_completed(futures):
            pair_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "✅" if result.passed else "❌"
                logger.info(f"  {status} {pair_name}: Avg Sharpe = {result.avg_oos_sharpe:.2f}, "
                          f"Pos folds = {result.positive_folds}/{result.total_folds}")
            except Exception as e:
                logger.error(f"  ❌ {pair_name}: {e}")
    
    # Print results
    print_results_table(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "total_days": args.days,
                "train_days": 180,
                "test_days": 30,
                "step_days": 30,
            },
            "results": [
                {
                    "pair_name": r.pair_name,
                    "pair": r.pair,
                    "params": r.params,
                    "avg_oos_sharpe": r.avg_oos_sharpe,
                    "avg_oos_return": r.avg_oos_return,
                    "avg_oos_trades": r.avg_oos_trades,
                    "avg_win_rate": r.avg_win_rate,
                    "sharpe_std": r.sharpe_std,
                    "positive_folds": r.positive_folds,
                    "total_folds": r.total_folds,
                    "passed": r.passed,
                    "folds": [
                        {
                            "fold_num": f.fold_num,
                            "test_start": f.test_start.isoformat(),
                            "test_end": f.test_end.isoformat(),
                            "test_return": f.test_return,
                            "test_sharpe": f.test_sharpe,
                            "test_trades": f.test_trades,
                            "test_win_rate": f.test_win_rate,
                        }
                        for f in r.folds
                    ]
                }
                for r in results
            ],
            "summary": {
                "passed": [r.pair_name for r in results if r.passed],
                "failed": [r.pair_name for r in results if not r.passed and r.total_folds > 0],
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()

