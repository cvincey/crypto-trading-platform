#!/usr/bin/env python3
"""
Ratio Pair Screening Script

Systematically screens all candidate pairs from config/research/ratio_universe.yaml 
to find promising ratio mean reversion opportunities.

Screening steps:
1. Load historical data for all candidates
2. Calculate correlation, volatility, and mean reversion metrics
3. Run quick backtests on passing pairs
4. Rank by OOS Sharpe and generate recommendations

Usage:
    python scripts/screen_ratio_pairs.py --days 365
    python scripts/screen_ratio_pairs.py --pair bnb_btc --verbose
    python scripts/screen_ratio_pairs.py --all --output notes/ratio_screening_results.json
"""

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Setup path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.data.repository import DataRepository
from crypto.backtesting.engine import BacktestEngine
from crypto.strategies.cross_symbol import RatioReversionStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_ratio_universe(config_path: str = "config/research/ratio_universe.yaml") -> dict:
    """Load ratio universe configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["ratio_universe"]


def calculate_half_life(ratio_series: pd.Series) -> float:
    """
    Calculate the half-life of mean reversion for a ratio series.
    
    Uses Ornstein-Uhlenbeck regression to estimate mean reversion speed.
    Lower half-life = faster reversion = better for trading.
    """
    if len(ratio_series) < 100:
        return float("inf")
    
    # Compute log ratio
    log_ratio = np.log(ratio_series.dropna())
    
    # Calculate spread from rolling mean
    spread = log_ratio - log_ratio.rolling(72).mean()
    spread = spread.dropna()
    
    if len(spread) < 50:
        return float("inf")
    
    # Lagged spread for regression
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    # Align series
    common_idx = spread_lag.index.intersection(spread_diff.index)
    if len(common_idx) < 30:
        return float("inf")
        
    spread_lag = spread_lag.loc[common_idx]
    spread_diff = spread_diff.loc[common_idx]
    
    # OLS regression: d(spread) = theta * spread_lag + epsilon
    # Half-life = -ln(2) / theta
    try:
        theta = np.cov(spread_diff, spread_lag)[0, 1] / np.var(spread_lag)
        if theta >= 0:
            return float("inf")  # No mean reversion
        half_life = -np.log(2) / theta
        return max(1, min(half_life, 1000))  # Clamp to reasonable range
    except Exception:
        return float("inf")


def calculate_adf_pvalue(ratio_series: pd.Series) -> float:
    """
    Calculate ADF test p-value for stationarity.
    Lower p-value = more stationary = better mean reversion.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(ratio_series.dropna(), maxlag=20, regression="c")
        return result[1]  # p-value
    except Exception:
        return 1.0  # Assume non-stationary on error


def screen_pair(
    target_symbol: str,
    reference_symbol: str,
    repo: DataRepository,
    days: int = 365,
    params: dict | None = None,
) -> dict:
    """
    Screen a single ratio pair for trading viability.
    
    Returns screening metrics:
    - correlation: rolling correlation between assets
    - ratio_volatility: standard deviation of ratio returns
    - half_life: mean reversion half-life in hours
    - adf_pvalue: stationarity test p-value
    - z_score_range: typical z-score range
    - backtest_sharpe: quick backtest Sharpe ratio
    """
    logger.info(f"Screening {target_symbol}/{reference_symbol}...")
    
    # Default params
    if params is None:
        params = {
            "lookback": 72,
            "entry_threshold": -1.2,
            "exit_threshold": -0.7,
            "max_hold_hours": 48,
        }
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days + 30)  # Extra for warmup
    
    # Load data
    try:
        target_data = repo.get_candles(target_symbol, "1h", start_date, end_date)
        ref_data = repo.get_candles(reference_symbol, "1h", start_date, end_date)
    except Exception as e:
        logger.warning(f"Failed to load data for {target_symbol}/{reference_symbol}: {e}")
        return {"status": "error", "error": str(e)}
    
    if target_data.empty or ref_data.empty:
        return {"status": "error", "error": "No data available"}
    
    # Align data
    common_idx = target_data.index.intersection(ref_data.index)
    if len(common_idx) < 500:
        return {"status": "error", "error": f"Insufficient overlapping data ({len(common_idx)} bars)"}
    
    target = target_data.loc[common_idx, "close"]
    ref = ref_data.loc[common_idx, "close"]
    
    # Calculate ratio
    ratio = target / ref
    ratio_returns = ratio.pct_change().dropna()
    
    # =========================================================================
    # Screening Metrics
    # =========================================================================
    
    # 1. Correlation
    correlation = target.pct_change().corr(ref.pct_change())
    
    # 2. Ratio volatility (annualized)
    ratio_volatility = ratio_returns.std() * np.sqrt(24 * 365)
    
    # 3. Half-life
    half_life = calculate_half_life(ratio)
    
    # 4. ADF test
    adf_pvalue = calculate_adf_pvalue(ratio)
    
    # 5. Z-score statistics
    lookback = params["lookback"]
    ratio_mean = ratio.rolling(lookback).mean()
    ratio_std = ratio.rolling(lookback).std()
    z_score = (ratio - ratio_mean) / ratio_std
    z_score_clean = z_score.dropna()
    
    z_score_stats = {
        "min": float(z_score_clean.min()),
        "max": float(z_score_clean.max()),
        "std": float(z_score_clean.std()),
        "below_entry_pct": float((z_score_clean < params["entry_threshold"]).mean() * 100),
    }
    
    # 6. Daily volume
    target_volume = target_data["volume"].mean() * target_data["close"].mean()
    
    # =========================================================================
    # Screening Pass/Fail
    # =========================================================================
    
    criteria = {
        "correlation_ok": 0.70 <= correlation <= 0.98,
        "volatility_ok": ratio_volatility > 0.05,
        "half_life_ok": 12 <= half_life <= 168,
        "stationarity_ok": adf_pvalue < 0.10,
        "volume_ok": target_volume > 50_000_000,
        "signal_frequency_ok": z_score_stats["below_entry_pct"] > 2.0,
    }
    
    passed_screening = all(criteria.values())
    
    result = {
        "pair": f"{target_symbol[:-4]}/{reference_symbol[:-4]}",
        "target": target_symbol,
        "reference": reference_symbol,
        "status": "passed" if passed_screening else "failed",
        "metrics": {
            "correlation": round(correlation, 3),
            "ratio_volatility": round(ratio_volatility, 3),
            "half_life_hours": round(half_life, 1),
            "adf_pvalue": round(adf_pvalue, 4),
            "avg_daily_volume_usd": round(target_volume, 0),
            "z_score_stats": z_score_stats,
        },
        "criteria": criteria,
        "data_points": len(common_idx),
    }
    
    # =========================================================================
    # Quick Backtest (only if passed screening)
    # =========================================================================
    
    if passed_screening:
        logger.info(f"  ✓ Passed screening, running backtest...")
        backtest_result = run_quick_backtest(
            target_symbol, reference_symbol, repo, params, days
        )
        result["backtest"] = backtest_result
        result["recommendation"] = generate_recommendation(result)
    else:
        failed_criteria = [k for k, v in criteria.items() if not v]
        logger.info(f"  ✗ Failed screening: {failed_criteria}")
        result["failed_criteria"] = failed_criteria
    
    return result


def run_quick_backtest(
    target_symbol: str,
    reference_symbol: str,
    repo: DataRepository,
    params: dict,
    days: int,
) -> dict:
    """Run a quick backtest on the ratio pair."""
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Get data
        target_data = repo.get_candles(target_symbol, "1h", start_date, end_date)
        ref_data = repo.get_candles(reference_symbol, "1h", start_date, end_date)
        
        # Create strategy
        strategy = RatioReversionStrategy()
        strategy._setup(
            reference_symbol=reference_symbol,
            target_symbol=target_symbol,
            **params
        )
        
        # Inject reference data
        strategy.set_reference_data(reference_symbol, ref_data)
        
        # Generate signals
        signals = strategy.generate_signals(target_data)
        
        # Simple PnL calculation
        returns = target_data["close"].pct_change()
        
        # Track position
        position = 0
        pnl = []
        trades = 0
        
        for idx in target_data.index:
            if idx not in signals.index:
                continue
            
            sig = signals.loc[idx]
            ret = returns.get(idx, 0)
            
            if position == 1:
                pnl.append(ret)
            else:
                pnl.append(0)
            
            if sig.value == 1 and position == 0:  # BUY
                position = 1
                trades += 1
            elif sig.value == -1 and position == 1:  # SELL
                position = 0
        
        pnl_series = pd.Series(pnl)
        
        # Calculate metrics
        if len(pnl_series) > 0 and pnl_series.std() > 0:
            total_return = (1 + pnl_series).prod() - 1
            sharpe = pnl_series.mean() / pnl_series.std() * np.sqrt(24 * 365)
            max_dd = (pnl_series.cumsum() - pnl_series.cumsum().cummax()).min()
        else:
            total_return = 0
            sharpe = 0
            max_dd = 0
        
        return {
            "total_return": round(float(total_return) * 100, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown": round(float(max_dd) * 100, 2),
            "trades": trades,
            "trades_per_month": round(trades / (days / 30), 1),
        }
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {"error": str(e)}


def generate_recommendation(result: dict) -> str:
    """Generate deployment recommendation based on results."""
    backtest = result.get("backtest", {})
    sharpe = backtest.get("sharpe_ratio", 0)
    trades_per_month = backtest.get("trades_per_month", 0)
    
    if sharpe >= 3.0 and trades_per_month >= 10:
        return "DEPLOY - High confidence"
    elif sharpe >= 1.5 and trades_per_month >= 5:
        return "VALIDATE - Run walk-forward test"
    elif sharpe >= 0.5:
        return "OPTIMIZE - Try parameter tuning"
    else:
        return "SKIP - Poor performance"


def screen_all_candidates(
    config: dict,
    repo: DataRepository,
    days: int = 365,
    status_filter: str | None = None,
) -> list[dict]:
    """Screen all candidate pairs from the configuration."""
    results = []
    
    pairs = config.get("pairs", {})
    default_params = config.get("default_params", {})
    
    for pair_name, pair_config in pairs.items():
        status = pair_config.get("status", "candidate")
        
        # Filter by status if specified
        if status_filter and status != status_filter:
            continue
            
        target = pair_config["target"]
        reference = pair_config["reference"]
        params = pair_config.get("params", default_params)
        
        result = screen_pair(target, reference, repo, days, params)
        result["pair_name"] = pair_name
        result["config_status"] = status
        results.append(result)
    
    # Sort by backtest Sharpe
    def sort_key(r):
        bt = r.get("backtest", {})
        return bt.get("sharpe_ratio", -999)
    
    results.sort(key=sort_key, reverse=True)
    
    return results


def print_screening_report(results: list[dict]) -> None:
    """Print a formatted screening report."""
    print("\n" + "=" * 80)
    print(" RATIO PAIR SCREENING REPORT")
    print("=" * 80 + "\n")
    
    passed = [r for r in results if r.get("status") == "passed"]
    failed = [r for r in results if r.get("status") == "failed"]
    errors = [r for r in results if r.get("status") == "error"]
    
    print(f"Screened: {len(results)} pairs")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Errors: {len(errors)}")
    
    if passed:
        print("\n" + "-" * 80)
        print(" PASSED PAIRS (ranked by Sharpe)")
        print("-" * 80)
        print(f"{'Pair':<12} {'Corr':>6} {'Half-Life':>10} {'Sharpe':>8} {'Return':>8} {'Trades/Mo':>10} {'Recommendation':<25}")
        print("-" * 80)
        
        for r in passed:
            pair = r.get("pair", "?")
            metrics = r.get("metrics", {})
            bt = r.get("backtest", {})
            rec = r.get("recommendation", "")
            
            print(f"{pair:<12} {metrics.get('correlation', 0):>6.2f} {metrics.get('half_life_hours', 0):>8.1f}h "
                  f"{bt.get('sharpe_ratio', 0):>8.2f} {bt.get('total_return', 0):>7.1f}% "
                  f"{bt.get('trades_per_month', 0):>10.1f} {rec:<25}")
    
    if failed:
        print("\n" + "-" * 80)
        print(" FAILED PAIRS")
        print("-" * 80)
        for r in failed:
            pair = r.get("pair", "?")
            failed_criteria = r.get("failed_criteria", [])
            print(f"  {pair}: {', '.join(failed_criteria)}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Screen ratio pairs for trading viability")
    parser.add_argument("--days", type=int, default=365, help="Days of history to analyze")
    parser.add_argument("--pair", type=str, help="Screen a specific pair (e.g., bnb_btc)")
    parser.add_argument("--all", action="store_true", help="Screen all pairs")
    parser.add_argument("--candidates", action="store_true", help="Screen only candidate pairs")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    config = load_ratio_universe()
    
    # Initialize repository
    repo = DataRepository()
    
    results = []
    
    if args.pair:
        # Screen specific pair
        pair_config = config["pairs"].get(args.pair)
        if not pair_config:
            print(f"Error: Pair '{args.pair}' not found in config")
            print(f"Available pairs: {list(config['pairs'].keys())}")
            return
        
        result = screen_pair(
            pair_config["target"],
            pair_config["reference"],
            repo,
            args.days,
            pair_config.get("params", config["default_params"])
        )
        result["pair_name"] = args.pair
        results = [result]
        
    elif args.candidates:
        # Screen only candidates
        results = screen_all_candidates(config, repo, args.days, status_filter="candidate")
        
    elif args.all:
        # Screen all pairs
        results = screen_all_candidates(config, repo, args.days)
        
    else:
        # Default: screen candidates
        results = screen_all_candidates(config, repo, args.days, status_filter="candidate")
    
    # Print report
    print_screening_report(results)
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "screening_date": datetime.now().isoformat(),
                "days_analyzed": args.days,
                "results": results,
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

