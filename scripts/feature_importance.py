#!/usr/bin/env python3
"""
Feature importance analysis for ML strategies.

Extracts and ranks feature importances from trained ML models.
Optionally uses SHAP for interpretability.

Configuration: config/optimization.yaml
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.config.settings import get_settings
from crypto.data.repository import CandleRepository
from crypto.indicators.base import indicator_registry
from crypto.strategies.registry import strategy_registry

# Import strategies to trigger registration
from crypto.strategies import ml
from crypto.strategies import ml_siblings

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


def compute_features(candles: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Compute feature matrix from candles."""
    features_df = pd.DataFrame(index=candles.index)

    for feature in feature_names:
        try:
            if feature.startswith("sma_"):
                period = int(feature.split("_")[1])
                raw = indicator_registry.compute("sma", candles, period=period)
                features_df[feature] = (candles["close"] - raw) / raw * 100
            elif feature.startswith("ema_"):
                period = int(feature.split("_")[1])
                raw = indicator_registry.compute("ema", candles, period=period)
                features_df[feature] = (candles["close"] - raw) / raw * 100
            elif feature.startswith("rsi_"):
                period = int(feature.split("_")[1])
                features_df[feature] = indicator_registry.compute(
                    "rsi", candles, period=period
                )
            elif feature == "macd":
                macd_df = indicator_registry.compute("macd", candles)
                features_df["macd"] = macd_df["macd"]
                features_df["macd_signal"] = macd_df["signal"]
                features_df["macd_hist"] = macd_df["histogram"]
            elif feature == "adx":
                features_df[feature] = indicator_registry.compute("adx", candles)
            elif feature == "atr_ratio":
                features_df[feature] = indicator_registry.compute("atr_ratio", candles)
            elif feature == "bb_width":
                features_df[feature] = indicator_registry.compute("bb_width", candles)
            elif feature == "obv":
                obv = indicator_registry.compute("obv", candles)
                features_df[feature] = obv.pct_change(10) * 100
            elif feature == "volume_momentum":
                features_df[feature] = indicator_registry.compute("volume_momentum", candles)
            elif feature.startswith("roc_"):
                period = int(feature.split("_")[1])
                features_df[feature] = indicator_registry.compute(
                    "roc", candles, period=period
                )
            else:
                features_df[feature] = indicator_registry.compute(feature, candles)
        except Exception as e:
            logger.warning(f"Could not compute feature {feature}: {e}")

    return features_df


def compute_target(candles: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Compute target variable (future price direction)."""
    future_returns = candles["close"].pct_change(horizon).shift(-horizon)
    return (future_returns > 0).astype(int)


async def analyze_feature_importance() -> dict[str, Any]:
    """Analyze feature importance for ML strategies."""
    settings = get_settings()
    config = settings.optimization.optimization.feature_analysis

    console.print(Panel(
        "[bold blue]Feature Importance Analysis[/bold blue]\n"
        f"Analyzing {len(config.features)} features for {len(config.strategies)} strategies",
        expand=False,
    ))

    # Load data
    repository = CandleRepository()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=config.days)

    console.print(f"\n[bold]Loading data for {len(config.symbols)} symbols...[/bold]")

    all_candles = []
    for symbol in config.symbols:
        candles = await repository.get_candles_df(
            symbol, config.interval, start, end
        )
        if not candles.empty:
            all_candles.append(candles)
            console.print(f"  {symbol}: {len(candles)} candles")

    if not all_candles:
        console.print("[red]No data available![/red]")
        return {}

    # Combine all candles for training
    combined = pd.concat(all_candles)
    console.print(f"\n[bold]Total samples: {len(combined)}[/bold]")

    results = {}

    for strategy_name in config.strategies:
        console.print(f"\n[bold]Analyzing {strategy_name}...[/bold]")

        try:
            # Compute features
            features_df = compute_features(combined, config.features)
            target = compute_target(combined)

            # Remove NaN and infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            valid_idx = features_df.dropna().index.intersection(target.dropna().index)
            X = features_df.loc[valid_idx].copy()
            y = target.loc[valid_idx].copy()
            
            # Double-check for any remaining NaN
            X = X.fillna(0)

            console.print(f"  Valid samples: {len(X)}")
            console.print(f"  Features: {list(X.columns)}")

            # Train model
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Try XGBoost first
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                    verbosity=0,
                )
                console.print("  Using XGBoost")
            except ImportError:
                model = GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                )
                console.print("  Using GradientBoosting")

            model.fit(X_scaled, y)

            # Get feature importances
            importances = model.feature_importances_
            feature_importance = dict(zip(X.columns, importances))

            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            results[strategy_name] = {
                "feature_importance": dict(sorted_features),
                "top_features": sorted_features[:config.top_n_features],
                "model_type": type(model).__name__,
            }

            # Try SHAP if enabled
            if config.use_shap:
                try:
                    import shap

                    console.print("  Computing SHAP values...")

                    # Use a sample for SHAP (it's slow)
                    sample_size = min(1000, len(X_scaled))
                    sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
                    X_sample = X_scaled[sample_idx]

                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)

                    # Mean absolute SHAP values
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Class 1 for binary

                    mean_shap = np.abs(shap_values).mean(axis=0)
                    shap_importance = dict(zip(X.columns, mean_shap))

                    sorted_shap = sorted(
                        shap_importance.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )

                    results[strategy_name]["shap_importance"] = dict(sorted_shap)
                    results[strategy_name]["top_shap_features"] = sorted_shap[:config.top_n_features]
                    console.print("  SHAP analysis complete")

                except ImportError:
                    console.print("  [yellow]SHAP not installed (pip install shap)[/yellow]")
                except Exception as e:
                    console.print(f"  [yellow]SHAP failed: {e}[/yellow]")

            # Display results
            display_importance(strategy_name, results[strategy_name])

        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")
            logger.exception(f"Failed to analyze {strategy_name}")

    return results


def display_importance(strategy_name: str, result: dict) -> None:
    """Display feature importance for a strategy."""
    table = Table(title=f"Feature Importance: {strategy_name}")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Feature", style="cyan")
    table.add_column("Model Importance", justify="right")

    if "shap_importance" in result:
        table.add_column("SHAP Importance", justify="right")

    top_features = result.get("top_features", [])
    shap_dict = result.get("shap_importance", {})

    for i, (feature, importance) in enumerate(top_features, 1):
        row = [str(i), feature, f"{importance:.4f}"]

        if shap_dict:
            shap_val = shap_dict.get(feature, 0)
            row.append(f"{shap_val:.4f}")

        table.add_row(*row)

    console.print(table)


def save_results(results: dict, output_dir: Path) -> None:
    """Save feature importance results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Recursively convert
    def deep_convert(d):
        if isinstance(d, dict):
            return {k: deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [deep_convert(v) for v in d]
        if isinstance(d, tuple):
            return [deep_convert(v) for v in d]
        return convert(d)

    results_converted = deep_convert(results)

    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(results_converted, f, indent=2)

    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")

    # Summary recommendations
    console.print("\n[bold]Feature Selection Recommendations:[/bold]")

    for strategy_name, result in results.items():
        top = result.get("top_features", [])[:5]
        if top:
            features_str = ", ".join(f[0] for f in top)
            console.print(f"\n[yellow]{strategy_name}[/yellow]")
            console.print(f"  Top 5: {features_str}")

            # Check for low-importance features
            all_importance = result.get("feature_importance", {})
            low_features = [k for k, v in all_importance.items() if v < 0.01]
            if low_features:
                console.print(f"  [dim]Consider removing: {', '.join(low_features[:3])}[/dim]")


async def main():
    """Main entry point."""
    console.print("[bold blue]═══ Feature Importance Analysis ═══[/bold blue]\n")

    results = await analyze_feature_importance()

    if results:
        settings = get_settings()
        output_dir = Path(settings.optimization.optimization.output.results_dir)
        save_results(results, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
