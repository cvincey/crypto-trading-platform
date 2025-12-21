"""
Cross-asset ML strategies that train on multiple symbols.

This module leverages the research finding that strategies generalize
well across symbols but not across time periods. By training on ALL
symbols together, we get:
- More training data
- Model learns general crypto patterns (not symbol-specific)
- Regularization effect from diverse data

Key strategies:
- MLCrossAssetStrategy: Universal model trained on all symbols
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


# =============================================================================
# ML Cross-Asset: Universal Model Trained on Multiple Symbols
# =============================================================================


@strategy_registry.register(
    "ml_cross_asset",
    description="Cross-asset ML - trains on multiple symbols for better generalization",
)
class MLCrossAssetStrategy(BaseStrategy):
    """
    Cross-asset ML Classifier that trains on multiple symbols.
    
    Key insight from research: strategies generalize well to new symbols
    but not to new time periods. This strategy exploits that finding by:
    
    1. Training on data from ALL available symbols combined
    2. Using normalized features (relative to price) that work across assets
    3. Learning general crypto market patterns rather than symbol-specific ones
    
    The model is trained once on combined data and applied to any symbol.
    
    Training modes:
    - 'combined': Concatenate all symbol data (more data, general patterns)
    - 'symbol': Train per-symbol but share model architecture (baseline)
    
    Features are already normalized (relative to SMA, etc.) so they
    work across assets with different price scales.
    """

    name = "ml_cross_asset"

    def _setup(
        self,
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 10,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        n_estimators: int = 50,
        max_depth: int = 3,
        min_samples_leaf: int = 50,
        learning_rate: float = 0.1,
        min_return_threshold: float = 0.02,
        training_mode: str = "combined",  # 'combined' or 'symbol'
        **kwargs,
    ) -> None:
        # Only use top features that generalize well
        self.features = features or ["volume_momentum", "adx", "rsi_14"]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_return_threshold = min_return_threshold
        self.training_mode = training_mode
        
        self._model_config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "random_state": 42,
        }

        self._model = GradientBoostingClassifier(**self._model_config)
        self._scaler = StandardScaler()
        self._is_trained = False
        
        # Cache for multi-symbol training
        self._training_data: dict[str, pd.DataFrame] = {}
        self._combined_trained = False

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute normalized features that work across assets."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
            try:
                if feature == "volume_momentum":
                    # Normalized by construction
                    features_df[feature] = indicator_registry.compute("volume_momentum", candles)
                elif feature == "adx":
                    # Scale-independent (0-100)
                    features_df[feature] = indicator_registry.compute("adx", candles)
                elif feature.startswith("rsi_"):
                    # Scale-independent (0-100)
                    period = int(feature.split("_")[1])
                    features_df[feature] = indicator_registry.compute("rsi", candles, period=period)
                elif feature.startswith("sma_"):
                    # Normalize relative to SMA (percentage deviation)
                    period = int(feature.split("_")[1])
                    raw = indicator_registry.compute("sma", candles, period=period)
                    features_df[feature] = (candles["close"] - raw) / raw * 100
                elif feature.startswith("ema_"):
                    # Normalize relative to EMA (percentage deviation)
                    period = int(feature.split("_")[1])
                    raw = indicator_registry.compute("ema", candles, period=period)
                    features_df[feature] = (candles["close"] - raw) / raw * 100
                elif feature == "macd":
                    macd_df = indicator_registry.compute("macd", candles)
                    # Normalize MACD by ATR for cross-asset comparison
                    atr = indicator_registry.compute("atr", candles)
                    features_df["macd_hist"] = macd_df["histogram"] / atr * 100
                elif feature == "atr_ratio":
                    # Already normalized (ATR / price)
                    features_df[feature] = indicator_registry.compute("atr_ratio", candles)
                elif feature == "bb_width":
                    # Already normalized
                    features_df[feature] = indicator_registry.compute("bb_width", candles)
                else:
                    features_df[feature] = indicator_registry.compute(feature, candles)
            except Exception as e:
                logger.warning(f"Could not compute feature {feature}: {e}")

        return features_df

    def _compute_target(self, candles: pd.DataFrame) -> pd.Series:
        """Compute target with minimum return threshold."""
        future_returns = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )
        # Use percentage threshold - same across all assets
        return (future_returns > self.min_return_threshold).astype(int)

    def add_training_data(self, symbol: str, candles: pd.DataFrame) -> None:
        """
        Add training data for a symbol.
        
        Call this for each symbol BEFORE calling generate_signals
        to enable cross-asset training.
        
        Args:
            symbol: Symbol identifier
            candles: OHLCV DataFrame for the symbol
        """
        self._training_data[symbol] = candles.copy()
        self._combined_trained = False
        logger.info(f"Added training data for {symbol}: {len(candles)} candles")

    def _train_combined(self) -> None:
        """Train on combined data from all symbols."""
        if not self._training_data:
            logger.warning("No training data available for cross-asset training")
            return
        
        all_features = []
        all_targets = []
        
        for symbol, candles in self._training_data.items():
            features_df = self._compute_features(candles)
            target = self._compute_target(candles)
            
            valid_idx = features_df.dropna().index.intersection(target.dropna().index)
            if len(valid_idx) > 0:
                all_features.append(features_df.loc[valid_idx])
                all_targets.append(target.loc[valid_idx])
        
        if not all_features:
            logger.warning("No valid features computed for any symbol")
            return
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        # Use train_size portion
        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        
        # Scale and train
        X_train_scaled = self._scaler.fit_transform(X_train)
        
        # Use balanced class weights
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight("balanced", y_train)
        
        self._model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        self._combined_trained = True
        self._is_trained = True
        
        logger.info(
            f"Cross-asset model trained on {len(X_train)} samples "
            f"from {len(self._training_data)} symbols"
        )

    def _train_single(self, candles: pd.DataFrame) -> None:
        """Train on single symbol (fallback mode)."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_train_scaled = self._scaler.fit_transform(X_train)
        
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight("balanced", y_train)
        
        self._model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        self._is_trained = True

        logger.info(f"Single-symbol model trained on {len(X_train)} samples")

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        # Train if needed
        if not self._is_trained:
            if self.training_mode == "combined" and self._training_data:
                self._train_combined()
            else:
                self._train_single(candles)

        features_df = self._compute_features(candles)
        valid_idx = features_df.dropna().index

        signals = self.create_signal_series(candles.index)

        if len(valid_idx) == 0:
            return signals

        X = features_df.loc[valid_idx]
        X_scaled = self._scaler.transform(X)

        probas = self._model.predict_proba(X_scaled)[:, 1]

        in_position = False
        for i, idx in enumerate(valid_idx):
            prob_up = probas[i]

            if not in_position and prob_up > self.buy_threshold:
                signals.loc[idx] = Signal.BUY
                in_position = True
            elif in_position and prob_up < self.sell_threshold:
                signals.loc[idx] = Signal.SELL
                in_position = False

        return signals

    def reset_training(self) -> None:
        """Reset training state to allow retraining."""
        self._model = GradientBoostingClassifier(**self._model_config)
        self._scaler = StandardScaler()
        self._is_trained = False
        self._combined_trained = False
        self._training_data.clear()


# =============================================================================
# Cross-Asset Ensemble: Multiple Models per Regime
# =============================================================================


@strategy_registry.register(
    "ml_cross_asset_regime",
    description="Cross-asset ML with regime-specific models",
)
class MLCrossAssetRegimeStrategy(BaseStrategy):
    """
    Cross-asset strategy with separate models for trending/ranging regimes.
    
    Trains two models:
    1. Trend model: Uses ADX > 25 periods
    2. Range model: Uses ADX <= 25 periods
    
    At prediction time, uses ADX to select the appropriate model.
    """

    name = "ml_cross_asset_regime"

    def _setup(
        self,
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 10,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        adx_threshold: int = 25,
        n_estimators: int = 50,
        max_depth: int = 3,
        min_samples_leaf: int = 50,
        min_return_threshold: float = 0.02,
        **kwargs,
    ) -> None:
        self.features = features or ["volume_momentum", "rsi_14"]  # Exclude ADX (used for regime)
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.adx_threshold = adx_threshold
        self.min_return_threshold = min_return_threshold
        
        self._model_config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        }

        self._trend_model = GradientBoostingClassifier(**self._model_config)
        self._range_model = GradientBoostingClassifier(**self._model_config)
        self._scaler = StandardScaler()
        self._is_trained = False

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute features (excluding ADX which is used for regime detection)."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
            try:
                if feature == "volume_momentum":
                    features_df[feature] = indicator_registry.compute("volume_momentum", candles)
                elif feature.startswith("rsi_"):
                    period = int(feature.split("_")[1])
                    features_df[feature] = indicator_registry.compute("rsi", candles, period=period)
                else:
                    features_df[feature] = indicator_registry.compute(feature, candles)
            except Exception as e:
                logger.warning(f"Could not compute feature {feature}: {e}")

        return features_df

    def _compute_target(self, candles: pd.DataFrame) -> pd.Series:
        """Compute target with minimum return threshold."""
        future_returns = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )
        return (future_returns > self.min_return_threshold).astype(int)

    def _train(self, candles: pd.DataFrame) -> None:
        """Train separate models for trend and range regimes."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)
        adx = indicator_registry.compute("adx", candles)

        valid_idx = features_df.dropna().index.intersection(target.dropna().index).intersection(adx.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]
        adx_vals = adx.loc[valid_idx]

        # Split into regimes
        trend_mask = adx_vals > self.adx_threshold
        range_mask = ~trend_mask

        # Use train_size portion
        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        trend_mask_train = trend_mask.iloc[:split_idx]
        range_mask_train = range_mask.iloc[:split_idx]

        # Fit scaler on all data
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index)

        # Train trend model
        X_trend = X_train_scaled_df.loc[trend_mask_train]
        y_trend = y_train.loc[trend_mask_train]
        if len(X_trend) > 50:
            from sklearn.utils.class_weight import compute_sample_weight
            weights = compute_sample_weight("balanced", y_trend)
            self._trend_model.fit(X_trend, y_trend, sample_weight=weights)
            logger.info(f"Trend model trained on {len(X_trend)} samples")

        # Train range model
        X_range = X_train_scaled_df.loc[range_mask_train]
        y_range = y_train.loc[range_mask_train]
        if len(X_range) > 50:
            from sklearn.utils.class_weight import compute_sample_weight
            weights = compute_sample_weight("balanced", y_range)
            self._range_model.fit(X_range, y_range, sample_weight=weights)
            logger.info(f"Range model trained on {len(X_range)} samples")

        self._is_trained = True

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        if not self._is_trained:
            self._train(candles)

        features_df = self._compute_features(candles)
        adx = indicator_registry.compute("adx", candles)
        valid_idx = features_df.dropna().index.intersection(adx.dropna().index)

        signals = self.create_signal_series(candles.index)

        if len(valid_idx) == 0:
            return signals

        X = features_df.loc[valid_idx]
        X_scaled = self._scaler.transform(X)

        in_position = False
        for i, idx in enumerate(valid_idx):
            adx_val = adx.loc[idx]
            
            # Select model based on regime
            if adx_val > self.adx_threshold:
                model = self._trend_model
            else:
                model = self._range_model

            try:
                prob_up = model.predict_proba([X_scaled[i]])[0, 1]
            except Exception:
                continue

            if not in_position and prob_up > self.buy_threshold:
                signals.loc[idx] = Signal.BUY
                in_position = True
            elif in_position and prob_up < self.sell_threshold:
                signals.loc[idx] = Signal.SELL
                in_position = False

        return signals
