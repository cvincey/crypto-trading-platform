"""
Online learning ML strategies with rolling retraining.

This module contains ML strategies that adapt to temporal drift by
retraining periodically on recent data. This addresses the key finding
that strategies fail to generalize across time periods.

Key strategies:
- MLClassifierOnlineStrategy: Retrains on rolling window
- MLClassifierDecayStrategy: Uses exponential decay sample weights
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
# ML Classifier Online: Rolling Retraining
# =============================================================================


@strategy_registry.register(
    "ml_classifier_online",
    description="Online ML Classifier - retrains on rolling window to adapt to temporal drift",
)
class MLClassifierOnlineStrategy(BaseStrategy):
    """
    Online learning ML Classifier that retrains periodically.
    
    Key features:
    - Rolling train window (only uses recent data)
    - Retrains every N bars
    - Uses simplified feature set (top 3 from feature importance)
    - Exponential decay sample weights (recent data weighted higher)
    
    This addresses the temporal overfitting problem by:
    1. Not using stale patterns from distant past
    2. Continuously adapting to market regime changes
    3. Treating older data as less relevant
    """

    name = "ml_classifier_online"

    def _setup(
        self,
        features: list[str] | None = None,
        train_window: int = 2160,      # 90 days of 1h candles
        retrain_every: int = 168,      # Retrain weekly
        prediction_horizon: int = 10,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        n_estimators: int = 50,
        max_depth: int = 3,
        min_samples_leaf: int = 50,
        learning_rate: float = 0.1,
        use_sample_weights: bool = True,
        decay_factor: float = 0.99,    # Exponential decay for sample weights
        min_return_threshold: float = 0.02,
        **kwargs,
    ) -> None:
        # Only use top 3 features
        self.features = features or ["volume_momentum", "adx", "rsi_14"]
        self.train_window = train_window
        self.retrain_every = retrain_every
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_sample_weights = use_sample_weights
        self.decay_factor = decay_factor
        self.min_return_threshold = min_return_threshold
        
        # Model configuration
        self._model_config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "random_state": 42,
        }
        
        self._model = None
        self._scaler = StandardScaler()
        self._last_train_idx = -self.retrain_every  # Force initial training
        self._train_count = 0

    def _create_model(self) -> GradientBoostingClassifier:
        """Create a fresh model instance."""
        return GradientBoostingClassifier(**self._model_config)

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute minimal feature set."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
            try:
                if feature == "volume_momentum":
                    features_df[feature] = indicator_registry.compute("volume_momentum", candles)
                elif feature == "adx":
                    features_df[feature] = indicator_registry.compute("adx", candles)
                elif feature.startswith("rsi_"):
                    period = int(feature.split("_")[1])
                    features_df[feature] = indicator_registry.compute("rsi", candles, period=period)
                elif feature.startswith("sma_"):
                    period = int(feature.split("_")[1])
                    raw = indicator_registry.compute("sma", candles, period=period)
                    features_df[feature] = (candles["close"] - raw) / raw * 100
                elif feature.startswith("ema_"):
                    period = int(feature.split("_")[1])
                    raw = indicator_registry.compute("ema", candles, period=period)
                    features_df[feature] = (candles["close"] - raw) / raw * 100
                elif feature == "macd":
                    macd_df = indicator_registry.compute("macd", candles)
                    features_df["macd_hist"] = macd_df["histogram"]
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

    def _compute_sample_weights(self, n_samples: int) -> np.ndarray:
        """Compute exponential decay sample weights."""
        if not self.use_sample_weights:
            return np.ones(n_samples)
        
        # Exponential decay: more recent samples have higher weight
        indices = np.arange(n_samples)
        weights = np.power(self.decay_factor, n_samples - 1 - indices)
        
        # Normalize
        weights = weights / weights.sum() * n_samples
        
        return weights

    def _train_on_window(self, candles: pd.DataFrame) -> bool:
        """Train model on the most recent train_window bars."""
        if len(candles) < self.train_window:
            logger.warning(f"Insufficient data for training: {len(candles)} < {self.train_window}")
            return False
        
        # Use only the most recent train_window bars
        train_candles = candles.iloc[-self.train_window:]
        
        features_df = self._compute_features(train_candles)
        target = self._compute_target(train_candles)
        
        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]
        
        if len(X) < 100:  # Minimum samples
            logger.warning(f"Insufficient valid samples: {len(X)}")
            return False
        
        # Create fresh model and scaler
        self._model = self._create_model()
        self._scaler = StandardScaler()
        
        X_scaled = self._scaler.fit_transform(X)
        
        # Compute sample weights with exponential decay
        sample_weights = self._compute_sample_weights(len(X))
        
        # Also apply class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Combine decay weights with class weights
        combined_weights = sample_weights * np.array([class_weight_dict[yi] for yi in y])
        
        self._model.fit(X_scaled, y, sample_weight=combined_weights)
        self._train_count += 1
        
        logger.info(
            f"Online model trained (#{self._train_count}) on {len(X)} samples, "
            f"window: {train_candles.index[0]} to {train_candles.index[-1]}"
        )
        
        return True

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        
        features_df = self._compute_features(candles)
        signals = self.create_signal_series(candles.index)
        
        valid_idx = features_df.dropna().index
        if len(valid_idx) == 0:
            return signals
        
        in_position = False
        
        for i, idx in enumerate(valid_idx):
            # Check if we need to retrain
            if i - self._last_train_idx >= self.retrain_every:
                # Get all candles up to this point for training
                train_end_loc = candles.index.get_loc(idx)
                if train_end_loc >= self.train_window:
                    train_candles = candles.iloc[:train_end_loc + 1]
                    if self._train_on_window(train_candles):
                        self._last_train_idx = i
            
            # Skip if model not trained yet
            if self._model is None:
                continue
            
            # Generate signal
            row = features_df.loc[[idx]]
            try:
                X_scaled = self._scaler.transform(row)
                prob_up = self._model.predict_proba(X_scaled)[0, 1]
                
                if not in_position and prob_up > self.buy_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                elif in_position and prob_up < self.sell_threshold:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
            except Exception as e:
                logger.debug(f"Prediction error at {idx}: {e}")
                continue
        
        return signals


# =============================================================================
# ML Classifier Decay: Simplified with Decay Weighting
# =============================================================================


@strategy_registry.register(
    "ml_classifier_decay",
    description="ML Classifier with exponential decay sample weighting",
)
class MLClassifierDecayStrategy(BaseStrategy):
    """
    ML Classifier that uses exponential decay sample weights.
    
    Unlike the online strategy, this trains once but weighs recent
    samples more heavily, reducing the influence of old patterns.
    
    This is simpler than online retraining but still addresses
    temporal drift by trusting recent data more.
    """

    name = "ml_classifier_decay"

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
        decay_factor: float = 0.995,   # Slower decay for single training
        min_return_threshold: float = 0.02,
        **kwargs,
    ) -> None:
        self.features = features or ["volume_momentum", "adx", "rsi_14"]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.decay_factor = decay_factor
        self.min_return_threshold = min_return_threshold

        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._is_trained = False

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute minimal feature set."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
            try:
                if feature == "volume_momentum":
                    features_df[feature] = indicator_registry.compute("volume_momentum", candles)
                elif feature == "adx":
                    features_df[feature] = indicator_registry.compute("adx", candles)
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
        """Train with exponential decay sample weights."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_train_scaled = self._scaler.fit_transform(X_train)
        
        # Compute exponential decay weights
        n_samples = len(X_train)
        indices = np.arange(n_samples)
        decay_weights = np.power(self.decay_factor, n_samples - 1 - indices)
        decay_weights = decay_weights / decay_weights.sum() * n_samples
        
        # Combine with class weights
        from sklearn.utils.class_weight import compute_sample_weight
        class_weights = compute_sample_weight("balanced", y_train)
        
        combined_weights = decay_weights * class_weights
        
        self._model.fit(X_train_scaled, y_train, sample_weight=combined_weights)
        self._is_trained = True

        logger.info(
            f"ML decay model trained on {len(X_train)} samples, "
            f"decay_factor: {self.decay_factor}"
        )

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        if not self._is_trained:
            self._train(candles)

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
