"""Machine learning-based trading strategies."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "ml_classifier",
    description="Machine Learning Classifier Strategy",
)
class MLClassifierStrategy(BaseStrategy):
    """
    Machine Learning Classifier Strategy.
    
    Uses sklearn classifiers to predict price direction.
    Trains on historical data and generates signals based on predictions.
    """

    name = "ml_classifier"

    MODELS = {
        "gradient_boosting": GradientBoostingClassifier,
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
    }

    def _setup(
        self,
        model: str = "gradient_boosting",
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 1,
        **kwargs,
    ) -> None:
        self.model_name = model
        self.features = features or ["sma_20", "rsi_14", "macd"]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.model_params = kwargs

        # Initialize model
        model_class = self.MODELS.get(model)
        if model_class is None:
            raise ValueError(
                f"Unknown model: {model}. Available: {list(self.MODELS.keys())}"
            )

        # Filter out non-model params
        valid_params = {
            k: v
            for k, v in kwargs.items()
            if k not in ["symbols", "interval", "enabled"]
        }
        self._model = model_class(**valid_params)
        self._scaler = StandardScaler()
        self._is_trained = False

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute feature matrix from candles."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
            if feature.startswith("sma_"):
                period = int(feature.split("_")[1])
                features_df[feature] = indicator_registry.compute(
                    "sma", candles, period=period
                )
            elif feature.startswith("ema_"):
                period = int(feature.split("_")[1])
                features_df[feature] = indicator_registry.compute(
                    "ema", candles, period=period
                )
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
            elif feature == "volume_sma":
                features_df[feature] = indicator_registry.compute(
                    "volume_sma", candles, period=20
                )
            elif feature.startswith("momentum_"):
                period = int(feature.split("_")[1])
                features_df[feature] = indicator_registry.compute(
                    "momentum", candles, period=period
                )
            elif feature.startswith("roc_"):
                period = int(feature.split("_")[1])
                features_df[feature] = indicator_registry.compute(
                    "roc", candles, period=period
                )
            elif feature == "atr":
                features_df[feature] = indicator_registry.compute(
                    "atr", candles, period=14
                )
            else:
                # Try to compute directly
                try:
                    features_df[feature] = indicator_registry.compute(
                        feature, candles
                    )
                except Exception as e:
                    logger.warning(f"Could not compute feature {feature}: {e}")

        return features_df

    def _compute_target(self, candles: pd.DataFrame) -> pd.Series:
        """Compute target variable (future price direction)."""
        future_returns = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )
        # 1 for positive return, 0 for negative
        return (future_returns > 0).astype(int)

    def _train(self, candles: pd.DataFrame) -> None:
        """Train the ML model on historical data."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        # Remove NaN rows
        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        # Train/test split
        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        # Scale features
        X_train_scaled = self._scaler.fit_transform(X_train)

        # Train model
        self._model.fit(X_train_scaled, y_train)
        self._is_trained = True

        logger.info(
            f"ML model trained on {len(X_train)} samples, "
            f"features: {list(X.columns)}"
        )

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        # Train if not already trained
        if not self._is_trained:
            self._train(candles)

        # Compute features
        features_df = self._compute_features(candles)
        valid_idx = features_df.dropna().index

        signals = self.create_signal_series(candles.index)

        if len(valid_idx) == 0:
            return signals

        X = features_df.loc[valid_idx]
        X_scaled = self._scaler.transform(X)

        # Get predictions
        predictions = self._model.predict(X_scaled)

        # Convert predictions to signals
        for i, idx in enumerate(valid_idx):
            if i == 0:
                continue
            
            # Look for changes in prediction
            if predictions[i] == 1 and predictions[i - 1] == 0:
                signals.loc[idx] = Signal.BUY
            elif predictions[i] == 0 and predictions[i - 1] == 1:
                signals.loc[idx] = Signal.SELL

        return signals


@strategy_registry.register(
    "ml_regression",
    description="ML Regression Strategy (predicts returns)",
)
class MLRegressionStrategy(BaseStrategy):
    """
    Machine Learning Regression Strategy.
    
    Predicts expected returns and generates signals based on
    predicted return magnitude.
    """

    name = "ml_regression"

    def _setup(
        self,
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 1,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
        **kwargs,
    ) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        self.features = features or ["sma_20", "rsi_14", "macd"]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        valid_params = {
            k: v
            for k, v in kwargs.items()
            if k not in ["symbols", "interval", "enabled"]
        }
        self._model = GradientBoostingRegressor(**valid_params)
        self._scaler = StandardScaler()
        self._is_trained = False

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute feature matrix from candles."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
            if feature.startswith("sma_"):
                period = int(feature.split("_")[1])
                features_df[feature] = indicator_registry.compute(
                    "sma", candles, period=period
                )
            elif feature.startswith("rsi_"):
                period = int(feature.split("_")[1])
                features_df[feature] = indicator_registry.compute(
                    "rsi", candles, period=period
                )
            elif feature == "macd":
                macd_df = indicator_registry.compute("macd", candles)
                features_df["macd"] = macd_df["macd"]
                features_df["macd_signal"] = macd_df["signal"]

        return features_df

    def _train(self, candles: pd.DataFrame) -> None:
        """Train the ML model on historical data."""
        features_df = self._compute_features(candles)
        
        # Target is future returns
        target = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )

        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_train_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_train_scaled, y_train)
        self._is_trained = True

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

        # Get predicted returns
        predicted_returns = self._model.predict(X_scaled)

        # Generate signals based on predicted returns
        for i, idx in enumerate(valid_idx):
            if predicted_returns[i] > self.buy_threshold:
                signals.loc[idx] = Signal.BUY
            elif predicted_returns[i] < self.sell_threshold:
                signals.loc[idx] = Signal.SELL

        return signals


@strategy_registry.register(
    "ml_enhanced",
    description="Enhanced ML Strategy with walk-forward validation",
)
class MLEnhancedStrategy(BaseStrategy):
    """
    Enhanced Machine Learning Strategy.
    
    Features:
    - Extended feature set (OBV, ATR ratio, BB width, volume momentum)
    - Walk-forward validation for more realistic backtesting
    - Configurable retrain frequency
    
    Config params:
        model: ML model type (gradient_boosting, random_forest, xgboost)
        features: List of feature names to compute
        walk_forward: Enable walk-forward validation
        train_window: Training window size (bars)
        test_window: Test window size before retraining
        retrain_frequency: How often to retrain (bars)
    """

    name = "ml_enhanced"

    MODELS = {
        "gradient_boosting": GradientBoostingClassifier,
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
    }

    def _setup(
        self,
        model: str = "gradient_boosting",
        features: list[str] | None = None,
        walk_forward: bool = True,
        train_window: int = 500,
        test_window: int = 100,
        retrain_frequency: int = 100,
        prediction_horizon: int = 1,
        **kwargs,
    ) -> None:
        self.model_name = model
        self.features = features or [
            "sma_20", "rsi_14", "macd", "obv", "atr_ratio", "bb_width", "volume_momentum"
        ]
        self.walk_forward = walk_forward
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_frequency = retrain_frequency
        self.prediction_horizon = prediction_horizon

        model_class = self.MODELS.get(model)
        if model_class is None:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.MODELS.keys())}")

        valid_params = {
            k: v for k, v in kwargs.items()
            if k not in ["symbols", "interval", "enabled"]
        }
        self._model = model_class(**valid_params)
        self._scaler = StandardScaler()
        self._last_train_idx = 0

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute extended feature matrix from candles."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
            try:
                if feature.startswith("sma_"):
                    period = int(feature.split("_")[1])
                    features_df[feature] = indicator_registry.compute("sma", candles, period=period)
                elif feature.startswith("ema_"):
                    period = int(feature.split("_")[1])
                    features_df[feature] = indicator_registry.compute("ema", candles, period=period)
                elif feature.startswith("rsi_"):
                    period = int(feature.split("_")[1])
                    features_df[feature] = indicator_registry.compute("rsi", candles, period=period)
                elif feature == "macd":
                    macd_df = indicator_registry.compute("macd", candles)
                    features_df["macd"] = macd_df["macd"]
                    features_df["macd_signal"] = macd_df["signal"]
                    features_df["macd_hist"] = macd_df["histogram"]
                elif feature == "obv":
                    features_df[feature] = indicator_registry.compute("obv", candles)
                elif feature == "atr_ratio":
                    features_df[feature] = indicator_registry.compute("atr_ratio", candles)
                elif feature == "bb_width":
                    features_df[feature] = indicator_registry.compute("bb_width", candles)
                elif feature == "volume_momentum":
                    features_df[feature] = indicator_registry.compute("volume_momentum", candles)
                elif feature == "adx":
                    features_df[feature] = indicator_registry.compute("adx", candles)
                elif feature == "atr":
                    features_df[feature] = indicator_registry.compute("atr", candles)
                elif feature == "volume_sma":
                    features_df[feature] = indicator_registry.compute("volume_sma", candles)
                else:
                    # Try direct computation
                    features_df[feature] = indicator_registry.compute(feature, candles)
            except Exception as e:
                logger.warning(f"Could not compute feature {feature}: {e}")

        return features_df

    def _compute_target(self, candles: pd.DataFrame) -> pd.Series:
        """Compute target variable (future price direction)."""
        future_returns = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )
        return (future_returns > 0).astype(int)

    def _train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on given data."""
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        # Get valid indices
        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        signals = self.create_signal_series(candles.index)

        if len(valid_idx) < self.train_window + 10:
            logger.warning("Insufficient data for ML training")
            return signals

        if self.walk_forward:
            # Walk-forward validation
            predictions = np.zeros(len(valid_idx))
            
            for i in range(self.train_window, len(valid_idx)):
                # Check if we need to retrain
                if (i - self._last_train_idx) >= self.retrain_frequency or self._last_train_idx == 0:
                    train_start = max(0, i - self.train_window)
                    train_end = i
                    
                    X_train = features_df.loc[valid_idx[train_start:train_end]]
                    y_train = target.loc[valid_idx[train_start:train_end]]
                    
                    self._train_model(X_train, y_train)
                    self._last_train_idx = i

                # Predict
                X_pred = features_df.loc[[valid_idx[i]]]
                X_pred_scaled = self._scaler.transform(X_pred)
                predictions[i] = self._model.predict(X_pred_scaled)[0]

            # Convert to signals
            for i in range(self.train_window + 1, len(valid_idx)):
                idx = valid_idx[i]
                if predictions[i] == 1 and predictions[i - 1] == 0:
                    signals.loc[idx] = Signal.BUY
                elif predictions[i] == 0 and predictions[i - 1] == 1:
                    signals.loc[idx] = Signal.SELL
        else:
            # Standard train/test split
            split_idx = int(len(valid_idx) * 0.8)
            X_train = features_df.loc[valid_idx[:split_idx]]
            y_train = target.loc[valid_idx[:split_idx]]
            
            self._train_model(X_train, y_train)
            
            X_test = features_df.loc[valid_idx]
            X_test_scaled = self._scaler.transform(X_test)
            predictions = self._model.predict(X_test_scaled)
            
            for i, idx in enumerate(valid_idx):
                if i == 0:
                    continue
                if predictions[i] == 1 and predictions[i - 1] == 0:
                    signals.loc[idx] = Signal.BUY
                elif predictions[i] == 0 and predictions[i - 1] == 1:
                    signals.loc[idx] = Signal.SELL

        return self.apply_filters(signals, candles)
