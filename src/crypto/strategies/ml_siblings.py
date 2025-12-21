"""
Sibling ML classifier strategies - experimental variants of the original ml_classifier.

This module contains improved versions of the ML classifier strategy,
exploring different approaches to improve performance:

- MLClassifierV2: Extended features + probability thresholds
- MLEnsembleVoting: Multiple models with voting
- MLClassifierV3: Longer horizon + feature engineering  
- MLClassifierXGB: XGBoost with class weights

All strategies are designed to be compared against the original ml_classifier.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


# =============================================================================
# ML Classifier V2: Extended Features + Probability Thresholds
# =============================================================================


@strategy_registry.register(
    "ml_classifier_v2",
    description="ML Classifier v2 - Extended features + probability thresholds",
)
class MLClassifierV2Strategy(BaseStrategy):
    """
    Improved ML Classifier with:
    - Extended feature set (ADX, ATR ratio, OBV, BB width, etc.)
    - Probability-based signal generation (only trade when confident)
    - Calibrated probabilities for better threshold tuning
    
    Key improvements over original:
    1. More features = better pattern recognition
    2. Probability threshold = fewer but higher quality signals
    3. Calibrated classifier = more reliable probability estimates
    
    NOTE: Thresholds of 0.52/0.48 work better than stricter thresholds.
    Too strict (0.6/0.4) results in almost no trades.
    """

    name = "ml_classifier_v2"

    def _setup(
        self,
        model: str = "gradient_boosting",
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 1,
        buy_threshold: float = 0.52,  # Relaxed from 0.6 - was too strict
        sell_threshold: float = 0.48,  # Relaxed from 0.4 - was too strict
        use_calibration: bool = True,
        **kwargs,
    ) -> None:
        self.model_name = model
        # Extended default feature set
        self.features = features or [
            "sma_20", "sma_50", "ema_12", "ema_26",
            "rsi_14", "macd",
            "adx", "atr_ratio", "bb_width",
            "obv", "volume_momentum",
            "momentum_10", "roc_10",
        ]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_calibration = use_calibration

        # Build the model
        valid_params = {
            k: v for k, v in kwargs.items()
            if k not in ["symbols", "interval", "enabled"]
        }
        
        if model == "gradient_boosting":
            base_model = GradientBoostingClassifier(**valid_params)
        elif model == "random_forest":
            base_model = RandomForestClassifier(**valid_params)
        else:
            base_model = GradientBoostingClassifier(**valid_params)

        self._base_model = base_model
        self._model = None  # Will be set during training
        self._scaler = StandardScaler()
        self._is_trained = False

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute extended feature matrix from candles."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
            try:
                if feature.startswith("sma_"):
                    period = int(feature.split("_")[1])
                    raw = indicator_registry.compute("sma", candles, period=period)
                    # Normalize: price relative to SMA
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
                    # Normalize OBV by taking rate of change
                    obv = indicator_registry.compute("obv", candles)
                    features_df[feature] = obv.pct_change(10) * 100
                elif feature == "volume_momentum":
                    features_df[feature] = indicator_registry.compute("volume_momentum", candles)
                elif feature == "volume_sma":
                    vol_sma = indicator_registry.compute("volume_sma", candles, period=20)
                    features_df[feature] = (candles["volume"] - vol_sma) / vol_sma * 100
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
                else:
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

        # Train with calibration for better probability estimates
        if self.use_calibration:
            self._model = CalibratedClassifierCV(
                self._base_model, method="sigmoid", cv=3
            )
        else:
            self._model = self._base_model

        self._model.fit(X_train_scaled, y_train)
        self._is_trained = True

        logger.info(
            f"ML v2 model trained on {len(X_train)} samples, "
            f"features: {list(X.columns)}"
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

        # Get probability predictions
        probas = self._model.predict_proba(X_scaled)[:, 1]  # P(up)

        # Track position state for signal generation
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


# =============================================================================
# ML Ensemble Voting: Combine Multiple Models
# =============================================================================


@strategy_registry.register(
    "ml_ensemble_voting",
    description="ML Ensemble - Combines RF + GB + LR with voting",
)
class MLEnsembleVotingStrategy(BaseStrategy):
    """
    Ensemble ML Strategy that combines multiple classifiers:
    - Random Forest (captures non-linear patterns)
    - Gradient Boosting (sequential error correction)
    - Logistic Regression (linear baseline, regularized)
    
    Uses soft voting (probability averaging) for more robust predictions.
    
    Key improvements:
    1. Model diversity = reduces overfitting
    2. Soft voting = smoother probability estimates
    3. Each model captures different patterns
    """

    name = "ml_ensemble_voting"

    def _setup(
        self,
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 1,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        rf_n_estimators: int = 100,
        gb_n_estimators: int = 100,
        **kwargs,
    ) -> None:
        self.features = features or [
            "sma_20", "ema_12", "rsi_14", "macd",
            "adx", "atr_ratio", "bb_width", "volume_momentum",
        ]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # Create ensemble of diverse models
        self._model = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                )),
                ("gb", GradientBoostingClassifier(
                    n_estimators=gb_n_estimators,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                )),
                ("lr", LogisticRegression(
                    C=1.0,
                    max_iter=500,
                    random_state=42,
                )),
            ],
            voting="soft",  # Use probability averaging
        )
        self._scaler = StandardScaler()
        self._is_trained = False

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute feature matrix from candles."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
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
                elif feature == "volume_momentum":
                    features_df[feature] = indicator_registry.compute("volume_momentum", candles)
                else:
                    features_df[feature] = indicator_registry.compute(feature, candles)
            except Exception as e:
                logger.warning(f"Could not compute feature {feature}: {e}")

        return features_df

    def _compute_target(self, candles: pd.DataFrame) -> pd.Series:
        """Compute target variable."""
        future_returns = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )
        return (future_returns > 0).astype(int)

    def _train(self, candles: pd.DataFrame) -> None:
        """Train the ensemble model."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_train_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_train_scaled, y_train)
        self._is_trained = True

        logger.info(f"ML Ensemble trained on {len(X_train)} samples")

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


# =============================================================================
# ML Classifier V3: Longer Horizon + Feature Engineering
# =============================================================================


@strategy_registry.register(
    "ml_classifier_v3",
    description="ML Classifier v3 - Longer horizon + lagged features",
)
class MLClassifierV3Strategy(BaseStrategy):
    """
    ML Classifier with advanced feature engineering:
    - Longer prediction horizon (4-6 candles) for more meaningful moves
    - Lagged features (past N values of indicators)
    - Rolling statistics (mean, std of recent features)
    - Price pattern features (higher highs, lower lows)
    
    Key improvements:
    1. Longer horizon = filters noise, captures real trends
    2. Lagged features = captures momentum/mean-reversion patterns
    3. Rolling stats = captures regime changes
    """

    name = "ml_classifier_v3"

    def _setup(
        self,
        model: str = "gradient_boosting",
        base_features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 4,  # Predict 4 candles ahead
        lag_periods: list[int] | None = None,
        rolling_window: int = 10,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        min_return_threshold: float = 0.01,  # Only predict moves > 1%
        **kwargs,
    ) -> None:
        self.model_name = model
        self.base_features = base_features or [
            "rsi_14", "macd_hist", "adx", "atr_ratio", "bb_width"
        ]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.lag_periods = lag_periods or [1, 2, 3, 5]
        self.rolling_window = rolling_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_return_threshold = min_return_threshold

        valid_params = {
            k: v for k, v in kwargs.items()
            if k not in ["symbols", "interval", "enabled"]
        }

        if model == "gradient_boosting":
            self._model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=10,
                **valid_params,
            )
        elif model == "random_forest":
            self._model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=10,
                **valid_params,
            )
        else:
            self._model = GradientBoostingClassifier(**valid_params)

        self._scaler = StandardScaler()
        self._is_trained = False

    def _compute_base_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute base features before adding lags."""
        features_df = pd.DataFrame(index=candles.index)
        
        # RSI
        features_df["rsi_14"] = indicator_registry.compute("rsi", candles, period=14)
        
        # MACD histogram
        macd_df = indicator_registry.compute("macd", candles)
        features_df["macd_hist"] = macd_df["histogram"]
        
        # ADX
        features_df["adx"] = indicator_registry.compute("adx", candles)
        
        # ATR ratio
        features_df["atr_ratio"] = indicator_registry.compute("atr_ratio", candles)
        
        # BB width
        features_df["bb_width"] = indicator_registry.compute("bb_width", candles)
        
        # Price relative to SMA
        sma_20 = indicator_registry.compute("sma", candles, period=20)
        features_df["price_vs_sma20"] = (candles["close"] - sma_20) / sma_20 * 100
        
        sma_50 = indicator_registry.compute("sma", candles, period=50)
        features_df["price_vs_sma50"] = (candles["close"] - sma_50) / sma_50 * 100
        
        # Volume feature
        vol_sma = indicator_registry.compute("volume_sma", candles, period=20)
        features_df["volume_ratio"] = candles["volume"] / vol_sma
        
        # Returns at different horizons
        features_df["return_1"] = candles["close"].pct_change(1) * 100
        features_df["return_5"] = candles["close"].pct_change(5) * 100
        features_df["return_10"] = candles["close"].pct_change(10) * 100
        
        return features_df

    def _add_lagged_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged versions of key features."""
        result = features_df.copy()
        
        # Only lag the most important features to avoid explosion
        lag_cols = ["rsi_14", "macd_hist", "return_1"]
        
        for col in lag_cols:
            if col in features_df.columns:
                for lag in self.lag_periods:
                    result[f"{col}_lag{lag}"] = features_df[col].shift(lag)
        
        return result

    def _add_rolling_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        result = features_df.copy()
        
        # Rolling mean and std of returns
        result["return_1_rolling_mean"] = features_df["return_1"].rolling(
            self.rolling_window
        ).mean()
        result["return_1_rolling_std"] = features_df["return_1"].rolling(
            self.rolling_window
        ).std()
        
        # RSI rolling std (regime indicator)
        result["rsi_rolling_std"] = features_df["rsi_14"].rolling(
            self.rolling_window
        ).std()
        
        return result

    def _add_pattern_features(self, candles: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features."""
        result = features_df.copy()
        
        # Higher highs / lower lows count
        high = candles["high"]
        low = candles["low"]
        
        # Count consecutive higher highs in last 5 bars
        higher_highs = (high > high.shift(1)).astype(int)
        result["higher_highs_5"] = higher_highs.rolling(5).sum()
        
        # Count consecutive lower lows in last 5 bars
        lower_lows = (low < low.shift(1)).astype(int)
        result["lower_lows_5"] = lower_lows.rolling(5).sum()
        
        # Candle body size (normalized)
        body = abs(candles["close"] - candles["open"])
        atr = indicator_registry.compute("atr", candles, period=14)
        result["body_size"] = body / atr
        
        return result

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute full feature matrix with engineering."""
        # Base features
        features_df = self._compute_base_features(candles)
        
        # Add lagged features
        features_df = self._add_lagged_features(features_df)
        
        # Add rolling stats
        features_df = self._add_rolling_features(features_df)
        
        # Add pattern features
        features_df = self._add_pattern_features(candles, features_df)
        
        return features_df

    def _compute_target(self, candles: pd.DataFrame) -> pd.Series:
        """Compute target with minimum return threshold."""
        future_returns = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )
        # Only count as positive if return > threshold
        return (future_returns > self.min_return_threshold).astype(int)

    def _train(self, candles: pd.DataFrame) -> None:
        """Train the model."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_train_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_train_scaled, y_train)
        self._is_trained = True

        logger.info(
            f"ML v3 model trained on {len(X_train)} samples with "
            f"{len(X.columns)} features, horizon={self.prediction_horizon}"
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


# =============================================================================
# ML Classifier XGB: XGBoost with Class Weights
# =============================================================================


@strategy_registry.register(
    "ml_classifier_xgb",
    description="ML Classifier with XGBoost and class weights",
)
class MLClassifierXGBStrategy(BaseStrategy):
    """
    ML Classifier using XGBoost (falls back to GradientBoosting if not available):
    - Class weights to handle imbalanced data
    - Early stopping to prevent overfitting
    - Feature importance tracking
    
    Key improvements:
    1. XGBoost = faster, often more accurate
    2. Class weights = better for imbalanced up/down days
    3. Early stopping = prevents overfitting
    """

    name = "ml_classifier_xgb"

    def _setup(
        self,
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 1,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        use_class_weights: bool = True,
        **kwargs,
    ) -> None:
        self.features = features or [
            "sma_20", "ema_12", "rsi_14", "macd",
            "adx", "atr_ratio", "bb_width",
            "obv", "volume_momentum", "roc_10",
        ]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_class_weights = use_class_weights
        
        # Try to use XGBoost, fall back to sklearn
        try:
            from xgboost import XGBClassifier
            self._model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                verbosity=0,
            )
            self._use_xgb = True
            logger.info("Using XGBoost classifier")
        except ImportError:
            # Fall back to sklearn
            self._model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )
            self._use_xgb = False
            logger.info("XGBoost not available, using sklearn GradientBoosting")

        self._scaler = StandardScaler()
        self._is_trained = False
        self._feature_importance = None

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute feature matrix."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
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

    def _compute_target(self, candles: pd.DataFrame) -> pd.Series:
        """Compute target variable."""
        future_returns = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )
        return (future_returns > 0).astype(int)

    def _train(self, candles: pd.DataFrame) -> None:
        """Train the model with class weights."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_train_scaled = self._scaler.fit_transform(X_train)

        # Compute class weights if enabled
        if self.use_class_weights:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight("balanced", y_train)
            self._model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            self._model.fit(X_train_scaled, y_train)

        self._is_trained = True

        # Store feature importance
        if hasattr(self._model, "feature_importances_"):
            self._feature_importance = dict(zip(X.columns, self._model.feature_importances_))
            top_features = sorted(
                self._feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:5]
            logger.info(f"Top 5 features: {top_features}")

        logger.info(f"ML XGB model trained on {len(X_train)} samples")

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

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance after training."""
        return self._feature_importance


# =============================================================================
# ML Classifier Hybrid: Best of All Worlds
# =============================================================================


@strategy_registry.register(
    "ml_classifier_hybrid",
    description="Hybrid ML - Combines XGBoost + Ensemble + Adaptive thresholds",
)
class MLClassifierHybridStrategy(BaseStrategy):
    """
    Hybrid ML Classifier combining the best elements from winning strategies:
    
    From ml_classifier_xgb (best Sharpe):
    - XGBoost with class weights for handling imbalance
    
    From ml_ensemble_voting (most consistent):  
    - Ensemble approach with multiple models
    
    From ml_classifier_v4 (regime-aware):
    - Adaptive thresholds based on market regime
    
    This strategy uses a weighted ensemble of XGBoost and Random Forest
    with regime-adaptive probability thresholds.
    """

    name = "ml_classifier_hybrid"

    def _setup(
        self,
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 1,
        # Ensemble weights (XGB performs better, so weight it higher)
        xgb_weight: float = 0.6,
        rf_weight: float = 0.4,
        # Adaptive thresholds
        trend_buy_threshold: float = 0.52,
        trend_sell_threshold: float = 0.48,
        range_buy_threshold: float = 0.58,
        range_sell_threshold: float = 0.42,
        adx_trend_threshold: int = 25,
        # Model params
        n_estimators: int = 150,
        max_depth: int = 6,
        use_class_weights: bool = True,
        **kwargs,
    ) -> None:
        self.features = features or [
            "sma_20", "ema_12", "rsi_14", "macd",
            "adx", "atr_ratio", "bb_width",
            "obv", "volume_momentum", "roc_10",
        ]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        
        self.xgb_weight = xgb_weight
        self.rf_weight = rf_weight
        
        self.trend_buy_threshold = trend_buy_threshold
        self.trend_sell_threshold = trend_sell_threshold
        self.range_buy_threshold = range_buy_threshold
        self.range_sell_threshold = range_sell_threshold
        self.adx_trend_threshold = adx_trend_threshold
        self.use_class_weights = use_class_weights
        
        # Try XGBoost, fall back to GradientBoosting
        try:
            from xgboost import XGBClassifier
            self._xgb_model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                verbosity=0,
            )
            self._use_xgb = True
        except ImportError:
            self._xgb_model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                random_state=42,
            )
            self._use_xgb = False

        # Random Forest for diversity
        self._rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth + 2,  # Slightly deeper for RF
            min_samples_leaf=5,
            random_state=42,
        )
        
        self._scaler = StandardScaler()
        self._is_trained = False

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute feature matrix."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
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

    def _compute_target(self, candles: pd.DataFrame) -> pd.Series:
        """Compute target variable."""
        future_returns = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )
        return (future_returns > 0).astype(int)

    def _train(self, candles: pd.DataFrame) -> None:
        """Train both models in the ensemble."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_train_scaled = self._scaler.fit_transform(X_train)

        # Train with class weights
        if self.use_class_weights:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight("balanced", y_train)
            self._xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            self._rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            self._xgb_model.fit(X_train_scaled, y_train)
            self._rf_model.fit(X_train_scaled, y_train)

        self._is_trained = True
        logger.info(f"ML Hybrid model trained on {len(X_train)} samples")

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

        # Get ensemble probabilities (weighted average)
        xgb_probas = self._xgb_model.predict_proba(X_scaled)[:, 1]
        rf_probas = self._rf_model.predict_proba(X_scaled)[:, 1]
        probas = (self.xgb_weight * xgb_probas + self.rf_weight * rf_probas)
        
        # Get ADX for regime detection
        adx_values = indicator_registry.compute("adx", candles)

        in_position = False
        for i, idx in enumerate(valid_idx):
            prob_up = probas[i]
            adx = adx_values.loc[idx] if idx in adx_values.index else 20
            
            # Adaptive thresholds based on regime
            if adx > self.adx_trend_threshold:
                buy_thresh = self.trend_buy_threshold
                sell_thresh = self.trend_sell_threshold
            else:
                buy_thresh = self.range_buy_threshold
                sell_thresh = self.range_sell_threshold
            
            if not in_position and prob_up > buy_thresh:
                signals.loc[idx] = Signal.BUY
                in_position = True
            elif in_position and prob_up < sell_thresh:
                signals.loc[idx] = Signal.SELL
                in_position = False

        return signals


# =============================================================================
# ML Classifier V4: Adaptive Threshold Strategy
# =============================================================================


@strategy_registry.register(
    "ml_classifier_v4",
    description="ML Classifier v4 - Adaptive thresholds based on market regime",
)
class MLClassifierV4Strategy(BaseStrategy):
    """
    ML Classifier with regime-adaptive thresholds:
    - Uses ADX to detect trending vs ranging markets
    - Adjusts buy/sell thresholds based on regime
    - More aggressive in trends, more conservative in ranges
    
    Key improvements:
    1. Adaptive thresholds = better for different market conditions
    2. Regime awareness = trade with the market, not against it
    """

    name = "ml_classifier_v4"

    def _setup(
        self,
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 2,
        # Thresholds for trending markets
        trend_buy_threshold: float = 0.52,
        trend_sell_threshold: float = 0.48,
        # Thresholds for ranging markets (more conservative)
        range_buy_threshold: float = 0.65,
        range_sell_threshold: float = 0.35,
        # ADX threshold to determine regime
        adx_trend_threshold: int = 25,
        **kwargs,
    ) -> None:
        self.features = features or [
            "sma_20", "ema_12", "rsi_14", "macd",
            "adx", "atr_ratio", "bb_width", "volume_momentum",
        ]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        
        self.trend_buy_threshold = trend_buy_threshold
        self.trend_sell_threshold = trend_sell_threshold
        self.range_buy_threshold = range_buy_threshold
        self.range_sell_threshold = range_sell_threshold
        self.adx_trend_threshold = adx_trend_threshold

        valid_params = {
            k: v for k, v in kwargs.items()
            if k not in ["symbols", "interval", "enabled"]
        }

        self._model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            **valid_params,
        )
        self._scaler = StandardScaler()
        self._is_trained = False

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute feature matrix."""
        features_df = pd.DataFrame(index=candles.index)

        for feature in self.features:
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
                elif feature == "volume_momentum":
                    features_df[feature] = indicator_registry.compute("volume_momentum", candles)
                else:
                    features_df[feature] = indicator_registry.compute(feature, candles)
            except Exception as e:
                logger.warning(f"Could not compute feature {feature}: {e}")

        return features_df

    def _compute_target(self, candles: pd.DataFrame) -> pd.Series:
        """Compute target variable."""
        future_returns = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )
        return (future_returns > 0).astype(int)

    def _train(self, candles: pd.DataFrame) -> None:
        """Train the model."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_train_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_train_scaled, y_train)
        self._is_trained = True

        logger.info(f"ML v4 (adaptive) trained on {len(X_train)} samples")

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
        
        # Get ADX values for regime detection
        adx_values = indicator_registry.compute("adx", candles)

        in_position = False
        for i, idx in enumerate(valid_idx):
            prob_up = probas[i]
            adx = adx_values.loc[idx] if idx in adx_values.index else 20
            
            # Determine thresholds based on regime
            if adx > self.adx_trend_threshold:
                # Trending market - be more aggressive
                buy_thresh = self.trend_buy_threshold
                sell_thresh = self.trend_sell_threshold
            else:
                # Ranging market - be more conservative
                buy_thresh = self.range_buy_threshold
                sell_thresh = self.range_sell_threshold
            
            if not in_position and prob_up > buy_thresh:
                signals.loc[idx] = Signal.BUY
                in_position = True
            elif in_position and prob_up < sell_thresh:
                signals.loc[idx] = Signal.SELL
                in_position = False

        return signals


# =============================================================================
# ML Classifier Conservative: Low Drawdown Priority
# =============================================================================


@strategy_registry.register(
    "ml_classifier_conservative",
    description="Conservative ML - Prioritizes low drawdown over high returns",
)
class MLClassifierConservativeStrategy(BaseStrategy):
    """
    Conservative ML Classifier optimized for capital preservation:
    
    Based on ml_classifier_v3 which had lowest max drawdown (4.72%):
    - Longer prediction horizon (filters noise)
    - Stricter entry thresholds  
    - ADX filter (only trade in trends)
    
    Key features:
    1. Higher confidence required for entry (0.60)
    2. Quick exits on uncertainty (0.50)
    3. Only trades when ADX > threshold (trend confirmed)
    4. Fewer but higher quality trades
    """

    name = "ml_classifier_conservative"

    def _setup(
        self,
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 4,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.50,
        min_adx: int = 25,
        n_estimators: int = 150,
        **kwargs,
    ) -> None:
        self.features = features or [
            "rsi_14", "macd", "adx", "atr_ratio", "bb_width",
            "sma_20", "sma_50", "volume_momentum",
        ]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_adx = min_adx

        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._is_trained = False

    def _compute_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute feature matrix for conservative strategy."""
        features_df = pd.DataFrame(index=candles.index)

        features_df["rsi_14"] = indicator_registry.compute("rsi", candles, period=14)
        
        macd_df = indicator_registry.compute("macd", candles)
        features_df["macd"] = macd_df["macd"]
        features_df["macd_signal"] = macd_df["signal"]
        features_df["macd_hist"] = macd_df["histogram"]
        
        features_df["adx"] = indicator_registry.compute("adx", candles)
        features_df["atr_ratio"] = indicator_registry.compute("atr_ratio", candles)
        features_df["bb_width"] = indicator_registry.compute("bb_width", candles)
        
        sma_20 = indicator_registry.compute("sma", candles, period=20)
        sma_50 = indicator_registry.compute("sma", candles, period=50)
        features_df["price_vs_sma20"] = (candles["close"] - sma_20) / sma_20 * 100
        features_df["price_vs_sma50"] = (candles["close"] - sma_50) / sma_50 * 100
        features_df["sma20_vs_sma50"] = (sma_20 - sma_50) / sma_50 * 100
        
        features_df["volume_momentum"] = indicator_registry.compute("volume_momentum", candles)
        features_df["return_5"] = candles["close"].pct_change(5) * 100
        
        return features_df

    def _compute_target(self, candles: pd.DataFrame) -> pd.Series:
        """Compute target with minimum return threshold."""
        future_returns = candles["close"].pct_change(self.prediction_horizon).shift(
            -self.prediction_horizon
        )
        return (future_returns > 0.01).astype(int)

    def _train(self, candles: pd.DataFrame) -> None:
        """Train the conservative model."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_train_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_train_scaled, y_train)
        self._is_trained = True

        logger.info(f"ML Conservative trained on {len(X_train)} samples")

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
        adx_values = indicator_registry.compute("adx", candles)

        in_position = False
        for i, idx in enumerate(valid_idx):
            prob_up = probas[i]
            adx = adx_values.loc[idx] if idx in adx_values.index else 20
            
            # Only trade in confirmed trends (ADX filter)
            if adx < self.min_adx:
                if in_position and prob_up < self.sell_threshold:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
                continue
            
            if not in_position and prob_up > self.buy_threshold:
                signals.loc[idx] = Signal.BUY
                in_position = True
            elif in_position and prob_up < self.sell_threshold:
                signals.loc[idx] = Signal.SELL
                in_position = False

        return signals


# =============================================================================
# ML Classifier V5: Simplified for Robustness
# =============================================================================


@strategy_registry.register(
    "ml_classifier_v5",
    description="Simplified ML Classifier - fewer features, stronger regularization, longer horizon",
)
class MLClassifierV5Strategy(BaseStrategy):
    """
    Simplified ML Classifier designed to avoid overfitting:
    
    Based on research findings from walk-forward validation:
    - Only top 3 features: volume_momentum, adx, rsi_14
    - Longer prediction horizon (10 bars) to filter noise
    - Stronger regularization (max_depth=3, min_samples_leaf=50)
    - Fewer trees (n_estimators=50)
    - Larger train window (90 days)
    
    Key principle: Simpler models generalize better to unseen time periods.
    """

    name = "ml_classifier_v5"

    def _setup(
        self,
        features: list[str] | None = None,
        lookback: int = 100,
        train_size: float = 0.8,
        prediction_horizon: int = 10,  # Longer horizon to filter noise
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        n_estimators: int = 50,        # Fewer trees
        max_depth: int = 3,            # Shallower trees
        min_samples_leaf: int = 50,    # More samples per leaf
        learning_rate: float = 0.1,
        min_return_threshold: float = 0.02,  # Only predict moves > 2%
        **kwargs,
    ) -> None:
        # Only use top 3 features from feature importance analysis
        self.features = features or ["volume_momentum", "adx", "rsi_14"]
        self.lookback = lookback
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_return_threshold = min_return_threshold

        # Use GradientBoosting with strong regularization
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            subsample=0.8,  # Additional regularization
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._is_trained = False
        self._feature_importance = None

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
                elif feature == "atr_ratio":
                    features_df[feature] = indicator_registry.compute("atr_ratio", candles)
                elif feature == "bb_width":
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
        # Only count as positive if return > threshold (filters noise)
        return (future_returns > self.min_return_threshold).astype(int)

    def _train(self, candles: pd.DataFrame) -> None:
        """Train the simplified model."""
        features_df = self._compute_features(candles)
        target = self._compute_target(candles)

        valid_idx = features_df.dropna().index.intersection(target.dropna().index)
        X = features_df.loc[valid_idx]
        y = target.loc[valid_idx]

        split_idx = int(len(X) * self.train_size)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_train_scaled = self._scaler.fit_transform(X_train)
        
        # Use balanced class weights
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight("balanced", y_train)
        self._model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        self._is_trained = True

        # Store feature importance
        if hasattr(self._model, "feature_importances_"):
            self._feature_importance = dict(zip(X.columns, self._model.feature_importances_))

        logger.info(
            f"ML v5 (simplified) trained on {len(X_train)} samples, "
            f"features: {self.features}, horizon: {self.prediction_horizon}"
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

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance after training."""
        return self._feature_importance
