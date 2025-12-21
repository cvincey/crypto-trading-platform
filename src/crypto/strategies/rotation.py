"""Relative strength and rotation trading strategies."""

import logging
from typing import Any

import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "relative_strength",
    description="Relative Strength Rotation Strategy",
)
class RelativeStrengthStrategy(BaseStrategy):
    """
    Relative Strength Rotation Strategy.
    
    For single-asset backtesting: Generates BUY signal when the asset
    has positive momentum above a threshold, SELL when momentum is weak.
    
    Note: Full cross-asset rotation requires multi-asset backtesting support.
    This implementation focuses on momentum-based signals for a single asset.
    
    Config params:
        lookback_period: Period for momentum calculation
        top_n: Number of top performers to be considered "strong" (for future multi-asset)
        momentum_type: "roc" (rate of change) or "returns"
        momentum_threshold: Minimum momentum to be considered strong
        weak_threshold: Momentum below this triggers sell
    """

    name = "relative_strength"

    def _setup(
        self,
        lookback_period: int = 20,
        top_n: int = 5,
        momentum_type: str = "roc",
        momentum_threshold: float = 0.0,
        weak_threshold: float = -0.05,
        **kwargs,
    ) -> None:
        self.lookback_period = lookback_period
        self.top_n = top_n
        self.momentum_type = momentum_type
        self.momentum_threshold = momentum_threshold
        self.weak_threshold = weak_threshold

    def _compute_momentum(self, candles: pd.DataFrame) -> pd.Series:
        """Compute momentum based on configured type."""
        if self.momentum_type == "roc":
            return indicator_registry.compute("roc", candles, period=self.lookback_period) / 100
        else:
            # Simple returns
            return candles["close"].pct_change(self.lookback_period)

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        momentum = self._compute_momentum(candles)
        signals = self.create_signal_series(candles.index)

        # BUY when momentum crosses above threshold (strengthening)
        buy_condition = (momentum > self.momentum_threshold) & (
            momentum.shift(1) <= self.momentum_threshold
        )
        signals.loc[buy_condition] = Signal.BUY

        # SELL when momentum crosses below weak threshold
        sell_condition = (momentum < self.weak_threshold) & (
            momentum.shift(1) >= self.weak_threshold
        )
        signals.loc[sell_condition] = Signal.SELL

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "momentum_ranking",
    description="Momentum Ranking Strategy - ranks by strength",
)
class MomentumRankingStrategy(BaseStrategy):
    """
    Momentum Ranking Strategy.
    
    Similar to relative strength but uses a percentile-based approach.
    Generates signals based on how the current momentum compares to
    historical momentum values.
    
    Config params:
        lookback_period: Period for momentum calculation
        ranking_window: Window for computing momentum percentile
        buy_percentile: Percentile above which to buy (e.g., 80 = top 20%)
        sell_percentile: Percentile below which to sell
    """

    name = "momentum_ranking"

    def _setup(
        self,
        lookback_period: int = 20,
        ranking_window: int = 100,
        buy_percentile: float = 80,
        sell_percentile: float = 20,
        **kwargs,
    ) -> None:
        self.lookback_period = lookback_period
        self.ranking_window = ranking_window
        self.buy_percentile = buy_percentile
        self.sell_percentile = sell_percentile

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        # Compute momentum
        momentum = indicator_registry.compute("roc", candles, period=self.lookback_period)

        # Compute rolling percentile
        def rolling_percentile(x):
            if len(x) < 2:
                return 50
            return (x.rank().iloc[-1] / len(x)) * 100

        momentum_percentile = momentum.rolling(window=self.ranking_window).apply(
            rolling_percentile, raw=False
        )

        signals = self.create_signal_series(candles.index)

        # BUY when momentum is in top percentile
        buy_condition = (momentum_percentile > self.buy_percentile) & (
            momentum_percentile.shift(1) <= self.buy_percentile
        )
        signals.loc[buy_condition] = Signal.BUY

        # SELL when momentum drops to bottom percentile
        sell_condition = (momentum_percentile < self.sell_percentile) & (
            momentum_percentile.shift(1) >= self.sell_percentile
        )
        signals.loc[sell_condition] = Signal.SELL

        return self.apply_filters(signals, candles)
