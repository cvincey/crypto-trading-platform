"""Multi-timeframe trading strategies."""

import asyncio
import logging
from typing import Any

import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "multi_timeframe",
    description="Multi-Timeframe Strategy - aligns signals across timeframes",
)
class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-Timeframe Strategy.
    
    Checks trend alignment across multiple timeframes before generating signals.
    Only trades when higher timeframes confirm the direction.
    
    Config params:
        base_interval: The interval of the input candles (e.g., "1h")
        higher_intervals: List of higher intervals to check (e.g., ["4h", "1d"])
        signal_strategy: Strategy type to use for base signals
        require_alignment: If True, all timeframes must align for entry
        trend_ma_period: Moving average period for trend detection
    
    Note: This strategy resamples base candles to create higher timeframe data.
    """

    name = "multi_timeframe"

    # Mapping of interval strings to pandas resample rules
    RESAMPLE_RULES = {
        "1h": "1h",
        "2h": "2h", 
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1D",
        "3d": "3D",
        "1w": "1W",
    }

    def _setup(
        self,
        base_interval: str = "1h",
        higher_intervals: list[str] | None = None,
        signal_strategy: str = "sma_crossover",
        require_alignment: bool = True,
        trend_ma_period: int = 20,
        fast_period: int = 10,
        slow_period: int = 30,
        **kwargs,
    ) -> None:
        self.base_interval = base_interval
        self.higher_intervals = higher_intervals or ["4h", "1d"]
        self.signal_strategy_type = signal_strategy
        self.require_alignment = require_alignment
        self.trend_ma_period = trend_ma_period
        self.fast_period = fast_period
        self.slow_period = slow_period

    def _resample_candles(self, candles: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Resample candles to a higher timeframe."""
        rule = self.RESAMPLE_RULES.get(target_interval)
        if rule is None:
            logger.warning(f"Unknown interval: {target_interval}, using 1D")
            rule = "1D"

        resampled = candles.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled

    def _get_trend(self, candles: pd.DataFrame) -> pd.Series:
        """
        Determine trend direction for each bar.
        
        Returns:
            Series with 1 for uptrend, -1 for downtrend, 0 for neutral
        """
        fast_ma = indicator_registry.compute("sma", candles, period=self.fast_period)
        slow_ma = indicator_registry.compute("sma", candles, period=self.slow_period)

        trend = pd.Series(0, index=candles.index)
        trend[fast_ma > slow_ma] = 1  # Uptrend
        trend[fast_ma < slow_ma] = -1  # Downtrend

        return trend

    def _get_higher_tf_trends(self, candles: pd.DataFrame) -> dict[str, pd.Series]:
        """Get trend for each higher timeframe."""
        trends = {}
        
        for interval in self.higher_intervals:
            try:
                resampled = self._resample_candles(candles, interval)
                if len(resampled) >= self.slow_period:
                    trend = self._get_trend(resampled)
                    # Forward fill to match base timeframe index
                    trend = trend.reindex(candles.index, method="ffill")
                    trends[interval] = trend
            except Exception as e:
                logger.warning(f"Failed to compute trend for {interval}: {e}")

        return trends

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        # Get base timeframe signals using SMA crossover
        fast_sma = indicator_registry.compute("sma", candles, period=self.fast_period)
        slow_sma = indicator_registry.compute("sma", candles, period=self.slow_period)

        signals = self.create_signal_series(candles.index)

        # Base signals
        fast_above = fast_sma > slow_sma
        fast_below = fast_sma < slow_sma

        # Handle shift NaN without triggering deprecation warning
        fast_below_shifted = fast_below.shift(1)
        fast_below_shifted = fast_below_shifted.where(pd.notna(fast_below_shifted), False)
        fast_above_shifted = fast_above.shift(1)
        fast_above_shifted = fast_above_shifted.where(pd.notna(fast_above_shifted), False)
        buy_signals = fast_above & fast_below_shifted
        sell_signals = fast_below & fast_above_shifted

        # Get higher timeframe trends
        htf_trends = self._get_higher_tf_trends(candles)

        if not htf_trends:
            # No higher timeframe data, use base signals only
            signals.loc[buy_signals] = Signal.BUY
            signals.loc[sell_signals] = Signal.SELL
            return self.apply_filters(signals, candles)

        # Apply higher timeframe filter
        for idx in candles.index:
            if buy_signals.get(idx, False):
                # Check if all higher TFs are in uptrend
                if self.require_alignment:
                    all_bullish = all(
                        htf_trends[tf].get(idx, 0) >= 0
                        for tf in htf_trends
                    )
                    if all_bullish:
                        signals.loc[idx] = Signal.BUY
                else:
                    # At least one higher TF bullish
                    any_bullish = any(
                        htf_trends[tf].get(idx, 0) > 0
                        for tf in htf_trends
                    )
                    if any_bullish:
                        signals.loc[idx] = Signal.BUY

            elif sell_signals.get(idx, False):
                # Check if all higher TFs are in downtrend
                if self.require_alignment:
                    all_bearish = all(
                        htf_trends[tf].get(idx, 0) <= 0
                        for tf in htf_trends
                    )
                    if all_bearish:
                        signals.loc[idx] = Signal.SELL
                else:
                    any_bearish = any(
                        htf_trends[tf].get(idx, 0) < 0
                        for tf in htf_trends
                    )
                    if any_bearish:
                        signals.loc[idx] = Signal.SELL

        return self.apply_filters(signals, candles)

    def get_parameters(self) -> dict[str, Any]:
        """Get parameters including timeframe info."""
        params = super().get_parameters()
        params["higher_intervals"] = self.higher_intervals
        return params
