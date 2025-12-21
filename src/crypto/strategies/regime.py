"""Market regime detection and adaptive trading strategies."""

import logging
from typing import Any

import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "regime_adaptive",
    description="Regime Adaptive Strategy - switches between trending and ranging",
)
class RegimeAdaptiveStrategy(BaseStrategy):
    """
    Regime Adaptive Strategy.
    
    Detects market regime (trending vs ranging) using ADX and volatility,
    then delegates to appropriate sub-strategy.
    
    Config params:
        adx_trending_threshold: ADX above this = trending market (default: 25)
        volatility_lookback: Period for volatility calculation (default: 20)
        trending_strategy: Strategy config name to use in trending market
        ranging_strategy: Strategy config name to use in ranging market
    
    Example config:
        regime_adaptive_btc:
          type: regime_adaptive
          params:
            adx_trending_threshold: 25
            volatility_lookback: 20
            trending_strategy: momentum_breakout
            ranging_strategy: rsi_mean_reversion
    """

    name = "regime_adaptive"

    def _setup(
        self,
        adx_trending_threshold: int = 25,
        volatility_lookback: int = 20,
        trending_strategy: str | None = None,
        ranging_strategy: str | None = None,
        adx_period: int = 14,
        **kwargs,
    ) -> None:
        self.adx_trending_threshold = adx_trending_threshold
        self.volatility_lookback = volatility_lookback
        self.trending_strategy_name = trending_strategy
        self.ranging_strategy_name = ranging_strategy
        self.adx_period = adx_period
        
        self._trending_strategy: BaseStrategy | None = None
        self._ranging_strategy: BaseStrategy | None = None

    def _ensure_strategies_loaded(self) -> None:
        """Lazy load sub-strategies from config names."""
        if self.trending_strategy_name and self._trending_strategy is None:
            try:
                self._trending_strategy = strategy_registry.create_from_config(
                    self.trending_strategy_name
                )
                logger.debug(f"Loaded trending strategy: {self.trending_strategy_name}")
            except Exception as e:
                logger.warning(f"Failed to load trending strategy: {e}")

        if self.ranging_strategy_name and self._ranging_strategy is None:
            try:
                self._ranging_strategy = strategy_registry.create_from_config(
                    self.ranging_strategy_name
                )
                logger.debug(f"Loaded ranging strategy: {self.ranging_strategy_name}")
            except Exception as e:
                logger.warning(f"Failed to load ranging strategy: {e}")

    def _detect_regime(self, candles: pd.DataFrame) -> pd.Series:
        """
        Detect market regime for each bar.
        
        Returns:
            Series with True for trending, False for ranging
        """
        adx = indicator_registry.compute("adx", candles, period=self.adx_period)
        
        # Trending when ADX > threshold
        is_trending = adx > self.adx_trending_threshold
        
        return is_trending

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        self._ensure_strategies_loaded()

        # Detect regime
        is_trending = self._detect_regime(candles)

        # Get signals from both strategies
        trending_signals = self.create_signal_series(candles.index)
        ranging_signals = self.create_signal_series(candles.index)

        if self._trending_strategy:
            try:
                trending_signals = self._trending_strategy.generate_signals(candles)
            except Exception as e:
                logger.warning(f"Trending strategy failed: {e}")

        if self._ranging_strategy:
            try:
                ranging_signals = self._ranging_strategy.generate_signals(candles)
            except Exception as e:
                logger.warning(f"Ranging strategy failed: {e}")

        # Combine based on regime
        result = self.create_signal_series(candles.index)
        
        for idx in candles.index:
            if is_trending.loc[idx]:
                result.loc[idx] = trending_signals.loc[idx]
            else:
                result.loc[idx] = ranging_signals.loc[idx]

        return self.apply_filters(result, candles)

    def get_parameters(self) -> dict[str, Any]:
        """Get parameters including regime info."""
        params = super().get_parameters()
        params["trending_strategy"] = self.trending_strategy_name
        params["ranging_strategy"] = self.ranging_strategy_name
        return params
