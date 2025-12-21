"""Ensemble and voting-based trading strategies."""

import logging
from typing import Any

import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "ensemble_voting",
    description="Ensemble Voting Strategy - combines multiple strategies",
)
class EnsembleVotingStrategy(BaseStrategy):
    """
    Ensemble Voting Strategy.
    
    Combines signals from multiple sub-strategies and generates
    a signal when enough strategies agree.
    
    Config params:
        strategies: List of strategy config names to combine
        min_agreement: Minimum number of strategies that must agree (default: 2)
        mode: "majority" (>50% agree) or "threshold" (>= min_agreement)
    
    Example config:
        ensemble_consensus:
          type: ensemble_voting
          params:
            strategies: [sma_crossover_btc, rsi_mean_reversion, macd_crossover]
            min_agreement: 2
            mode: majority
    """

    name = "ensemble_voting"

    def _setup(
        self,
        strategies: list[str] | None = None,
        min_agreement: int = 2,
        mode: str = "threshold",
        **kwargs,
    ) -> None:
        self.strategy_names = strategies or []
        self.min_agreement = min_agreement
        self.mode = mode
        self._sub_strategies: list[BaseStrategy] = []

    def _ensure_strategies_loaded(self) -> None:
        """Lazy load sub-strategies from config names."""
        if self._sub_strategies:
            return
            
        for name in self.strategy_names:
            try:
                strategy = strategy_registry.create_from_config(name)
                self._sub_strategies.append(strategy)
                logger.debug(f"Loaded sub-strategy: {name}")
            except Exception as e:
                logger.warning(f"Failed to load strategy '{name}': {e}")

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        self._ensure_strategies_loaded()

        if not self._sub_strategies:
            logger.warning("No sub-strategies loaded for ensemble")
            return self.create_signal_series(candles.index)

        # Collect signals from all sub-strategies
        all_signals: list[pd.Series] = []
        for strategy in self._sub_strategies:
            try:
                signals = strategy.generate_signals(candles)
                all_signals.append(signals)
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")

        if not all_signals:
            return self.create_signal_series(candles.index)

        # Count votes for each signal type
        signals_df = pd.DataFrame({
            f"s{i}": s for i, s in enumerate(all_signals)
        })

        result = self.create_signal_series(candles.index)
        n_strategies = len(all_signals)

        for idx in candles.index:
            row_signals = signals_df.loc[idx]
            
            buy_count = sum(1 for s in row_signals if s == Signal.BUY)
            sell_count = sum(1 for s in row_signals if s == Signal.SELL)

            if self.mode == "majority":
                threshold = n_strategies / 2
            else:
                threshold = self.min_agreement

            if buy_count >= threshold:
                result.loc[idx] = Signal.BUY
            elif sell_count >= threshold:
                result.loc[idx] = Signal.SELL

        return self.apply_filters(result, candles)

    def get_parameters(self) -> dict[str, Any]:
        """Get parameters including sub-strategy names."""
        params = super().get_parameters()
        params["sub_strategies"] = self.strategy_names
        return params
