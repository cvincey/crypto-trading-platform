"""
Meta-strategies that control other strategies or manage risk.

These strategies operate at a higher level, deciding when to trade
rather than what to trade.
"""

import logging
from typing import Any

import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "regime_gate",
    description="Only trade in clear market regimes",
)
class RegimeGateStrategy(BaseStrategy):
    """
    Regime Gate Strategy.
    
    Only allow trading when market regime is clear:
    - Strong trend (ADX > 30): Use trend-following strategy
    - Clear range (ADX < 15): Use mean-reversion strategy
    - Ambiguous (ADX 15-30): No trading (too unpredictable)
    """

    name = "regime_gate"

    def _setup(
        self,
        trend_strategy: str = "momentum_breakout",
        range_strategy: str = "rsi_mean_reversion",
        adx_trend_threshold: int = 30,
        adx_range_threshold: int = 15,
        adx_period: int = 14,
        block_ambiguous: bool = True,
        **kwargs,
    ) -> None:
        self.trend_strategy_name = trend_strategy
        self.range_strategy_name = range_strategy
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.adx_period = adx_period
        self.block_ambiguous = block_ambiguous
        self._trend_strategy = None
        self._range_strategy = None

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Create sub-strategies
        if self._trend_strategy is None:
            try:
                self._trend_strategy = strategy_registry.create_from_config(
                    self.trend_strategy_name
                )
            except Exception as e:
                logger.warning(f"Could not create trend strategy: {e}")

        if self._range_strategy is None:
            try:
                self._range_strategy = strategy_registry.create_from_config(
                    self.range_strategy_name
                )
            except Exception as e:
                logger.warning(f"Could not create range strategy: {e}")

        # Get signals from both strategies
        trend_signals = (
            self._trend_strategy.generate_signals(candles)
            if self._trend_strategy
            else self.create_signal_series(candles.index)
        )
        range_signals = (
            self._range_strategy.generate_signals(candles)
            if self._range_strategy
            else self.create_signal_series(candles.index)
        )

        # Calculate ADX for regime detection
        adx = indicator_registry.compute("adx", candles, period=self.adx_period)

        in_position = False

        for idx in candles.index:
            adx_val = adx.get(idx, 25)

            if pd.isna(adx_val):
                continue

            # Determine regime and select appropriate signal
            if adx_val > self.adx_trend_threshold:
                # Strong trend - use trend strategy
                signal = trend_signals.loc[idx]
            elif adx_val < self.adx_range_threshold:
                # Clear range - use range strategy
                signal = range_signals.loc[idx]
            else:
                # Ambiguous regime
                if self.block_ambiguous:
                    signal = Signal.HOLD
                else:
                    # Default to trend strategy
                    signal = trend_signals.loc[idx]

            # Apply signal with position tracking
            if signal == Signal.BUY and not in_position:
                signals.loc[idx] = Signal.BUY
                in_position = True
            elif signal == Signal.SELL and in_position:
                signals.loc[idx] = Signal.SELL
                in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "drawdown_pause",
    description="Pause trading after significant drawdown",
)
class DrawdownPauseStrategy(BaseStrategy):
    """
    Drawdown Pause Strategy.
    
    Pause trading after the strategy hits a drawdown threshold.
    Resume after a cooling-off period.
    
    Prevents cascading losses during adverse conditions.
    """

    name = "drawdown_pause"

    def _setup(
        self,
        base_strategy: str = "ml_classifier_xgb",
        pause_threshold: float = 0.05,
        pause_duration: int = 48,
        recovery_threshold: float = 0.02,
        max_consecutive_losses: int = 5,
        **kwargs,
    ) -> None:
        self.base_strategy_name = base_strategy
        self.pause_threshold = pause_threshold
        self.pause_duration = pause_duration
        self.recovery_threshold = recovery_threshold
        self.max_consecutive_losses = max_consecutive_losses
        self._base_strategy = None

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        # Get base strategy signals
        if self._base_strategy is None:
            try:
                self._base_strategy = strategy_registry.create_from_config(
                    self.base_strategy_name
                )
            except Exception as e:
                logger.warning(f"Could not create base strategy: {e}")
                return self.create_signal_series(candles.index)

        base_signals = self._base_strategy.generate_signals(candles)
        signals = self.create_signal_series(candles.index)

        # Track equity for drawdown calculation
        equity = 1.0
        peak_equity = 1.0
        paused = False
        pause_start = 0
        consecutive_losses = 0
        in_position = False
        entry_price = 0.0

        for i, idx in enumerate(candles.index):
            price = float(candles.loc[idx, "close"])

            # Update equity if in position
            if in_position and entry_price > 0:
                pnl = (price - entry_price) / entry_price
                current_equity = equity * (1 + pnl)
            else:
                current_equity = equity

            # Update peak and calculate drawdown
            peak_equity = max(peak_equity, current_equity)
            drawdown = (peak_equity - current_equity) / peak_equity

            # Check pause conditions
            if not paused:
                if drawdown > self.pause_threshold:
                    paused = True
                    pause_start = i
                    logger.debug(f"Pausing at {idx} due to {drawdown:.2%} drawdown")
                elif consecutive_losses >= self.max_consecutive_losses:
                    paused = True
                    pause_start = i
                    logger.debug(f"Pausing at {idx} due to {consecutive_losses} consecutive losses")

            # Check resume conditions
            if paused:
                if i - pause_start >= self.pause_duration:
                    paused = False
                    consecutive_losses = 0
                    logger.debug(f"Resuming at {idx} after pause")

            # Generate signals
            base_signal = base_signals.loc[idx]

            if paused:
                # Force exit if in position during pause
                if in_position:
                    signals.loc[idx] = Signal.SELL
                    # Track trade result
                    if price < entry_price:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    equity = current_equity
                    in_position = False
                # Otherwise hold
            else:
                if base_signal == Signal.BUY and not in_position:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_price = price
                elif base_signal == Signal.SELL and in_position:
                    signals.loc[idx] = Signal.SELL
                    # Track trade result
                    if price < entry_price:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    equity = current_equity
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "strategy_momentum",
    description="Use the recently best-performing strategy",
)
class StrategyMomentumStrategy(BaseStrategy):
    """
    Strategy Momentum Strategy.
    
    Track performance of multiple strategies.
    Use the one that performed best recently.
    
    Adapts to changing market conditions.
    """

    name = "strategy_momentum"

    def _setup(
        self,
        strategies: list[str] | None = None,
        evaluation_period: int = 168,
        selection_count: int = 1,
        min_sharpe: float = 0.0,
        reeval_period: int = 24,
        **kwargs,
    ) -> None:
        self.strategy_names = strategies or [
            "momentum_breakout",
            "rsi_mean_reversion",
            "sma_crossover_btc",
        ]
        self.evaluation_period = evaluation_period
        self.selection_count = selection_count
        self.min_sharpe = min_sharpe
        self.reeval_period = reeval_period
        self._strategies: dict[str, BaseStrategy] = {}

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Initialize strategies
        for name in self.strategy_names:
            if name not in self._strategies:
                try:
                    self._strategies[name] = strategy_registry.create_from_config(name)
                except Exception as e:
                    logger.warning(f"Could not create strategy {name}: {e}")

        if not self._strategies:
            return signals

        # Get signals from all strategies
        all_signals = {}
        for name, strategy in self._strategies.items():
            try:
                all_signals[name] = strategy.generate_signals(candles)
            except Exception:
                pass

        if not all_signals:
            return signals

        # Track strategy performance
        strategy_returns: dict[str, list[float]] = {name: [] for name in all_signals}
        current_strategy = list(all_signals.keys())[0]
        last_eval = 0
        in_position = False
        entry_price = 0.0

        for i, idx in enumerate(candles.index):
            price = float(candles.loc[idx, "close"])

            # Re-evaluate strategies periodically
            if i - last_eval >= self.reeval_period and i >= self.evaluation_period:
                last_eval = i

                # Calculate recent returns for each strategy
                best_strategy = current_strategy
                best_return = float("-inf")

                for name, strat_signals in all_signals.items():
                    # Simple return calculation over evaluation period
                    recent = strat_signals.iloc[max(0, i - self.evaluation_period) : i]
                    if len(recent) == 0:
                        continue

                    # Count profitable signals
                    buy_signals = recent[recent == Signal.BUY]
                    if len(buy_signals) > 0:
                        # Estimate return (simplified)
                        strat_return = len(buy_signals) / len(recent)
                        if strat_return > best_return:
                            best_return = strat_return
                            best_strategy = name

                if best_strategy != current_strategy:
                    logger.debug(f"Switching to {best_strategy} at {idx}")
                    current_strategy = best_strategy

            # Use selected strategy's signal
            signal = all_signals[current_strategy].loc[idx]

            if signal == Signal.BUY and not in_position:
                signals.loc[idx] = Signal.BUY
                in_position = True
                entry_price = price
            elif signal == Signal.SELL and in_position:
                signals.loc[idx] = Signal.SELL
                in_position = False

        return self.apply_filters(signals, candles)
