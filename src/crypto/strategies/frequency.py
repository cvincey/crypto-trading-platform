"""
Frequency reduction strategies.

These strategies use hourly data but trade less frequently to reduce costs.
Key insight: High trading frequency destroys value through transaction costs.
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
    "weekly_momentum",
    description="Weekly rebalance based on 7-day momentum",
)
class WeeklyMomentumStrategy(BaseStrategy):
    """
    Weekly Momentum Strategy.
    
    Rebalance weekly to hold assets with positive momentum.
    Uses hourly data for precision but only trades once per week.
    """

    name = "weekly_momentum"

    def _setup(
        self,
        momentum_period: int = 168,
        hold_period: int = 168,
        top_n: int = 3,
        rebalance_hour: int = 0,
        rebalance_day: int = 0,
        min_momentum: float = 0.0,
        **kwargs,
    ) -> None:
        self.momentum_period = momentum_period
        self.hold_period = hold_period
        self.top_n = top_n
        self.rebalance_hour = rebalance_hour
        self.rebalance_day = rebalance_day
        self.min_momentum = min_momentum

    def _is_rebalance_time(self, dt) -> bool:
        """Check if it's time to rebalance."""
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        return dt.weekday() == self.rebalance_day and dt.hour == self.rebalance_hour

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Calculate momentum
        momentum = candles["close"].pct_change(self.momentum_period)

        in_position = False
        last_rebalance = None

        for idx in candles.index:
            if self._is_rebalance_time(idx):
                mom = momentum.get(idx, 0)

                if pd.isna(mom):
                    continue

                # Rebalance decision
                if mom > self.min_momentum:
                    if not in_position:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                else:
                    if in_position:
                        signals.loc[idx] = Signal.SELL
                        in_position = False

                last_rebalance = idx

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "signal_confirmation_delay",
    description="Require signal to persist before acting",
)
class SignalConfirmationDelayStrategy(BaseStrategy):
    """
    Signal Confirmation Delay Strategy.
    
    Only act on signals that persist for N bars.
    This filters out noise and false signals.
    """

    name = "signal_confirmation_delay"

    def _setup(
        self,
        base_strategy: str = "rsi_mean_reversion",
        confirmation_delay: int = 4,
        require_consistent: bool = True,
        allow_strengthen: bool = True,
        **kwargs,
    ) -> None:
        self.base_strategy_name = base_strategy
        self.confirmation_delay = confirmation_delay
        self.require_consistent = require_consistent
        self.allow_strengthen = allow_strengthen
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

        # Track pending signals
        pending_signal = None
        pending_count = 0
        in_position = False

        for i, idx in enumerate(candles.index):
            current = base_signals.loc[idx]

            if current == Signal.BUY:
                if pending_signal == Signal.BUY:
                    pending_count += 1
                else:
                    pending_signal = Signal.BUY
                    pending_count = 1

                # Check if confirmed
                if pending_count >= self.confirmation_delay:
                    if not in_position:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                    pending_signal = None
                    pending_count = 0

            elif current == Signal.SELL:
                if pending_signal == Signal.SELL:
                    pending_count += 1
                else:
                    pending_signal = Signal.SELL
                    pending_count = 1

                # Check if confirmed
                if pending_count >= self.confirmation_delay:
                    if in_position:
                        signals.loc[idx] = Signal.SELL
                        in_position = False
                    pending_signal = None
                    pending_count = 0

            else:
                # HOLD signal
                if self.require_consistent:
                    # Reset pending if not consistent
                    pending_signal = None
                    pending_count = 0

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "signal_strength_filter",
    description="Only trade on extreme signals",
)
class SignalStrengthFilterStrategy(BaseStrategy):
    """
    Signal Strength Filter Strategy.
    
    Only act on extreme signals (RSI < 20 instead of < 30).
    Reduces noise by requiring stronger conditions.
    """

    name = "signal_strength_filter"

    def _setup(
        self,
        base_strategy: str = "rsi_mean_reversion",
        rsi_extreme_oversold: int = 20,
        rsi_extreme_overbought: int = 80,
        volume_confirmation: bool = True,
        volume_multiplier: float = 1.5,
        **kwargs,
    ) -> None:
        self.base_strategy_name = base_strategy
        self.rsi_extreme_oversold = rsi_extreme_oversold
        self.rsi_extreme_overbought = rsi_extreme_overbought
        self.volume_confirmation = volume_confirmation
        self.volume_multiplier = volume_multiplier

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Calculate RSI
        rsi = indicator_registry.compute("rsi", candles, period=14)

        # Calculate volume average for confirmation
        volume = candles["volume"]
        volume_avg = volume.rolling(20).mean()
        volume_high = volume > (volume_avg * self.volume_multiplier)

        in_position = False

        for idx in candles.index:
            r = rsi.get(idx, 50)
            vol_ok = volume_high.get(idx, False) if self.volume_confirmation else True

            if pd.isna(r):
                continue

            if not in_position:
                # Only buy on extreme oversold + volume confirmation
                if r < self.rsi_extreme_oversold and vol_ok:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
            else:
                # Exit on overbought or RSI normalizes
                if r > self.rsi_extreme_overbought or r > 50:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)
