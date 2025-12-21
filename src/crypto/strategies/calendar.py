"""
Calendar and time-based strategies.

These strategies exploit temporal patterns and market structure anomalies.
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "weekend_effect",
    description="Exit positions before weekend, re-enter Monday",
)
class WeekendEffectStrategy(BaseStrategy):
    """
    Weekend Effect Strategy.
    
    Crypto markets are open 24/7 but liquidity drops on weekends.
    Exit positions Friday evening, re-enter Monday morning.
    
    Can be applied as a filter to any base strategy.
    """

    name = "weekend_effect"

    def _setup(
        self,
        exit_day: int = 4,  # Friday (0=Monday)
        exit_hour: int = 20,
        reentry_day: int = 0,  # Monday
        reentry_hour: int = 8,
        base_strategy: str | None = None,
        invert: bool = False,
        **kwargs,
    ) -> None:
        self.exit_day = exit_day
        self.exit_hour = exit_hour
        self.reentry_day = reentry_day
        self.reentry_hour = reentry_hour
        self.base_strategy_name = base_strategy
        self.invert = invert
        self._base_strategy = None

    def _is_weekend(self, dt: datetime) -> bool:
        """Check if datetime is in weekend period."""
        day = dt.weekday()
        hour = dt.hour

        # After Friday exit time
        if day == self.exit_day and hour >= self.exit_hour:
            return True
        # Saturday or Sunday
        if day > self.exit_day or day < self.reentry_day:
            return True
        # Monday before reentry time
        if day == self.reentry_day and hour < self.reentry_hour:
            return True

        return False

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)

        # Get base strategy signals if specified
        if self.base_strategy_name and self._base_strategy is None:
            try:
                self._base_strategy = strategy_registry.create_from_config(
                    self.base_strategy_name
                )
            except Exception:
                pass

        if self._base_strategy:
            signals = self._base_strategy.generate_signals(candles)
        else:
            # Simple buy-and-hold that avoids weekends
            signals = self.create_signal_series(candles.index)

        in_position = False
        was_in_position = False

        for idx in candles.index:
            dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            is_weekend = self._is_weekend(dt)

            if self.invert:
                is_weekend = not is_weekend

            if is_weekend:
                # Force exit during weekend
                if in_position:
                    signals.loc[idx] = Signal.SELL
                    was_in_position = True
                    in_position = False
                else:
                    # Block any buy signals during weekend
                    if signals.loc[idx] == Signal.BUY:
                        signals.loc[idx] = Signal.HOLD
            else:
                # Re-enter if we exited for weekend
                if was_in_position and not in_position:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    was_in_position = False
                elif signals.loc[idx] == Signal.BUY:
                    in_position = True
                elif signals.loc[idx] == Signal.SELL:
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "hour_of_day_filter",
    description="Only trade during high-liquidity hours",
)
class HourOfDayFilterStrategy(BaseStrategy):
    """
    Hour of Day Filter Strategy.
    
    Only allow trading during specified hours (typically high-liquidity).
    Block signals outside these hours to reduce noise.
    """

    name = "hour_of_day_filter"

    def _setup(
        self,
        active_hours: list[int] | None = None,
        base_strategy: str = "momentum_breakout",
        block_entries_only: bool = True,
        **kwargs,
    ) -> None:
        self.active_hours = active_hours or [14, 15, 16, 17, 18, 19, 20, 21, 22]
        self.base_strategy_name = base_strategy
        self.block_entries_only = block_entries_only
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

        signals = self._base_strategy.generate_signals(candles)

        # Filter signals by hour
        for idx in candles.index:
            dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            hour = dt.hour

            if hour not in self.active_hours:
                if self.block_entries_only:
                    # Only block BUY signals
                    if signals.loc[idx] == Signal.BUY:
                        signals.loc[idx] = Signal.HOLD
                else:
                    # Block all signals
                    signals.loc[idx] = Signal.HOLD

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "month_end_rebalancing",
    description="Trade anticipating institutional month-end flows",
)
class MonthEndRebalancingStrategy(BaseStrategy):
    """
    Month End Rebalancing Strategy.
    
    Institutions often rebalance portfolios around month-end.
    Enter positions in the last days of the month, exit in the first days.
    """

    name = "month_end_rebalancing"

    def _setup(
        self,
        entry_days_before_eom: int = 3,
        exit_days_after_eom: int = 3,
        selection_mode: str = "momentum",
        momentum_period: int = 168,
        top_n: int = 3,
        **kwargs,
    ) -> None:
        self.entry_days_before_eom = entry_days_before_eom
        self.exit_days_after_eom = exit_days_after_eom
        self.selection_mode = selection_mode
        self.momentum_period = momentum_period
        self.top_n = top_n

    def _days_from_month_end(self, dt: datetime) -> int:
        """Calculate days from end of month (negative = before, positive = after)."""
        import calendar

        last_day = calendar.monthrange(dt.year, dt.month)[1]
        day = dt.day

        if day > last_day - self.entry_days_before_eom:
            # Before month end
            return day - last_day  # Negative
        elif day <= self.exit_days_after_eom:
            # After month start
            return day  # Positive
        else:
            return 999  # Not in window

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Calculate momentum for selection
        momentum = candles["close"].pct_change(self.momentum_period)

        in_position = False

        for idx in candles.index:
            dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            days_from_eom = self._days_from_month_end(dt)

            # Entry window: last N days of month
            if -self.entry_days_before_eom <= days_from_eom < 0:
                if not in_position:
                    # Check momentum if using momentum selection
                    mom = momentum.get(idx, 0)
                    if self.selection_mode == "momentum" and not pd.isna(mom) and mom > 0:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                    elif self.selection_mode == "equal_weight":
                        signals.loc[idx] = Signal.BUY
                        in_position = True

            # Exit window: first N days of new month
            elif 0 < days_from_eom <= self.exit_days_after_eom:
                if in_position:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)
