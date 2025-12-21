"""
Alternative data strategies using funding rates, open interest, etc.

These strategies use non-price data sources for alpha generation.
Key insight: Price-derived indicators are arbitraged away; alternative data may not be.
"""

import logging
from typing import Any

import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


class AlternativeDataBaseStrategy(BaseStrategy):
    """
    Base class for strategies that use alternative data.
    
    Alternative data includes funding rates, open interest, liquidations, etc.
    """

    def __init__(self, **params: Any):
        super().__init__(**params)
        self._funding_data: pd.Series | None = None
        self._oi_data: pd.Series | None = None

    def set_funding_data(self, funding: pd.Series) -> None:
        """Set funding rate data aligned to candle index."""
        self._funding_data = funding

    def set_open_interest_data(self, oi: pd.Series) -> None:
        """Set open interest data aligned to candle index."""
        self._oi_data = oi


@strategy_registry.register(
    "funding_rate_fade",
    description="Avoid/fade positions when funding is extreme",
)
class FundingRateFadeStrategy(AlternativeDataBaseStrategy):
    """
    Funding Rate Fade Strategy.
    
    When funding rate is extremely positive (longs pay shorts),
    the market is overleveraged long. Fade by avoiding longs or going short.
    
    When funding is extremely negative, the opposite applies.
    """

    name = "funding_rate_fade"

    def _setup(
        self,
        extreme_positive: float = 0.0005,
        extreme_negative: float = -0.0005,
        normal_threshold: float = 0.0002,
        action_mode: str = "avoid_longs",
        lookback_periods: int = 3,
        **kwargs,
    ) -> None:
        self.extreme_positive = extreme_positive
        self.extreme_negative = extreme_negative
        self.normal_threshold = normal_threshold
        self.action_mode = action_mode
        self.lookback_periods = lookback_periods

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        if self._funding_data is None or self._funding_data.empty:
            logger.warning("No funding rate data available")
            return signals

        # Calculate average funding over lookback
        funding_avg = self._funding_data.rolling(
            window=self.lookback_periods, min_periods=1
        ).mean()

        in_position = False

        for idx in candles.index:
            funding = funding_avg.get(idx, 0)

            if pd.isna(funding):
                continue

            if self.action_mode == "avoid_longs":
                # Only go long when funding is not extreme positive
                if not in_position and funding < self.extreme_positive:
                    # Check for mean reversion opportunity after extreme
                    if funding < -self.normal_threshold:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                elif in_position and funding > self.extreme_positive:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

            elif self.action_mode == "contrarian":
                # Fade extreme funding
                if not in_position:
                    if funding > self.extreme_positive:
                        # Market overleveraged long, expect reversal
                        signals.loc[idx] = Signal.SELL
                        in_position = True
                    elif funding < self.extreme_negative:
                        # Market overleveraged short, expect reversal
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                else:
                    if abs(funding) < self.normal_threshold:
                        # Funding normalized, exit
                        signals.loc[idx] = Signal.SELL if in_position else Signal.BUY
                        in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "funding_rate_carry",
    description="Long spot when funding is positive to collect carry",
)
class FundingRateCarryStrategy(AlternativeDataBaseStrategy):
    """
    Funding Rate Carry Strategy.
    
    When funding rate is consistently positive, being long spot
    implicitly collects the funding (vs being short perpetual).
    
    This is a carry strategy, not a directional prediction.
    """

    name = "funding_rate_carry"

    def _setup(
        self,
        funding_avg_period: int = 9,
        min_avg_funding: float = 0.0003,
        exit_threshold: float = 0.0,
        min_hold_hours: int = 24,
        **kwargs,
    ) -> None:
        self.funding_avg_period = funding_avg_period
        self.min_avg_funding = min_avg_funding
        self.exit_threshold = exit_threshold
        self.min_hold_hours = min_hold_hours

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        if self._funding_data is None or self._funding_data.empty:
            logger.warning("No funding rate data available")
            return signals

        # Calculate average funding
        funding_avg = self._funding_data.rolling(
            window=self.funding_avg_period, min_periods=1
        ).mean()

        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            funding = funding_avg.get(idx, 0)

            if pd.isna(funding):
                continue

            if not in_position:
                # Enter when average funding is positive enough
                if funding > self.min_avg_funding:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                # Exit when funding turns negative (after min hold)
                if i - entry_bar >= self.min_hold_hours:
                    if funding < self.exit_threshold:
                        signals.loc[idx] = Signal.SELL
                        in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "open_interest_divergence",
    description="Fade moves when price and OI diverge",
)
class OpenInterestDivergenceStrategy(AlternativeDataBaseStrategy):
    """
    Open Interest Divergence Strategy.
    
    When price rises but OI declines, the rally is weak (distribution).
    When price falls but OI rises, sellers are being absorbed.
    
    Trade against moves with negative OI confirmation.
    """

    name = "open_interest_divergence"

    def _setup(
        self,
        price_lookback: int = 24,
        oi_lookback: int = 24,
        price_threshold: float = 0.02,
        oi_threshold: float = -0.01,
        signal_mode: str = "fade",
        **kwargs,
    ) -> None:
        self.price_lookback = price_lookback
        self.oi_lookback = oi_lookback
        self.price_threshold = price_threshold
        self.oi_threshold = oi_threshold
        self.signal_mode = signal_mode

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        if self._oi_data is None or self._oi_data.empty:
            logger.warning("No open interest data available")
            return signals

        # Calculate price and OI changes
        price_change = candles["close"].pct_change(self.price_lookback)
        oi_change = self._oi_data.pct_change(self.oi_lookback)

        in_position = False

        for idx in candles.index:
            price_chg = price_change.get(idx, 0)
            oi_chg = oi_change.get(idx, 0)

            if pd.isna(price_chg) or pd.isna(oi_chg):
                continue

            # Detect divergence
            # Price up + OI down = weak rally (fade long)
            weak_rally = price_chg > self.price_threshold and oi_chg < self.oi_threshold

            # Price down + OI up = absorption (fade short / go long)
            absorption = price_chg < -self.price_threshold and oi_chg > -self.oi_threshold

            if self.signal_mode == "fade":
                if not in_position:
                    if absorption:
                        # Sellers being absorbed, go long
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                    elif weak_rally:
                        # Weak rally, could short if enabled
                        pass  # For now, just avoid longs
                else:
                    if weak_rally or price_chg < 0:
                        signals.loc[idx] = Signal.SELL
                        in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "liquidation_cascade_fade",
    description="Mean reversion after large liquidation events",
)
class LiquidationCascadeFadeStrategy(BaseStrategy):
    """
    Liquidation Cascade Fade Strategy.
    
    After large liquidation events, prices often overshoot.
    Fade the move expecting mean reversion.
    
    Note: Requires external liquidation data (disabled by default).
    """

    name = "liquidation_cascade_fade"

    def _setup(
        self,
        liquidation_threshold: float = 10_000_000,
        lookback_hours: int = 1,
        fade_delay: int = 1,
        hold_period: int = 24,
        min_cascade_size: float = 5_000_000,
        **kwargs,
    ) -> None:
        self.liquidation_threshold = liquidation_threshold
        self.lookback_hours = lookback_hours
        self.fade_delay = fade_delay
        self.hold_period = hold_period
        self.min_cascade_size = min_cascade_size
        self._liquidation_data: pd.DataFrame | None = None

    def set_liquidation_data(self, data: pd.DataFrame) -> None:
        """Set liquidation data with columns: long_liquidations, short_liquidations."""
        self._liquidation_data = data

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        if self._liquidation_data is None or self._liquidation_data.empty:
            logger.warning("No liquidation data available - strategy disabled")
            return signals

        # Detect liquidation cascades
        long_liqs = self._liquidation_data.get("long_liquidations", pd.Series())
        short_liqs = self._liquidation_data.get("short_liquidations", pd.Series())

        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            long_liq = long_liqs.get(idx, 0)
            short_liq = short_liqs.get(idx, 0)

            if pd.isna(long_liq):
                long_liq = 0
            if pd.isna(short_liq):
                short_liq = 0

            if not in_position:
                # Long cascade: longs got liquidated, price probably oversold
                if long_liq > self.liquidation_threshold:
                    # Wait for fade delay
                    if i >= self.fade_delay:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
                # Short cascade: shorts got liquidated, price probably overbought
                elif short_liq > self.liquidation_threshold:
                    # Could short here if enabled
                    pass
            else:
                if i - entry_bar >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)
