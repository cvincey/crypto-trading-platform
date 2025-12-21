"""
Market microstructure strategies based on volume and order flow.

These strategies use volume patterns to confirm or filter signals.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "volume_breakout_confirmation",
    description="Only trade volume-confirmed breakouts",
)
class VolumeBreakoutConfirmationStrategy(BaseStrategy):
    """
    Volume Breakout Confirmation Strategy.
    
    Only trade breakouts that are confirmed by above-average volume.
    Low-volume breakouts are more likely to fail.
    """

    name = "volume_breakout_confirmation"

    def _setup(
        self,
        breakout_period: int = 24,
        volume_multiplier: float = 2.0,
        volume_average_period: int = 48,
        require_close_confirmation: bool = True,
        **kwargs,
    ) -> None:
        self.breakout_period = breakout_period
        self.volume_multiplier = volume_multiplier
        self.volume_average_period = volume_average_period
        self.require_close_confirmation = require_close_confirmation

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Calculate breakout levels
        high_rolling = candles["high"].rolling(self.breakout_period).max().shift(1)
        low_rolling = candles["low"].rolling(self.breakout_period).min().shift(1)

        # Calculate volume average
        volume = candles["volume"]
        volume_avg = volume.rolling(self.volume_average_period).mean()
        volume_high = volume > (volume_avg * self.volume_multiplier)

        close = candles["close"]
        in_position = False

        for idx in candles.index:
            c = close.get(idx)
            h = high_rolling.get(idx)
            l = low_rolling.get(idx)
            vol_ok = volume_high.get(idx, False)

            if pd.isna(c) or pd.isna(h) or pd.isna(l):
                continue

            if not in_position:
                # Bullish breakout with volume confirmation
                if c > h and vol_ok:
                    if self.require_close_confirmation:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                    else:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
            else:
                # Exit on bearish breakout
                if c < l:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "volume_divergence",
    description="Fade price moves with declining volume",
)
class VolumeDivergenceStrategy(BaseStrategy):
    """
    Volume Divergence Strategy.
    
    When price makes new highs but volume is declining,
    the move is weak and likely to reverse.
    """

    name = "volume_divergence"

    def _setup(
        self,
        price_lookback: int = 48,
        volume_lookback: int = 48,
        price_threshold: float = 0.05,
        volume_decline_threshold: float = 0.3,
        signal_mode: str = "fade",
        **kwargs,
    ) -> None:
        self.price_lookback = price_lookback
        self.volume_lookback = volume_lookback
        self.price_threshold = price_threshold
        self.volume_decline_threshold = volume_decline_threshold
        self.signal_mode = signal_mode

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Calculate price change
        price_change = candles["close"].pct_change(self.price_lookback)

        # Calculate volume change (use sum for better comparison)
        volume = candles["volume"]
        volume_recent = volume.rolling(self.volume_lookback // 2).sum()
        volume_prior = volume.shift(self.volume_lookback // 2).rolling(self.volume_lookback // 2).sum()
        volume_change = (volume_recent - volume_prior) / volume_prior

        in_position = False
        entry_type = None

        for idx in candles.index:
            price_chg = price_change.get(idx, 0)
            vol_chg = volume_change.get(idx, 0)

            if pd.isna(price_chg) or pd.isna(vol_chg):
                continue

            # Detect bearish divergence: price up, volume down
            bearish_div = (
                price_chg > self.price_threshold
                and vol_chg < -self.volume_decline_threshold
            )

            # Detect bullish divergence: price down, volume down (less selling pressure)
            bullish_div = (
                price_chg < -self.price_threshold
                and vol_chg < -self.volume_decline_threshold
            )

            if self.signal_mode == "fade":
                if not in_position:
                    if bullish_div:
                        # Price falling but selling pressure declining - buy
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_type = "bullish"
                else:
                    # Exit on opposite divergence or normalization
                    if bearish_div or price_chg > 0:
                        signals.loc[idx] = Signal.SELL
                        in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "buy_sell_imbalance",
    description="Trade extreme order flow imbalance",
)
class BuySellImbalanceStrategy(BaseStrategy):
    """
    Buy/Sell Imbalance Strategy.
    
    Estimate buy/sell volume from candle data.
    Trade when imbalance is extreme.
    
    Buy volume approximated by: volume * (close - low) / (high - low)
    """

    name = "buy_sell_imbalance"

    def _setup(
        self,
        imbalance_lookback: int = 12,
        buy_threshold: float = 0.70,
        sell_threshold: float = 0.30,
        smoothing_period: int = 3,
        min_volume_percentile: float = 50,
        **kwargs,
    ) -> None:
        self.imbalance_lookback = imbalance_lookback
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.smoothing_period = smoothing_period
        self.min_volume_percentile = min_volume_percentile

    def _estimate_buy_volume_ratio(self, candles: pd.DataFrame) -> pd.Series:
        """
        Estimate the ratio of buy volume to total volume.
        
        Uses the position of close relative to high-low range.
        If close is near high, most volume was buying.
        If close is near low, most volume was selling.
        """
        high = candles["high"]
        low = candles["low"]
        close = candles["close"]

        # Avoid division by zero
        range_size = high - low
        range_size = range_size.replace(0, np.nan)

        # Buy ratio: how close is the close to the high?
        buy_ratio = (close - low) / range_size

        return buy_ratio

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Estimate buy volume ratio
        buy_ratio = self._estimate_buy_volume_ratio(candles)

        # Smooth the ratio
        buy_ratio_smooth = buy_ratio.rolling(self.smoothing_period).mean()

        # Calculate rolling average imbalance
        imbalance = buy_ratio_smooth.rolling(self.imbalance_lookback).mean()

        # Volume filter
        volume = candles["volume"]
        volume_pct = volume.rolling(100).apply(
            lambda x: (x.rank().iloc[-1] / len(x)) * 100 if len(x) > 1 else 50,
            raw=False
        )

        in_position = False

        for idx in candles.index:
            imb = imbalance.get(idx, 0.5)
            vol_pct = volume_pct.get(idx, 50)

            if pd.isna(imb):
                continue

            # Volume filter
            if vol_pct < self.min_volume_percentile:
                continue

            if not in_position:
                # Strong buying imbalance
                if imb > self.buy_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
            else:
                # Exit on selling imbalance
                if imb < self.sell_threshold:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)
