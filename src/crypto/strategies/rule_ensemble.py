"""
Rule-based ensemble strategy - no ML, cannot overfit.

This module contains rule-based trading strategies that use fixed,
interpretable rules rather than learned patterns. These strategies
cannot overfit to temporal patterns because they don't learn from data.

Key strategies:
- RuleEnsembleStrategy: Voting-based ensemble of technical rules
- TrendFollowingRulesStrategy: Classic trend following with multiple confirmations
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


# =============================================================================
# Rule-Based Ensemble: No ML, Pure Technical Rules
# =============================================================================


@strategy_registry.register(
    "rule_ensemble",
    description="Rule-based ensemble - voting across ADX, RSI, Volume, MA rules",
)
class RuleEnsembleStrategy(BaseStrategy):
    """
    Rule-based ensemble strategy using voting mechanism.
    
    This strategy CANNOT overfit because it uses fixed rules rather than
    learning from historical data. Rules are based on classic technical analysis.
    
    Rules for BUY signal:
    1. ADX trend confirmation: ADX > threshold (trending market)
    2. RSI not overbought: RSI < 70 (room to go up)
    3. RSI momentum: RSI > previous RSI (momentum building)
    4. Volume spike: Volume > multiplier * average (conviction)
    5. MA alignment: EMA12 > EMA26 (trend direction)
    6. Price above SMA: Close > SMA20 (uptrend)
    
    Rules for SELL signal:
    1. RSI overbought: RSI > 70 OR
    2. MA cross down: EMA12 < EMA26 OR
    3. Price below SMA: Close < SMA20
    
    Signal is generated when min_agreement rules agree.
    """

    name = "rule_ensemble"

    def _setup(
        self,
        # ADX parameters
        adx_threshold: int = 25,
        adx_period: int = 14,
        # RSI parameters
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        # Volume parameters
        volume_period: int = 20,
        volume_multiplier: float = 1.5,
        # MA parameters
        fast_ema: int = 12,
        slow_ema: int = 26,
        sma_period: int = 20,
        # Voting parameters
        min_buy_agreement: int = 4,   # Need 4 of 6 rules for buy
        min_sell_agreement: int = 2,  # Need 2 of 3 rules for sell
        **kwargs,
    ) -> None:
        self.adx_threshold = adx_threshold
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.sma_period = sma_period
        self.min_buy_agreement = min_buy_agreement
        self.min_sell_agreement = min_sell_agreement

    def _compute_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators needed for rules."""
        df = pd.DataFrame(index=candles.index)
        
        # ADX
        df["adx"] = indicator_registry.compute("adx", candles, period=self.adx_period)
        
        # RSI
        df["rsi"] = indicator_registry.compute("rsi", candles, period=self.rsi_period)
        df["rsi_prev"] = df["rsi"].shift(1)
        
        # Volume
        vol_sma = indicator_registry.compute("volume_sma", candles, period=self.volume_period)
        df["volume_ratio"] = candles["volume"] / vol_sma
        
        # EMAs
        df["ema_fast"] = indicator_registry.compute("ema", candles, period=self.fast_ema)
        df["ema_slow"] = indicator_registry.compute("ema", candles, period=self.slow_ema)
        
        # SMA
        df["sma"] = indicator_registry.compute("sma", candles, period=self.sma_period)
        
        # Price
        df["close"] = candles["close"]
        
        return df

    def _evaluate_buy_rules(self, row: pd.Series) -> int:
        """Count how many buy rules are satisfied. Returns count 0-6."""
        count = 0
        
        # Rule 1: ADX trend confirmation
        if row["adx"] > self.adx_threshold:
            count += 1
        
        # Rule 2: RSI not overbought
        if row["rsi"] < self.rsi_overbought:
            count += 1
        
        # Rule 3: RSI momentum (rising)
        if pd.notna(row["rsi_prev"]) and row["rsi"] > row["rsi_prev"]:
            count += 1
        
        # Rule 4: Volume spike
        if row["volume_ratio"] > self.volume_multiplier:
            count += 1
        
        # Rule 5: MA alignment (fast > slow)
        if row["ema_fast"] > row["ema_slow"]:
            count += 1
        
        # Rule 6: Price above SMA
        if row["close"] > row["sma"]:
            count += 1
        
        return count

    def _evaluate_sell_rules(self, row: pd.Series) -> int:
        """Count how many sell rules are satisfied. Returns count 0-3."""
        count = 0
        
        # Rule 1: RSI overbought
        if row["rsi"] > self.rsi_overbought:
            count += 1
        
        # Rule 2: MA cross down (fast < slow)
        if row["ema_fast"] < row["ema_slow"]:
            count += 1
        
        # Rule 3: Price below SMA
        if row["close"] < row["sma"]:
            count += 1
        
        return count

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        
        indicators = self._compute_indicators(candles)
        signals = self.create_signal_series(candles.index)
        
        valid_idx = indicators.dropna().index
        
        in_position = False
        for idx in valid_idx:
            row = indicators.loc[idx]
            
            if not in_position:
                # Check buy rules
                buy_votes = self._evaluate_buy_rules(row)
                if buy_votes >= self.min_buy_agreement:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
            else:
                # Check sell rules
                sell_votes = self._evaluate_sell_rules(row)
                if sell_votes >= self.min_sell_agreement:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return signals


# =============================================================================
# Trend Following Rules Strategy
# =============================================================================


@strategy_registry.register(
    "trend_following_rules",
    description="Classic trend following with multiple confirmations",
)
class TrendFollowingRulesStrategy(BaseStrategy):
    """
    Classic trend following strategy with multiple confirmations.
    
    Entry conditions (ALL must be true):
    1. ADX > threshold (confirmed trend)
    2. +DI > -DI (bullish trend)
    3. Price > EMA20 (above trend)
    4. EMA20 > EMA50 (trend alignment)
    
    Exit conditions (ANY triggers exit):
    1. ADX < exit_threshold (trend weakening)
    2. -DI > +DI (bearish crossover)
    3. Price < EMA20 (trend break)
    """

    name = "trend_following_rules"

    def _setup(
        self,
        adx_threshold: int = 25,
        adx_exit_threshold: int = 20,
        adx_period: int = 14,
        fast_ema: int = 20,
        slow_ema: int = 50,
        **kwargs,
    ) -> None:
        self.adx_threshold = adx_threshold
        self.adx_exit_threshold = adx_exit_threshold
        self.adx_period = adx_period
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema

    def _compute_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute trend indicators."""
        df = pd.DataFrame(index=candles.index)
        
        # ADX and directional indicators
        df["adx"] = indicator_registry.compute("adx", candles, period=self.adx_period)
        
        # Compute +DI and -DI manually
        high = candles["high"]
        low = candles["low"]
        close = candles["close"]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.adx_period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm, index=candles.index).rolling(self.adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=candles.index).rolling(self.adx_period).mean() / atr
        
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        
        # EMAs
        df["ema_fast"] = indicator_registry.compute("ema", candles, period=self.fast_ema)
        df["ema_slow"] = indicator_registry.compute("ema", candles, period=self.slow_ema)
        
        # Price
        df["close"] = candles["close"]
        
        return df

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        
        indicators = self._compute_indicators(candles)
        signals = self.create_signal_series(candles.index)
        
        valid_idx = indicators.dropna().index
        
        in_position = False
        for idx in valid_idx:
            row = indicators.loc[idx]
            
            if not in_position:
                # Check all entry conditions
                entry_conditions = [
                    row["adx"] > self.adx_threshold,      # Confirmed trend
                    row["plus_di"] > row["minus_di"],     # Bullish
                    row["close"] > row["ema_fast"],       # Above fast EMA
                    row["ema_fast"] > row["ema_slow"],    # EMA alignment
                ]
                
                if all(entry_conditions):
                    signals.loc[idx] = Signal.BUY
                    in_position = True
            else:
                # Check any exit condition
                exit_conditions = [
                    row["adx"] < self.adx_exit_threshold,  # Trend weakening
                    row["minus_di"] > row["plus_di"],      # Bearish crossover
                    row["close"] < row["ema_fast"],        # Below fast EMA
                ]
                
                if any(exit_conditions):
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return signals


# =============================================================================
# Mean Reversion Rules Strategy
# =============================================================================


@strategy_registry.register(
    "mean_reversion_rules",
    description="Mean reversion with RSI and Bollinger Band rules",
)
class MeanReversionRulesStrategy(BaseStrategy):
    """
    Mean reversion strategy for ranging markets.
    
    Entry conditions:
    1. ADX < threshold (ranging market)
    2. RSI < oversold OR price < lower Bollinger Band
    3. Volume confirmation
    
    Exit conditions:
    1. RSI > neutral (50) OR
    2. Price > middle Bollinger Band
    """

    name = "mean_reversion_rules"

    def _setup(
        self,
        adx_threshold: int = 25,  # Below this = ranging
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_neutral: int = 50,
        bb_period: int = 20,
        bb_std: float = 2.0,
        **kwargs,
    ) -> None:
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_neutral = rsi_neutral
        self.bb_period = bb_period
        self.bb_std = bb_std

    def _compute_indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute mean reversion indicators."""
        df = pd.DataFrame(index=candles.index)
        
        # ADX for regime detection
        df["adx"] = indicator_registry.compute("adx", candles)
        
        # RSI
        df["rsi"] = indicator_registry.compute("rsi", candles, period=self.rsi_period)
        
        # Bollinger Bands
        sma = candles["close"].rolling(self.bb_period).mean()
        std = candles["close"].rolling(self.bb_period).std()
        df["bb_upper"] = sma + self.bb_std * std
        df["bb_middle"] = sma
        df["bb_lower"] = sma - self.bb_std * std
        
        # Price
        df["close"] = candles["close"]
        
        return df

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        
        indicators = self._compute_indicators(candles)
        signals = self.create_signal_series(candles.index)
        
        valid_idx = indicators.dropna().index
        
        in_position = False
        for idx in valid_idx:
            row = indicators.loc[idx]
            
            if not in_position:
                # Only trade in ranging markets
                if row["adx"] < self.adx_threshold:
                    # Entry conditions
                    oversold = row["rsi"] < self.rsi_oversold
                    below_bb = row["close"] < row["bb_lower"]
                    
                    if oversold or below_bb:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
            else:
                # Exit conditions
                neutral_rsi = row["rsi"] > self.rsi_neutral
                above_middle = row["close"] > row["bb_middle"]
                
                if neutral_rsi or above_middle:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return signals
