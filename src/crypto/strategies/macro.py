"""
Macro market structure strategies.

These strategies use macro crypto market indicators:
- BTC Dominance (BTC market cap / total market cap)
- Dominance trends and momentum
- Macro indicators (DXY, VIX, etc.)
"""

import logging

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register("btc_dominance_rotation")
class BTCDominanceRotationStrategy(BaseStrategy):
    """
    Rotate between BTC and alts based on dominance trend.
    
    BTC Dominance indicates capital flow:
    - Rising dominance = capital flowing to BTC (safe haven)
    - Falling dominance = capital rotating to alts (risk-on)
    
    Strategy:
    - For BTC: Buy when dominance rising
    - For Alts: Buy when dominance falling (alt season)
    - Hold for 1 week, rebalance
    
    Requires BTC dominance data - run ingest_tier2_data.py first.
    """
    
    name = "btc_dominance_rotation"
    
    def _setup(self, **params):
        self.dominance_window = params.get("dominance_window", 168)  # 7 days in hours
        self.rising_threshold = params.get("rising_threshold", 0.5)  # 0.5% weekly change
        self.falling_threshold = params.get("falling_threshold", -0.5)  # -0.5%
        self.hold_period = params.get("hold_period", 168)
        self.trade_alts = params.get("trade_alts", True)  # If True, buy alts on falling dom
        self._dominance_data: pd.Series | None = None
        self._symbol: str = ""
    
    def set_dominance_data(self, dom_series: pd.Series) -> None:
        """Set BTC dominance data aligned to candle index."""
        self._dominance_data = dom_series
    
    def set_symbol(self, symbol: str) -> None:
        """Set the symbol being traded (to determine if BTC or alt)."""
        self._symbol = symbol
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on BTC dominance rotation."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._dominance_data is None or self._dominance_data.empty:
            logger.warning(
                f"{self.name}: BTC dominance data not available. "
                "Run: python scripts/ingest_tier2_data.py --dominance-only"
            )
            return signals
        
        # Align to candles (daily data forward-filled to hourly)
        dominance = self._dominance_data.reindex(candles.index, method="ffill")
        
        # Calculate weekly change in dominance
        dom_change = dominance.pct_change(self.dominance_window) * 100  # As percentage
        
        is_btc = "BTC" in self._symbol.upper() and "ETH" not in self._symbol.upper()
        
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            dc = dom_change.get(idx)
            
            if pd.isna(dc):
                continue
            
            if not in_position:
                if is_btc:
                    # For BTC: buy when dominance rising
                    if dc > self.rising_threshold:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_idx = i
                else:
                    # For alts: buy when dominance falling (alt season)
                    if self.trade_alts and dc < self.falling_threshold:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_idx = i
            else:
                bars_held = i - entry_idx
                
                # Exit conditions
                should_exit = False
                
                if bars_held >= self.hold_period:
                    should_exit = True
                
                # Exit if trend reverses
                if is_btc and dc < self.falling_threshold:
                    should_exit = True
                elif not is_btc and dc > self.rising_threshold:
                    should_exit = True
                
                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)


@strategy_registry.register("dominance_momentum")
class DominanceMomentumStrategy(BaseStrategy):
    """
    Trade strong dominance momentum shifts.
    
    Large dominance changes signal regime shifts:
    - >2% increase in 7 days = strong BTC preference
    - >2% decrease in 7 days = strong alt preference
    
    Strategy:
    - Wait for strong momentum (>2% move)
    - Confirm with persistence
    - Hold for 1 week
    
    Requires BTC dominance data.
    """
    
    name = "dominance_momentum"
    
    def _setup(self, **params):
        self.momentum_period = params.get("momentum_period", 168)  # 7 days
        self.strong_move_threshold = params.get("strong_move_threshold", 2.0)  # 2%
        self.hold_period = params.get("hold_period", 168)
        self.confirmation_bars = params.get("confirmation_bars", 24)  # 1 day
        self._dominance_data: pd.Series | None = None
        self._symbol: str = ""
    
    def set_dominance_data(self, dom_series: pd.Series) -> None:
        """Set BTC dominance data aligned to candle index."""
        self._dominance_data = dom_series
    
    def set_symbol(self, symbol: str) -> None:
        """Set the symbol being traded."""
        self._symbol = symbol
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on dominance momentum."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._dominance_data is None or self._dominance_data.empty:
            logger.warning(
                f"{self.name}: BTC dominance data not available. "
                "Run: python scripts/ingest_tier2_data.py --dominance-only"
            )
            return signals
        
        # Align to candles
        dominance = self._dominance_data.reindex(candles.index, method="ffill")
        
        # Calculate momentum (absolute change in dominance percentage)
        dom_momentum = dominance.diff(self.momentum_period)
        
        is_btc = "BTC" in self._symbol.upper() and "ETH" not in self._symbol.upper()
        
        in_position = False
        entry_idx = None
        confirmation_count = 0
        
        for i, idx in enumerate(candles.index):
            mom = dom_momentum.get(idx)
            
            if pd.isna(mom):
                continue
            
            if not in_position:
                # Strong momentum detected
                if abs(mom) > self.strong_move_threshold:
                    confirmation_count += 1
                    
                    if confirmation_count >= self.confirmation_bars:
                        if is_btc and mom > 0:
                            # Rising dominance = BTC outperforming
                            signals.loc[idx] = Signal.BUY
                            in_position = True
                            entry_idx = i
                        elif not is_btc and mom < 0:
                            # Falling dominance = alts outperforming
                            signals.loc[idx] = Signal.BUY
                            in_position = True
                            entry_idx = i
                        
                        confirmation_count = 0
                else:
                    confirmation_count = 0
            else:
                bars_held = i - entry_idx
                
                if bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)


@strategy_registry.register("macro_correlation")
class MacroCorrelationStrategy(BaseStrategy):
    """
    Trade based on macro indicator correlations.
    
    Uses DXY (Dollar Index) and VIX (Volatility Index) as regime signals:
    - DXY rising + VIX rising = risk-off, avoid crypto longs
    - DXY falling + VIX falling = risk-on, favor crypto longs
    
    Requires macro indicator data - run ingest_tier2_data.py --macro-only first.
    """
    
    name = "macro_correlation"
    
    def _setup(self, **params):
        self.dxy_lookback = params.get("dxy_lookback", 120)  # 5 days
        self.vix_threshold = params.get("vix_threshold", 20)
        self.dxy_change_threshold = params.get("dxy_change_threshold", 0.01)  # 1%
        self.hold_period = params.get("hold_period", 72)
        self._dxy_data: pd.Series | None = None
        self._vix_data: pd.Series | None = None
    
    def set_macro_data(self, dxy: pd.Series | None, vix: pd.Series | None) -> None:
        """Set macro indicator data aligned to candle index."""
        self._dxy_data = dxy
        self._vix_data = vix
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on macro correlations."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._dxy_data is None or self._dxy_data.empty:
            logger.warning(
                f"{self.name}: DXY data not available. "
                "Run: python scripts/ingest_tier2_data.py --macro-only"
            )
            return signals
        
        # Align to candles
        dxy = self._dxy_data.reindex(candles.index, method="ffill")
        dxy_change = dxy.pct_change(self.dxy_lookback)
        
        vix = None
        if self._vix_data is not None and not self._vix_data.empty:
            vix = self._vix_data.reindex(candles.index, method="ffill")
        
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            dxy_chg = dxy_change.get(idx)
            vix_val = vix.get(idx) if vix is not None else None
            
            if pd.isna(dxy_chg):
                continue
            
            # Risk assessment
            risk_off = False
            risk_on = False
            
            if dxy_chg > self.dxy_change_threshold:
                # Dollar strengthening = risk-off
                if vix_val is not None and vix_val > self.vix_threshold:
                    risk_off = True
            elif dxy_chg < -self.dxy_change_threshold:
                # Dollar weakening = risk-on
                if vix_val is None or vix_val < self.vix_threshold:
                    risk_on = True
            
            if not in_position:
                if risk_on:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_idx = i
            else:
                bars_held = i - entry_idx
                
                if risk_off or bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)


@strategy_registry.register("dxy_inverse")
class DXYInverseStrategy(BaseStrategy):
    """
    Simple DXY inverse correlation strategy.
    
    BTC tends to move inversely to the US Dollar:
    - DXY down = BTC up (weak dollar = risk assets rally)
    - DXY up = BTC down (strong dollar = risk assets sell)
    
    Strategy:
    - Buy when DXY shows weakness (below moving average, falling)
    - Sell when DXY shows strength
    """
    
    name = "dxy_inverse"
    
    def _setup(self, **params):
        self.dxy_ma_period = params.get("dxy_ma_period", 120)  # 5 days
        self.dxy_momentum_period = params.get("dxy_momentum_period", 24)
        self.hold_period = params.get("hold_period", 72)
        self._dxy_data: pd.Series | None = None
    
    def set_dxy_data(self, dxy: pd.Series) -> None:
        """Set DXY data aligned to candle index."""
        self._dxy_data = dxy
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on DXY inverse correlation."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if self._dxy_data is None or self._dxy_data.empty:
            logger.warning(
                f"{self.name}: DXY data not available. "
                "Run: python scripts/ingest_tier2_data.py --macro-only"
            )
            return signals
        
        # Align to candles
        dxy = self._dxy_data.reindex(candles.index, method="ffill")
        dxy_ma = dxy.rolling(self.dxy_ma_period, min_periods=24).mean()
        dxy_momentum = dxy.pct_change(self.dxy_momentum_period)
        
        in_position = False
        entry_idx = None
        
        for i, idx in enumerate(candles.index):
            dxy_val = dxy.get(idx)
            dxy_ma_val = dxy_ma.get(idx)
            dxy_mom = dxy_momentum.get(idx)
            
            if pd.isna(dxy_val) or pd.isna(dxy_ma_val):
                continue
            
            # Dollar weakness conditions
            dxy_weak = dxy_val < dxy_ma_val and (pd.isna(dxy_mom) or dxy_mom < 0)
            dxy_strong = dxy_val > dxy_ma_val and (not pd.isna(dxy_mom) and dxy_mom > 0)
            
            if not in_position:
                if dxy_weak:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_idx = i
            else:
                bars_held = i - entry_idx
                
                if dxy_strong or bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
        
        return self.apply_filters(signals, candles)
