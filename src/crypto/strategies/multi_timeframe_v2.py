"""
Multi-timeframe and cross-asset dynamics strategies.

These strategies exploit disagreements across timeframes or leadership dynamics:
- Multi-timeframe momentum divergence
- Leader-follower rotation detection
- Sector dispersion trades
"""

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.cross_symbol_base import CrossSymbolBaseStrategy
from crypto.strategies.registry import strategy_registry


@strategy_registry.register("mtf_momentum_divergence")
class MultiTimeframeMomentumDivergenceStrategy(BaseStrategy):
    """
    Trade when short and long timeframe momentum disagree.
    
    Divergence signals potential reversals:
    - 1h momentum positive but 4h negative = short-term rally in downtrend → short
    - 1h momentum negative but 4h positive = short-term dip in uptrend → long
    
    Note: This implementation uses simulated multi-timeframe data from 1h candles.
    """
    
    name = "mtf_momentum_divergence"
    
    def _setup(self, **params):
        # Since we only have 1h data, we'll simulate by using different periods
        self.fast_period = params.get("fast_momentum_period", 12)  # ~12h
        self.slow_period = params.get("slow_momentum_period", 48)  # ~48h (2d)
        self.divergence_threshold = params.get("divergence_threshold", 0.01)
        self.hold_period = params.get("hold_period", 24)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on multi-timeframe divergence."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if len(candles) < self.slow_period + 10:
            return signals
        
        prices = candles["close"]
        
        # Calculate momentum on different timeframes
        fast_momentum = prices.pct_change(periods=self.fast_period)
        slow_momentum = prices.pct_change(periods=self.slow_period)
        
        # Track position
        in_position = False
        entry_idx = None
        
        for i in range(self.slow_period, len(candles)):
            if pd.isna(fast_momentum.iloc[i]) or pd.isna(slow_momentum.iloc[i]):
                continue
            
            fast_mom = fast_momentum.iloc[i]
            slow_mom = slow_momentum.iloc[i]
            
            # Bullish divergence: short-term weak but long-term strong
            if (slow_mom > self.divergence_threshold and 
                fast_mom < -self.divergence_threshold):
                signals.iloc[i] = Signal.BUY
                in_position = True
                entry_idx = i
            
            # Bearish divergence: short-term strong but long-term weak
            elif (slow_mom < -self.divergence_threshold and 
                  fast_mom > self.divergence_threshold):
                if in_position:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
            
            # Exit after hold period
            if in_position and entry_idx is not None:
                bars_held = i - entry_idx
                if bars_held >= self.hold_period:
                    signals.iloc[i] = Signal.SELL
                    in_position = False
        
        return signals


@strategy_registry.register("leader_follower_rotation")
class LeaderFollowerRotationStrategy(CrossSymbolBaseStrategy):
    """
    Detect dynamic leadership between BTC and ETH.
    
    Sometimes BTC leads (ETH follows with lag).
    Sometimes ETH leads (BTC follows).
    
    Strategy:
    - Calculate rolling cross-correlation with different lags
    - Positive lag = leader leads follower
    - Trade the follower when leader makes a move
    
    Note: Requires reference data for both BTC and ETH.
    """
    
    name = "leader_follower_rotation"
    
    def _setup(self, **params):
        self.leader_symbols = params.get("leader_symbols", ["BTCUSDT", "ETHUSDT"])
        self.correlation_window = params.get("correlation_window", 168)
        self.lead_lag_window = params.get("lead_lag_window", 24)
        self.min_correlation = params.get("min_correlation", 0.80)
        self.hold_period = params.get("hold_period", 48)
        
        # We'll need to determine which symbol leads dynamically
        self.max_lag = 6  # Check up to 6 hour lag
    
    def generate_signals(
        self,
        candles: pd.DataFrame,
        reference_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        """Generate signals based on leader-follower rotation."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if reference_data is None or len(self.leader_symbols) != 2:
            return signals
        
        # Get both leader candidate data
        btc_data = reference_data.get("BTCUSDT")
        eth_data = reference_data.get("ETHUSDT")
        
        if btc_data is None or eth_data is None:
            return signals
        
        if len(candles) < self.correlation_window:
            return signals
        
        # Align indices
        common_idx = candles.index.intersection(btc_data.index).intersection(eth_data.index)
        
        if len(common_idx) < self.correlation_window:
            return signals
        
        btc_returns = btc_data.loc[common_idx, "close"].pct_change()
        eth_returns = eth_data.loc[common_idx, "close"].pct_change()
        target_returns = candles.loc[common_idx, "close"].pct_change()
        
        # Determine current leader by checking cross-correlation at different lags
        for i in range(self.correlation_window, len(common_idx)):
            window_start = i - self.correlation_window
            
            btc_window = btc_returns.iloc[window_start:i]
            eth_window = eth_returns.iloc[window_start:i]
            
            # Find optimal lag for each potential leader
            best_btc_lag = 0
            best_btc_corr = 0
            best_eth_lag = 0
            best_eth_corr = 0
            
            for lag in range(1, min(self.max_lag + 1, len(btc_window) // 4)):
                # BTC leads ETH
                btc_lead = btc_window.iloc[:-lag] if lag > 0 else btc_window
                eth_follow = eth_window.iloc[lag:]
                
                if len(btc_lead) == len(eth_follow) and len(btc_lead) > 10:
                    corr = btc_lead.corr(eth_follow)
                    if abs(corr) > abs(best_btc_corr):
                        best_btc_corr = corr
                        best_btc_lag = lag
                
                # ETH leads BTC
                eth_lead = eth_window.iloc[:-lag] if lag > 0 else eth_window
                btc_follow = btc_window.iloc[lag:]
                
                if len(eth_lead) == len(btc_follow) and len(eth_lead) > 10:
                    corr = eth_lead.corr(btc_follow)
                    if abs(corr) > abs(best_eth_corr):
                        best_eth_corr = corr
                        best_eth_lag = lag
            
            # Determine current leader
            if abs(best_btc_corr) > abs(best_eth_corr) and abs(best_btc_corr) > self.min_correlation:
                # BTC leads - trade based on BTC recent move
                if best_btc_lag > 0:
                    btc_signal = btc_returns.iloc[i - best_btc_lag:i].mean()
                    
                    if btc_signal > 0.005:  # BTC up, expect ETH to follow
                        idx_pos = candles.index.get_loc(common_idx[i])
                        signals.iloc[idx_pos] = Signal.BUY
                    elif btc_signal < -0.005:
                        idx_pos = candles.index.get_loc(common_idx[i])
                        signals.iloc[idx_pos] = Signal.SELL
            
            elif abs(best_eth_corr) > self.min_correlation:
                # ETH leads - trade based on ETH recent move
                if best_eth_lag > 0:
                    eth_signal = eth_returns.iloc[i - best_eth_lag:i].mean()
                    
                    if eth_signal > 0.005:  # ETH up, expect BTC to follow
                        idx_pos = candles.index.get_loc(common_idx[i])
                        signals.iloc[idx_pos] = Signal.BUY
                    elif eth_signal < -0.005:
                        idx_pos = candles.index.get_loc(common_idx[i])
                        signals.iloc[idx_pos] = Signal.SELL
        
        # Apply hold period
        signals = self._apply_hold_period(signals, candles.index)
        
        return signals
    
    def _apply_hold_period(self, signals: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Apply minimum hold period."""
        result = signals.copy()
        in_position = False
        entry_idx = None
        
        for i in range(len(index)):
            if signals.iloc[i] == Signal.BUY and not in_position:
                in_position = True
                entry_idx = i
                result.iloc[i] = Signal.BUY
            elif in_position:
                bars_held = i - entry_idx
                if bars_held < self.hold_period:
                    result.iloc[i] = Signal.HOLD
                elif signals.iloc[i] == Signal.SELL:
                    result.iloc[i] = Signal.SELL
                    in_position = False
                else:
                    result.iloc[i] = Signal.HOLD
            else:
                result.iloc[i] = Signal.HOLD
        
        return result


@strategy_registry.register("sector_dispersion_trade")
class SectorDispersionTradeStrategy(CrossSymbolBaseStrategy):
    """
    Trade sector convergence/divergence.
    
    High dispersion among sector constituents = opportunity:
    - Convergence trade: Buy laggards, expecting catch-up
    - Divergence trade: Sector rotation, trade leaders
    
    Strategy:
    - Calculate dispersion (std dev of returns) across sector
    - High dispersion = trade opportunity
    - Mode: convergence (buy laggards) or divergence (buy leaders)
    """
    
    name = "sector_dispersion_trade"
    
    def _setup(self, **params):
        self.sectors = params.get("sectors", {
            "L1": ["SOLUSDT", "AVAXUSDT", "NEARUSDT", "APTUSDT"]
        })
        self.dispersion_window = params.get("dispersion_window", 168)
        self.high_dispersion_threshold = params.get("high_dispersion_threshold", 0.15)
        self.trade_mode = params.get("trade_mode", "convergence")  # or divergence
        self.hold_period = params.get("hold_period", 168)
    
    def generate_signals(
        self,
        candles: pd.DataFrame,
        reference_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        """Generate signals based on sector dispersion."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        if reference_data is None:
            return signals
        
        if len(candles) < self.dispersion_window:
            return signals
        
        # Find which sector this symbol belongs to
        current_symbol = None
        current_sector = None
        
        for sector_name, symbols in self.sectors.items():
            # Check if any reference data matches this sector
            if any(sym in reference_data for sym in symbols):
                current_sector = sector_name
                break
        
        if current_sector is None:
            return signals
        
        sector_symbols = self.sectors[current_sector]
        
        # Collect sector data
        sector_data = {}
        for sym in sector_symbols:
            if sym in reference_data:
                sector_data[sym] = reference_data[sym]
        
        # Need at least 3 symbols for dispersion calculation
        if len(sector_data) < 3:
            return signals
        
        # Align all indices
        common_idx = candles.index
        for sym_data in sector_data.values():
            common_idx = common_idx.intersection(sym_data.index)
        
        if len(common_idx) < self.dispersion_window:
            return signals
        
        # Calculate returns for each symbol
        sector_returns = {}
        for sym, sym_data in sector_data.items():
            returns = sym_data.loc[common_idx, "close"].pct_change()
            sector_returns[sym] = returns
        
        # Calculate rolling dispersion
        for i in range(self.dispersion_window, len(common_idx)):
            window_start = i - self.dispersion_window
            
            # Get window returns for all symbols
            window_returns = []
            for sym, returns in sector_returns.items():
                window_ret = returns.iloc[window_start:i].sum()
                window_returns.append(window_ret)
            
            # Calculate dispersion (std dev of sector returns)
            dispersion = np.std(window_returns)
            
            # High dispersion = opportunity
            if dispersion > self.high_dispersion_threshold:
                current_symbol_return = candles.loc[common_idx[window_start:i], "close"].pct_change().sum()
                
                if self.trade_mode == "convergence":
                    # Buy if we're a laggard
                    if current_symbol_return < np.mean(window_returns):
                        idx_pos = candles.index.get_loc(common_idx[i])
                        signals.iloc[idx_pos] = Signal.BUY
                
                elif self.trade_mode == "divergence":
                    # Buy if we're a leader
                    if current_symbol_return > np.mean(window_returns):
                        idx_pos = candles.index.get_loc(common_idx[i])
                        signals.iloc[idx_pos] = Signal.BUY
        
        # Apply hold period
        signals = self._apply_hold_period(signals, candles.index)
        
        return signals
    
    def _apply_hold_period(self, signals: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Apply minimum hold period."""
        result = signals.copy()
        in_position = False
        entry_idx = None
        
        for i in range(len(index)):
            if signals.iloc[i] == Signal.BUY and not in_position:
                in_position = True
                entry_idx = i
                result.iloc[i] = Signal.BUY
            elif in_position:
                bars_held = i - entry_idx
                if bars_held < self.hold_period:
                    result.iloc[i] = Signal.HOLD
                elif signals.iloc[i] == Signal.SELL:
                    result.iloc[i] = Signal.SELL
                    in_position = False
                elif bars_held >= self.hold_period:
                    # Auto-exit after hold period
                    result.iloc[i] = Signal.SELL
                    in_position = False
                else:
                    result.iloc[i] = Signal.HOLD
            else:
                result.iloc[i] = Signal.HOLD
        
        return result
