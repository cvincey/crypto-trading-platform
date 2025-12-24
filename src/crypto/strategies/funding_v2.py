"""
Enhanced funding rate strategies.

These strategies use funding rate data in more sophisticated ways:
- Cross-exchange funding arbitrage proxies
- Funding momentum as trend indicator
"""

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy
from crypto.strategies.registry import strategy_registry


@strategy_registry.register("funding_arbitrage_proxy")
class FundingArbitrageProxyStrategy(BaseStrategy):
    """
    Trade based on funding rate divergence across exchanges.
    
    When funding rates diverge across exchanges:
    - Binance: 0.05%, Bybit: 0.02% = arbitrage opportunity
    - Divergence signals market inefficiency
    - Trade toward convergence
    
    Strategy:
    - Detect divergence > 0.02% between exchanges
    - Enter on divergence
    - Exit when rates converge
    
    Note: Requires multi-exchange funding rate data.
    """
    
    name = "funding_arbitrage_proxy"
    
    def _setup(self, **params):
        self.exchanges = params.get("exchanges", ["binance", "bybit", "okx"])
        self.divergence_threshold = params.get("divergence_threshold", 0.0002)  # 0.02%
        self.lookback_periods = params.get("lookback_periods", 3)
        self.hold_period = params.get("hold_period", 24)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on funding arbitrage."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        # TODO: This strategy requires multi-exchange funding rate data
        # For now, return empty signals as placeholder
        
        logger = __import__('logging').getLogger(__name__)
        logger.warning(
            f"{self.name}: Multi-exchange funding rate data not available. "
            "Single-exchange funding rate data available but not sufficient for arbitrage."
        )
        
        return signals
        
        # Example implementation (when data is available):
        # funding_binance = get_funding_rates("binance", candles.index)
        # funding_bybit = get_funding_rates("bybit", candles.index)
        # funding_okx = get_funding_rates("okx", candles.index)
        # 
        # # Calculate divergence
        # funding_df = pd.DataFrame({
        #     "binance": funding_binance,
        #     "bybit": funding_bybit,
        #     "okx": funding_okx
        # })
        # 
        # # Rolling average divergence
        # funding_mean = funding_df.mean(axis=1)
        # funding_std = funding_df.std(axis=1)
        # 
        # in_position = False
        # entry_idx = None
        # 
        # for i in range(self.lookback_periods, len(candles)):
        #     divergence = funding_std.iloc[i]
        #     
        #     # High divergence detected
        #     if divergence > self.divergence_threshold:
        #         # Trade based on which exchange has extreme funding
        #         # If binance funding >> others, it will likely converge down
        #         signals.iloc[i] = Signal.BUY
        #         in_position = True
        #         entry_idx = i
        #     
        #     # Exit when divergence normalizes
        #     elif divergence < self.divergence_threshold / 2:
        #         if in_position:
        #             signals.iloc[i] = Signal.SELL
        #             in_position = False
        #     
        #     # Exit after hold period
        #     if in_position and entry_idx is not None:
        #         bars_held = i - entry_idx
        #         if bars_held >= self.hold_period:
        #             signals.iloc[i] = Signal.SELL
        #             in_position = False


@strategy_registry.register("funding_momentum")
class FundingMomentumStrategy(BaseStrategy):
    """
    Use funding rate momentum as trend indicator.
    
    Funding rate trends:
    - Consistently increasing funding = strong uptrend (longs willing to pay)
    - Consistently decreasing funding = weakening/downtrend
    
    Strategy:
    - Detect 3+ consecutive funding increases
    - Enter long (trend continuation)
    - Exit when funding stops increasing or becomes extreme
    
    Note: Requires funding rate history.
    """
    
    name = "funding_momentum"
    
    def _setup(self, **params):
        self.momentum_periods = params.get("momentum_periods", 3)  # 3 consecutive
        self.min_rate_change = params.get("min_rate_change", 0.0001)  # 0.01%
        self.extreme_threshold = params.get("extreme_threshold", 0.0005)  # 0.05%
        self.hold_period = params.get("hold_period", 24)
    
    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """Generate signals based on funding momentum."""
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)
        
        # TODO: This strategy requires funding rate data
        # For now, return empty signals as placeholder
        
        logger = __import__('logging').getLogger(__name__)
        logger.warning(
            f"{self.name}: Funding rate data not available. "
            "Run funding rate ingestion first."
        )
        
        return signals
        
        # Example implementation (when data is available):
        # from crypto.data.alternative_data import FundingRateRepository
        # 
        # # Get funding rates aligned to candles
        # repo = FundingRateRepository()
        # funding = await repo.get_funding_aligned_to_candles(
        #     symbol=candles.attrs.get("symbol", "BTCUSDT"),
        #     candle_index=candles.index
        # )
        # 
        # # Calculate funding rate changes
        # funding_diff = funding.diff()
        # 
        # in_position = False
        # entry_idx = None
        # consecutive_increases = 0
        # 
        # for i in range(self.momentum_periods * 8, len(candles)):  # 8h per funding
        #     # Check last N funding periods for consistent increases
        #     recent_changes = funding_diff.iloc[i - self.momentum_periods * 8:i:8]
        #     
        #     # All recent changes positive and above threshold
        #     if all(recent_changes > self.min_rate_change):
        #         consecutive_increases += 1
        #         
        #         if consecutive_increases >= self.momentum_periods:
        #             # Funding momentum confirmed
        #             current_funding = funding.iloc[i]
        #             
        #             # Don't enter if funding already extreme
        #             if abs(current_funding) < self.extreme_threshold:
        #                 signals.iloc[i] = Signal.BUY
        #                 in_position = True
        #                 entry_idx = i
        #             
        #             consecutive_increases = 0
        #     else:
        #         consecutive_increases = 0
        #         
        #         # Exit if funding momentum reverses
        #         if in_position and recent_changes.iloc[-1] < 0:
        #             signals.iloc[i] = Signal.SELL
        #             in_position = False
        #     
        #     # Exit if funding becomes extreme
        #     if in_position and abs(funding.iloc[i]) > self.extreme_threshold:
        #         signals.iloc[i] = Signal.SELL
        #         in_position = False
        #     
        #     # Exit after hold period
        #     if in_position and entry_idx is not None:
        #         bars_held = i - entry_idx
        #         if bars_held >= self.hold_period:
        #             signals.iloc[i] = Signal.SELL
        #             in_position = False
