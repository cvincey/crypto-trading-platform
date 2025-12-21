"""
Hybrid strategies combining multiple successful approaches.

These strategies combine the best elements from Phase 1 winners:
- eth_btc_ratio_reversion: Cross-asset mean reversion (2.53 Sharpe)
- signal_confirmation_delay: Noise filtering (100% pass rate)
"""

import logging
from typing import Any

import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.cross_symbol_base import CrossSymbolBaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "eth_btc_ratio_confirmed",
    description="ETH/BTC ratio reversion with signal confirmation",
)
class ETHBTCRatioConfirmedStrategy(CrossSymbolBaseStrategy):
    """
    ETH/BTC Ratio Reversion with Signal Confirmation.
    
    Combines the two top-performing strategies from Phase 1:
    1. ETH/BTC Ratio Reversion: Mean reversion when ETH underperforms BTC
    2. Signal Confirmation Delay: Only act when signal persists for N bars
    
    This hybrid approach aims to:
    - Capture the strong alpha from cross-asset ratio reversion
    - Filter out false signals using confirmation delay
    - Reduce noise while maintaining the structural edge
    
    Phase 1 Results:
    - eth_btc_ratio_reversion: 2.53 OOS Sharpe, 100% beats B&H
    - signal_confirmation_delay: 1.34 OOS Sharpe, 100% pass rate
    
    Expected improvement: Higher pass rate with maintained alpha.
    """

    name = "eth_btc_ratio_confirmed"

    def _setup(
        self,
        lookback: int = 168,
        entry_threshold: float = -2.0,
        exit_threshold: float = -0.5,
        max_hold_hours: int = 72,
        confirmation_delay: int = 3,
        **kwargs,
    ) -> None:
        """
        Initialize hybrid strategy parameters.
        
        Args:
            lookback: Bars for rolling z-score calculation (default 168 = 7 days)
            entry_threshold: Z-score below which to consider entry (default -2.0)
            exit_threshold: Z-score above which to exit (default -0.5)
            max_hold_hours: Maximum bars to hold position (default 72)
            confirmation_delay: Bars signal must persist before entry (default 3)
        """
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_hold_hours = max_hold_hours
        self.confirmation_delay = confirmation_delay

    def get_reference_symbols(self) -> list[str]:
        """Require BTC data for ratio calculation."""
        return ["BTCUSDT"]

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """
        Generate signals using ETH/BTC ratio reversion + confirmation.
        
        Logic:
        1. Calculate ETH/BTC ratio z-score
        2. When z-score < entry_threshold for confirmation_delay bars, BUY
        3. When z-score > exit_threshold or max_hold reached, SELL
        
        Args:
            candles: Target symbol OHLCV data (typically ETHUSDT)
            
        Returns:
            Series of Signal values
        """
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Check for required reference data
        if not self.has_reference_data("BTCUSDT"):
            logger.warning("No reference data for BTCUSDT - returning neutral signals")
            return signals

        # Get BTC close aligned to target candles
        btc = self.align_reference_to_target("BTCUSDT", candles.index, ["close"])

        if btc.empty or btc["close"].isna().all():
            logger.warning("BTC reference data is empty or all NaN")
            return signals

        # Calculate ETH/BTC ratio
        ratio = candles["close"].astype(float) / btc["close"].astype(float)

        # Calculate rolling z-score of ratio
        ratio_mean = ratio.rolling(self.lookback, min_periods=self.lookback // 2).mean()
        ratio_std = ratio.rolling(self.lookback, min_periods=self.lookback // 2).std()
        z_score = (ratio - ratio_mean) / ratio_std

        # Track state
        in_position = False
        entry_bar = 0
        pending_entry_count = 0  # Count of consecutive bars below threshold

        for i, idx in enumerate(candles.index):
            z = z_score.get(idx, float("nan"))

            if pd.isna(z):
                continue

            if not in_position:
                # Check for entry signal
                if z < self.entry_threshold:
                    pending_entry_count += 1
                    
                    # Enter only after confirmation_delay consecutive bars
                    if pending_entry_count >= self.confirmation_delay:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
                        pending_entry_count = 0  # Reset counter
                else:
                    # Reset counter if signal doesn't persist
                    pending_entry_count = 0
            else:
                # Check for exit conditions
                bars_held = i - entry_bar
                
                # Exit when ratio normalizes OR max hold period reached
                if z > self.exit_threshold or bars_held >= self.max_hold_hours:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
                    pending_entry_count = 0

        return self.apply_filters(signals, candles)

    def get_strategy_info(self) -> dict[str, Any]:
        """Return strategy metadata for logging/reporting."""
        return {
            "name": self.name,
            "type": "hybrid",
            "components": ["eth_btc_ratio_reversion", "signal_confirmation_delay"],
            "parameters": {
                "lookback": self.lookback,
                "entry_threshold": self.entry_threshold,
                "exit_threshold": self.exit_threshold,
                "max_hold_hours": self.max_hold_hours,
                "confirmation_delay": self.confirmation_delay,
            },
            "expected_trades_per_month": 3,  # Low frequency
            "description": (
                "Combines ETH/BTC ratio mean reversion with signal confirmation. "
                "Enters long ETH when ratio z-score is below threshold for N "
                "consecutive bars, exits when ratio normalizes or max hold reached."
            ),
        }
