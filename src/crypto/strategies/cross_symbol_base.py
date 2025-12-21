"""Base class for cross-symbol strategies that use signals from other symbols."""

from abc import ABC
from typing import Any

import pandas as pd

from crypto.core.types import Signal
from crypto.strategies.base import BaseStrategy


class CrossSymbolBaseStrategy(BaseStrategy, ABC):
    """
    Base class for strategies that use signals from other symbols.
    
    Cross-symbol strategies can access data from reference symbols (e.g., BTC, ETH)
    to make trading decisions on target symbols. This exploits the finding that
    cross-asset patterns generalize well in crypto markets.
    
    Key features:
    - Reference data storage and retrieval
    - Lookahead-safe data access (only uses data up to current timestamp)
    - Support for multiple reference symbols
    
    Usage:
        1. Create strategy inheriting from CrossSymbolBaseStrategy
        2. Override get_reference_symbols() to specify which symbols are needed
        3. Before backtesting, call set_reference_data() for each reference symbol
        4. In generate_signals(), use get_reference_candles() to access ref data
    
    Example:
        class BTCLeadAltFollowStrategy(CrossSymbolBaseStrategy):
            def get_reference_symbols(self) -> list[str]:
                return ["BTCUSDT"]
            
            def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
                btc = self.get_reference_candles("BTCUSDT", candles.index)
                # Use BTC data to generate signals for target symbol
                ...
    """

    name = "cross_symbol_base"

    def __init__(self, **params: Any):
        """Initialize cross-symbol strategy."""
        super().__init__(**params)
        self._reference_data: dict[str, pd.DataFrame] = {}
        self._reference_symbols: list[str] = []

    def get_reference_symbols(self) -> list[str]:
        """
        Return list of reference symbols needed by this strategy.
        
        Override this method to specify which symbols are required.
        The BacktestRunner will load these symbols and call set_reference_data().
        
        Returns:
            List of symbol strings (e.g., ["BTCUSDT", "ETHUSDT"])
        """
        return self._params.get("reference_symbols", [])

    def set_reference_data(self, symbol: str, candles: pd.DataFrame) -> None:
        """
        Set reference data for a symbol.
        
        This is called by the BacktestRunner before generate_signals().
        
        Args:
            symbol: Reference symbol (e.g., "BTCUSDT")
            candles: OHLCV DataFrame for the reference symbol
        """
        if not candles.empty:
            # Ensure index is datetime
            if not isinstance(candles.index, pd.DatetimeIndex):
                candles = candles.copy()
                candles.index = pd.to_datetime(candles.index)
            
            self._reference_data[symbol] = candles
            
            if symbol not in self._reference_symbols:
                self._reference_symbols.append(symbol)

    def get_reference_candles(
        self,
        symbol: str,
        up_to: pd.DatetimeIndex | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Get reference candles, optionally filtered to avoid lookahead.
        
        If up_to is provided, only returns data with timestamps <= up_to.
        This prevents lookahead bias in backtesting.
        
        Args:
            symbol: Reference symbol to get
            up_to: Maximum timestamp to include (prevents lookahead)
            
        Returns:
            DataFrame of reference candles
            
        Raises:
            KeyError: If reference symbol not found
        """
        if symbol not in self._reference_data:
            raise KeyError(
                f"Reference symbol '{symbol}' not found. "
                f"Available: {list(self._reference_data.keys())}"
            )
        
        ref_data = self._reference_data[symbol]
        
        if up_to is None:
            return ref_data
        
        # Handle both DatetimeIndex and single Timestamp
        if isinstance(up_to, pd.DatetimeIndex):
            # For index, use the max value
            max_ts = up_to.max()
        else:
            max_ts = up_to
        
        # Filter to avoid lookahead
        return ref_data[ref_data.index <= max_ts]

    def get_reference_value(
        self,
        symbol: str,
        column: str,
        at: pd.Timestamp,
        lookback: int = 0,
    ) -> pd.Series | float:
        """
        Get a specific value from reference data at a given timestamp.
        
        Args:
            symbol: Reference symbol
            column: Column name (e.g., "close", "high")
            at: Timestamp to look up
            lookback: Number of bars before 'at' to include (0 = just that bar)
            
        Returns:
            Single value if lookback=0, Series if lookback>0
        """
        ref_data = self.get_reference_candles(symbol, at)
        
        if ref_data.empty:
            return float("nan")
        
        if lookback == 0:
            # Get the most recent value at or before 'at'
            valid = ref_data[ref_data.index <= at]
            if valid.empty:
                return float("nan")
            return valid[column].iloc[-1]
        else:
            # Get lookback bars
            valid = ref_data[ref_data.index <= at]
            return valid[column].iloc[-lookback:] if len(valid) >= lookback else valid[column]

    def align_reference_to_target(
        self,
        symbol: str,
        target_index: pd.DatetimeIndex,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Align reference data to target symbol's index.
        
        Uses forward-fill to handle any misaligned timestamps.
        Only uses data available at each target timestamp (no lookahead).
        
        Args:
            symbol: Reference symbol
            target_index: Target symbol's DatetimeIndex
            columns: Columns to include (None = all)
            
        Returns:
            DataFrame aligned to target_index
        """
        ref_data = self.get_reference_candles(symbol)
        
        if columns:
            ref_data = ref_data[columns]
        
        # Reindex with forward fill (uses last known value)
        aligned = ref_data.reindex(target_index, method="ffill")
        
        return aligned

    def has_reference_data(self, symbol: str) -> bool:
        """Check if reference data is available for a symbol."""
        return symbol in self._reference_data and not self._reference_data[symbol].empty

    def clear_reference_data(self) -> None:
        """Clear all reference data."""
        self._reference_data.clear()
        self._reference_symbols.clear()

    def get_parameters(self) -> dict[str, Any]:
        """Get strategy parameters including reference symbols."""
        params = super().get_parameters()
        params["reference_symbols"] = self._reference_symbols
        return params
