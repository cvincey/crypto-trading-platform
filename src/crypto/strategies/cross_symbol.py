"""
Cross-symbol strategies that use signals from other symbols.

These strategies exploit the finding that cross-asset patterns generalize well
in crypto markets. BTC and ETH typically lead price movements, with alts following.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from crypto.core.types import Signal
from crypto.indicators.base import indicator_registry
from crypto.strategies.cross_symbol_base import CrossSymbolBaseStrategy
from crypto.strategies.registry import strategy_registry

logger = logging.getLogger(__name__)


@strategy_registry.register(
    "btc_lead_alt_follow",
    description="BTC breakout triggers alt entry with delay",
)
class BTCLeadAltFollowStrategy(CrossSymbolBaseStrategy):
    """
    BTC Lead Alt Follow Strategy.
    
    When BTC breaks above its N-bar high, buy the target alt.
    Hold for a fixed period or until stop/take profit.
    
    Key insight: Information propagates from BTC to alts with a delay.
    Alts are less liquid and slower to react.
    """

    name = "btc_lead_alt_follow"

    def _setup(
        self,
        leader_symbol: str = "BTCUSDT",
        breakout_period: int = 24,
        entry_delay: int = 1,
        hold_period: int = 48,
        use_trailing_exit: bool = False,
        **kwargs,
    ) -> None:
        self.leader_symbol = leader_symbol
        self.breakout_period = breakout_period
        self.entry_delay = entry_delay
        self.hold_period = hold_period
        self.use_trailing_exit = use_trailing_exit

    def get_reference_symbols(self) -> list[str]:
        return [self.leader_symbol]

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        if not self.has_reference_data(self.leader_symbol):
            logger.warning(f"No reference data for {self.leader_symbol}")
            return signals

        # Get BTC data aligned to target candles
        btc = self.align_reference_to_target(
            self.leader_symbol, candles.index, ["high", "close"]
        )

        if btc.empty:
            return signals

        # Calculate BTC breakout: close above N-bar high
        btc_rolling_high = btc["high"].rolling(self.breakout_period).max().shift(1)
        btc_breakout = btc["close"] > btc_rolling_high

        # Apply entry delay
        btc_breakout_delayed = btc_breakout.shift(self.entry_delay).fillna(False)

        # Track position for hold period exit
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            if not in_position:
                if btc_breakout_delayed.get(idx, False):
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                # Exit after hold period
                if i - entry_bar >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "eth_btc_ratio_reversion",
    description="Trade ETH when ETH/BTC ratio is extreme",
)
class ETHBTCRatioReversionStrategy(CrossSymbolBaseStrategy):
    """
    ETH/BTC Ratio Reversion Strategy.
    
    When ETH underperforms BTC significantly (z-score < -2),
    go long ETH expecting mean reversion.
    
    The ETH/BTC ratio tends to mean-revert over medium timeframes.
    """

    name = "eth_btc_ratio_reversion"

    def _setup(
        self,
        lookback: int = 168,
        entry_threshold: float = -2.0,
        exit_threshold: float = -0.5,
        max_hold_hours: int = 72,
        **kwargs,
    ) -> None:
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_hold_hours = max_hold_hours

    def get_reference_symbols(self) -> list[str]:
        return ["BTCUSDT"]

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        if not self.has_reference_data("BTCUSDT"):
            logger.warning("No reference data for BTCUSDT")
            return signals

        # Get BTC close aligned to ETH candles
        btc = self.align_reference_to_target("BTCUSDT", candles.index, ["close"])

        if btc.empty:
            return signals

        # Calculate ETH/BTC ratio
        ratio = candles["close"] / btc["close"]

        # Calculate z-score of ratio
        ratio_mean = ratio.rolling(self.lookback).mean()
        ratio_std = ratio.rolling(self.lookback).std()
        z_score = (ratio - ratio_mean) / ratio_std

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            z = z_score.get(idx, 0)
            
            if pd.isna(z):
                continue

            if not in_position:
                # Enter when ETH is very oversold vs BTC
                if z < self.entry_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                # Exit when ratio normalizes or max hold reached
                if z > self.exit_threshold or (i - entry_bar >= self.max_hold_hours):
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "correlation_breakdown",
    description="Trade laggard when correlated assets diverge",
)
class CorrelationBreakdownStrategy(CrossSymbolBaseStrategy):
    """
    Correlation Breakdown Strategy.
    
    When two highly correlated assets diverge, trade the laggard
    expecting convergence.
    
    Temporary divergences in correlated assets tend to revert.
    """

    name = "correlation_breakdown"

    def _setup(
        self,
        reference_symbol: str = "BTCUSDT",
        correlation_window: int = 720,
        min_correlation: float = 0.85,
        divergence_threshold: float = 0.03,
        max_hold_hours: int = 48,
        **kwargs,
    ) -> None:
        self.reference_symbol = reference_symbol
        self.correlation_window = correlation_window
        self.min_correlation = min_correlation
        self.divergence_threshold = divergence_threshold
        self.max_hold_hours = max_hold_hours

    def get_reference_symbols(self) -> list[str]:
        return [self.reference_symbol]

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        if not self.has_reference_data(self.reference_symbol):
            logger.warning(f"No reference data for {self.reference_symbol}")
            return signals

        # Get reference close aligned to target
        ref = self.align_reference_to_target(
            self.reference_symbol, candles.index, ["close"]
        )

        if ref.empty:
            return signals

        # Calculate returns
        target_returns = candles["close"].pct_change()
        ref_returns = ref["close"].pct_change()

        # Calculate rolling correlation
        rolling_corr = target_returns.rolling(self.correlation_window).corr(ref_returns)

        # Calculate 24h returns for divergence detection
        target_24h = candles["close"].pct_change(24)
        ref_24h = ref["close"].pct_change(24)
        divergence = target_24h - ref_24h

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            corr = rolling_corr.get(idx, 0)
            div = divergence.get(idx, 0)

            if pd.isna(corr) or pd.isna(div):
                continue

            if not in_position:
                # Enter when highly correlated but target is lagging
                if corr > self.min_correlation and div < -self.divergence_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                # Exit when divergence closes or max hold
                if div > -self.divergence_threshold / 2 or (i - entry_bar >= self.max_hold_hours):
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "sector_momentum_rotation",
    description="Weekly rebalance to top performing sector",
)
class SectorMomentumRotationStrategy(CrossSymbolBaseStrategy):
    """
    Sector Momentum Rotation Strategy.
    
    Divide crypto into sectors (L1, DeFi, Major).
    Every week, go long the sector with highest momentum.
    
    Crypto capital rotates between narratives/sectors.
    """

    name = "sector_momentum_rotation"

    def _setup(
        self,
        sectors: dict[str, list[str]] | None = None,
        momentum_period: int = 168,
        rebalance_period: int = 168,
        top_n_sectors: int = 1,
        **kwargs,
    ) -> None:
        self.sectors = sectors or {
            "L1": ["SOLUSDT", "AVAXUSDT", "NEARUSDT", "APTUSDT"],
            "DeFi": ["UNIUSDT", "AAVEUSDT", "LINKUSDT"],
            "Major": ["BTCUSDT", "ETHUSDT"],
        }
        self.momentum_period = momentum_period
        self.rebalance_period = rebalance_period
        self.top_n_sectors = top_n_sectors

    def get_reference_symbols(self) -> list[str]:
        # Need all sector symbols as reference
        all_symbols = []
        for symbols in self.sectors.values():
            all_symbols.extend(symbols)
        return list(set(all_symbols))

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Determine which sector the target symbol belongs to
        target_sector = None
        for sector_name, symbols in self.sectors.items():
            # We don't know the target symbol here, so we'll use the candle data
            # to determine momentum-based entry
            pass

        # Calculate sector momentum using available reference data
        sector_momentum = {}

        for sector_name, symbols in self.sectors.items():
            momentum_values = []
            for symbol in symbols:
                if self.has_reference_data(symbol):
                    ref = self.align_reference_to_target(symbol, candles.index, ["close"])
                    if not ref.empty:
                        mom = ref["close"].pct_change(self.momentum_period)
                        momentum_values.append(mom)

            if momentum_values:
                # Average momentum across sector
                sector_mom = pd.concat(momentum_values, axis=1).mean(axis=1)
                sector_momentum[sector_name] = sector_mom

        if not sector_momentum:
            return signals

        # Combine sector momentums into DataFrame
        sector_df = pd.DataFrame(sector_momentum)

        # Generate signals based on rebalancing
        in_position = False
        last_rebalance = 0

        for i, idx in enumerate(candles.index):
            # Rebalance check
            if i - last_rebalance >= self.rebalance_period or i == 0:
                last_rebalance = i

                # Get current sector momentums
                current_momentum = sector_df.loc[idx] if idx in sector_df.index else None

                if current_momentum is not None and not current_momentum.isna().all():
                    top_sector = current_momentum.idxmax()

                    # Check if target is in top sector
                    # For now, always signal BUY at rebalance if momentum is positive
                    if current_momentum[top_sector] > 0:
                        if not in_position:
                            signals.loc[idx] = Signal.BUY
                            in_position = True
                    else:
                        if in_position:
                            signals.loc[idx] = Signal.SELL
                            in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "btc_volatility_filter",
    description="Filter signals when BTC volatility is high",
)
class BTCVolatilityFilterStrategy(CrossSymbolBaseStrategy):
    """
    BTC Volatility Filter Strategy.
    
    Only allow trading when BTC volatility is below a threshold.
    High BTC volatility = uncertain market = avoid trading alts.
    
    This is a meta-strategy that filters another strategy's signals.
    """

    name = "btc_volatility_filter"

    def _setup(
        self,
        base_strategy: str = "momentum_breakout",
        btc_atr_period: int = 24,
        btc_atr_percentile_window: int = 168,
        max_volatility_percentile: float = 70,
        **kwargs,
    ) -> None:
        self.base_strategy_name = base_strategy
        self.btc_atr_period = btc_atr_period
        self.btc_atr_percentile_window = btc_atr_percentile_window
        self.max_volatility_percentile = max_volatility_percentile
        self._base_strategy = None

    def get_reference_symbols(self) -> list[str]:
        return ["BTCUSDT"]

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

        if not self.has_reference_data("BTCUSDT"):
            logger.warning("No BTC reference data, returning unfiltered signals")
            return base_signals

        # Get BTC data for volatility calculation
        btc = self.align_reference_to_target("BTCUSDT", candles.index)

        if btc.empty:
            return base_signals

        # Calculate BTC ATR
        btc_atr = indicator_registry.compute("atr", btc, period=self.btc_atr_period)

        # Calculate ATR percentile
        def rolling_percentile(x):
            if len(x) < 2:
                return 50
            return (x.rank().iloc[-1] / len(x)) * 100

        atr_percentile = btc_atr.rolling(self.btc_atr_percentile_window).apply(
            rolling_percentile, raw=False
        )

        # Filter signals when volatility is high
        signals = base_signals.copy()
        high_vol = atr_percentile > self.max_volatility_percentile

        # Set BUY signals to HOLD when volatility is high
        buy_mask = signals == Signal.BUY
        signals.loc[buy_mask & high_vol] = Signal.HOLD

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "ratio_reversion",
    description="Generic X/BTC ratio reversion strategy",
)
class RatioReversionStrategy(CrossSymbolBaseStrategy):
    """
    Generic Ratio Reversion Strategy.
    
    Trades mean reversion on the ratio between any asset and a reference asset
    (typically BTC). When the target underperforms the reference significantly
    (z-score below threshold), go long expecting convergence.
    
    This is a generalized version of ETHBTCRatioReversionStrategy that works
    for any trading pair.
    
    Use cases:
    - SOL/BTC ratio reversion
    - BNB/BTC ratio reversion  
    - LINK/BTC ratio reversion
    - Any alt vs BTC or ETH
    """

    name = "ratio_reversion"

    def _setup(
        self,
        reference_symbol: str = "BTCUSDT",
        target_symbol: str = "SOLUSDT",
        lookback: int = 168,
        entry_threshold: float = -1.5,
        exit_threshold: float = -0.7,
        max_hold_hours: int = 72,
        **kwargs,
    ) -> None:
        """
        Initialize ratio reversion strategy.
        
        Args:
            reference_symbol: Symbol to use as reference (e.g., BTCUSDT)
            target_symbol: Symbol being traded (e.g., SOLUSDT)
            lookback: Bars for rolling z-score calculation
            entry_threshold: Z-score below which to enter long
            exit_threshold: Z-score above which to exit
            max_hold_hours: Maximum bars to hold position
        """
        self.reference_symbol = reference_symbol
        self.target_symbol = target_symbol
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_hold_hours = max_hold_hours

    def get_reference_symbols(self) -> list[str]:
        """Return the reference symbol for ratio calculation."""
        return [self.reference_symbol]

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on ratio z-score.
        
        Logic:
        1. Calculate target/reference price ratio
        2. Compute rolling z-score of ratio
        3. Enter long when z-score < entry_threshold
        4. Exit when z-score > exit_threshold or max hold reached
        """
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        if not self.has_reference_data(self.reference_symbol):
            logger.warning(f"No reference data for {self.reference_symbol}")
            return signals

        # Get reference close aligned to target candles
        ref = self.align_reference_to_target(
            self.reference_symbol, candles.index, ["close"]
        )

        if ref.empty or ref["close"].isna().all():
            logger.warning(f"Reference data for {self.reference_symbol} is empty")
            return signals

        # Calculate ratio
        ratio = candles["close"].astype(float) / ref["close"].astype(float)

        # Calculate rolling z-score
        ratio_mean = ratio.rolling(
            self.lookback, min_periods=self.lookback // 2
        ).mean()
        ratio_std = ratio.rolling(
            self.lookback, min_periods=self.lookback // 2
        ).std()
        z_score = (ratio - ratio_mean) / ratio_std

        # Generate signals with state tracking
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            z = z_score.get(idx, float("nan"))

            if pd.isna(z):
                continue

            if not in_position:
                # Enter when target severely underperforms reference
                if z < self.entry_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                # Exit when ratio normalizes or max hold reached
                bars_held = i - entry_bar
                if z > self.exit_threshold or bars_held >= self.max_hold_hours:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)
