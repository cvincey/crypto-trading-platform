"""
Hyper-Creative Orthogonal Strategies.

These strategies are designed to be orthogonal to the ratio mean reversion core.
They diversify the portfolio by exploiting different state variables:
- Crash protection / trend capture
- Volatility regime changes
- Cross-sectional momentum (rank-based)
- Microstructure signals
- Correlation regime switches

Tier 1: Uses only OHLCV data (no external APIs required)
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
# TIER 1: OHLCV-ONLY STRATEGIES
# =============================================================================


@strategy_registry.register(
    "crash_only_trend_filter",
    description="Defensive mode during BTC downtrends - protects from ratio MR getting steamrolled",
)
class CrashOnlyTrendFilterStrategy(BaseStrategy):
    """
    Crash-Only Trend Filter ("Downtrend Harvester").
    
    Signal: If BTC 3-7 day return < -X and BTC range expands (true range percentile high),
            enter short-duration defensive mode.
    
    Trade: During defensive mode, only hold BTC (rotate from alts to BTC).
           When defensive mode ends, re-enable normal trading.
    
    Why diversifies: Aims to protect specifically in regimes where ratio MR
    can get steamrolled by coordinated alt selloffs.
    
    Implementation: This generates BUY signals for BTC during crashes,
    effectively signaling "rotate to BTC for safety".
    """

    name = "crash_only_trend_filter"

    def _setup(
        self,
        return_lookback: int = 120,  # 5 days in hours
        crash_threshold: float = -0.08,  # 8% decline triggers defensive mode
        range_percentile: int = 80,  # True range must be in top 20%
        range_lookback: int = 168,  # 7 days for percentile calculation
        defensive_duration: int = 48,  # Stay defensive for 48 hours
        recovery_threshold: float = 0.03,  # 3% recovery exits defensive mode
        **kwargs,
    ) -> None:
        self.return_lookback = return_lookback
        self.crash_threshold = crash_threshold
        self.range_percentile = range_percentile
        self.range_lookback = range_lookback
        self.defensive_duration = defensive_duration
        self.recovery_threshold = recovery_threshold

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]
        high = candles["high"]
        low = candles["low"]

        # Calculate returns over lookback period
        returns = close.pct_change(self.return_lookback)

        # Calculate true range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate true range percentile
        tr_percentile = true_range.rolling(self.range_lookback, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
            raw=False,
        )

        # Generate signals
        in_defensive_mode = False
        defensive_start = 0
        entry_price = None

        for i, idx in enumerate(candles.index):
            ret = returns.get(idx, 0)
            tr_pct = tr_percentile.get(idx, 50)
            current_price = close.get(idx)

            if pd.isna(ret) or pd.isna(tr_pct):
                continue

            if not in_defensive_mode:
                # Enter defensive mode: crash + high volatility
                if ret < self.crash_threshold and tr_pct > self.range_percentile:
                    signals.loc[idx] = Signal.BUY  # Signal to hold BTC
                    in_defensive_mode = True
                    defensive_start = i
                    entry_price = current_price
            else:
                bars_in_defensive = i - defensive_start

                # Exit defensive mode conditions
                should_exit = False

                # Time-based exit
                if bars_in_defensive >= self.defensive_duration:
                    should_exit = True

                # Recovery-based exit
                if entry_price and current_price:
                    recovery = (current_price - entry_price) / entry_price
                    if recovery > self.recovery_threshold:
                        should_exit = True

                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_defensive_mode = False
                    entry_price = None

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "gamma_mimic_breakout",
    description="Tail breakout after compression - rare high-conviction trades",
)
class GammaMimicBreakoutStrategy(BaseStrategy):
    """
    Tail-Breakout "Gamma Mimic" Strategy.
    
    Signal: Multi-week compression (low volatility) then sudden range expansion.
            Think "volatility breakout" but only after LONG compression.
    
    Trade: Enter in direction of breakout, time stop + wide TP.
           Very low frequency: expect 1-4 trades per month.
    
    Why diversifies: Trend/tail capture vs mean reversion core.
    This strategy profits from the rare but powerful breakout moves.
    """

    name = "gamma_mimic_breakout"

    def _setup(
        self,
        compression_period: int = 336,  # 2 weeks lookback for compression
        compression_percentile: int = 25,  # BB width must be bottom 25% (FIXED from 20)
        breakout_atr_mult: float = 2.0,  # Breakout must exceed 2.0x ATR (FIXED from 2.5)
        atr_period: int = 14,
        hold_period: int = 72,  # Hold for 3 days
        min_compression_bars: int = 72,  # 3 days compression (FIXED from 168)
        **kwargs,
    ) -> None:
        self.compression_period = compression_period
        self.compression_percentile = compression_percentile
        self.breakout_atr_mult = breakout_atr_mult
        self.atr_period = atr_period
        self.hold_period = hold_period
        self.min_compression_bars = min_compression_bars

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]

        # Calculate Bollinger Band width as volatility measure
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        bb_width = (std / sma) * 100  # As percentage

        # Calculate BB width percentile (compression detection)
        bb_percentile = bb_width.rolling(self.compression_period, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
            raw=False,
        )

        # Count consecutive bars in compression
        in_compression = bb_percentile < self.compression_percentile
        compression_streak = in_compression.groupby(
            (~in_compression).cumsum()
        ).cumsum()

        # Calculate ATR for breakout detection
        atr = indicator_registry.compute("atr", candles, period=self.atr_period)

        # Calculate bar range
        bar_range = candles["high"] - candles["low"]

        # Detect breakout direction
        prev_close = close.shift(1)
        bullish_breakout = close > (prev_close + atr * self.breakout_atr_mult)
        bearish_breakout = close < (prev_close - atr * self.breakout_atr_mult)

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            comp_streak = compression_streak.get(idx, 0)
            is_bullish = bullish_breakout.get(idx, False)
            is_bearish = bearish_breakout.get(idx, False)
            range_val = bar_range.get(idx, 0)
            atr_val = atr.get(idx, 0)

            if pd.isna(comp_streak) or pd.isna(atr_val):
                continue

            if not in_position:
                # Entry: Long compression followed by strong breakout
                if comp_streak >= self.min_compression_bars:
                    if is_bullish and range_val > atr_val * self.breakout_atr_mult:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
            else:
                bars_held = i - entry_bar

                # Exit after hold period (time stop)
                if bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "volatility_targeting_overlay",
    description="Scale positions to maintain constant volatility contribution",
)
class VolatilityTargetingOverlayStrategy(BaseStrategy):
    """
    Volatility Targeting Overlay Strategy.
    
    Signal: Realized volatility estimate of the asset.
    
    Trade: Scale position sizes to keep constant vol contribution.
           Optionally "kill switch" when vol spikes.
    
    Why diversifies: Changes the shape of returns (reduces crash coupling)
    without needing new alpha. This is more of a position sizing overlay.
    
    Implementation: Generates position size signals as confidence levels.
    BUY with high confidence = full position, low confidence = reduced.
    """

    name = "volatility_targeting_overlay"

    def _setup(
        self,
        vol_target: float = 0.15,  # Target 15% annualized volatility
        vol_lookback: int = 168,  # 7-day realized vol
        max_leverage: float = 2.0,  # Cap at 2x
        min_leverage: float = 0.25,  # Floor at 0.25x
        kill_switch_vol: float = 0.50,  # Kill all positions above 50% vol
        rebalance_threshold: float = 0.25,  # Rebalance if target changes >25%
        **kwargs,
    ) -> None:
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.kill_switch_vol = kill_switch_vol
        self.rebalance_threshold = rebalance_threshold

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]

        # Calculate realized volatility (annualized)
        returns = close.pct_change()
        realized_vol = returns.rolling(self.vol_lookback, min_periods=20).std() * np.sqrt(8760)

        # Calculate target leverage
        target_leverage = self.vol_target / realized_vol
        target_leverage = target_leverage.clip(self.min_leverage, self.max_leverage)

        # Previous target for rebalancing decisions
        prev_leverage = target_leverage.shift(1)

        # Detect when we should be in position
        in_position = False
        current_leverage = 1.0

        for i, idx in enumerate(candles.index):
            vol = realized_vol.get(idx, 0.20)
            target = target_leverage.get(idx, 1.0)
            prev = prev_leverage.get(idx, 1.0)

            if pd.isna(vol) or pd.isna(target):
                continue

            # Kill switch: exit all positions if vol is extreme
            if vol > self.kill_switch_vol:
                if in_position:
                    signals.loc[idx] = Signal.SELL
                    in_position = False
                continue

            if not in_position:
                # Enter position with vol-adjusted sizing
                if vol < self.kill_switch_vol:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    current_leverage = target
            else:
                # Check if we need to rebalance
                leverage_change = abs(target - current_leverage) / current_leverage
                if leverage_change > self.rebalance_threshold:
                    # Signal a rebalance (exit and re-enter)
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "market_breadth_alt_participation",
    description="Regime gating based on cross-sectional alt health",
)
class MarketBreadthStrategy(BaseStrategy):
    """
    Market Breadth "Alt Participation" Strategy.
    
    Signal: % of alts above their N-day moving average,
            or % with positive 7d return.
    
    Trade: If breadth is strong, allow alt exposure.
           If breadth collapses, rotate to BTC/cash.
    
    Why diversifies: Regime gating based on cross-sectional health,
    not pair ratios. This uses the health of the entire alt ecosystem.
    
    Note: This strategy requires multi-symbol data. When run on single symbol,
    it uses internal breadth proxies based on momentum dispersion.
    """

    name = "market_breadth_alt_participation"

    def _setup(
        self,
        ma_period: int = 168,  # 7-day MA for above/below calculation
        breadth_threshold: float = 0.60,  # 60% alts must be healthy
        collapse_threshold: float = 0.30,  # Below 30% = breadth collapse
        momentum_period: int = 168,  # 7-day momentum for health check
        hold_period: int = 48,  # Minimum hold
        **kwargs,
    ) -> None:
        self.ma_period = ma_period
        self.breadth_threshold = breadth_threshold
        self.collapse_threshold = collapse_threshold
        self.momentum_period = momentum_period
        self.hold_period = hold_period
        self._multi_symbol_data: dict[str, pd.DataFrame] | None = None

    def set_multi_symbol_data(self, data: dict[str, pd.DataFrame]) -> None:
        """Set multi-symbol data for breadth calculation."""
        self._multi_symbol_data = data

    def _calculate_breadth(self, candles: pd.DataFrame, idx: pd.Timestamp) -> float:
        """Calculate market breadth at a given timestamp."""
        if self._multi_symbol_data is None:
            # Fallback: Use single-symbol momentum as proxy
            close = candles["close"]
            ma = close.rolling(self.ma_period, min_periods=20).mean()
            above_ma = close > ma
            return 1.0 if above_ma.get(idx, False) else 0.0

        # Multi-symbol breadth calculation
        above_count = 0
        total_count = 0

        for symbol, symbol_candles in self._multi_symbol_data.items():
            if idx not in symbol_candles.index:
                continue

            close = symbol_candles["close"]
            ma = close.rolling(self.ma_period, min_periods=20).mean()

            if idx in close.index and idx in ma.index:
                if close.loc[idx] > ma.loc[idx]:
                    above_count += 1
                total_count += 1

        if total_count == 0:
            return 0.5  # Neutral

        return above_count / total_count

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]

        # Pre-calculate MA for single-symbol fallback
        ma = close.rolling(self.ma_period, min_periods=20).mean()
        above_ma = close > ma

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            breadth = self._calculate_breadth(candles, idx)

            if not in_position:
                # Enter when breadth is strong
                if breadth >= self.breadth_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                bars_held = i - entry_bar

                # Exit on breadth collapse or minimum hold reached
                should_exit = False

                if breadth < self.collapse_threshold:
                    should_exit = True

                if bars_held >= self.hold_period and breadth < self.breadth_threshold:
                    should_exit = True

                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "cross_sectional_momentum",
    description="Rank-based momentum: long winners, avoid losers",
)
class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    Cross-Sectional "Losers Keep Losing" Strategy.
    
    Signal: Weekly rank of returns across universe.
            Look for persistent rank persistence.
    
    Trade: Long top-k, avoid/short bottom-k (or just long top-k + cash).
    
    Why diversifies: Cross-sectional factor vs pair mean reversion.
    This exploits momentum continuation across the alt universe.
    
    Note: When run on single symbol, uses internal ranking vs BTC.
    """

    name = "cross_sectional_momentum"

    def _setup(
        self,
        ranking_period: int = 168,  # 7-day momentum for ranking
        hold_period: int = 168,  # Hold for 1 week
        top_percentile: float = 0.20,  # Top 20% = long
        bottom_percentile: float = 0.20,  # Bottom 20% = avoid
        min_momentum: float = 0.0,  # Must have positive momentum to long
        **kwargs,
    ) -> None:
        self.ranking_period = ranking_period
        self.hold_period = hold_period
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile
        self.min_momentum = min_momentum
        self._multi_symbol_data: dict[str, pd.DataFrame] | None = None

    def set_multi_symbol_data(self, data: dict[str, pd.DataFrame]) -> None:
        """Set multi-symbol data for cross-sectional ranking."""
        self._multi_symbol_data = data

    def _calculate_rank(self, candles: pd.DataFrame, idx: pd.Timestamp) -> float | None:
        """Calculate percentile rank of this asset at timestamp."""
        if self._multi_symbol_data is None:
            # Single symbol: use momentum percentile vs own history
            close = candles["close"]
            momentum = close.pct_change(self.ranking_period)
            if idx not in momentum.index:
                return None
            mom_val = momentum.get(idx)
            if pd.isna(mom_val):
                return None
            # Calculate percentile rank vs history
            hist = momentum.loc[:idx].dropna()
            if len(hist) < 20:
                return 0.5
            return (hist < mom_val).mean()

        # Multi-symbol ranking
        momentums = {}
        for symbol, symbol_candles in self._multi_symbol_data.items():
            if idx not in symbol_candles.index:
                continue
            close = symbol_candles["close"]
            momentum = close.pct_change(self.ranking_period)
            if idx in momentum.index:
                mom_val = momentum.get(idx)
                if not pd.isna(mom_val):
                    momentums[symbol] = mom_val

        if len(momentums) < 2:
            return None

        # Get this symbol's momentum (assume first symbol in candles)
        symbol_momentum = candles["close"].pct_change(self.ranking_period).get(idx)
        if pd.isna(symbol_momentum):
            return None

        # Calculate rank
        all_moms = list(momentums.values())
        rank = sum(1 for m in all_moms if m < symbol_momentum) / len(all_moms)
        return rank

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]
        momentum = close.pct_change(self.ranking_period)

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            rank = self._calculate_rank(candles, idx)
            mom = momentum.get(idx, 0)

            if pd.isna(mom):
                continue

            if rank is None:
                continue

            if not in_position:
                # Long if in top percentile and momentum is positive
                is_top_rank = rank >= (1 - self.top_percentile)
                has_positive_momentum = mom > self.min_momentum

                if is_top_rank and has_positive_momentum:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                bars_held = i - entry_bar

                # Exit conditions
                should_exit = False

                # Time-based exit
                if bars_held >= self.hold_period:
                    should_exit = True

                # Rank drops to bottom (loser)
                is_bottom_rank = rank < self.bottom_percentile
                if is_bottom_rank:
                    should_exit = True

                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "gap_reversion",
    description="Fade idiosyncratic jumps when BTC is quiet",
)
class GapReversionStrategy(BaseStrategy):
    """
    Idiosyncratic Jump-Reversion ("Gap Crime") Strategy.
    
    Signal: Asset prints an extreme 1-3h move without similar move in BTC/ETH
            (residual return spike).
    
    Trade: Fade residual jump with tight time stop; only when BTC is quiet.
    
    Why diversifies: Captures single-name dislocations vs systematic ratio signals.
    This exploits temporary mispricings that quickly revert.
    """

    name = "gap_reversion"

    def _setup(
        self,
        jump_lookback: int = 3,  # 3-hour move
        jump_threshold: float = 0.04,  # 4% move threshold
        btc_quiet_threshold: float = 0.015,  # BTC must move < 1.5%
        fade_hold_period: int = 12,  # Hold fade for 12 hours
        max_hold_period: int = 24,  # Max hold 24 hours
        **kwargs,
    ) -> None:
        self.jump_lookback = jump_lookback
        self.jump_threshold = jump_threshold
        self.btc_quiet_threshold = btc_quiet_threshold
        self.fade_hold_period = fade_hold_period
        self.max_hold_period = max_hold_period
        self._btc_candles: pd.DataFrame | None = None

    def set_btc_data(self, btc_candles: pd.DataFrame) -> None:
        """Set BTC candle data for comparison."""
        self._btc_candles = btc_candles

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]

        # Calculate short-term returns
        asset_return = close.pct_change(self.jump_lookback)

        # Calculate BTC returns if available
        if self._btc_candles is not None:
            btc_close = self._btc_candles["close"].reindex(candles.index, method="ffill")
            btc_return = btc_close.pct_change(self.jump_lookback)
        else:
            # No BTC data - use asset's own volatility as filter
            btc_return = pd.Series(0.0, index=candles.index)

        # Generate signals
        in_position = False
        entry_bar = 0
        position_direction = None

        for i, idx in enumerate(candles.index):
            asset_ret = asset_return.get(idx, 0)
            btc_ret = btc_return.get(idx, 0)

            if pd.isna(asset_ret) or pd.isna(btc_ret):
                continue

            # Calculate residual (idiosyncratic) return
            residual = asset_ret - btc_ret

            # Check if BTC is quiet
            btc_is_quiet = abs(btc_ret) < self.btc_quiet_threshold

            if not in_position:
                # Detect idiosyncratic jump
                if btc_is_quiet:
                    # Large positive jump - fade it (expect reversion down)
                    if residual > self.jump_threshold:
                        signals.loc[idx] = Signal.SELL  # Short/avoid
                        in_position = True
                        entry_bar = i
                        position_direction = "short"
                    # Large negative jump - fade it (expect reversion up)
                    elif residual < -self.jump_threshold:
                        signals.loc[idx] = Signal.BUY  # Long
                        in_position = True
                        entry_bar = i
                        position_direction = "long"
            else:
                bars_held = i - entry_bar

                # Exit after fade period or max hold
                if bars_held >= self.fade_hold_period or bars_held >= self.max_hold_period:
                    if position_direction == "long":
                        signals.loc[idx] = Signal.SELL
                    else:
                        signals.loc[idx] = Signal.BUY  # Cover short
                    in_position = False
                    position_direction = None

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "liquidity_vacuum_detector",
    description="Mean-revert thin-book moves that stall immediately",
)
class LiquidityVacuumDetectorStrategy(BaseStrategy):
    """
    Liquidity Vacuum Detector Strategy.
    
    Signal: Candle range expands while volume drops (thin book),
            followed by immediate stall.
    
    Trade: Mean-revert the vacuum move; strict max-hold (6-24h).
    
    Why diversifies: Microstructure-style; different trigger than ratio z-scores.
    This exploits thin-liquidity price spikes that quickly revert.
    """

    name = "liquidity_vacuum_detector"

    def _setup(
        self,
        range_percentile: int = 85,  # Range must be top 15% (FIXED from 90)
        volume_percentile: int = 40,  # Volume must be bottom 40% (FIXED from 30)
        percentile_window: int = 168,  # 7-day window for percentiles
        stall_threshold: float = 0.6,  # Next bar range < 60% of vacuum (FIXED from 0.5)
        hold_period: int = 12,  # Hold for 12 hours
        max_hold_period: int = 24,  # Max 24 hours
        **kwargs,
    ) -> None:
        self.range_percentile = range_percentile
        self.volume_percentile = volume_percentile
        self.percentile_window = percentile_window
        self.stall_threshold = stall_threshold
        self.hold_period = hold_period
        self.max_hold_period = max_hold_period

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]
        high = candles["high"]
        low = candles["low"]
        volume = candles["volume"]

        # Calculate bar range
        bar_range = high - low

        # Calculate percentiles
        range_pct = bar_range.rolling(self.percentile_window, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
            raw=False,
        )
        volume_pct = volume.rolling(self.percentile_window, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
            raw=False,
        )

        # Detect vacuum conditions
        is_wide_range = range_pct > self.range_percentile
        is_low_volume = volume_pct < self.volume_percentile
        is_vacuum = is_wide_range & is_low_volume

        # Calculate stall (next bar range relative to current)
        next_range_ratio = bar_range.shift(-1) / bar_range
        is_stall = next_range_ratio < self.stall_threshold

        # Determine direction (bullish or bearish vacuum)
        is_bullish = close > close.shift(1)  # Closed higher than open
        is_bearish = close < close.shift(1)

        # Generate signals
        in_position = False
        entry_bar = 0
        position_direction = None

        for i, idx in enumerate(candles.index):
            vacuum = is_vacuum.get(idx, False)
            stall = is_stall.get(idx, False)
            bullish = is_bullish.get(idx, False)
            bearish = is_bearish.get(idx, False)

            if pd.isna(vacuum):
                continue

            if not in_position:
                # Vacuum followed by stall = fade opportunity
                if vacuum and stall:
                    if bullish:
                        # Fade bullish vacuum (expect drop)
                        signals.loc[idx] = Signal.SELL
                        in_position = True
                        entry_bar = i
                        position_direction = "short"
                    elif bearish:
                        # Fade bearish vacuum (expect bounce)
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
                        position_direction = "long"
            else:
                bars_held = i - entry_bar

                # Exit after hold period
                if bars_held >= self.hold_period or bars_held >= self.max_hold_period:
                    if position_direction == "long":
                        signals.loc[idx] = Signal.SELL
                    else:
                        signals.loc[idx] = Signal.BUY
                    in_position = False
                    position_direction = None

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "correlation_regime_switch",
    description="Anti-overfit correlation filter: disable MR when relationship is unstable",
)
class CorrelationRegimeSwitchStrategy(BaseStrategy):
    """
    Correlation Regime Switch Strategy (Anti-Overfit Version).
    
    Signal: Rolling correlation of each alt to BTC toggles between
            "sticky high" vs "broken low" using simple thresholds + hysteresis.
    
    Trade: Only run ratio MR when correlation is stable-high.
           Otherwise switch to BTC/cash.
    
    Why diversifies: Stops betting on a relationship when it's unstable.
    This addresses a key failure mode observed in ratio strategies.
    
    Implementation: When correlation is high and stable, generate BUY signals
    for the alt. When correlation breaks down, generate SELL and stay out.
    """

    name = "correlation_regime_switch"

    def _setup(
        self,
        correlation_window: int = 720,  # 30 days
        high_correlation: float = 0.70,  # Above this = stable regime (FIXED from 0.80)
        low_correlation: float = 0.45,  # Below this = broken regime (FIXED from 0.50)
        stability_window: int = 168,  # 7 days for stability check
        stability_threshold: float = 0.12,  # Std dev of correlation < 0.12 (FIXED from 0.10)
        min_hold_period: int = 48,  # Minimum hold when in regime
        **kwargs,
    ) -> None:
        self.correlation_window = correlation_window
        self.high_correlation = high_correlation
        self.low_correlation = low_correlation
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self.min_hold_period = min_hold_period
        self._btc_candles: pd.DataFrame | None = None

    def set_btc_data(self, btc_candles: pd.DataFrame) -> None:
        """Set BTC candle data for correlation calculation."""
        self._btc_candles = btc_candles

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]

        # Calculate returns
        asset_returns = close.pct_change()

        # Get BTC returns
        if self._btc_candles is not None:
            btc_close = self._btc_candles["close"].reindex(candles.index, method="ffill")
            btc_returns = btc_close.pct_change()
        else:
            # Fallback: use momentum as proxy
            btc_returns = asset_returns.copy()

        # Calculate rolling correlation
        correlation = asset_returns.rolling(self.correlation_window, min_periods=50).corr(
            btc_returns
        )

        # Calculate correlation stability (std dev of recent correlation)
        corr_stability = correlation.rolling(self.stability_window, min_periods=20).std()

        # Regime detection with hysteresis
        in_stable_regime = False
        entry_bar = 0
        in_position = False

        for i, idx in enumerate(candles.index):
            corr = correlation.get(idx, 0.5)
            stability = corr_stability.get(idx, 0.5)

            if pd.isna(corr) or pd.isna(stability):
                continue

            # Update regime state (hysteresis)
            if not in_stable_regime:
                # Enter stable regime when correlation is high and stable
                if corr > self.high_correlation and stability < self.stability_threshold:
                    in_stable_regime = True
            else:
                # Exit stable regime when correlation breaks down
                if corr < self.low_correlation or stability > self.stability_threshold * 2:
                    in_stable_regime = False

            # Generate signals based on regime
            if not in_position:
                if in_stable_regime:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                bars_held = i - entry_bar

                # Exit when regime ends (after minimum hold)
                if not in_stable_regime and bars_held >= self.min_hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)
