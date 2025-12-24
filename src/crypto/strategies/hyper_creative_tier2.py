"""
Hyper-Creative Orthogonal Strategies - Tier 2.

These strategies require external data sources:
- Funding rates (existing in DB)
- BTC dominance (CoinGecko / DB)
- Stablecoin supply (CoinGecko)
- Liquidations (mock data for backtest)

Tier 2: Alternative data required
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
# TIER 2: ALTERNATIVE DATA STRATEGIES
# =============================================================================


@strategy_registry.register(
    "funding_term_structure",
    description="Trade funding dispersion across assets - carry curve strategy",
)
class FundingTermStructureStrategy(BaseStrategy):
    """
    Funding Term Structure ("Carry Curve") Strategy.
    
    Signal: Funding level + persistence + cross-asset dispersion
            (e.g., SOL funding extreme vs others).
    
    Trade: Go contrarian on the most crowded names; or "pair" crowded long vs uncrowded.
    
    Why diversifies: Positioning mean reversion vs price/ratio mean reversion.
    This exploits crowded positioning rather than price patterns.
    """

    name = "funding_term_structure"

    def _setup(
        self,
        lookback_periods: int = 3,  # 3 funding periods (24h)
        dispersion_threshold: float = 0.0003,  # 0.03% dispersion triggers signal
        extreme_funding: float = 0.0005,  # 0.05% = crowded
        hold_period: int = 24,
        use_contrarian: bool = True,  # Fade crowded positions
        **kwargs,
    ) -> None:
        self.lookback_periods = lookback_periods
        self.dispersion_threshold = dispersion_threshold
        self.extreme_funding = extreme_funding
        self.hold_period = hold_period
        self.use_contrarian = use_contrarian
        self._funding_data: pd.Series | None = None
        self._multi_funding: dict[str, pd.Series] | None = None

    def set_funding_data(self, funding_series: pd.Series) -> None:
        """Set funding rate data aligned to candle index."""
        self._funding_data = funding_series

    def set_multi_funding_data(self, funding_dict: dict[str, pd.Series]) -> None:
        """Set funding data for multiple symbols for dispersion calculation."""
        self._multi_funding = funding_dict

    def _calculate_funding_dispersion(self, idx: pd.Timestamp) -> float:
        """Calculate cross-asset funding dispersion at timestamp."""
        if self._multi_funding is None:
            return 0.0

        funding_values = []
        for symbol, funding in self._multi_funding.items():
            if idx in funding.index:
                val = funding.get(idx)
                if not pd.isna(val):
                    funding_values.append(val)

        if len(funding_values) < 2:
            return 0.0

        return np.std(funding_values)

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        # Get funding data
        if self._funding_data is not None:
            funding_avg = self._funding_data.rolling(
                self.lookback_periods, min_periods=1
            ).mean()
        else:
            # Estimate from price action
            close = candles["close"]
            short_ma = close.rolling(24).mean()
            long_ma = close.rolling(168).mean()
            deviation = (close - long_ma) / long_ma
            funding_avg = deviation * 0.01  # Scale to funding range

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            funding = funding_avg.get(idx, 0)
            dispersion = self._calculate_funding_dispersion(idx)

            if pd.isna(funding):
                continue

            if not in_position:
                # Contrarian entry: fade extreme funding
                if self.use_contrarian:
                    # Extreme positive funding = crowded longs = expect pullback
                    # But for spot-only, we can't short, so avoid longs
                    # Extreme negative funding = crowded shorts = expect squeeze = BUY
                    if funding < -self.extreme_funding:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
                else:
                    # Non-contrarian: follow funding (carry trade)
                    if funding > self.extreme_funding:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
            else:
                bars_held = i - entry_bar

                # Exit after hold period or funding normalizes
                should_exit = False

                if bars_held >= self.hold_period:
                    should_exit = True

                # Exit if funding reverses direction
                if self.use_contrarian:
                    if funding > self.extreme_funding / 2:
                        should_exit = True
                else:
                    if funding < -self.extreme_funding / 2:
                        should_exit = True

                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "funding_vol_interaction",
    description="Crowding + fragility: reduce exposure when funding extreme + vol rising",
)
class FundingVolInteractionStrategy(BaseStrategy):
    """
    Funding-Vol Interaction ("Crowding + Fragility") Strategy.
    
    Signal: Extreme funding AND rising realized volatility = crowded + unstable.
    
    Trade: Reduce exposure or take contrarian trades only when fragility flag is on.
    
    Why diversifies: Uses second-order condition (crowding becomes actionable when
    instability appears). This is about timing the unwind of crowded trades.
    """

    name = "funding_vol_interaction"

    def _setup(
        self,
        funding_lookback: int = 3,
        extreme_funding: float = 0.0005,  # 0.05%
        vol_lookback: int = 168,  # 7-day realized vol
        vol_spike_threshold: float = 0.30,  # 30% annualized = spike
        fragility_mode: str = "reduce",  # "reduce" or "fade"
        hold_period: int = 24,
        **kwargs,
    ) -> None:
        self.funding_lookback = funding_lookback
        self.extreme_funding = extreme_funding
        self.vol_lookback = vol_lookback
        self.vol_spike_threshold = vol_spike_threshold
        self.fragility_mode = fragility_mode
        self.hold_period = hold_period
        self._funding_data: pd.Series | None = None

    def set_funding_data(self, funding_series: pd.Series) -> None:
        """Set funding rate data aligned to candle index."""
        self._funding_data = funding_series

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]

        # Calculate realized volatility
        returns = close.pct_change()
        realized_vol = returns.rolling(self.vol_lookback, min_periods=20).std() * np.sqrt(8760)

        # Vol trend (is vol rising?)
        vol_change = realized_vol.pct_change(24)  # 24-hour vol change

        # Get funding data
        if self._funding_data is not None:
            funding_avg = self._funding_data.rolling(
                self.funding_lookback, min_periods=1
            ).mean()
        else:
            # Estimate from price momentum
            momentum = close.pct_change(168)
            funding_avg = momentum * 0.01

        # Detect fragility: extreme funding + rising vol
        is_fragile_long = (funding_avg > self.extreme_funding) & (realized_vol > self.vol_spike_threshold) & (vol_change > 0)
        is_fragile_short = (funding_avg < -self.extreme_funding) & (realized_vol > self.vol_spike_threshold) & (vol_change > 0)

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            frag_long = is_fragile_long.get(idx, False)
            frag_short = is_fragile_short.get(idx, False)

            if pd.isna(frag_long):
                continue

            if not in_position:
                if self.fragility_mode == "fade":
                    # Fade fragile positions
                    if frag_short:  # Crowded shorts + fragile = expect squeeze
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
                else:
                    # Reduce mode: only enter when NOT fragile
                    if not frag_long and not frag_short:
                        # Normal conditions - enter on momentum
                        momentum = close.pct_change(24).get(idx, 0)
                        if momentum > 0.02:  # 2% 24h move
                            signals.loc[idx] = Signal.BUY
                            in_position = True
                            entry_bar = i
            else:
                bars_held = i - entry_bar

                # Exit on fragility or hold period
                should_exit = False

                if bars_held >= self.hold_period:
                    should_exit = True

                # Exit if fragility appears (for non-fade mode)
                if self.fragility_mode == "reduce" and frag_long:
                    should_exit = True

                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "liquidation_cluster_fade",
    description="Fade large liquidation cascades after delay",
)
class LiquidationClusterFadeStrategy(BaseStrategy):
    """
    Liquidation Cluster Fade Strategy.
    
    Signal: "Liquidation storm" in 1-3h that is extreme relative to its own history.
    
    Trade: Fade the move after a short delay; scale with event magnitude; strict time stop.
    
    Why diversifies: Event-driven, non-linear payoff.
    
    Note: Uses mock liquidation data for backtesting.
    """

    name = "liquidation_cluster_fade"

    def _setup(
        self,
        liquidation_threshold: float = 5_000_000,  # $5M triggers signal
        lookback_hours: int = 1,
        fade_delay: int = 2,  # Wait 2 bars after cascade
        hold_period: int = 24,
        min_cascade_size: float = 3_000_000,  # $3M minimum
        **kwargs,
    ) -> None:
        self.liquidation_threshold = liquidation_threshold
        self.lookback_hours = lookback_hours
        self.fade_delay = fade_delay
        self.hold_period = hold_period
        self.min_cascade_size = min_cascade_size
        self._liquidation_data: pd.DataFrame | None = None

    def set_liquidation_data(self, liq_data: pd.DataFrame) -> None:
        """Set liquidation data with long_liquidations and short_liquidations columns."""
        self._liquidation_data = liq_data

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        if self._liquidation_data is None:
            logger.warning("No liquidation data available for liquidation_cluster_fade")
            return signals

        close = candles["close"]

        # Align liquidation data to candles
        long_liqs = self._liquidation_data["long_liquidations"].reindex(
            candles.index, fill_value=0
        )
        short_liqs = self._liquidation_data["short_liquidations"].reindex(
            candles.index, fill_value=0
        )

        # Rolling sum of liquidations
        long_liq_sum = long_liqs.rolling(self.lookback_hours, min_periods=1).sum()
        short_liq_sum = short_liqs.rolling(self.lookback_hours, min_periods=1).sum()

        # Detect cascade events
        long_cascade = long_liq_sum > self.liquidation_threshold
        short_cascade = short_liq_sum > self.liquidation_threshold

        # Generate signals
        in_position = False
        entry_bar = 0
        pending_signal = None
        pending_direction = None
        pending_countdown = 0

        for i, idx in enumerate(candles.index):
            lc = long_cascade.get(idx, False)
            sc = short_cascade.get(idx, False)

            # Check for cascade trigger
            if not in_position and pending_signal is None:
                if lc and long_liq_sum.get(idx, 0) >= self.min_cascade_size:
                    # Large long liquidation cascade = price dropped
                    # Fade by going long after delay
                    pending_signal = Signal.BUY
                    pending_direction = "long"
                    pending_countdown = self.fade_delay
                elif sc and short_liq_sum.get(idx, 0) >= self.min_cascade_size:
                    # Large short liquidation cascade = price spiked
                    # We can't short, so skip (or could sell if already long)
                    pass

            # Handle pending signal countdown
            if pending_signal is not None:
                pending_countdown -= 1
                if pending_countdown <= 0:
                    signals.loc[idx] = pending_signal
                    in_position = True
                    entry_bar = i
                    pending_signal = None
                    pending_direction = None

            # Exit logic
            if in_position:
                bars_held = i - entry_bar

                if bars_held >= self.hold_period:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "stablecoin_liquidity_pulse",
    description="Risk on/off based on stablecoin supply changes",
)
class StablecoinLiquidityPulseStrategy(BaseStrategy):
    """
    Stablecoin Liquidity Pulse Strategy.
    
    Signal: Net stablecoin supply change (USDT + USDC + DAI market cap).
    
    Trade: Risk-on when liquidity expands, risk-off when it contracts; slow moving (weekly).
    
    Why diversifies: Macro liquidity driver, not technicals.
    This is a regime signal based on capital flows into the crypto ecosystem.
    
    Requires stablecoin supply data - run ingest_tier2_data.py --stablecoin-only first.
    """

    name = "stablecoin_liquidity_pulse"

    def _setup(
        self,
        supply_change_threshold: float = 0.01,  # 1% supply change
        lookback_days: int = 7,  # Weekly granularity
        risk_on_mode: str = "expand",  # "expand" = risk on when supply grows
        hold_period: int = 168,  # Hold 1 week
        **kwargs,
    ) -> None:
        self.supply_change_threshold = supply_change_threshold
        self.lookback_days = lookback_days
        self.risk_on_mode = risk_on_mode
        self.hold_period = hold_period
        self._stablecoin_data: pd.Series | None = None

    def set_stablecoin_data(self, supply_series: pd.Series) -> None:
        """Set stablecoin total supply data (USDT + USDC + DAI)."""
        self._stablecoin_data = supply_series

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]

        if self._stablecoin_data is not None and not self._stablecoin_data.empty:
            # Use actual stablecoin data
            supply = self._stablecoin_data.reindex(candles.index, method="ffill")
            supply_change = supply.pct_change(self.lookback_days * 24)  # Weekly change
            logger.debug(f"Using real stablecoin data with {len(supply.dropna())} points")
        else:
            # Proxy: Use volume trend as liquidity proxy
            logger.warning(
                f"{self.name}: Stablecoin supply data not available. "
                "Using volume as proxy. Run: python scripts/ingest_tier2_data.py --stablecoin-only"
            )
            volume = candles["volume"]
            volume_ma = volume.rolling(self.lookback_days * 24, min_periods=20).mean()
            supply_change = volume_ma.pct_change(self.lookback_days * 24)

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            sc = supply_change.get(idx, 0)

            if pd.isna(sc):
                continue

            if not in_position:
                # Risk on: liquidity expanding
                if self.risk_on_mode == "expand":
                    if sc > self.supply_change_threshold:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
                else:
                    # Risk on when liquidity contracting (contrarian)
                    if sc < -self.supply_change_threshold:
                        signals.loc[idx] = Signal.BUY
                        in_position = True
                        entry_bar = i
            else:
                bars_held = i - entry_bar

                # Exit on hold period or liquidity reversal
                should_exit = False

                if bars_held >= self.hold_period:
                    should_exit = True

                # Exit if liquidity condition reverses
                if self.risk_on_mode == "expand" and sc < -self.supply_change_threshold:
                    should_exit = True
                elif self.risk_on_mode != "expand" and sc > self.supply_change_threshold:
                    should_exit = True

                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)


@strategy_registry.register(
    "btc_dominance_rotation",
    description="Rotate to alts when BTC dominance is falling",
)
class BTCDominanceRotationStrategy(BaseStrategy):
    """
    BTC Dominance "Risk Budget Switch" Strategy.
    
    Signal: Dominance trend up = risk-off (rotate to BTC/cash).
            Dominance trend down = alt risk-on.
    
    Trade: Dynamically allocate between BTC vs alt basket vs cash; can run alongside ratios.
    
    Why diversifies: Captures rotation regimes your ratios may miss.
    
    Implementation: When applied to an alt, BUY when dominance is falling (alt season),
    SELL when dominance is rising (BTC dominance).
    """

    name = "btc_dominance_rotation"

    def _setup(
        self,
        dominance_trend_period: int = 168,  # 7-day trend
        dominance_momentum_period: int = 24,  # 24h momentum
        risk_on_threshold: float = -0.01,  # Dominance falling 1% = alt risk on
        risk_off_threshold: float = 0.01,  # Dominance rising 1% = risk off
        hold_period: int = 72,
        **kwargs,
    ) -> None:
        self.dominance_trend_period = dominance_trend_period
        self.dominance_momentum_period = dominance_momentum_period
        self.risk_on_threshold = risk_on_threshold
        self.risk_off_threshold = risk_off_threshold
        self.hold_period = hold_period
        self._dominance_data: pd.Series | None = None

    def set_dominance_data(self, dominance_series: pd.Series) -> None:
        """Set BTC dominance percentage series (0-100 or 0-1)."""
        self._dominance_data = dominance_series

    def generate_signals(self, candles: pd.DataFrame) -> pd.Series:
        self.validate_candles(candles)
        signals = self.create_signal_series(candles.index)

        close = candles["close"]

        if self._dominance_data is not None:
            # Use actual dominance data
            dominance = self._dominance_data.reindex(candles.index, method="ffill")
        else:
            # Proxy: Use BTC strength relative to this alt as dominance proxy
            # If BTC is outperforming, dominance is likely rising
            btc_proxy = close.rolling(self.dominance_trend_period).mean()
            dominance = btc_proxy  # This is a rough proxy

        # Calculate dominance change
        dom_change = dominance.pct_change(self.dominance_momentum_period)
        dom_trend = dominance.rolling(self.dominance_trend_period, min_periods=20).mean()
        dom_vs_trend = (dominance - dom_trend) / dom_trend

        # Generate signals
        in_position = False
        entry_bar = 0

        for i, idx in enumerate(candles.index):
            dc = dom_change.get(idx, 0)
            dvt = dom_vs_trend.get(idx, 0)

            if pd.isna(dc) or pd.isna(dvt):
                continue

            if not in_position:
                # Alt risk on: Dominance falling (alt season)
                if dc < self.risk_on_threshold:
                    signals.loc[idx] = Signal.BUY
                    in_position = True
                    entry_bar = i
            else:
                bars_held = i - entry_bar

                # Exit conditions
                should_exit = False

                if bars_held >= self.hold_period:
                    should_exit = True

                # Exit if dominance starts rising (risk off)
                if dc > self.risk_off_threshold:
                    should_exit = True

                if should_exit:
                    signals.loc[idx] = Signal.SELL
                    in_position = False

        return self.apply_filters(signals, candles)
