"""
Tier 2 Alternative Data Loader.

Helper module to load and inject alternative data into strategies
for backtesting and live trading.

Usage:
    from crypto.data.tier2_loader import Tier2DataLoader
    
    loader = Tier2DataLoader()
    
    # Load all data for a time range
    await loader.load_all(start, end, symbols=["BTCUSDT", "ETHUSDT"])
    
    # Inject into a strategy
    loader.inject_into_strategy(strategy, candle_index, symbol="BTCUSDT")
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import pandas as pd

from crypto.data.alternative_data import (
    FearGreedRepository,
    LongShortRatioRepository,
    BTCDominanceRepository,
    StablecoinSupplyRepository,
    MacroIndicatorRepository,
    FundingRateRepository,
    OpenInterestRepository,
)
from crypto.data.mock_liquidations import generate_mock_liquidations

logger = logging.getLogger(__name__)


class Tier2DataLoader:
    """
    Loader for Tier 2 alternative data.
    
    Handles loading data from repositories and injecting it into strategies.
    """
    
    def __init__(self):
        self._fear_greed_repo = FearGreedRepository()
        self._long_short_repo = LongShortRatioRepository()
        self._dominance_repo = BTCDominanceRepository()
        self._stablecoin_repo = StablecoinSupplyRepository()
        self._macro_repo = MacroIndicatorRepository()
        self._funding_repo = FundingRateRepository()
        self._oi_repo = OpenInterestRepository()
        
        # Cached data
        self._fear_greed: pd.Series | None = None
        self._long_short: dict[str, pd.Series] = {}
        self._dominance: pd.Series | None = None
        self._stablecoin: pd.Series | None = None
        self._dxy: pd.Series | None = None
        self._vix: pd.Series | None = None
        self._funding: dict[str, pd.Series] = {}
        self._oi: dict[str, pd.Series] = {}
    
    async def load_fear_greed(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.Series:
        """Load Fear & Greed Index data."""
        data = await self._fear_greed_repo.get_fear_greed(start, end)
        
        if not data:
            logger.warning("No Fear & Greed data found in database")
            return pd.Series(dtype=float)
        
        df = pd.DataFrame([
            {"timestamp": d.timestamp, "value": d.value}
            for d in data
        ])
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        
        self._fear_greed = df["value"]
        logger.info(f"Loaded {len(self._fear_greed)} Fear & Greed data points")
        
        return self._fear_greed
    
    async def load_long_short_ratio(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        ratio_type: str = "accounts",
    ) -> pd.Series:
        """Load Long/Short ratio data for a symbol."""
        data = await self._long_short_repo.get_long_short_ratio(
            symbol, start, end, ratio_type
        )
        
        if not data:
            logger.warning(f"No Long/Short ratio data found for {symbol}")
            return pd.Series(dtype=float)
        
        df = pd.DataFrame([
            {"timestamp": d.timestamp, "ratio": float(d.long_short_ratio)}
            for d in data
        ])
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        
        self._long_short[symbol] = df["ratio"]
        logger.info(f"Loaded {len(df)} Long/Short ratio points for {symbol}")
        
        return self._long_short[symbol]
    
    async def load_dominance(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.Series:
        """Load BTC dominance data."""
        data = await self._dominance_repo.get_btc_dominance(start, end)
        
        if not data:
            logger.warning("No BTC dominance data found in database")
            return pd.Series(dtype=float)
        
        df = pd.DataFrame([
            {"timestamp": d.timestamp, "dominance": float(d.btc_dominance)}
            for d in data
        ])
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        
        self._dominance = df["dominance"]
        logger.info(f"Loaded {len(self._dominance)} BTC dominance data points")
        
        return self._dominance
    
    async def load_stablecoin_supply(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.Series:
        """Load total stablecoin supply data."""
        df = await self._stablecoin_repo.get_total_stablecoin_supply(start, end)
        
        if df.empty:
            logger.warning("No stablecoin supply data found in database")
            return pd.Series(dtype=float)
        
        self._stablecoin = df["total_supply"]
        logger.info(f"Loaded {len(self._stablecoin)} stablecoin supply data points")
        
        return self._stablecoin
    
    async def load_macro_indicators(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.Series]:
        """Load macro indicators (DXY, VIX)."""
        result = {}
        
        for indicator in ["DXY", "VIX", "SPX", "GOLD"]:
            df = await self._macro_repo.get_macro_indicator_df(indicator, start, end)
            
            if not df.empty:
                result[indicator] = df["value"]
                logger.info(f"Loaded {len(df)} {indicator} data points")
        
        self._dxy = result.get("DXY")
        self._vix = result.get("VIX")
        
        return result
    
    async def load_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.Series:
        """Load funding rate data for a symbol."""
        df = await self._funding_repo.get_funding_rates_df(symbol, start, end)
        
        if df.empty:
            logger.warning(f"No funding rate data found for {symbol}")
            return pd.Series(dtype=float)
        
        self._funding[symbol] = df["funding_rate"]
        logger.info(f"Loaded {len(df)} funding rate points for {symbol}")
        
        return self._funding[symbol]
    
    async def load_open_interest(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.Series:
        """Load open interest data for a symbol."""
        df = await self._oi_repo.get_open_interest_df(symbol, start, end)
        
        if df.empty:
            logger.warning(f"No open interest data found for {symbol}")
            return pd.Series(dtype=float)
        
        self._oi[symbol] = df["open_interest"]
        logger.info(f"Loaded {len(df)} open interest points for {symbol}")
        
        return self._oi[symbol]
    
    async def load_all(
        self,
        start: datetime,
        end: datetime,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Load all available Tier 2 data for a time range.
        
        Args:
            start: Start datetime
            end: End datetime
            symbols: List of symbols for symbol-specific data (L/S ratio, funding)
        
        Returns:
            Dict with loaded data counts
        """
        symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        results = {}
        
        # Load global data
        try:
            fg = await self.load_fear_greed(start, end)
            results["fear_greed"] = len(fg)
        except Exception as e:
            logger.error(f"Error loading Fear & Greed: {e}")
            results["fear_greed"] = 0
        
        try:
            dom = await self.load_dominance(start, end)
            results["dominance"] = len(dom)
        except Exception as e:
            logger.error(f"Error loading dominance: {e}")
            results["dominance"] = 0
        
        try:
            stable = await self.load_stablecoin_supply(start, end)
            results["stablecoin"] = len(stable)
        except Exception as e:
            logger.error(f"Error loading stablecoin supply: {e}")
            results["stablecoin"] = 0
        
        try:
            macro = await self.load_macro_indicators(start, end)
            results["macro"] = {k: len(v) for k, v in macro.items()}
        except Exception as e:
            logger.error(f"Error loading macro indicators: {e}")
            results["macro"] = {}
        
        # Load symbol-specific data
        results["long_short"] = {}
        results["funding"] = {}
        results["open_interest"] = {}
        
        for symbol in symbols:
            try:
                ls = await self.load_long_short_ratio(symbol, start, end)
                results["long_short"][symbol] = len(ls)
            except Exception as e:
                logger.error(f"Error loading L/S ratio for {symbol}: {e}")
            
            try:
                fr = await self.load_funding_rates(symbol, start, end)
                results["funding"][symbol] = len(fr)
            except Exception as e:
                logger.error(f"Error loading funding for {symbol}: {e}")
            
            try:
                oi = await self.load_open_interest(symbol, start, end)
                results["open_interest"][symbol] = len(oi)
            except Exception as e:
                logger.error(f"Error loading OI for {symbol}: {e}")
        
        return results
    
    def inject_into_strategy(
        self,
        strategy: Any,
        candle_index: pd.DatetimeIndex,
        symbol: str | None = None,
        candles: pd.DataFrame | None = None,
    ) -> None:
        """
        Inject loaded alternative data into a strategy.
        
        Detects what data setters the strategy has and injects appropriate data.
        
        Args:
            strategy: Strategy instance
            candle_index: Index to align data to
            symbol: Symbol for symbol-specific data
            candles: Candle data (for mock liquidations)
        """
        # Fear & Greed
        if hasattr(strategy, "set_fear_greed_data") and self._fear_greed is not None:
            aligned = self._fear_greed.reindex(candle_index, method="ffill")
            strategy.set_fear_greed_data(aligned)
            logger.debug(f"Injected Fear & Greed data into {strategy.name}")
        
        # Long/Short Ratio
        if hasattr(strategy, "set_long_short_data") and symbol and symbol in self._long_short:
            aligned = self._long_short[symbol].reindex(candle_index, method="ffill")
            strategy.set_long_short_data(aligned)
            logger.debug(f"Injected L/S ratio data into {strategy.name}")
        
        # BTC Dominance
        if hasattr(strategy, "set_dominance_data") and self._dominance is not None:
            aligned = self._dominance.reindex(candle_index, method="ffill")
            strategy.set_dominance_data(aligned)
            logger.debug(f"Injected BTC dominance data into {strategy.name}")
        
        # Set symbol for dominance strategies
        if hasattr(strategy, "set_symbol") and symbol:
            strategy.set_symbol(symbol)
        
        # Stablecoin Supply
        if hasattr(strategy, "set_stablecoin_data") and self._stablecoin is not None:
            aligned = self._stablecoin.reindex(candle_index, method="ffill")
            strategy.set_stablecoin_data(aligned)
            logger.debug(f"Injected stablecoin supply data into {strategy.name}")
        
        # Macro indicators
        if hasattr(strategy, "set_macro_data"):
            dxy_aligned = self._dxy.reindex(candle_index, method="ffill") if self._dxy is not None else None
            vix_aligned = self._vix.reindex(candle_index, method="ffill") if self._vix is not None else None
            strategy.set_macro_data(dxy_aligned, vix_aligned)
            logger.debug(f"Injected macro data into {strategy.name}")
        
        if hasattr(strategy, "set_dxy_data") and self._dxy is not None:
            aligned = self._dxy.reindex(candle_index, method="ffill")
            strategy.set_dxy_data(aligned)
            logger.debug(f"Injected DXY data into {strategy.name}")
        
        # Funding rates
        if hasattr(strategy, "set_funding_data") and symbol and symbol in self._funding:
            aligned = self._funding[symbol].reindex(candle_index, method="ffill")
            strategy.set_funding_data(aligned)
            logger.debug(f"Injected funding rate data into {strategy.name}")
        
        # Open Interest
        if hasattr(strategy, "set_open_interest_data") and symbol and symbol in self._oi:
            aligned = self._oi[symbol].reindex(candle_index, method="ffill")
            strategy.set_open_interest_data(aligned)
            logger.debug(f"Injected open interest data into {strategy.name}")
        
        # Liquidation data (mock)
        if hasattr(strategy, "set_liquidation_data") and candles is not None:
            mock_liqs = generate_mock_liquidations(candles)
            strategy.set_liquidation_data(mock_liqs)
            logger.debug(f"Injected mock liquidation data into {strategy.name}")
    
    def get_data_summary(self) -> dict[str, int]:
        """Get summary of loaded data."""
        return {
            "fear_greed": len(self._fear_greed) if self._fear_greed is not None else 0,
            "dominance": len(self._dominance) if self._dominance is not None else 0,
            "stablecoin": len(self._stablecoin) if self._stablecoin is not None else 0,
            "dxy": len(self._dxy) if self._dxy is not None else 0,
            "vix": len(self._vix) if self._vix is not None else 0,
            "long_short_symbols": list(self._long_short.keys()),
            "funding_symbols": list(self._funding.keys()),
            "oi_symbols": list(self._oi.keys()),
        }


# Convenience function for quick loading
async def load_tier2_data(
    start: datetime,
    end: datetime,
    symbols: list[str] | None = None,
) -> Tier2DataLoader:
    """
    Load all Tier 2 data and return configured loader.
    
    Usage:
        loader = await load_tier2_data(start, end, ["BTCUSDT", "ETHUSDT"])
        loader.inject_into_strategy(strategy, candles.index, "BTCUSDT")
    """
    loader = Tier2DataLoader()
    await loader.load_all(start, end, symbols)
    return loader
