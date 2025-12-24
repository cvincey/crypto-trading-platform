"""Repositories for alternative data: funding rates, open interest, etc."""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Sequence

import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from crypto.data.database import get_async_session
from crypto.data.models import (
    FundingRateModel,
    OpenInterestModel,
    FearGreedIndexModel,
    MacroIndicatorModel,
    LongShortRatioModel,
    BTCDominanceModel,
    ExchangeFlowModel,
    StablecoinSupplyModel,
)

logger = logging.getLogger(__name__)


@dataclass
class FundingRate:
    """Funding rate data point."""
    
    symbol: str
    funding_time: datetime
    funding_rate: Decimal
    mark_price: Decimal | None = None


@dataclass
class OpenInterest:
    """Open interest data point."""
    
    symbol: str
    timestamp: datetime
    open_interest: Decimal
    open_interest_value: Decimal | None = None


@dataclass
class FearGreedIndex:
    """Fear & Greed Index data point."""
    
    timestamp: datetime
    value: int  # 0-100
    value_classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"


@dataclass
class MacroIndicator:
    """Macro indicator data point (DXY, VIX, SPX, etc.)."""
    
    indicator: str
    timestamp: datetime
    value: Decimal
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    close: Decimal | None = None


@dataclass
class LongShortRatio:
    """Long/Short ratio data point."""
    
    symbol: str
    timestamp: datetime
    ratio_type: str  # "accounts" or "positions"
    long_ratio: Decimal
    short_ratio: Decimal
    long_short_ratio: Decimal


@dataclass
class BTCDominance:
    """BTC dominance data point."""
    
    timestamp: datetime
    btc_dominance: Decimal
    eth_dominance: Decimal | None = None
    total_market_cap: Decimal | None = None
    btc_market_cap: Decimal | None = None


@dataclass
class ExchangeFlow:
    """Exchange flow data point."""
    
    symbol: str
    timestamp: datetime
    inflow: Decimal
    outflow: Decimal
    netflow: Decimal
    exchange: str | None = None
    inflow_usd: Decimal | None = None
    outflow_usd: Decimal | None = None
    netflow_usd: Decimal | None = None


@dataclass
class StablecoinSupply:
    """Stablecoin supply data point."""
    
    symbol: str  # USDT, USDC, DAI
    timestamp: datetime
    market_cap: Decimal
    supply_change_24h: Decimal | None = None


class FundingRateRepository:
    """Repository for funding rate data operations."""

    async def save_funding_rates(
        self,
        rates: list[FundingRate],
        session: AsyncSession | None = None,
    ) -> int:
        """
        Save funding rates to database, upserting on conflict.
        
        Args:
            rates: List of FundingRate objects to save
            session: Optional session to use
            
        Returns:
            Number of records saved
        """
        if not rates:
            return 0

        async def _save(sess: AsyncSession) -> int:
            values = [
                {
                    "symbol": r.symbol,
                    "funding_time": r.funding_time,
                    "funding_rate": r.funding_rate,
                    "mark_price": r.mark_price,
                }
                for r in rates
            ]

            stmt = insert(FundingRateModel).values(values)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_funding_rate")
            result = await sess.execute(stmt)
            return result.rowcount

        if session:
            return await _save(session)
        else:
            async with get_async_session() as sess:
                result = await _save(sess)
                return result

    async def get_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> list[FundingRate]:
        """
        Get funding rates for a symbol in a time range.
        
        Args:
            symbol: Trading pair symbol
            start: Start time
            end: End time
            session: Optional session to use
            
        Returns:
            List of FundingRate objects
        """
        async def _get(sess: AsyncSession) -> list[FundingRate]:
            stmt = (
                select(FundingRateModel)
                .where(
                    and_(
                        FundingRateModel.symbol == symbol,
                        FundingRateModel.funding_time >= start,
                        FundingRateModel.funding_time <= end,
                    )
                )
                .order_by(FundingRateModel.funding_time)
            )
            result = await sess.execute(stmt)
            rows = result.scalars().all()

            return [
                FundingRate(
                    symbol=row.symbol,
                    funding_time=row.funding_time,
                    funding_rate=row.funding_rate,
                    mark_price=row.mark_price,
                )
                for row in rows
            ]

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)

    async def get_funding_rates_df(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> pd.DataFrame:
        """
        Get funding rates as a DataFrame.
        
        Args:
            symbol: Trading pair symbol
            start: Start time
            end: End time
            session: Optional session to use
            
        Returns:
            DataFrame with funding_time index and funding_rate column
        """
        rates = await self.get_funding_rates(symbol, start, end, session)
        
        if not rates:
            return pd.DataFrame(columns=["funding_rate", "mark_price"])
        
        df = pd.DataFrame([
            {
                "funding_time": r.funding_time,
                "funding_rate": float(r.funding_rate),
                "mark_price": float(r.mark_price) if r.mark_price else None,
            }
            for r in rates
        ])
        df.set_index("funding_time", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    async def get_funding_aligned_to_candles(
        self,
        symbol: str,
        candle_index: pd.DatetimeIndex,
        session: AsyncSession | None = None,
    ) -> pd.Series:
        """
        Get funding rates aligned to candle timestamps.
        
        Since funding is every 8 hours, we forward-fill to hourly candles.
        
        Args:
            symbol: Trading pair symbol
            candle_index: DatetimeIndex of candle timestamps
            session: Optional session to use
            
        Returns:
            Series of funding rates aligned to candle index
        """
        if candle_index.empty:
            return pd.Series(dtype=float)
        
        start = candle_index.min().to_pydatetime()
        end = candle_index.max().to_pydatetime()
        
        df = await self.get_funding_rates_df(symbol, start, end, session)
        
        if df.empty:
            return pd.Series(0.0, index=candle_index, name="funding_rate")
        
        # Reindex to candle timestamps, forward-fill
        aligned = df["funding_rate"].reindex(candle_index, method="ffill")
        aligned = aligned.fillna(0.0)
        return aligned


class OpenInterestRepository:
    """Repository for open interest data operations."""

    async def save_open_interest(
        self,
        data: list[OpenInterest],
        session: AsyncSession | None = None,
    ) -> int:
        """
        Save open interest data to database, upserting on conflict.
        
        Args:
            data: List of OpenInterest objects to save
            session: Optional session to use
            
        Returns:
            Number of records saved
        """
        if not data:
            return 0

        async def _save(sess: AsyncSession) -> int:
            values = [
                {
                    "symbol": d.symbol,
                    "timestamp": d.timestamp,
                    "open_interest": d.open_interest,
                    "open_interest_value": d.open_interest_value,
                }
                for d in data
            ]

            stmt = insert(OpenInterestModel).values(values)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_open_interest")
            result = await sess.execute(stmt)
            return result.rowcount

        if session:
            return await _save(session)
        else:
            async with get_async_session() as sess:
                result = await _save(sess)
                return result

    async def get_open_interest(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> list[OpenInterest]:
        """
        Get open interest for a symbol in a time range.
        
        Args:
            symbol: Trading pair symbol
            start: Start time
            end: End time
            session: Optional session to use
            
        Returns:
            List of OpenInterest objects
        """
        async def _get(sess: AsyncSession) -> list[OpenInterest]:
            stmt = (
                select(OpenInterestModel)
                .where(
                    and_(
                        OpenInterestModel.symbol == symbol,
                        OpenInterestModel.timestamp >= start,
                        OpenInterestModel.timestamp <= end,
                    )
                )
                .order_by(OpenInterestModel.timestamp)
            )
            result = await sess.execute(stmt)
            rows = result.scalars().all()

            return [
                OpenInterest(
                    symbol=row.symbol,
                    timestamp=row.timestamp,
                    open_interest=row.open_interest,
                    open_interest_value=row.open_interest_value,
                )
                for row in rows
            ]

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)

    async def get_open_interest_df(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> pd.DataFrame:
        """
        Get open interest as a DataFrame.
        
        Args:
            symbol: Trading pair symbol
            start: Start time
            end: End time
            session: Optional session to use
            
        Returns:
            DataFrame with timestamp index and open_interest column
        """
        data = await self.get_open_interest(symbol, start, end, session)
        
        if not data:
            return pd.DataFrame(columns=["open_interest", "open_interest_value"])
        
        df = pd.DataFrame([
            {
                "timestamp": d.timestamp,
                "open_interest": float(d.open_interest),
                "open_interest_value": float(d.open_interest_value) if d.open_interest_value else None,
            }
            for d in data
        ])
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    async def get_oi_aligned_to_candles(
        self,
        symbol: str,
        candle_index: pd.DatetimeIndex,
        session: AsyncSession | None = None,
    ) -> pd.Series:
        """
        Get open interest aligned to candle timestamps.
        
        Args:
            symbol: Trading pair symbol
            candle_index: DatetimeIndex of candle timestamps
            session: Optional session to use
            
        Returns:
            Series of open interest aligned to candle index
        """
        if candle_index.empty:
            return pd.Series(dtype=float)
        
        start = candle_index.min().to_pydatetime()
        end = candle_index.max().to_pydatetime()
        
        df = await self.get_open_interest_df(symbol, start, end, session)
        
        if df.empty:
            return pd.Series(dtype=float, index=candle_index, name="open_interest")
        
        # Reindex to candle timestamps, forward-fill
        aligned = df["open_interest"].reindex(candle_index, method="ffill")
        return aligned


class FearGreedRepository:
    """Repository for Fear & Greed Index data operations."""

    async def save_fear_greed(
        self,
        data: list[FearGreedIndex],
        session: AsyncSession | None = None,
    ) -> int:
        """Save Fear & Greed Index data to database."""
        if not data:
            return 0

        async def _save(sess: AsyncSession) -> int:
            values = [
                {
                    "timestamp": d.timestamp,
                    "value": d.value,
                    "value_classification": d.value_classification,
                }
                for d in data
            ]

            stmt = insert(FearGreedIndexModel).values(values)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_fear_greed_timestamp")
            result = await sess.execute(stmt)
            return result.rowcount

        if session:
            return await _save(session)
        else:
            async with get_async_session() as sess:
                result = await _save(sess)
                return result

    async def get_fear_greed(
        self,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> list[FearGreedIndex]:
        """Get Fear & Greed Index data in a time range."""
        async def _get(sess: AsyncSession) -> list[FearGreedIndex]:
            stmt = (
                select(FearGreedIndexModel)
                .where(
                    and_(
                        FearGreedIndexModel.timestamp >= start,
                        FearGreedIndexModel.timestamp <= end,
                    )
                )
                .order_by(FearGreedIndexModel.timestamp)
            )
            result = await sess.execute(stmt)
            rows = result.scalars().all()

            return [
                FearGreedIndex(
                    timestamp=row.timestamp,
                    value=row.value,
                    value_classification=row.value_classification,
                )
                for row in rows
            ]

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)

    async def get_fear_greed_df(
        self,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> pd.DataFrame:
        """Get Fear & Greed Index as a DataFrame."""
        data = await self.get_fear_greed(start, end, session)
        
        if not data:
            return pd.DataFrame(columns=["value", "value_classification"])
        
        df = pd.DataFrame([
            {
                "timestamp": d.timestamp,
                "value": d.value,
                "value_classification": d.value_classification,
            }
            for d in data
        ])
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    async def get_fear_greed_aligned_to_candles(
        self,
        candle_index: pd.DatetimeIndex,
        session: AsyncSession | None = None,
    ) -> pd.Series:
        """Get Fear & Greed Index aligned to candle timestamps."""
        if candle_index.empty:
            return pd.Series(dtype=float)
        
        start = candle_index.min().to_pydatetime()
        end = candle_index.max().to_pydatetime()
        
        df = await self.get_fear_greed_df(start, end, session)
        
        if df.empty:
            return pd.Series(50.0, index=candle_index, name="fear_greed")
        
        aligned = df["value"].reindex(candle_index, method="ffill")
        aligned = aligned.fillna(50.0)  # Neutral default
        return aligned


class LongShortRatioRepository:
    """Repository for Long/Short ratio data operations."""

    async def save_long_short_ratio(
        self,
        data: list[LongShortRatio],
        session: AsyncSession | None = None,
    ) -> int:
        """Save Long/Short ratio data to database."""
        if not data:
            return 0

        async def _save(sess: AsyncSession) -> int:
            values = [
                {
                    "symbol": d.symbol,
                    "timestamp": d.timestamp,
                    "ratio_type": d.ratio_type,
                    "long_ratio": d.long_ratio,
                    "short_ratio": d.short_ratio,
                    "long_short_ratio": d.long_short_ratio,
                }
                for d in data
            ]

            stmt = insert(LongShortRatioModel).values(values)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_long_short_ratio")
            result = await sess.execute(stmt)
            return result.rowcount

        if session:
            return await _save(session)
        else:
            async with get_async_session() as sess:
                result = await _save(sess)
                return result

    async def get_long_short_ratio(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        ratio_type: str = "accounts",
        session: AsyncSession | None = None,
    ) -> list[LongShortRatio]:
        """Get Long/Short ratio for a symbol in a time range."""
        async def _get(sess: AsyncSession) -> list[LongShortRatio]:
            stmt = (
                select(LongShortRatioModel)
                .where(
                    and_(
                        LongShortRatioModel.symbol == symbol,
                        LongShortRatioModel.ratio_type == ratio_type,
                        LongShortRatioModel.timestamp >= start,
                        LongShortRatioModel.timestamp <= end,
                    )
                )
                .order_by(LongShortRatioModel.timestamp)
            )
            result = await sess.execute(stmt)
            rows = result.scalars().all()

            return [
                LongShortRatio(
                    symbol=row.symbol,
                    timestamp=row.timestamp,
                    ratio_type=row.ratio_type,
                    long_ratio=row.long_ratio,
                    short_ratio=row.short_ratio,
                    long_short_ratio=row.long_short_ratio,
                )
                for row in rows
            ]

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)

    async def get_long_short_ratio_df(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        ratio_type: str = "accounts",
        session: AsyncSession | None = None,
    ) -> pd.DataFrame:
        """Get Long/Short ratio as a DataFrame."""
        data = await self.get_long_short_ratio(symbol, start, end, ratio_type, session)
        
        if not data:
            return pd.DataFrame(columns=["long_ratio", "short_ratio", "long_short_ratio"])
        
        df = pd.DataFrame([
            {
                "timestamp": d.timestamp,
                "long_ratio": float(d.long_ratio),
                "short_ratio": float(d.short_ratio),
                "long_short_ratio": float(d.long_short_ratio),
            }
            for d in data
        ])
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df


class BTCDominanceRepository:
    """Repository for BTC dominance data operations."""

    async def save_btc_dominance(
        self,
        data: list[BTCDominance],
        session: AsyncSession | None = None,
    ) -> int:
        """Save BTC dominance data to database."""
        if not data:
            return 0

        async def _save(sess: AsyncSession) -> int:
            values = [
                {
                    "timestamp": d.timestamp,
                    "btc_dominance": d.btc_dominance,
                    "eth_dominance": d.eth_dominance,
                    "total_market_cap": d.total_market_cap,
                    "btc_market_cap": d.btc_market_cap,
                }
                for d in data
            ]

            stmt = insert(BTCDominanceModel).values(values)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_btc_dominance_timestamp")
            result = await sess.execute(stmt)
            return result.rowcount

        if session:
            return await _save(session)
        else:
            async with get_async_session() as sess:
                result = await _save(sess)
                return result

    async def get_btc_dominance(
        self,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> list[BTCDominance]:
        """Get BTC dominance data in a time range."""
        async def _get(sess: AsyncSession) -> list[BTCDominance]:
            stmt = (
                select(BTCDominanceModel)
                .where(
                    and_(
                        BTCDominanceModel.timestamp >= start,
                        BTCDominanceModel.timestamp <= end,
                    )
                )
                .order_by(BTCDominanceModel.timestamp)
            )
            result = await sess.execute(stmt)
            rows = result.scalars().all()

            return [
                BTCDominance(
                    timestamp=row.timestamp,
                    btc_dominance=row.btc_dominance,
                    eth_dominance=row.eth_dominance,
                    total_market_cap=row.total_market_cap,
                    btc_market_cap=row.btc_market_cap,
                )
                for row in rows
            ]

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)

    async def get_btc_dominance_df(
        self,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> pd.DataFrame:
        """Get BTC dominance as a DataFrame."""
        data = await self.get_btc_dominance(start, end, session)
        
        if not data:
            return pd.DataFrame(columns=["btc_dominance", "eth_dominance"])
        
        df = pd.DataFrame([
            {
                "timestamp": d.timestamp,
                "btc_dominance": float(d.btc_dominance),
                "eth_dominance": float(d.eth_dominance) if d.eth_dominance else None,
                "total_market_cap": float(d.total_market_cap) if d.total_market_cap else None,
            }
            for d in data
        ])
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df


class ExchangeFlowRepository:
    """Repository for exchange flow data operations (requires API key)."""

    async def save_exchange_flows(
        self,
        data: list[ExchangeFlow],
        session: AsyncSession | None = None,
    ) -> int:
        """Save exchange flow data to database."""
        if not data:
            return 0

        async def _save(sess: AsyncSession) -> int:
            values = [
                {
                    "symbol": d.symbol,
                    "timestamp": d.timestamp,
                    "exchange": d.exchange,
                    "inflow": d.inflow,
                    "outflow": d.outflow,
                    "netflow": d.netflow,
                    "inflow_usd": d.inflow_usd,
                    "outflow_usd": d.outflow_usd,
                    "netflow_usd": d.netflow_usd,
                }
                for d in data
            ]

            stmt = insert(ExchangeFlowModel).values(values)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_exchange_flow")
            result = await sess.execute(stmt)
            return result.rowcount

        if session:
            return await _save(session)
        else:
            async with get_async_session() as sess:
                result = await _save(sess)
                return result

    async def get_exchange_flows(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> list[ExchangeFlow]:
        """Get exchange flow data in a time range."""
        async def _get(sess: AsyncSession) -> list[ExchangeFlow]:
            stmt = (
                select(ExchangeFlowModel)
                .where(
                    and_(
                        ExchangeFlowModel.symbol == symbol,
                        ExchangeFlowModel.timestamp >= start,
                        ExchangeFlowModel.timestamp <= end,
                    )
                )
                .order_by(ExchangeFlowModel.timestamp)
            )
            result = await sess.execute(stmt)
            rows = result.scalars().all()

            return [
                ExchangeFlow(
                    symbol=row.symbol,
                    timestamp=row.timestamp,
                    exchange=row.exchange,
                    inflow=row.inflow,
                    outflow=row.outflow,
                    netflow=row.netflow,
                    inflow_usd=row.inflow_usd,
                    outflow_usd=row.outflow_usd,
                    netflow_usd=row.netflow_usd,
                )
                for row in rows
            ]

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)

    async def get_exchange_flows_df(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> pd.DataFrame:
        """Get exchange flows as a DataFrame."""
        data = await self.get_exchange_flows(symbol, start, end, session)
        
        if not data:
            return pd.DataFrame(columns=["inflow", "outflow", "netflow"])
        
        df = pd.DataFrame([
            {
                "timestamp": d.timestamp,
                "inflow": float(d.inflow),
                "outflow": float(d.outflow),
                "netflow": float(d.netflow),
                "inflow_usd": float(d.inflow_usd) if d.inflow_usd else None,
                "outflow_usd": float(d.outflow_usd) if d.outflow_usd else None,
                "netflow_usd": float(d.netflow_usd) if d.netflow_usd else None,
            }
            for d in data
        ])
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df


class MacroIndicatorRepository:
    """Repository for macro indicator data operations (DXY, VIX, SPX, etc.)."""

    async def save_macro_indicators(
        self,
        data: list[MacroIndicator],
        session: AsyncSession | None = None,
    ) -> int:
        """Save macro indicator data to database."""
        if not data:
            return 0

        async def _save(sess: AsyncSession) -> int:
            values = [
                {
                    "indicator": d.indicator,
                    "timestamp": d.timestamp,
                    "value": d.value,
                    "open": d.open,
                    "high": d.high,
                    "low": d.low,
                    "close": d.close,
                }
                for d in data
            ]

            stmt = insert(MacroIndicatorModel).values(values)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_macro_indicator")
            result = await sess.execute(stmt)
            return result.rowcount

        if session:
            return await _save(session)
        else:
            async with get_async_session() as sess:
                result = await _save(sess)
                return result

    async def get_macro_indicator(
        self,
        indicator: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> list[MacroIndicator]:
        """Get macro indicator data in a time range."""
        async def _get(sess: AsyncSession) -> list[MacroIndicator]:
            stmt = (
                select(MacroIndicatorModel)
                .where(
                    and_(
                        MacroIndicatorModel.indicator == indicator,
                        MacroIndicatorModel.timestamp >= start,
                        MacroIndicatorModel.timestamp <= end,
                    )
                )
                .order_by(MacroIndicatorModel.timestamp)
            )
            result = await sess.execute(stmt)
            rows = result.scalars().all()

            return [
                MacroIndicator(
                    indicator=row.indicator,
                    timestamp=row.timestamp,
                    value=row.value,
                    open=row.open,
                    high=row.high,
                    low=row.low,
                    close=row.close,
                )
                for row in rows
            ]

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)

    async def get_macro_indicator_df(
        self,
        indicator: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> pd.DataFrame:
        """Get macro indicator as a DataFrame."""
        data = await self.get_macro_indicator(indicator, start, end, session)
        
        if not data:
            return pd.DataFrame(columns=["value", "open", "high", "low", "close"])
        
        df = pd.DataFrame([
            {
                "timestamp": d.timestamp,
                "value": float(d.value),
                "open": float(d.open) if d.open else None,
                "high": float(d.high) if d.high else None,
                "low": float(d.low) if d.low else None,
                "close": float(d.close) if d.close else None,
            }
            for d in data
        ])
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    async def get_macro_aligned_to_candles(
        self,
        indicator: str,
        candle_index: pd.DatetimeIndex,
        session: AsyncSession | None = None,
    ) -> pd.Series:
        """Get macro indicator aligned to candle timestamps."""
        if candle_index.empty:
            return pd.Series(dtype=float)
        
        start = candle_index.min().to_pydatetime()
        end = candle_index.max().to_pydatetime()
        
        df = await self.get_macro_indicator_df(indicator, start, end, session)
        
        if df.empty:
            return pd.Series(dtype=float, index=candle_index, name=indicator)
        
        # Reindex to candle timestamps, forward-fill (macro data is daily)
        aligned = df["value"].reindex(candle_index, method="ffill")
        return aligned


class StablecoinSupplyRepository:
    """Repository for stablecoin supply data operations."""

    async def save_stablecoin_supply(
        self,
        data: list[StablecoinSupply],
        session: AsyncSession | None = None,
    ) -> int:
        """Save stablecoin supply data to database."""
        if not data:
            return 0

        async def _save(sess: AsyncSession) -> int:
            values = [
                {
                    "symbol": d.symbol,
                    "timestamp": d.timestamp,
                    "market_cap": d.market_cap,
                    "supply_change_24h": d.supply_change_24h,
                }
                for d in data
            ]

            stmt = insert(StablecoinSupplyModel).values(values)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_stablecoin_supply")
            result = await sess.execute(stmt)
            return result.rowcount

        if session:
            return await _save(session)
        else:
            async with get_async_session() as sess:
                result = await _save(sess)
                return result

    async def get_stablecoin_supply(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> list[StablecoinSupply]:
        """Get stablecoin supply data in a time range."""
        async def _get(sess: AsyncSession) -> list[StablecoinSupply]:
            stmt = (
                select(StablecoinSupplyModel)
                .where(
                    and_(
                        StablecoinSupplyModel.symbol == symbol,
                        StablecoinSupplyModel.timestamp >= start,
                        StablecoinSupplyModel.timestamp <= end,
                    )
                )
                .order_by(StablecoinSupplyModel.timestamp)
            )
            result = await sess.execute(stmt)
            rows = result.scalars().all()

            return [
                StablecoinSupply(
                    symbol=row.symbol,
                    timestamp=row.timestamp,
                    market_cap=row.market_cap,
                    supply_change_24h=row.supply_change_24h,
                )
                for row in rows
            ]

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)

    async def get_total_stablecoin_supply(
        self,
        start: datetime,
        end: datetime,
        symbols: list[str] | None = None,
        session: AsyncSession | None = None,
    ) -> pd.DataFrame:
        """Get total stablecoin supply (USDT + USDC + DAI)."""
        symbols = symbols or ["USDT", "USDC", "DAI"]
        
        all_data = []
        for symbol in symbols:
            data = await self.get_stablecoin_supply(symbol, start, end, session)
            for d in data:
                all_data.append({
                    "timestamp": d.timestamp,
                    "symbol": d.symbol,
                    "market_cap": float(d.market_cap),
                })
        
        if not all_data:
            return pd.DataFrame(columns=["total_supply"])
        
        df = pd.DataFrame(all_data)
        
        # Pivot to get each stablecoin as a column
        pivot = df.pivot_table(
            index="timestamp", 
            columns="symbol", 
            values="market_cap",
            aggfunc="first"
        )
        
        # Sum across all stablecoins
        pivot["total_supply"] = pivot.sum(axis=1)
        pivot.index = pd.to_datetime(pivot.index, utc=True)
        
        return pivot

    async def get_supply_aligned_to_candles(
        self,
        candle_index: pd.DatetimeIndex,
        session: AsyncSession | None = None,
    ) -> pd.Series:
        """Get total stablecoin supply aligned to candle timestamps."""
        if candle_index.empty:
            return pd.Series(dtype=float)
        
        start = candle_index.min().to_pydatetime()
        end = candle_index.max().to_pydatetime()
        
        df = await self.get_total_stablecoin_supply(start, end, session=session)
        
        if df.empty:
            return pd.Series(dtype=float, index=candle_index, name="total_supply")
        
        # Reindex to candle timestamps, forward-fill (daily data)
        aligned = df["total_supply"].reindex(candle_index, method="ffill")
        return aligned
