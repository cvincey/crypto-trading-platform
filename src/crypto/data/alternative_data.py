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
from crypto.data.models import FundingRateModel, OpenInterestModel

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
