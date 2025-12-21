"""Data access layer for the crypto trading platform."""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Sequence

import pandas as pd
from sqlalchemy import delete, select, and_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from crypto.core.types import Candle
from crypto.data.database import get_async_session
from crypto.data.models import CandleModel, BacktestResultModel, TradeModel

logger = logging.getLogger(__name__)


class CandleRepository:
    """Repository for candle data operations."""

    async def save_candles(
        self,
        candles: list[Candle],
        session: AsyncSession | None = None,
    ) -> int:
        """
        Save candles to database, upserting on conflict.
        
        Args:
            candles: List of candles to save
            session: Optional session to use
            
        Returns:
            Number of candles saved
        """
        if not candles:
            return 0

        async def _save(sess: AsyncSession) -> int:
            values = [
                {
                    "symbol": c.symbol,
                    "interval": c.interval,
                    "open_time": c.open_time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "close_time": c.close_time,
                    "quote_volume": c.quote_volume,
                    "trades": c.trades,
                }
                for c in candles
            ]

            # PostgreSQL upsert - use do_nothing to avoid "cannot affect row twice" error
            # when batch has duplicates or conflicts with existing data
            stmt = insert(CandleModel).values(values)
            stmt = stmt.on_conflict_do_nothing(constraint="uq_candle")
            result = await sess.execute(stmt)
            return result.rowcount

        if session:
            return await _save(session)
        else:
            async with get_async_session() as sess:
                result = await _save(sess)
                return result

    async def get_candles(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> list[Candle]:
        """
        Get candles for a symbol/interval in a time range.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            start: Start time
            end: End time
            session: Optional session to use
            
        Returns:
            List of Candle objects
        """

        async def _get(sess: AsyncSession) -> list[Candle]:
            stmt = (
                select(CandleModel)
                .where(
                    and_(
                        CandleModel.symbol == symbol,
                        CandleModel.interval == interval,
                        CandleModel.open_time >= start,
                        CandleModel.open_time <= end,
                    )
                )
                .order_by(CandleModel.open_time)
            )
            result = await sess.execute(stmt)
            rows = result.scalars().all()

            return [
                Candle(
                    symbol=row.symbol,
                    interval=row.interval,
                    open_time=row.open_time,
                    open=row.open,
                    high=row.high,
                    low=row.low,
                    close=row.close,
                    volume=row.volume,
                    close_time=row.close_time,
                    quote_volume=row.quote_volume,
                    trades=row.trades,
                )
                for row in rows
            ]

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)

    async def get_candles_df(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        session: AsyncSession | None = None,
    ) -> pd.DataFrame:
        """
        Get candles as a pandas DataFrame.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            start: Start time
            end: End time
            session: Optional session to use
            
        Returns:
            DataFrame with OHLCV data
        """
        candles = await self.get_candles(symbol, interval, start, end, session)

        if not candles:
            return pd.DataFrame(
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )

        df = pd.DataFrame([c.to_dict() for c in candles])
        df.set_index("open_time", inplace=True)
        
        # Convert Decimal to float for calculations
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df

    async def get_latest_candle(
        self,
        symbol: str,
        interval: str,
        session: AsyncSession | None = None,
    ) -> Candle | None:
        """
        Get the latest candle for a symbol/interval.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            session: Optional session to use
            
        Returns:
            Latest candle or None
        """

        async def _get(sess: AsyncSession) -> Candle | None:
            stmt = (
                select(CandleModel)
                .where(
                    and_(
                        CandleModel.symbol == symbol,
                        CandleModel.interval == interval,
                    )
                )
                .order_by(CandleModel.open_time.desc())
                .limit(1)
            )
            result = await sess.execute(stmt)
            row = result.scalar_one_or_none()

            if row is None:
                return None

            return Candle(
                symbol=row.symbol,
                interval=row.interval,
                open_time=row.open_time,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                close_time=row.close_time,
                quote_volume=row.quote_volume,
                trades=row.trades,
            )

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)

    async def delete_candles(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        session: AsyncSession | None = None,
    ) -> int:
        """
        Delete candles for a symbol/interval, optionally in a time range.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            start: Optional start time
            end: Optional end time
            session: Optional session to use
            
        Returns:
            Number of candles deleted
        """

        async def _delete(sess: AsyncSession) -> int:
            conditions = [
                CandleModel.symbol == symbol,
                CandleModel.interval == interval,
            ]
            if start:
                conditions.append(CandleModel.open_time >= start)
            if end:
                conditions.append(CandleModel.open_time <= end)

            stmt = delete(CandleModel).where(and_(*conditions))
            result = await sess.execute(stmt)
            return result.rowcount

        if session:
            return await _delete(session)
        else:
            async with get_async_session() as sess:
                return await _delete(sess)

    async def get_available_data_range(
        self,
        symbol: str,
        interval: str,
        session: AsyncSession | None = None,
    ) -> tuple[datetime, datetime] | None:
        """
        Get the time range of available data.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            session: Optional session to use
            
        Returns:
            Tuple of (start, end) datetime or None if no data
        """
        from sqlalchemy import func

        async def _get(sess: AsyncSession) -> tuple[datetime, datetime] | None:
            stmt = select(
                func.min(CandleModel.open_time),
                func.max(CandleModel.open_time),
            ).where(
                and_(
                    CandleModel.symbol == symbol,
                    CandleModel.interval == interval,
                )
            )
            result = await sess.execute(stmt)
            row = result.one_or_none()

            if row is None or row[0] is None:
                return None

            return (row[0], row[1])

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)


class BacktestResultRepository:
    """Repository for backtest result operations."""

    async def save_result(
        self,
        result: dict,
        session: AsyncSession | None = None,
    ) -> str:
        """
        Save a backtest result.
        
        Args:
            result: Backtest result dictionary
            session: Optional session to use
            
        Returns:
            Backtest ID
        """

        async def _save(sess: AsyncSession) -> str:
            model = BacktestResultModel(**result)
            sess.add(model)
            await sess.flush()
            return model.backtest_id

        if session:
            return await _save(session)
        else:
            async with get_async_session() as sess:
                result_id = await _save(sess)
                return result_id

    async def get_results(
        self,
        strategy_name: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
        session: AsyncSession | None = None,
    ) -> Sequence[BacktestResultModel]:
        """Get backtest results with optional filters."""

        async def _get(sess: AsyncSession) -> Sequence[BacktestResultModel]:
            stmt = select(BacktestResultModel)

            if strategy_name:
                stmt = stmt.where(BacktestResultModel.strategy_name == strategy_name)
            if symbol:
                stmt = stmt.where(BacktestResultModel.symbol == symbol)

            stmt = stmt.order_by(BacktestResultModel.created_at.desc()).limit(limit)
            result = await sess.execute(stmt)
            return result.scalars().all()

        if session:
            return await _get(session)
        else:
            async with get_async_session() as sess:
                return await _get(sess)
