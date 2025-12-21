"""Data ingestion service for fetching and storing market data."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable

from crypto.core.types import Candle
from crypto.data.repository import CandleRepository
from crypto.exchanges.base import Exchange
from crypto.exchanges.registry import get_exchange

logger = logging.getLogger(__name__)


class DataIngestionService:
    """
    Service for ingesting market data from exchanges.
    
    Handles:
    - Fetching historical data with pagination
    - Real-time data streaming
    - Storing data in the database
    - Gap detection and filling
    """

    def __init__(
        self,
        exchange: Exchange | None = None,
        exchange_name: str = "binance_testnet",
        repository: CandleRepository | None = None,
    ):
        """
        Initialize the ingestion service.
        
        Args:
            exchange: Exchange instance to use
            exchange_name: Name of exchange in config (used if exchange is None)
            repository: Candle repository for storage
        """
        self._exchange = exchange
        self._exchange_name = exchange_name
        self._repository = repository or CandleRepository()
        self._streaming = False

    async def _get_exchange(self) -> Exchange:
        """Get or create exchange instance."""
        if self._exchange is None:
            self._exchange = get_exchange(self._exchange_name)
        return self._exchange

    async def ingest_historical(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime | None = None,
        batch_size: int = 1000,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """
        Ingest historical candle data.
        
        Fetches data from the exchange and stores it in the database.
        Handles pagination for large date ranges.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1h")
            start: Start time
            end: End time (defaults to now)
            batch_size: Number of candles per request
            progress_callback: Optional callback(fetched, total_estimate)
            
        Returns:
            Number of candles ingested
        """
        end = end or datetime.utcnow()
        exchange = await self._get_exchange()
        
        logger.info(
            f"Ingesting {symbol} {interval} from {start} to {end}"
        )

        total_candles = 0
        current_start = start

        # Estimate total candles for progress
        interval_seconds = self._interval_to_seconds(interval)
        total_seconds = (end - start).total_seconds()
        estimated_total = int(total_seconds / interval_seconds)

        while current_start < end:
            # Calculate batch end
            batch_seconds = batch_size * interval_seconds
            batch_end = min(
                current_start + timedelta(seconds=batch_seconds),
                end,
            )

            try:
                candles = await exchange.fetch_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    start=current_start,
                    end=batch_end,
                    limit=batch_size,
                )

                if not candles:
                    logger.warning(
                        f"No candles returned for {symbol} {interval} "
                        f"from {current_start} to {batch_end}"
                    )
                    break

                # Save to database
                saved = await self._repository.save_candles(candles)
                total_candles += saved

                # Progress callback
                if progress_callback:
                    progress_callback(total_candles, estimated_total)

                # Move to next batch
                current_start = candles[-1].open_time + timedelta(
                    seconds=interval_seconds
                )

                logger.debug(
                    f"Fetched {len(candles)} candles, total: {total_candles}"
                )

                # Rate limiting - small delay between batches
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"Error fetching candles for {symbol}: {e}"
                )
                # Wait and retry once
                await asyncio.sleep(1)
                continue

        logger.info(
            f"Ingestion complete: {total_candles} candles for {symbol} {interval}"
        )
        return total_candles

    async def ingest_days(
        self,
        symbol: str,
        interval: str,
        days: int,
        exchange_name: str | None = None,
    ) -> int:
        """
        Convenience method to ingest last N days of data.
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            days: Number of days to ingest
            exchange_name: Optional exchange name override
            
        Returns:
            Number of candles ingested
        """
        if exchange_name:
            self._exchange_name = exchange_name
            self._exchange = None

        end = datetime.utcnow()
        start = end - timedelta(days=days)
        
        return await self.ingest_historical(symbol, interval, start, end)

    async def fill_gaps(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> int:
        """
        Detect and fill gaps in historical data.
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            start: Start time
            end: End time
            
        Returns:
            Number of candles added to fill gaps
        """
        # Get existing data range
        data_range = await self._repository.get_available_data_range(
            symbol, interval
        )
        
        if data_range is None:
            # No data at all, fetch everything
            return await self.ingest_historical(symbol, interval, start, end)

        existing_start, existing_end = data_range
        filled = 0

        # Fill before existing data
        if start < existing_start:
            filled += await self.ingest_historical(
                symbol, interval, start, existing_start
            )

        # Fill after existing data
        if end > existing_end:
            filled += await self.ingest_historical(
                symbol, interval, existing_end, end
            )

        # TODO: Detect internal gaps by checking for missing candles

        return filled

    async def start_streaming(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Candle], None] | None = None,
    ) -> None:
        """
        Start streaming real-time candle data.
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            callback: Optional callback for each candle
        """
        exchange = await self._get_exchange()
        
        async def _on_candle(candle: Candle) -> None:
            # Save to database
            await self._repository.save_candles([candle])
            
            # Call user callback
            if callback:
                callback(candle)

        await exchange.subscribe_candles(symbol, interval, _on_candle)
        self._streaming = True
        
        logger.info(f"Started streaming {symbol} {interval}")

    async def stop_streaming(self) -> None:
        """Stop all streaming subscriptions."""
        if self._exchange:
            await self._exchange.unsubscribe_all()
        self._streaming = False
        logger.info("Stopped streaming")

    async def close(self) -> None:
        """Close the ingestion service."""
        await self.stop_streaming()
        if self._exchange:
            await self._exchange.close()

    @staticmethod
    def _interval_to_seconds(interval: str) -> int:
        """Convert interval string to seconds."""
        unit = interval[-1]
        value = int(interval[:-1])

        multipliers = {
            "m": 60,
            "h": 3600,
            "d": 86400,
            "w": 604800,
            "M": 2592000,  # Approximate month
        }

        return value * multipliers.get(unit, 60)


async def ingest_symbols(
    symbols: list[str],
    interval: str,
    days: int,
    exchange_name: str = "binance_testnet",
) -> dict[str, int]:
    """
    Convenience function to ingest multiple symbols.
    
    Args:
        symbols: List of trading pairs
        interval: Candle interval
        days: Number of days to ingest
        exchange_name: Exchange name in config
        
    Returns:
        Dict mapping symbol to number of candles ingested
    """
    service = DataIngestionService(exchange_name=exchange_name)
    results = {}

    try:
        for symbol in symbols:
            count = await service.ingest_days(symbol, interval, days)
            results[symbol] = count
            logger.info(f"Ingested {count} candles for {symbol}")
    finally:
        await service.close()

    return results


# =============================================================================
# Alternative Data Ingestion
# =============================================================================

class AlternativeDataIngestionService:
    """
    Service for ingesting alternative data: funding rates, open interest, etc.
    
    Uses Binance Futures API endpoints.
    """

    # Binance Futures API base URL
    FUTURES_BASE_URL = "https://fapi.binance.com"

    def __init__(self):
        """Initialize the alternative data ingestion service."""
        from crypto.data.alternative_data import (
            FundingRateRepository,
            OpenInterestRepository,
            FundingRate,
            OpenInterest,
        )
        self._funding_repo = FundingRateRepository()
        self._oi_repo = OpenInterestRepository()
        self._FundingRate = FundingRate
        self._OpenInterest = OpenInterest
        self._client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def ingest_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> int:
        """
        Ingest historical funding rates from Binance Futures.
        
        Binance returns funding rates every 8 hours.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start: Start time
            end: End time (defaults to now)
            limit: Max records per request
            
        Returns:
            Number of funding rates ingested
        """
        from decimal import Decimal
        
        end = end or datetime.utcnow()
        client = await self._get_client()
        
        logger.info(f"Ingesting funding rates for {symbol} from {start} to {end}")
        
        total_saved = 0
        current_start = start
        
        while current_start < end:
            start_ms = int(current_start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            
            url = f"{self.FUTURES_BASE_URL}/fapi/v1/fundingRate"
            params = {
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
            
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                # Convert to FundingRate objects
                rates = [
                    self._FundingRate(
                        symbol=symbol,
                        funding_time=datetime.fromtimestamp(d["fundingTime"] / 1000),
                        funding_rate=Decimal(d["fundingRate"]),
                        mark_price=Decimal(d.get("markPrice", "0")) if d.get("markPrice") else None,
                    )
                    for d in data
                ]
                
                # Save to database
                saved = await self._funding_repo.save_funding_rates(rates)
                total_saved += saved
                
                # Move to next batch
                if len(data) < limit:
                    break
                
                last_time = datetime.fromtimestamp(data[-1]["fundingTime"] / 1000)
                current_start = last_time + timedelta(seconds=1)
                
                await asyncio.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching funding rates for {symbol}: {e}")
                await asyncio.sleep(1)
                break
        
        logger.info(f"Ingested {total_saved} funding rates for {symbol}")
        return total_saved

    async def ingest_open_interest_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
        period: str = "1h",
        limit: int = 500,
    ) -> int:
        """
        Ingest historical open interest from Binance Futures.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start: Start time
            end: End time (defaults to now)
            period: Interval ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")
            limit: Max records per request
            
        Returns:
            Number of open interest records ingested
        """
        from decimal import Decimal
        
        end = end or datetime.utcnow()
        client = await self._get_client()
        
        logger.info(f"Ingesting open interest for {symbol} from {start} to {end}")
        
        total_saved = 0
        current_start = start
        
        while current_start < end:
            start_ms = int(current_start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            
            url = f"{self.FUTURES_BASE_URL}/futures/data/openInterestHist"
            params = {
                "symbol": symbol,
                "period": period,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
            
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                # Convert to OpenInterest objects
                oi_data = [
                    self._OpenInterest(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(d["timestamp"] / 1000),
                        open_interest=Decimal(d["sumOpenInterest"]),
                        open_interest_value=Decimal(d["sumOpenInterestValue"]) if d.get("sumOpenInterestValue") else None,
                    )
                    for d in data
                ]
                
                # Save to database
                saved = await self._oi_repo.save_open_interest(oi_data)
                total_saved += saved
                
                # Move to next batch
                if len(data) < limit:
                    break
                
                last_time = datetime.fromtimestamp(data[-1]["timestamp"] / 1000)
                current_start = last_time + timedelta(seconds=1)
                
                await asyncio.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching open interest for {symbol}: {e}")
                await asyncio.sleep(1)
                break
        
        logger.info(f"Ingested {total_saved} open interest records for {symbol}")
        return total_saved

    async def ingest_funding_rates_days(
        self,
        symbol: str,
        days: int,
    ) -> int:
        """
        Convenience method to ingest last N days of funding rates.
        
        Args:
            symbol: Trading pair
            days: Number of days to ingest
            
        Returns:
            Number of records ingested
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        return await self.ingest_funding_rates(symbol, start, end)

    async def ingest_open_interest_days(
        self,
        symbol: str,
        days: int,
        period: str = "1h",
    ) -> int:
        """
        Convenience method to ingest last N days of open interest.
        
        Args:
            symbol: Trading pair
            days: Number of days to ingest
            period: Data interval
            
        Returns:
            Number of records ingested
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        return await self.ingest_open_interest_history(symbol, start, end, period)


async def ingest_alternative_data(
    symbols: list[str],
    days: int,
    include_funding: bool = True,
    include_oi: bool = True,
) -> dict[str, dict[str, int]]:
    """
    Convenience function to ingest alternative data for multiple symbols.
    
    Args:
        symbols: List of trading pairs
        days: Number of days to ingest
        include_funding: Include funding rates
        include_oi: Include open interest
        
    Returns:
        Dict mapping symbol to counts for each data type
    """
    service = AlternativeDataIngestionService()
    results = {}

    try:
        for symbol in symbols:
            results[symbol] = {}
            
            if include_funding:
                count = await service.ingest_funding_rates_days(symbol, days)
                results[symbol]["funding_rates"] = count
                logger.info(f"Ingested {count} funding rates for {symbol}")
            
            if include_oi:
                count = await service.ingest_open_interest_days(symbol, days)
                results[symbol]["open_interest"] = count
                logger.info(f"Ingested {count} OI records for {symbol}")
                
    finally:
        await service.close()

    return results
