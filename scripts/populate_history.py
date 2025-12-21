#!/usr/bin/env python3
"""
Populate database with historical market data for backtesting.

This script fetches historical OHLCV data from Binance public API
and stores it in the database for all symbols/intervals needed by backtests.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal

import httpx

from crypto.core.types import Candle
from crypto.data.database import init_db, close_db
from crypto.data.repository import CandleRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Binance public API (no auth required for market data)
BINANCE_API = "https://api.binance.com"

# Data requirements based on backtests.yaml
DATA_REQUIREMENTS = [
    # btc_strategies_2024: BTCUSDT 1h from 2024-01-01 to 2024-12-01
    {"symbol": "BTCUSDT", "interval": "1h", "start": "2024-01-01", "end": "2024-12-21"},
    # ml_evaluation: BTCUSDT 1d from 2022-01-01 to 2024-01-01
    {"symbol": "BTCUSDT", "interval": "1d", "start": "2022-01-01", "end": "2024-12-21"},
    # multi_asset_momentum: ETHUSDT, SOLUSDT 1d from 2023-06-01 to 2024-06-01
    {"symbol": "ETHUSDT", "interval": "1d", "start": "2023-06-01", "end": "2024-12-21"},
    {"symbol": "SOLUSDT", "interval": "1d", "start": "2023-06-01", "end": "2024-12-21"},
]


async def fetch_klines(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> list[list]:
    """Fetch klines from Binance API."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    response = await client.get(f"{BINANCE_API}/api/v3/klines", params=params)
    response.raise_for_status()
    return response.json()


def parse_kline(symbol: str, interval: str, kline: list) -> Candle:
    """Parse a Binance kline into a Candle object."""
    return Candle(
        symbol=symbol,
        interval=interval,
        open_time=datetime.utcfromtimestamp(kline[0] / 1000),
        open=Decimal(str(kline[1])),
        high=Decimal(str(kline[2])),
        low=Decimal(str(kline[3])),
        close=Decimal(str(kline[4])),
        volume=Decimal(str(kline[5])),
        close_time=datetime.utcfromtimestamp(kline[6] / 1000),
        quote_volume=Decimal(str(kline[7])),
        trades=int(kline[8]),
    )


def interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    unit = interval[-1]
    value = int(interval[:-1])
    multipliers = {
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
        "w": 7 * 24 * 60 * 60 * 1000,
    }
    return value * multipliers.get(unit, 60 * 1000)


async def fetch_historical_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> list[Candle]:
    """Fetch all historical data for a symbol/interval range."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    interval_ms = interval_to_ms(interval)
    
    all_candles = []
    current_start = start_ms
    
    async with httpx.AsyncClient(timeout=30) as client:
        while current_start < end_ms:
            try:
                klines = await fetch_klines(
                    client, symbol, interval, current_start, end_ms
                )
                
                if not klines:
                    break
                
                for kline in klines:
                    candle = parse_kline(symbol, interval, kline)
                    all_candles.append(candle)
                
                # Move to next batch
                last_close_time = klines[-1][6]
                current_start = last_close_time + 1
                
                logger.info(
                    f"  Fetched {len(klines)} candles, total: {len(all_candles)}"
                )
                
                # Rate limiting
                await asyncio.sleep(0.2)
                
                if len(klines) < 1000:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol} {interval}: {e}")
                await asyncio.sleep(1)
                continue
    
    return all_candles


async def populate_data():
    """Main function to populate historical data."""
    logger.info("=" * 60)
    logger.info("Populating database with historical market data")
    logger.info("=" * 60)
    
    # Initialize database
    await init_db()
    repo = CandleRepository()
    
    total_candles = 0
    
    for req in DATA_REQUIREMENTS:
        symbol = req["symbol"]
        interval = req["interval"]
        start = req["start"]
        end = req["end"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Fetching {symbol} {interval} from {start} to {end}")
        logger.info("=" * 60)
        
        # Check if we already have data
        existing_range = await repo.get_available_data_range(symbol, interval)
        if existing_range:
            ex_start, ex_end = existing_range
            logger.info(f"Existing data: {ex_start} to {ex_end}")
        
        # Fetch data
        candles = await fetch_historical_data(symbol, interval, start, end)
        
        if candles:
            # Deduplicate candles by (symbol, interval, open_time)
            seen = set()
            unique_candles = []
            for c in candles:
                key = (c.symbol, c.interval, c.open_time)
                if key not in seen:
                    seen.add(key)
                    unique_candles.append(c)
            
            logger.info(f"Unique candles: {len(unique_candles)} (deduped from {len(candles)})")
            
            # Save to database in batches (PostgreSQL has 32767 param limit)
            # With 11 fields per candle, max ~2900 per batch
            BATCH_SIZE = 2000
            saved = 0
            for i in range(0, len(unique_candles), BATCH_SIZE):
                batch = unique_candles[i:i + BATCH_SIZE]
                saved += await repo.save_candles(batch)
                logger.info(f"  Batch saved: {len(batch)} candles")
            total_candles += saved
            logger.info(f"✓ Saved {saved} candles for {symbol} {interval}")
        else:
            logger.warning(f"No data fetched for {symbol} {interval}")
    
    await close_db()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE! Total candles saved: {total_candles}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(populate_data())
