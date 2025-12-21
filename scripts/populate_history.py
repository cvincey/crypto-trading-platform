#!/usr/bin/env python3
"""
Populate database with historical market data for backtesting.

This script fetches historical OHLCV data from Binance public API
and stores it in the database for all symbols/intervals needed by backtests.

Usage:
    python populate_history.py                    # Fetch all Top 50
    python populate_history.py --top20            # Fetch only Top 20
    python populate_history.py --new              # Fetch only new Top 21-50
    python populate_history.py --symbols BTC ETH  # Fetch specific symbols
"""

import argparse
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

# =============================================================================
# Top 50 Cryptocurrencies (by market cap, USDT pairs on Binance)
# =============================================================================
# Top 20 (existing)
TOP_20_SYMBOLS = [
    "BTCUSDT",    # 1. Bitcoin
    "ETHUSDT",    # 2. Ethereum
    "BNBUSDT",    # 3. BNB
    "XRPUSDT",    # 4. XRP
    "SOLUSDT",    # 5. Solana
    "ADAUSDT",    # 6. Cardano
    "DOGEUSDT",   # 7. Dogecoin
    "TRXUSDT",    # 8. TRON
    "AVAXUSDT",   # 9. Avalanche
    "LINKUSDT",   # 10. Chainlink
    "DOTUSDT",    # 11. Polkadot
    "XLMUSDT",    # 12. Stellar
    "SUIUSDT",    # 13. Sui
    "LTCUSDT",    # 14. Litecoin
    "BCHUSDT",    # 15. Bitcoin Cash
    "NEARUSDT",   # 16. NEAR Protocol
    "APTUSDT",    # 17. Aptos
    "UNIUSDT",    # 18. Uniswap
    "ICPUSDT",    # 19. Internet Computer
    "ETCUSDT",    # 20. Ethereum Classic
]

# Extended Top 21-50
TOP_21_50_SYMBOLS = [
    "RENDERUSDT", # 21. Render
    "AAVEUSDT",   # 22. Aave
    "FILUSDT",    # 23. Filecoin
    "ATOMUSDT",   # 24. Cosmos
    "ARBUSDT",    # 25. Arbitrum
    "OPUSDT",     # 26. Optimism
    "INJUSDT",    # 27. Injective
    "IMXUSDT",    # 28. Immutable X
    "VETUSDT",    # 29. VeChain
    "GRTUSDT",    # 30. The Graph
    "ALGOUSDT",   # 31. Algorand
    "FTMUSDT",    # 32. Fantom
    "RUNEUSDT",   # 33. THORChain
    "SEIUSDT",    # 34. Sei
    "TIAUSDT",    # 35. Celestia
    "LDOUSDT",    # 36. Lido DAO
    "MKRUSDT",    # 37. Maker
    "THETAUSDT",  # 38. Theta Network
    "FLOKIUSDT",  # 39. FLOKI
    "BONKUSDT",   # 40. Bonk
    "JUPUSDT",    # 41. Jupiter
    "SANDUSDT",   # 42. The Sandbox
    "AXSUSDT",    # 43. Axie Infinity
    "WLDUSDT",    # 44. Worldcoin
    "SNXUSDT",    # 45. Synthetix
    "FLOWUSDT",   # 46. Flow
    "EGLDUSDT",   # 47. MultiversX
    "CHZUSDT",    # 48. Chiliz
    "APEUSDT",    # 49. ApeCoin
    "MANAUSDT",   # 50. Decentraland
]

# Combined Top 50
TOP_50_SYMBOLS = TOP_20_SYMBOLS + TOP_21_50_SYMBOLS


def build_data_requirements(
    symbols: list[str],
    intervals: list[str] = ["1h", "1d"],
    start: str = "2024-01-01",
    end: str = "2024-12-21",
) -> list[dict]:
    """Build data requirements for multiple symbols and intervals."""
    requirements = []
    for symbol in symbols:
        for interval in intervals:
            requirements.append({
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
            })
    return requirements


# Default data requirements - Top 50 with 1h and 1d intervals
# Can be overridden via CLI arguments
# Note: Binance API will return data from each coin's listing date
DATA_REQUIREMENTS = build_data_requirements(
    symbols=TOP_50_SYMBOLS,
    intervals=["1h", "1d"],
    start="2017-01-01",  # Early date - API returns from listing date
    end="2025-12-20",
)


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


async def populate_data(requirements: list[dict] | None = None):
    """Main function to populate historical data."""
    reqs = requirements if requirements is not None else DATA_REQUIREMENTS
    
    logger.info("=" * 60)
    logger.info("Populating database with historical market data")
    logger.info("=" * 60)
    
    # Initialize database
    await init_db()
    repo = CandleRepository()
    
    total_candles = 0
    
    for req in reqs:
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Populate database with historical market data"
    )
    parser.add_argument(
        "--top20", action="store_true",
        help="Fetch only Top 20 symbols"
    )
    parser.add_argument(
        "--new", action="store_true",
        help="Fetch only new symbols (Top 21-50)"
    )
    parser.add_argument(
        "--symbols", nargs="+", metavar="SYMBOL",
        help="Fetch specific symbols (e.g., BTC ETH SOL)"
    )
    parser.add_argument(
        "--interval", choices=["1h", "1d", "both"], default="both",
        help="Interval to fetch (default: both)"
    )
    parser.add_argument(
        "--start", default="2024-01-01",
        help="Start date YYYY-MM-DD (default: 2024-01-01)"
    )
    parser.add_argument(
        "--end", default="2024-12-21",
        help="End date YYYY-MM-DD (default: 2024-12-21)"
    )
    return parser.parse_args()


def get_symbols_from_args(args) -> list[str]:
    """Determine which symbols to fetch based on args."""
    if args.symbols:
        # Convert shorthand to full pair names (BTC -> BTCUSDT)
        return [
            s.upper() if s.upper().endswith("USDT") else f"{s.upper()}USDT"
            for s in args.symbols
        ]
    elif args.top20:
        return TOP_20_SYMBOLS
    elif args.new:
        return TOP_21_50_SYMBOLS
    else:
        return TOP_50_SYMBOLS


if __name__ == "__main__":
    args = parse_args()
    
    # Build requirements based on args
    symbols = get_symbols_from_args(args)
    intervals = ["1h", "1d"] if args.interval == "both" else [args.interval]
    
    # Override global DATA_REQUIREMENTS
    DATA_REQUIREMENTS = build_data_requirements(
        symbols=symbols,
        intervals=intervals,
        start=args.start,
        end=args.end,
    )
    
    logger.info(f"Symbols to fetch: {len(symbols)}")
    logger.info(f"Intervals: {intervals}")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Total jobs: {len(DATA_REQUIREMENTS)}")
    
    # Pass requirements to avoid global variable issues
    asyncio.run(populate_data(DATA_REQUIREMENTS))
