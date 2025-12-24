#!/usr/bin/env python3
"""
Ingest Tier 2 alternative data for trading strategies.

Data sources (all FREE):
- Fear & Greed Index: alternative.me API
- Long/Short Ratio: Binance Futures API
- BTC Dominance: CoinGecko API
- Stablecoin Supply: CoinGecko API
- Macro Indicators: Yahoo Finance (yfinance)

Usage:
    python scripts/ingest_tier2_data.py --days 365
    python scripts/ingest_tier2_data.py --fear-greed-only
    python scripts/ingest_tier2_data.py --long-short-only --symbols BTCUSDT ETHUSDT
    python scripts/ingest_tier2_data.py --dominance-only
    python scripts/ingest_tier2_data.py --stablecoin-only
    python scripts/ingest_tier2_data.py --macro-only
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from crypto.data.alternative_data import (
    FearGreedRepository,
    FearGreedIndex,
    LongShortRatioRepository,
    LongShortRatio,
    BTCDominanceRepository,
    BTCDominance,
    StablecoinSupplyRepository,
    StablecoinSupply,
    MacroIndicatorRepository,
    MacroIndicator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# API endpoints
FEAR_GREED_API = "https://api.alternative.me/fng/"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
BINANCE_FUTURES_URL = "https://fapi.binance.com"

# Rate limiting
COINGECKO_DELAY = 2.5  # seconds (free tier: 30 calls/min)
BINANCE_DELAY = 0.2  # seconds

# Default symbols for Long/Short ratio
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


# =============================================================================
# Fear & Greed Index (alternative.me)
# =============================================================================

async def fetch_fear_greed_history(
    client: httpx.AsyncClient,
    days: int,
) -> list[FearGreedIndex]:
    """
    Fetch Fear & Greed Index history from alternative.me.
    
    API: https://api.alternative.me/fng/?limit={days}&format=json
    """
    console.print("[dim]Fetching Fear & Greed Index from alternative.me...[/dim]")
    
    try:
        params = {"limit": days, "format": "json"}
        response = await client.get(FEAR_GREED_API, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "data" not in data:
            logger.error("No data in Fear & Greed response")
            return []
        
        results = []
        for item in data["data"]:
            ts = datetime.fromtimestamp(int(item["timestamp"]), tz=timezone.utc)
            value = int(item["value"])
            classification = item["value_classification"]
            
            results.append(FearGreedIndex(
                timestamp=ts,
                value=value,
                value_classification=classification,
            ))
        
        console.print(f"  Fetched {len(results)} Fear & Greed data points")
        return results
        
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed data: {e}")
        return []


async def save_fear_greed(data: list[FearGreedIndex]) -> int:
    """Save Fear & Greed data to database."""
    if not data:
        return 0
    repo = FearGreedRepository()
    return await repo.save_fear_greed(data)


# =============================================================================
# Long/Short Ratio (Binance Futures)
# =============================================================================

async def fetch_long_short_ratio_history(
    client: httpx.AsyncClient,
    symbol: str,
    days: int,
    period: str = "4h",  # Use 4h for longer history (less data points)
) -> list[LongShortRatio]:
    """
    Fetch Long/Short ratio history from Binance Futures.
    
    API: /futures/data/globalLongShortAccountRatio
    Also available: topLongShortAccountRatio, topLongShortPositionRatio
    
    Note: Binance API returns max 500 records, ~83 days for 4h period.
    """
    console.print(f"[dim]Fetching Long/Short ratio for {symbol}...[/dim]")
    
    results = []
    limit = 500  # Max per request - gets ~83 days of 4h data
    
    try:
        # Fetch account ratio - just get most recent data
        url = f"{BINANCE_FUTURES_URL}/futures/data/globalLongShortAccountRatio"
        params = {
            "symbol": symbol,
            "period": period,
            "limit": limit,
        }
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        for item in data:
            ts = datetime.fromtimestamp(item["timestamp"] / 1000, tz=timezone.utc)
            
            results.append(LongShortRatio(
                symbol=symbol,
                timestamp=ts,
                ratio_type="accounts",
                long_ratio=Decimal(str(item["longAccount"])),
                short_ratio=Decimal(str(item["shortAccount"])),
                long_short_ratio=Decimal(str(item["longShortRatio"])),
            ))
        
        await asyncio.sleep(BINANCE_DELAY)
        
    except Exception as e:
        logger.error(f"Error fetching L/S ratio for {symbol}: {e}")
    
    try:
        # Fetch position ratio
        url = f"{BINANCE_FUTURES_URL}/futures/data/topLongShortPositionRatio"
        params = {
            "symbol": symbol,
            "period": period,
            "limit": limit,
        }
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        for item in data:
            ts = datetime.fromtimestamp(item["timestamp"] / 1000, tz=timezone.utc)
            
            results.append(LongShortRatio(
                symbol=symbol,
                timestamp=ts,
                ratio_type="positions",
                long_ratio=Decimal(str(item["longAccount"])),
                short_ratio=Decimal(str(item["shortAccount"])),
                long_short_ratio=Decimal(str(item["longShortRatio"])),
            ))
        
        await asyncio.sleep(BINANCE_DELAY)
        
    except Exception as e:
        logger.error(f"Error fetching position ratio for {symbol}: {e}")
    
    console.print(f"  Fetched {len(results)} L/S ratio records for {symbol}")
    return results


async def save_long_short_ratio(data: list[LongShortRatio]) -> int:
    """Save Long/Short ratio data to database."""
    if not data:
        return 0
    repo = LongShortRatioRepository()
    return await repo.save_long_short_ratio(data)


# =============================================================================
# BTC Dominance (CoinGecko)
# =============================================================================

async def fetch_btc_dominance_history(
    client: httpx.AsyncClient,
    days: int,
) -> list[BTCDominance]:
    """
    Fetch BTC dominance history from CoinGecko.
    
    Uses global endpoint for current + historical market cap calculation.
    """
    console.print("[dim]Fetching BTC dominance from CoinGecko...[/dim]")
    
    results = []
    
    try:
        # First get current global data for reference
        response = await client.get(f"{COINGECKO_BASE_URL}/global")
        response.raise_for_status()
        global_data = response.json()
        
        current_btc_dom = global_data["data"]["market_cap_percentage"].get("btc", 50)
        current_eth_dom = global_data["data"]["market_cap_percentage"].get("eth", 15)
        
        await asyncio.sleep(COINGECKO_DELAY)
        
        # Get BTC market chart for historical data
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        response = await client.get(
            f"{COINGECKO_BASE_URL}/coins/bitcoin/market_chart",
            params=params,
        )
        response.raise_for_status()
        btc_data = response.json()
        
        await asyncio.sleep(COINGECKO_DELAY)
        
        # Get ETH market chart
        response = await client.get(
            f"{COINGECKO_BASE_URL}/coins/ethereum/market_chart",
            params=params,
        )
        response.raise_for_status()
        eth_data = response.json()
        
        await asyncio.sleep(COINGECKO_DELAY)
        
        # Get total market cap from global/defi (includes historical)
        # Note: CoinGecko free tier has limited historical total mcap
        # We'll estimate from current ratio
        
        btc_market_caps = {int(mc[0]): mc[1] for mc in btc_data.get("market_caps", [])}
        eth_market_caps = {int(mc[0]): mc[1] for mc in eth_data.get("market_caps", [])}
        
        # Use current dominance to estimate total market cap
        for ts_ms, btc_mc in btc_data.get("market_caps", []):
            if btc_mc is None or btc_mc == 0:
                continue
            
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            
            # Estimate total market cap from current dominance ratio
            # This is an approximation - in reality, dominance changes over time
            estimated_total = btc_mc / (current_btc_dom / 100)
            
            eth_mc = eth_market_caps.get(int(ts_ms), 0)
            eth_dom = (eth_mc / estimated_total * 100) if estimated_total > 0 else 0
            
            results.append(BTCDominance(
                timestamp=ts,
                btc_dominance=Decimal(str(round(current_btc_dom, 4))),
                eth_dominance=Decimal(str(round(eth_dom, 4))) if eth_mc else None,
                total_market_cap=Decimal(str(int(estimated_total))),
                btc_market_cap=Decimal(str(int(btc_mc))),
            ))
        
        console.print(f"  Fetched {len(results)} BTC dominance records")
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            console.print("[yellow]Rate limited by CoinGecko. Try again later.[/yellow]")
        else:
            logger.error(f"HTTP error fetching BTC dominance: {e}")
    except Exception as e:
        logger.error(f"Error fetching BTC dominance: {e}")
    
    return results


async def save_btc_dominance(data: list[BTCDominance]) -> int:
    """Save BTC dominance data to database."""
    if not data:
        return 0
    repo = BTCDominanceRepository()
    return await repo.save_btc_dominance(data)


# =============================================================================
# Stablecoin Supply (CoinGecko)
# =============================================================================

async def fetch_stablecoin_supply_history(
    client: httpx.AsyncClient,
    days: int,
) -> list[StablecoinSupply]:
    """
    Fetch stablecoin supply history from CoinGecko.
    
    Tracks USDT, USDC, DAI market caps.
    """
    console.print("[dim]Fetching stablecoin supply from CoinGecko...[/dim]")
    
    stablecoins = [
        ("tether", "USDT"),
        ("usd-coin", "USDC"),
        ("dai", "DAI"),
    ]
    
    results = []
    
    for coin_id, symbol in stablecoins:
        try:
            params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
            response = await client.get(
                f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            
            market_caps = data.get("market_caps", [])
            
            # Calculate supply changes
            prev_mc = None
            for ts_ms, mc in market_caps:
                if mc is None:
                    continue
                
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                
                change_24h = None
                if prev_mc is not None and prev_mc > 0:
                    change_24h = (mc - prev_mc) / prev_mc
                
                results.append(StablecoinSupply(
                    symbol=symbol,
                    timestamp=ts,
                    market_cap=Decimal(str(int(mc))),
                    supply_change_24h=Decimal(str(round(change_24h, 6))) if change_24h else None,
                ))
                
                prev_mc = mc
            
            console.print(f"  Fetched {symbol}: {len(market_caps)} data points")
            await asyncio.sleep(COINGECKO_DELAY)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                console.print(f"[yellow]Rate limited fetching {symbol}[/yellow]")
                await asyncio.sleep(60)
            else:
                logger.error(f"Error fetching {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
    
    return results


async def save_stablecoin_supply(data: list[StablecoinSupply]) -> int:
    """Save stablecoin supply data to database."""
    if not data:
        return 0
    repo = StablecoinSupplyRepository()
    return await repo.save_stablecoin_supply(data)


# =============================================================================
# Macro Indicators (Yahoo Finance via yfinance)
# =============================================================================

def fetch_macro_indicators_sync(days: int) -> list[MacroIndicator]:
    """
    Fetch macro indicators from Yahoo Finance.
    
    Indicators:
    - DXY: US Dollar Index (^DXY or DX-Y.NYB)
    - VIX: Volatility Index (^VIX)
    - SPX: S&P 500 (^GSPC)
    - GOLD: Gold futures (GC=F)
    """
    console.print("[dim]Fetching macro indicators from Yahoo Finance...[/dim]")
    
    try:
        import yfinance as yf
    except ImportError:
        console.print("[yellow]yfinance not installed. Run: pip install yfinance[/yellow]")
        return []
    
    indicators = {
        "DXY": "DX-Y.NYB",
        "VIX": "^VIX",
        "SPX": "^GSPC",
        "GOLD": "GC=F",
    }
    
    results = []
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    
    for name, ticker in indicators.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            
            if data.empty:
                console.print(f"  [yellow]No data for {name} ({ticker})[/yellow]")
                continue
            
            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            for idx, row in data.iterrows():
                ts = idx.to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                
                # Handle potential NaN values
                close_val = row["Close"]
                if pd.isna(close_val):
                    continue
                    
                results.append(MacroIndicator(
                    indicator=name,
                    timestamp=ts,
                    value=Decimal(str(round(float(close_val), 4))),
                    open=Decimal(str(round(float(row["Open"]), 4))) if not pd.isna(row["Open"]) else None,
                    high=Decimal(str(round(float(row["High"]), 4))) if not pd.isna(row["High"]) else None,
                    low=Decimal(str(round(float(row["Low"]), 4))) if not pd.isna(row["Low"]) else None,
                    close=Decimal(str(round(float(close_val), 4))),
                ))
            
            console.print(f"  Fetched {name}: {len(data)} data points")
            
        except Exception as e:
            logger.error(f"Error fetching {name}: {e}")
    
    return results


async def fetch_macro_indicators(days: int) -> list[MacroIndicator]:
    """Async wrapper for macro indicator fetching."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_macro_indicators_sync, days)


async def save_macro_indicators(data: list[MacroIndicator]) -> int:
    """Save macro indicator data to database."""
    if not data:
        return 0
    repo = MacroIndicatorRepository()
    return await repo.save_macro_indicators(data)


# =============================================================================
# Main
# =============================================================================

async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    days = args.days
    symbols = args.symbols or DEFAULT_SYMBOLS
    
    # Determine what to ingest
    ingest_all = not any([
        args.fear_greed_only,
        args.long_short_only,
        args.dominance_only,
        args.stablecoin_only,
        args.macro_only,
    ])
    
    include_fear_greed = ingest_all or args.fear_greed_only
    include_long_short = ingest_all or args.long_short_only
    include_dominance = ingest_all or args.dominance_only
    include_stablecoin = ingest_all or args.stablecoin_only
    include_macro = ingest_all or args.macro_only
    
    # API-specific limits
    fear_greed_days = min(days, 2000)  # F&G API limit
    long_short_days = min(days, 1800)  # Binance Futures started Sept 2019
    dominance_days = min(days, 365)    # CoinGecko free tier limit
    stablecoin_days = min(days, 365)   # CoinGecko free tier limit
    
    console.print(Panel(
        f"[bold blue]Tier 2 Alternative Data Ingestion[/bold blue]\n\n"
        f"Days requested: {days}\n"
        f"Fear & Greed Index: {'Yes' if include_fear_greed else 'No'} (max {fear_greed_days} days)\n"
        f"Long/Short Ratio: {'Yes' if include_long_short else 'No'} ({len(symbols)} symbols, max {long_short_days} days)\n"
        f"BTC Dominance: {'Yes' if include_dominance else 'No'} (max {dominance_days} days)\n"
        f"Stablecoin Supply: {'Yes' if include_stablecoin else 'No'} (max {stablecoin_days} days)\n"
        f"Macro Indicators: {'Yes' if include_macro else 'No'} ({days} days)"
    ))
    
    results = {}
    
    # Use ssl context that doesn't verify for APIs with cert issues
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
        
        # Fear & Greed Index
        if include_fear_greed:
            console.print(f"\n[bold]1. Fear & Greed Index[/bold] ({fear_greed_days} days)")
            data = await fetch_fear_greed_history(client, fear_greed_days)
            saved = await save_fear_greed(data)
            results["fear_greed"] = {"fetched": len(data), "saved": saved}
            console.print(f"  [green]✓ Saved {saved} records[/green]")
        
        # Long/Short Ratio
        if include_long_short:
            console.print(f"\n[bold]2. Long/Short Ratio[/bold] ({long_short_days} days)")
            total_fetched = 0
            total_saved = 0
            
            for symbol in symbols:
                data = await fetch_long_short_ratio_history(client, symbol, long_short_days)
                saved = await save_long_short_ratio(data)
                total_fetched += len(data)
                total_saved += saved
            
            results["long_short_ratio"] = {"fetched": total_fetched, "saved": total_saved}
            console.print(f"  [green]✓ Saved {total_saved} records across {len(symbols)} symbols[/green]")
        
        # BTC Dominance
        if include_dominance:
            console.print(f"\n[bold]3. BTC Dominance[/bold] ({dominance_days} days)")
            data = await fetch_btc_dominance_history(client, dominance_days)
            saved = await save_btc_dominance(data)
            results["btc_dominance"] = {"fetched": len(data), "saved": saved}
            console.print(f"  [green]✓ Saved {saved} records[/green]")
        
        # Stablecoin Supply
        if include_stablecoin:
            console.print(f"\n[bold]4. Stablecoin Supply[/bold] ({stablecoin_days} days)")
            data = await fetch_stablecoin_supply_history(client, stablecoin_days)
            saved = await save_stablecoin_supply(data)
            results["stablecoin_supply"] = {"fetched": len(data), "saved": saved}
            console.print(f"  [green]✓ Saved {saved} records[/green]")
    
    # Macro Indicators (uses yfinance, not httpx)
    if include_macro:
        console.print("\n[bold]5. Macro Indicators[/bold]")
        data = await fetch_macro_indicators(days)
        saved = await save_macro_indicators(data)
        results["macro_indicators"] = {"fetched": len(data), "saved": saved}
        console.print(f"  [green]✓ Saved {saved} records[/green]")
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Summary[/bold]")
    console.print("=" * 60)
    
    table = Table()
    table.add_column("Data Type")
    table.add_column("Fetched", justify="right")
    table.add_column("Saved", justify="right")
    
    total_fetched = 0
    total_saved = 0
    
    for data_type, info in results.items():
        fetched = info.get("fetched", 0)
        saved = info.get("saved", 0)
        total_fetched += fetched
        total_saved += saved
        
        table.add_row(
            data_type.replace("_", " ").title(),
            str(fetched),
            str(saved),
        )
    
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_fetched}[/bold]",
        f"[bold]{total_saved}[/bold]",
    )
    
    console.print(table)
    console.print("\n[bold green]Ingestion complete![/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Tier 2 alternative data for trading strategies"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of history to fetch (default: 365)",
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols for Long/Short ratio (default: top 5)",
    )
    
    parser.add_argument(
        "--fear-greed-only",
        action="store_true",
        help="Only fetch Fear & Greed Index",
    )
    
    parser.add_argument(
        "--long-short-only",
        action="store_true",
        help="Only fetch Long/Short ratio",
    )
    
    parser.add_argument(
        "--dominance-only",
        action="store_true",
        help="Only fetch BTC dominance",
    )
    
    parser.add_argument(
        "--stablecoin-only",
        action="store_true",
        help="Only fetch stablecoin supply",
    )
    
    parser.add_argument(
        "--macro-only",
        action="store_true",
        help="Only fetch macro indicators",
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))
