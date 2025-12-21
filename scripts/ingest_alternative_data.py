#!/usr/bin/env python3
"""
Ingest alternative data: funding rates and open interest.

This script fetches historical funding rates and open interest data
from Binance Futures API and stores them in the database.

Usage:
    python scripts/ingest_alternative_data.py --days 180
    python scripts/ingest_alternative_data.py --symbols BTCUSDT ETHUSDT --days 365
    python scripts/ingest_alternative_data.py --funding-only --days 90
    python scripts/ingest_alternative_data.py --oi-only --days 90
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.data.ingestion import AlternativeDataIngestionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Default symbols for alternative data
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "DOTUSDT",
]


async def ingest_symbol(
    service: AlternativeDataIngestionService,
    symbol: str,
    days: int,
    include_funding: bool,
    include_oi: bool,
) -> dict[str, int]:
    """Ingest data for a single symbol."""
    results = {}
    
    if include_funding:
        count = await service.ingest_funding_rates_days(symbol, days)
        results["funding_rates"] = count
    
    if include_oi:
        count = await service.ingest_open_interest_days(symbol, days)
        results["open_interest"] = count
    
    return results


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    symbols = args.symbols or DEFAULT_SYMBOLS
    days = args.days
    include_funding = not args.oi_only
    include_oi = not args.funding_only
    
    # Display configuration
    console.print(Panel(
        f"[bold blue]Alternative Data Ingestion[/bold blue]\n\n"
        f"Symbols: {len(symbols)}\n"
        f"Days: {days}\n"
        f"Funding Rates: {'Yes' if include_funding else 'No'}\n"
        f"Open Interest: {'Yes' if include_oi else 'No'}",
        expand=False,
    ))
    
    service = AlternativeDataIngestionService()
    results = {}
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting...", total=len(symbols))
            
            for symbol in symbols:
                progress.update(task, description=f"Processing {symbol}...")
                
                try:
                    result = await ingest_symbol(
                        service, symbol, days, include_funding, include_oi
                    )
                    results[symbol] = result
                except Exception as e:
                    console.print(f"[red]Error processing {symbol}: {e}[/red]")
                    results[symbol] = {"error": str(e)}
                
                progress.advance(task)
    
    finally:
        await service.close()
    
    # Display results
    console.print("\n")
    table = Table(title="Ingestion Results")
    table.add_column("Symbol", style="cyan")
    if include_funding:
        table.add_column("Funding Rates", justify="right")
    if include_oi:
        table.add_column("Open Interest", justify="right")
    table.add_column("Status")
    
    total_funding = 0
    total_oi = 0
    
    for symbol, result in results.items():
        if "error" in result:
            table.add_row(
                symbol,
                *(["-"] * (int(include_funding) + int(include_oi))),
                f"[red]Error: {result['error'][:30]}[/red]"
            )
        else:
            row = [symbol]
            if include_funding:
                count = result.get("funding_rates", 0)
                total_funding += count
                row.append(str(count))
            if include_oi:
                count = result.get("open_interest", 0)
                total_oi += count
                row.append(str(count))
            row.append("[green]OK[/green]")
            table.add_row(*row)
    
    # Add totals row
    total_row = ["[bold]TOTAL[/bold]"]
    if include_funding:
        total_row.append(f"[bold]{total_funding}[/bold]")
    if include_oi:
        total_row.append(f"[bold]{total_oi}[/bold]")
    total_row.append("")
    table.add_row(*total_row)
    
    console.print(table)
    console.print(f"\n[green]Ingestion complete![/green]")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest alternative data (funding rates, open interest)"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to ingest (default: top 8 by volume)",
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days of history to ingest (default: 180)",
    )
    
    parser.add_argument(
        "--funding-only",
        action="store_true",
        help="Only ingest funding rates",
    )
    
    parser.add_argument(
        "--oi-only",
        action="store_true",
        help="Only ingest open interest",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
