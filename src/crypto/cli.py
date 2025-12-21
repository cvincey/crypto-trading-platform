"""Command-line interface for the crypto trading platform."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="crypto",
    help="Crypto Trading Platform - Backtest and trade cryptocurrencies",
    add_completion=False,
)

console = Console()

# Sub-commands
ingest_app = typer.Typer(help="Data ingestion commands")
backtest_app = typer.Typer(help="Backtesting commands")
strategies_app = typer.Typer(help="Strategy management commands")
config_app = typer.Typer(help="Configuration commands")
trade_app = typer.Typer(help="Live trading commands")

app.add_typer(ingest_app, name="ingest")
app.add_typer(backtest_app, name="backtest")
app.add_typer(strategies_app, name="strategies")
app.add_typer(config_app, name="config")
app.add_typer(trade_app, name="trade")


def run_async(coro):
    """Run an async function."""
    return asyncio.get_event_loop().run_until_complete(coro)


# =============================================================================
# Ingest Commands
# =============================================================================


@ingest_app.command("fetch")
def ingest_fetch(
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s", help="Trading pair"),
    interval: str = typer.Option("1h", "--interval", "-i", help="Candle interval"),
    days: int = typer.Option(30, "--days", "-d", help="Number of days to fetch"),
    exchange: str = typer.Option(
        "binance_testnet", "--exchange", "-e", help="Exchange name from config"
    ),
):
    """Fetch historical market data from an exchange."""
    from crypto.data.ingestion import DataIngestionService

    console.print(f"[bold]Fetching {symbol} {interval} for {days} days...[/bold]")

    async def _fetch():
        service = DataIngestionService(exchange_name=exchange)
        try:
            count = await service.ingest_days(symbol, interval, days)
            console.print(f"[green]✓ Ingested {count} candles[/green]")
        finally:
            await service.close()

    run_async(_fetch())


@ingest_app.command("status")
def ingest_status(
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s", help="Trading pair"),
    interval: str = typer.Option("1h", "--interval", "-i", help="Candle interval"),
):
    """Check data availability for a symbol."""
    from crypto.data.repository import CandleRepository

    async def _status():
        repo = CandleRepository()
        data_range = await repo.get_available_data_range(symbol, interval)

        if data_range:
            start, end = data_range
            days = (end - start).days
            console.print(f"[bold]{symbol} {interval}[/bold]")
            console.print(f"  Start: {start}")
            console.print(f"  End:   {end}")
            console.print(f"  Days:  {days}")
        else:
            console.print(f"[yellow]No data found for {symbol} {interval}[/yellow]")

    run_async(_status())


# =============================================================================
# Backtest Commands
# =============================================================================


@backtest_app.command("run")
def backtest_run(
    name: str = typer.Argument(..., help="Backtest name from config/backtests.yaml"),
):
    """Run a backtest from configuration."""
    from crypto.backtesting.runner import BacktestRunner

    console.print(f"[bold]Running backtest: {name}[/bold]")

    async def _run():
        runner = BacktestRunner()
        results = await runner.run(name)

        for result in results:
            console.print("")
            result.print_report()

        # Show comparison table
        if len(results) > 1:
            console.print("")
            console.print("[bold]Strategy Comparison:[/bold]")
            df = runner.compare(results)
            table = Table()
            for col in df.columns:
                table.add_column(col)
            for _, row in df.iterrows():
                table.add_row(*[f"{v:.2f}" if isinstance(v, float) else str(v) for v in row])
            console.print(table)

    run_async(_run())


@backtest_app.command("run-all")
def backtest_run_all():
    """Run all configured backtests."""
    from crypto.backtesting.runner import BacktestRunner

    console.print("[bold]Running all backtests...[/bold]")

    async def _run():
        runner = BacktestRunner()
        all_results = await runner.run_all()

        for name, results in all_results.items():
            console.print(f"\n[bold]{name}[/bold]")
            for result in results:
                console.print(
                    f"  {result.strategy_name}: "
                    f"{result.metrics.total_return_pct:.2f}% "
                    f"(Sharpe: {result.metrics.sharpe_ratio:.2f})"
                )

    run_async(_run())


@backtest_app.command("quick")
def backtest_quick(
    strategy: str = typer.Argument(..., help="Strategy name from config"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s"),
    interval: str = typer.Option("1h", "--interval", "-i"),
    days: int = typer.Option(30, "--days", "-d"),
    capital: float = typer.Option(10000.0, "--capital", "-c"),
):
    """Quick backtest a single strategy."""
    from crypto.backtesting.runner import run_quick_backtest

    console.print(f"[bold]Quick backtest: {strategy} on {symbol}[/bold]")

    async def _run():
        try:
            result = await run_quick_backtest(
                strategy_name=strategy,
                symbol=symbol,
                interval=interval,
                days=days,
                initial_capital=Decimal(str(capital)),
            )
            result.print_report()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    run_async(_run())


# =============================================================================
# Strategy Commands
# =============================================================================


@strategies_app.command("list")
def strategies_list():
    """List all available strategies."""
    # Import strategies to trigger registration
    from crypto.strategies import technical, statistical, momentum, ml
    from crypto.strategies.registry import strategy_registry

    table = Table(title="Available Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Class", style="green")
    table.add_column("Description")

    for name, meta in strategy_registry.list_with_metadata().items():
        table.add_row(
            name,
            meta.get("class", ""),
            meta.get("description", ""),
        )

    console.print(table)


@strategies_app.command("show")
def strategies_show(name: str = typer.Argument(..., help="Strategy name")):
    """Show strategy configuration."""
    from crypto.config.settings import get_settings

    settings = get_settings()
    try:
        config = settings.get_strategy(name)
        console.print(f"[bold]{name}[/bold]")
        console.print(f"  Type:     {config.type}")
        console.print(f"  Interval: {config.interval}")
        console.print(f"  Symbols:  {', '.join(config.symbols)}")
        console.print(f"  Params:   {config.params}")
        console.print(f"  Enabled:  {config.enabled}")
    except KeyError:
        console.print(f"[red]Strategy '{name}' not found in config[/red]")


# =============================================================================
# Config Commands
# =============================================================================


@config_app.command("validate")
def config_validate():
    """Validate all configuration files."""
    from crypto.config.settings import get_settings

    try:
        settings = get_settings()
        console.print("[green]✓ settings.yaml valid[/green]")
        
        _ = settings.exchanges
        console.print("[green]✓ exchanges.yaml valid[/green]")
        
        _ = settings.strategies
        console.print("[green]✓ strategies.yaml valid[/green]")
        
        _ = settings.backtests
        console.print("[green]✓ backtests.yaml valid[/green]")
        
        console.print("\n[bold green]All configuration files are valid![/bold green]")
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")


@config_app.command("show")
def config_show(section: str = typer.Argument("all", help="Config section to show")):
    """Show configuration values."""
    from crypto.config.settings import get_settings

    settings = get_settings()

    if section in ("all", "database"):
        console.print("[bold]Database:[/bold]")
        console.print(f"  Host: {settings.database.host}")
        console.print(f"  Port: {settings.database.port}")
        console.print(f"  Name: {settings.database.name}")

    if section in ("all", "trading"):
        console.print("[bold]Trading:[/bold]")
        console.print(f"  Commission: {settings.trading.default_commission}")
        console.print(f"  Slippage:   {settings.trading.default_slippage}")


# =============================================================================
# Trade Commands
# =============================================================================


@trade_app.command("start")
def trade_start(
    strategy: str = typer.Argument(..., help="Strategy name from config"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s"),
    interval: str = typer.Option("1h", "--interval", "-i"),
    exchange: str = typer.Option("binance_testnet", "--exchange", "-e"),
    paper: bool = typer.Option(True, "--paper/--live", help="Paper trading mode"),
    capital: float = typer.Option(10000.0, "--capital", "-c"),
):
    """Start live trading with a strategy."""
    from crypto.trading.live import start_trading

    mode = "PAPER" if paper else "LIVE"
    console.print(
        f"[bold]Starting {mode} trading: {strategy} on {symbol}[/bold]"
    )

    if not paper:
        confirm = typer.confirm(
            "⚠️  You are about to start LIVE trading with real money. Continue?"
        )
        if not confirm:
            raise typer.Abort()

    async def _trade():
        trader = await start_trading(
            strategy_name=strategy,
            symbol=symbol,
            interval=interval,
            exchange_name=exchange,
            paper=paper,
            initial_capital=Decimal(str(capital)),
        )

        console.print("[green]Trading started. Press Ctrl+C to stop.[/green]")

        try:
            while True:
                await asyncio.sleep(60)
                status = trader.get_status()
                console.print(f"Status: {status}")
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping trader...[/yellow]")
            await trader.stop()
            console.print("[green]Trader stopped.[/green]")

    run_async(_trade())


# =============================================================================
# Main Entry Point
# =============================================================================


@app.command()
def version():
    """Show version information."""
    from crypto import __version__
    console.print(f"Crypto Trading Platform v{__version__}")


@app.callback()
def main():
    """
    Crypto Trading Platform
    
    A config-first, extensible platform for backtesting and trading cryptocurrencies.
    """
    pass


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
