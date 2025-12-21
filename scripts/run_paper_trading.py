#!/usr/bin/env python3
"""
Paper Trading Launcher.

Starts paper trading with winning strategies from Phase 2 validation.
Uses live market data but simulated execution (no real money at risk).

Configuration: config/paper_trading.yaml

Usage:
    python scripts/run_paper_trading.py                    # Start paper trading
    python scripts/run_paper_trading.py --strategy eth_btc_ratio_optimized
    python scripts/run_paper_trading.py --dry-run          # Show config without starting
    python scripts/run_paper_trading.py --status           # Show current status
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto.exchanges.registry import get_exchange
from crypto.strategies.registry import strategy_registry
from crypto.trading.live import LiveTrader
from crypto.trading.risk import RiskLimits

# Import strategies to register them
from crypto.strategies import cross_symbol, hybrid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()

# State file for persistence
STATE_FILE = Path("notes/test_logs/paper_trading_state.json")
LOG_FILE = Path("notes/test_logs/paper_trading.log")


def load_config() -> dict:
    """Load paper trading configuration."""
    config_path = Path("config/paper_trading.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)["paper_trading"]


def setup_logging(config: dict) -> None:
    """Configure logging based on config."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    
    # Create log directory
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Add file handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)


def get_risk_limits(config: dict) -> RiskLimits:
    """Create RiskLimits from config."""
    risk_config = config.get("risk_limits", {})
    return RiskLimits(
        max_position_size=Decimal(str(risk_config.get("max_position_pct", 0.25))),
        max_single_trade=Decimal(str(risk_config.get("max_position_pct", 0.25))),
        max_daily_loss=Decimal(str(risk_config.get("daily_loss_limit_pct", 0.05))),
        max_drawdown=Decimal(str(risk_config.get("max_drawdown_pct", 0.15))),
        max_open_positions=risk_config.get("max_positions", 3),
    )


def save_state(trader: LiveTrader, strategy_name: str) -> None:
    """Save trading state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy_name,
        "status": trader.get_status(),
    }
    
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_state() -> dict | None:
    """Load saved state if exists."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def show_status() -> None:
    """Show current paper trading status."""
    state = load_state()
    
    if state is None:
        console.print("[yellow]No paper trading session found[/yellow]")
        return
    
    console.print(Panel(
        f"[bold]Paper Trading Status[/bold]\n\n"
        f"Strategy: {state.get('strategy', 'N/A')}\n"
        f"Last Update: {state.get('timestamp', 'N/A')}\n"
        f"Running: {state.get('status', {}).get('running', False)}",
        expand=False,
    ))
    
    status = state.get("status", {})
    risk = status.get("risk", {})
    
    if risk:
        table = Table(title="Risk Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        for key, value in risk.items():
            table.add_row(key, str(value))
        
        console.print(table)


def show_config(config: dict) -> None:
    """Display configuration without starting."""
    console.print(Panel("[bold]Paper Trading Configuration[/bold]", expand=False))
    
    # Strategies
    strategies = config.get("strategies", {})
    table = Table(title="Configured Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Symbol")
    table.add_column("Enabled")
    
    for name, strat_config in strategies.items():
        enabled = "✓" if strat_config.get("enabled", True) else "✗"
        table.add_row(
            name,
            strat_config.get("type", "N/A"),
            strat_config.get("symbol", "N/A"),
            enabled,
        )
    
    console.print(table)
    
    # Risk limits
    risk = config.get("risk_limits", {})
    console.print("\n[bold]Risk Limits:[/bold]")
    for key, value in risk.items():
        console.print(f"  {key}: {value}")
    
    console.print(f"\n[bold]Initial Capital:[/bold] ${config.get('initial_capital', 10000)}")
    console.print(f"[bold]Exchange:[/bold] {config.get('exchange', 'binance_testnet')}")


async def run_paper_trading(strategy_name: str | None = None) -> None:
    """Run paper trading with specified strategy."""
    config = load_config()
    setup_logging(config)
    
    # Get strategy config
    strategies = config.get("strategies", {})
    
    if strategy_name:
        if strategy_name not in strategies:
            console.print(f"[red]Strategy '{strategy_name}' not found in config[/red]")
            console.print(f"Available: {list(strategies.keys())}")
            return
        strat_config = strategies[strategy_name]
    else:
        # Find first enabled strategy
        for name, strat_config in strategies.items():
            if strat_config.get("enabled", True):
                strategy_name = name
                break
        else:
            console.print("[red]No enabled strategies found[/red]")
            return
    
    console.print(Panel(
        f"[bold blue]Starting Paper Trading[/bold blue]\n\n"
        f"Strategy: {strategy_name}\n"
        f"Type: {strat_config.get('type')}\n"
        f"Symbol: {strat_config.get('symbol')}\n"
        f"Interval: {strat_config.get('interval', '1h')}",
        expand=False,
    ))
    
    # Create strategy
    try:
        strategy = strategy_registry.create(
            strat_config.get("type"),
            **strat_config.get("params", {}),
        )
        console.print(f"[green]✓ Strategy created: {strategy.name}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to create strategy: {e}[/red]")
        return
    
    # Get exchange
    exchange_name = config.get("exchange", "binance_testnet")
    try:
        exchange = get_exchange(exchange_name)
        console.print(f"[green]✓ Exchange connected: {exchange_name}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to connect exchange: {e}[/red]")
        return
    
    # Create trader
    risk_limits = get_risk_limits(config)
    initial_capital = Decimal(str(config.get("initial_capital", 10000)))
    
    trader = LiveTrader(
        exchange=exchange,
        strategy=strategy,
        symbol=strat_config.get("symbol", "ETHUSDT"),
        interval=strat_config.get("interval", "1h"),
        paper=True,
        initial_capital=initial_capital,
        risk_limits=risk_limits,
    )
    
    # Add logging callback
    def log_event(event: dict[str, Any]) -> None:
        event_type = event.get("event", "unknown")
        logger.info(f"Event: {event_type} - {event}")
        
        if event_type == "trade":
            console.print(
                f"[bold]TRADE:[/bold] {event.get('side')} "
                f"{event.get('quantity')} @ {event.get('price')}"
            )
        elif event_type == "signal":
            console.print(
                f"[dim]Signal: {event.get('signal')} @ {event.get('price')}[/dim]"
            )
    
    trader.add_callback(log_event)
    
    # Handle shutdown
    shutdown_event = asyncio.Event()
    
    def handle_shutdown(sig, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Start trading
    console.print("\n[bold green]Paper trading started![/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    
    # Run in background and monitor
    trading_task = asyncio.create_task(trader.start())
    
    try:
        # Periodically save state and show status
        while not shutdown_event.is_set():
            save_state(trader, strategy_name)
            
            status = trader.get_status()
            console.print(
                f"[dim]{datetime.now().strftime('%H:%M:%S')} | "
                f"Signal: {status.get('last_signal', 'N/A')} | "
                f"Buffer: {status.get('candle_buffer_size', 0)} candles[/dim]"
            )
            
            await asyncio.sleep(60)  # Update every minute
            
    except asyncio.CancelledError:
        pass
    finally:
        await trader.stop()
        save_state(trader, strategy_name)
        console.print("[green]Paper trading stopped[/green]")


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    config = load_config()
    
    if args.status:
        show_status()
        return
    
    if args.dry_run:
        show_config(config)
        return
    
    await run_paper_trading(args.strategy)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run paper trading")
    
    parser.add_argument(
        "--strategy",
        type=str,
        help="Specific strategy to run",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show config without starting",
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
