#!/usr/bin/env python3
"""
Multi-Strategy Paper Trading (Single Process).

Runs ALL enabled strategies concurrently in one process.
Designed for cloud deployment (Railway, etc.).

Usage:
    python scripts/run_all_strategies.py
"""

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

# State file for dashboard
STATE_FILE = Path("notes/test_logs/trading_state.json")

# Global state shared between strategies
GLOBAL_STATE: dict[str, Any] = {
    "start_time": None,
    "last_update": None,
    "strategies": {},
    "trades": [],
}


def save_state() -> None:
    """Save current state to file for dashboard."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_STATE["last_update"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(GLOBAL_STATE, f, indent=2, default=str)


def load_config() -> dict:
    """Load paper trading configuration."""
    config_path = Path("config/paper_trading.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)["paper_trading"]


def get_enabled_strategies(config: dict) -> list[tuple[str, dict]]:
    """Get list of enabled strategies."""
    strategies = config.get("strategies", {})
    enabled = []
    for name, strat_config in strategies.items():
        if strat_config.get("enabled", False):
            enabled.append((name, strat_config))
    return enabled


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


async def run_strategy(
    name: str,
    strat_config: dict,
    exchange,
    risk_limits: RiskLimits,
    initial_capital: Decimal,
    shutdown_event: asyncio.Event,
) -> None:
    """Run a single strategy."""
    symbol = strat_config.get("symbol", "ETHUSDT")
    interval = strat_config.get("interval", "1h")
    
    try:
        # Create strategy
        strategy = strategy_registry.create(
            strat_config.get("type"),
            **strat_config.get("params", {}),
        )
        logger.info(f"[{name}] Strategy created: {strategy.name}")
        
        # Create trader
        trader = LiveTrader(
            exchange=exchange,
            strategy=strategy,
            symbol=symbol,
            interval=interval,
            paper=True,
            initial_capital=initial_capital,
            risk_limits=risk_limits,
        )
        
        # Add logging callback
        def log_event(event: dict) -> None:
            event_type = event.get("event", "unknown")
            if event_type == "trade":
                console.print(
                    f"[bold green][{name}] TRADE:[/bold green] {event.get('side')} "
                    f"{event.get('quantity'):.4f} @ {event.get('price'):.2f}"
                )
                # Save trade to global state
                GLOBAL_STATE["trades"].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "strategy": name,
                    "symbol": symbol,
                    "side": event.get("side"),
                    "price": event.get("price"),
                    "quantity": event.get("quantity"),
                })
                save_state()
            elif event_type == "signal":
                logger.info(f"[{name}] Signal: {event.get('signal')} @ {event.get('price')}")
        
        trader.add_callback(log_event)
        
        # Start trading in background
        trading_task = asyncio.create_task(trader.start())
        
        # Wait for shutdown
        while not shutdown_event.is_set():
            status = trader.get_status()
            
            # Update global state
            GLOBAL_STATE["strategies"][name] = {
                "symbol": symbol,
                "signal": status.get("last_signal", "HOLD"),
                "buffer_size": status.get("candle_buffer_size", 0),
                "running": status.get("running", False),
            }
            save_state()
            
            logger.info(
                f"[{name}] {symbol} | Signal: {status.get('last_signal', 'N/A')} | "
                f"Buffer: {status.get('candle_buffer_size', 0)} candles"
            )
            await asyncio.sleep(60)
        
        # Stop trader
        await trader.stop()
        trading_task.cancel()
        
    except Exception as e:
        logger.error(f"[{name}] Error: {e}")


async def main():
    """Main entry point."""
    config = load_config()
    enabled = get_enabled_strategies(config)
    
    if not enabled:
        console.print("[red]No enabled strategies found[/red]")
        return
    
    # Display startup info
    table = Table(title="🚀 Starting Multi-Strategy Paper Trading")
    table.add_column("Strategy", style="cyan")
    table.add_column("Symbol", style="yellow")
    table.add_column("Type")
    table.add_column("Sharpe")
    
    for name, strat_config in enabled:
        validation = strat_config.get("validation", {})
        sharpe = validation.get("sharpe_2025", validation.get("sharpe_2024", "N/A"))
        table.add_row(
            name,
            strat_config.get("symbol", "???"),
            strat_config.get("type", "???"),
            str(sharpe),
        )
    
    console.print(table)
    console.print(f"\n[bold green]Starting {len(enabled)} strategies...[/bold green]\n")
    
    # Get shared resources
    exchange_name = config.get("exchange", "binance_testnet")
    exchange = get_exchange(exchange_name)
    risk_limits = get_risk_limits(config)
    initial_capital = Decimal(str(config.get("initial_capital", 10000)))
    
    console.print(f"[green]✓ Exchange connected: {exchange_name}[/green]")
    console.print(f"[green]✓ Initial capital: ${initial_capital}[/green]")
    console.print(f"[green]✓ Max positions: {risk_limits.max_open_positions}[/green]")
    console.print()
    
    # Initialize global state
    GLOBAL_STATE["start_time"] = datetime.now(timezone.utc).isoformat()
    GLOBAL_STATE["strategies"] = {}
    GLOBAL_STATE["trades"] = []
    save_state()
    
    # Shutdown handling
    shutdown_event = asyncio.Event()
    
    def handle_shutdown(sig, frame):
        console.print("\n[yellow]Shutting down all strategies...[/yellow]")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Start all strategies concurrently
    tasks = []
    for name, strat_config in enabled:
        task = asyncio.create_task(
            run_strategy(
                name,
                strat_config,
                exchange,
                risk_limits,
                initial_capital / len(enabled),  # Split capital
                shutdown_event,
            )
        )
        tasks.append(task)
    
    console.print(f"[bold]Paper trading started for {len(tasks)} strategies![/bold]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    
    # Wait for all strategies
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    
    console.print("[green]All strategies stopped[/green]")


if __name__ == "__main__":
    asyncio.run(main())
