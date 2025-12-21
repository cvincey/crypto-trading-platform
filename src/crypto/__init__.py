"""
Crypto Trading Platform

A config-first, extensible platform for backtesting and trading cryptocurrencies.
"""

__version__ = "0.1.0"

# Import registries to trigger strategy/exchange registration
from crypto.strategies import technical, statistical, momentum, ml
from crypto.exchanges import binance

# Public API
from crypto.config.settings import get_settings, Settings
from crypto.core.types import Signal, OrderSide, OrderType, Candle, Order
from crypto.strategies.registry import strategy_registry, get_strategy
from crypto.exchanges.registry import exchange_registry, get_exchange
from crypto.backtesting.engine import BacktestEngine, BacktestResult
from crypto.backtesting.runner import BacktestRunner
from crypto.trading.live import LiveTrader, start_trading

__all__ = [
    # Version
    "__version__",
    # Config
    "get_settings",
    "Settings",
    # Types
    "Signal",
    "OrderSide",
    "OrderType",
    "Candle",
    "Order",
    # Registries
    "strategy_registry",
    "exchange_registry",
    "get_strategy",
    "get_exchange",
    # Backtesting
    "BacktestEngine",
    "BacktestResult",
    "BacktestRunner",
    # Trading
    "LiveTrader",
    "start_trading",
]


def main() -> None:
    """Main entry point - runs the CLI."""
    from crypto.cli import cli
    cli()


if __name__ == "__main__":
    main()
