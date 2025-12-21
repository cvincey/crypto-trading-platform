"""Live trading loop with cross-symbol strategy support."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable

import pandas as pd

from crypto.config.settings import get_settings
from crypto.core.types import Candle, OrderSide, Signal
from crypto.data.repository import CandleRepository
from crypto.exchanges.base import Exchange
from crypto.exchanges.registry import get_exchange
from crypto.strategies.base import Strategy
from crypto.strategies.cross_symbol_base import CrossSymbolBaseStrategy
from crypto.strategies.registry import strategy_registry
from crypto.trading.executor import ExchangeOrderExecutor, OrderExecutor, PaperOrderExecutor
from crypto.trading.risk import RiskLimits, RiskManager

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Live trading loop.
    
    Connects to exchange, receives real-time data,
    generates signals, and executes trades with risk management.
    """

    def __init__(
        self,
        exchange: Exchange,
        strategy: Strategy,
        symbol: str,
        interval: str = "1h",
        paper: bool = True,
        initial_capital: Decimal = Decimal("10000"),
        risk_limits: RiskLimits | None = None,
    ):
        """
        Initialize live trader.
        
        Args:
            exchange: Exchange adapter
            strategy: Trading strategy
            symbol: Trading pair
            interval: Candle interval
            paper: Use paper trading if True
            initial_capital: Starting capital
            risk_limits: Risk management limits
        """
        self.exchange = exchange
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.paper = paper
        self.initial_capital = initial_capital

        # Initialize components
        if paper:
            self.executor: OrderExecutor = PaperOrderExecutor(exchange)
        else:
            self.executor = ExchangeOrderExecutor(exchange)

        self.risk_manager = RiskManager(
            limits=risk_limits or RiskLimits(),
            initial_capital=initial_capital,
        )

        self.repository = CandleRepository()

        # State
        self._running = False
        self._candle_buffer: list[Candle] = []
        self._last_signal: Signal = Signal.HOLD
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []
        
        # Cross-symbol support
        self._reference_data: dict[str, pd.DataFrame] = {}
        self._reference_update_interval = 3600  # Update reference data hourly
        self._last_reference_update: datetime | None = None

    async def _load_reference_data(self, lookback_days: int = 30) -> None:
        """
        Load reference data for cross-symbol strategies.
        
        This method loads historical data for reference symbols (e.g., BTC, ETH)
        that cross-symbol strategies need to generate signals.
        
        Args:
            lookback_days: Number of days of historical data to load
        """
        if not isinstance(self.strategy, CrossSymbolBaseStrategy):
            return
        
        ref_symbols = self.strategy.get_reference_symbols()
        if not ref_symbols:
            return
        
        logger.info(f"Loading reference data for: {ref_symbols}")
        
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        
        for symbol in ref_symbols:
            try:
                candles = await self.repository.get_candles_df(
                    symbol, self.interval, start, end
                )
                if not candles.empty:
                    self._reference_data[symbol] = candles
                    self.strategy.set_reference_data(symbol, candles)
                    logger.info(f"  Loaded {len(candles)} candles for {symbol}")
                else:
                    logger.warning(f"  No data found for {symbol}")
            except Exception as e:
                logger.error(f"  Error loading {symbol}: {e}")
        
        self._last_reference_update = datetime.now(timezone.utc)

    async def _maybe_update_reference_data(self) -> None:
        """Update reference data if enough time has passed."""
        if not isinstance(self.strategy, CrossSymbolBaseStrategy):
            return
        
        if self._last_reference_update is None:
            await self._load_reference_data()
            return
        
        elapsed = (datetime.now(timezone.utc) - self._last_reference_update).total_seconds()
        if elapsed >= self._reference_update_interval:
            await self._load_reference_data()

    async def start(self) -> None:
        """Start the live trading loop."""
        logger.info(
            f"Starting live trader: {self.strategy.name} on {self.symbol} "
            f"({'PAPER' if self.paper else 'LIVE'})"
        )

        self._running = True
        
        # Load reference data for cross-symbol strategies
        await self._load_reference_data()

        # Subscribe to candle updates
        await self.exchange.subscribe_candles(
            self.symbol,
            self.interval,
            self._on_candle,
        )

        # Keep running and periodically update reference data
        while self._running:
            await self._maybe_update_reference_data()
            await asyncio.sleep(60)  # Check every minute

    async def stop(self) -> None:
        """Stop the live trading loop."""
        logger.info("Stopping live trader...")
        self._running = False
        await self.exchange.unsubscribe_all()

        # Cancel any pending orders
        if hasattr(self.executor, "cancel_all"):
            await self.executor.cancel_all(self.symbol)

    def _on_candle(self, candle: Candle) -> None:
        """Handle new candle data."""
        asyncio.create_task(self._process_candle(candle))

    async def _process_candle(self, candle: Candle) -> None:
        """Process a new candle and potentially trade."""
        try:
            # Add to buffer
            self._candle_buffer.append(candle)

            # Keep buffer at reasonable size
            max_buffer = 500
            if len(self._candle_buffer) > max_buffer:
                self._candle_buffer = self._candle_buffer[-max_buffer:]

            # Convert to DataFrame for strategy
            import pandas as pd
            
            df = pd.DataFrame([c.to_dict() for c in self._candle_buffer])
            df.set_index("open_time", inplace=True)
            
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            # Generate signals
            signals = self.strategy.generate_signals(df)

            if signals.empty:
                return

            # Get latest signal
            current_signal = signals.iloc[-1]

            # Only act on signal changes
            if current_signal == self._last_signal:
                return

            self._last_signal = current_signal
            current_price = Decimal(str(candle.close))

            # Execute based on signal
            if current_signal == Signal.BUY:
                await self._handle_buy_signal(current_price)
            elif current_signal == Signal.SELL:
                await self._handle_sell_signal(current_price)

            # Notify callbacks
            self._notify({
                "event": "signal",
                "signal": current_signal.name,
                "price": float(current_price),
                "timestamp": candle.open_time.isoformat(),
            })

        except Exception as e:
            logger.error(f"Error processing candle: {e}")

    async def _handle_buy_signal(self, price: Decimal) -> None:
        """Handle a buy signal."""
        # Calculate position size
        quantity = self.risk_manager.calculate_position_size(
            self.symbol, price
        )

        if quantity <= 0:
            logger.info("Position size is zero, skipping buy")
            return

        # Check risk limits
        check = self.risk_manager.check_trade(
            self.symbol, OrderSide.BUY, quantity, price
        )

        if not check.allowed:
            logger.warning(f"Buy rejected: {check.reason}")
            return

        if check.adjusted_quantity:
            quantity = check.adjusted_quantity

        # Execute order
        result = await self.executor.execute(
            symbol=self.symbol,
            side=OrderSide.BUY,
            quantity=quantity,
        )

        if result.success:
            logger.info(
                f"BUY executed: {result.filled_quantity} {self.symbol} "
                f"@ {result.avg_price}"
            )

            # Update risk manager
            self.risk_manager.update_position(
                self.symbol,
                result.filled_quantity,
                result.avg_price or price,
                price,
            )

            self._notify({
                "event": "trade",
                "side": "BUY",
                "quantity": float(result.filled_quantity),
                "price": float(result.avg_price) if result.avg_price else float(price),
            })
        else:
            logger.error(f"BUY failed: {result.error}")

    async def _handle_sell_signal(self, price: Decimal) -> None:
        """Handle a sell signal."""
        position = self.risk_manager.positions.get(self.symbol)

        if not position or position.quantity <= 0:
            logger.info("No position to sell")
            return

        # Execute sell order
        result = await self.executor.execute(
            symbol=self.symbol,
            side=OrderSide.SELL,
            quantity=position.quantity,
        )

        if result.success:
            logger.info(
                f"SELL executed: {result.filled_quantity} {self.symbol} "
                f"@ {result.avg_price}"
            )

            # Update risk manager
            self.risk_manager.update_position(
                self.symbol,
                Decimal("0"),
                Decimal("0"),
                price,
            )

            self._notify({
                "event": "trade",
                "side": "SELL",
                "quantity": float(result.filled_quantity),
                "price": float(result.avg_price) if result.avg_price else float(price),
            })
        else:
            logger.error(f"SELL failed: {result.error}")

    def add_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Add a callback for trading events."""
        self._callbacks.append(callback)

    def _notify(self, event: dict[str, Any]) -> None:
        """Notify all callbacks of an event."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current trading status."""
        return {
            "running": self._running,
            "strategy": self.strategy.name,
            "symbol": self.symbol,
            "interval": self.interval,
            "paper": self.paper,
            "last_signal": self._last_signal.name,
            "candle_buffer_size": len(self._candle_buffer),
            "risk": self.risk_manager.get_risk_summary(),
        }


async def start_trading(
    strategy_name: str,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    exchange_name: str = "binance_testnet",
    paper: bool = True,
    initial_capital: Decimal = Decimal("10000"),
) -> LiveTrader:
    """
    Convenience function to start live trading.
    
    Args:
        strategy_name: Strategy name from config
        symbol: Trading pair
        interval: Candle interval
        exchange_name: Exchange name from config
        paper: Use paper trading
        initial_capital: Starting capital
        
    Returns:
        LiveTrader instance
    """
    # Load strategy from config
    strategy = strategy_registry.create_from_config(strategy_name)

    # Get exchange
    exchange = get_exchange(exchange_name)

    # Create trader
    trader = LiveTrader(
        exchange=exchange,
        strategy=strategy,
        symbol=symbol,
        interval=interval,
        paper=paper,
        initial_capital=initial_capital,
    )

    # Start trading
    asyncio.create_task(trader.start())

    return trader
