"""Order execution abstraction for live trading."""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from crypto.core.types import Order, OrderSide, OrderStatus, OrderType
from crypto.exchanges.base import Exchange

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of an order execution attempt."""

    success: bool
    order: Order | None = None
    error: str | None = None
    filled_quantity: Decimal = Decimal("0")
    avg_price: Decimal | None = None
    commission: Decimal = Decimal("0")


class OrderExecutor(ABC):
    """
    Abstract order executor.
    
    Handles order placement, tracking, and management.
    """

    @abstractmethod
    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> ExecutionResult:
        """Execute an order."""
        pass

    @abstractmethod
    async def cancel(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        """Get order status."""
        pass


class ExchangeOrderExecutor(OrderExecutor):
    """
    Order executor that uses an exchange adapter.
    
    Supports:
    - Market and limit orders
    - Order tracking
    - Retry logic for failed orders
    """

    def __init__(
        self,
        exchange: Exchange,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the executor.
        
        Args:
            exchange: Exchange adapter instance
            max_retries: Maximum retry attempts for failed orders
            retry_delay: Delay between retries in seconds
        """
        self.exchange = exchange
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._pending_orders: dict[str, Order] = {}

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> ExecutionResult:
        """
        Execute an order on the exchange.
        
        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Amount to trade
            order_type: MARKET, LIMIT, etc.
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            ExecutionResult with order details
        """
        import asyncio

        client_order_id = f"crypto_{uuid.uuid4().hex[:16]}"
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Executing order: {side.value} {quantity} {symbol} "
                    f"@ {price or 'MARKET'} (attempt {attempt + 1})"
                )

                order = await self.exchange.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price,
                    client_order_id=client_order_id,
                )

                self._pending_orders[order.id] = order

                # For market orders, wait for fill
                if order_type == OrderType.MARKET:
                    order = await self._wait_for_fill(symbol, order.id)

                logger.info(
                    f"Order executed: {order.id} - Status: {order.status}"
                )

                return ExecutionResult(
                    success=True,
                    order=order,
                    filled_quantity=order.filled_quantity,
                    avg_price=order.avg_fill_price,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Order execution failed (attempt {attempt + 1}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        logger.error(f"Order execution failed after {self.max_retries} attempts")
        return ExecutionResult(
            success=False,
            error=last_error,
        )

    async def _wait_for_fill(
        self,
        symbol: str,
        order_id: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Order:
        """Wait for an order to be filled."""
        import asyncio

        start_time = asyncio.get_event_loop().time()

        while True:
            order = await self.exchange.get_order(symbol, order_id)

            if order.status == OrderStatus.FILLED:
                return order

            if order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                raise Exception(f"Order {order_id} was {order.status.value}")

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Order {order_id} not filled within {timeout}s")

            await asyncio.sleep(poll_interval)

    async def cancel(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        try:
            success = await self.exchange.cancel_order(symbol, order_id)
            if success and order_id in self._pending_orders:
                del self._pending_orders[order_id]
            return success
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all open orders."""
        open_orders = await self.exchange.get_open_orders(symbol)
        cancelled = 0

        for order in open_orders:
            if await self.cancel(order.symbol, order.id):
                cancelled += 1

        return cancelled

    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        """Get current order status."""
        return await self.exchange.get_order(symbol, order_id)

    def get_pending_orders(self) -> list[Order]:
        """Get all pending orders."""
        return list(self._pending_orders.values())


class PaperOrderExecutor(OrderExecutor):
    """
    Paper trading order executor.
    
    Simulates order execution without actually placing orders.
    Useful for testing strategies in paper trading mode.
    """

    def __init__(
        self,
        exchange: Exchange,
        slippage: Decimal = Decimal("0.001"),
    ):
        """
        Initialize paper executor.
        
        Args:
            exchange: Exchange for getting prices
            slippage: Simulated slippage rate
        """
        self.exchange = exchange
        self.slippage = slippage
        self._orders: dict[str, Order] = {}
        self._order_counter = 0

    async def execute(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> ExecutionResult:
        """Execute a paper order."""
        self._order_counter += 1
        order_id = f"PAPER_{self._order_counter:06d}"

        # Get current price
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker["price"]
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Could not get price: {e}",
            )

        # Apply slippage
        if side == OrderSide.BUY:
            fill_price = current_price * (Decimal("1") + self.slippage)
        else:
            fill_price = current_price * (Decimal("1") - self.slippage)

        # Create filled order
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            avg_fill_price=fill_price,
            created_at=datetime.utcnow(),
        )

        self._orders[order_id] = order

        logger.info(
            f"[PAPER] Executed: {side.value} {quantity} {symbol} @ {fill_price}"
        )

        return ExecutionResult(
            success=True,
            order=order,
            filled_quantity=quantity,
            avg_price=fill_price,
        )

    async def cancel(self, symbol: str, order_id: str) -> bool:
        """Cancel a paper order."""
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        """Get paper order status."""
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")
        return self._orders[order_id]
