"""Exchange protocol and base classes."""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Protocol, runtime_checkable

from crypto.core.types import Candle, Order, OrderSide, OrderType, Trade


@runtime_checkable
class Exchange(Protocol):
    """
    Exchange protocol - implement this to add new exchange support.
    
    This defines the interface that all exchange adapters must implement.
    The platform uses this protocol to interact with exchanges in a
    uniform way, regardless of the underlying exchange.
    """

    name: str

    # ==========================================================================
    # Market Data Methods
    # ==========================================================================

    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Candle]:
        """
        Fetch historical OHLCV candles.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1m", "1h", "1d")
            start: Start time
            end: End time
            limit: Maximum number of candles to fetch
            
        Returns:
            List of Candle objects
        """
        ...

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Fetch current ticker data for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Ticker data including price, volume, etc.
        """
        ...

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 100,
    ) -> dict[str, list]:
        """
        Fetch order book for a symbol.
        
        Args:
            symbol: Trading pair
            limit: Depth of order book
            
        Returns:
            Dict with 'bids' and 'asks' lists
        """
        ...

    # ==========================================================================
    # Trading Methods
    # ==========================================================================

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Place a new order.
        
        Args:
            symbol: Trading pair
            side: BUY or SELL
            order_type: MARKET, LIMIT, etc.
            quantity: Amount to trade
            price: Limit price (required for LIMIT orders)
            stop_price: Stop price (for stop orders)
            client_order_id: Optional client-specified order ID
            
        Returns:
            Order object with exchange order ID
        """
        ...

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
    ) -> bool:
        """
        Cancel an open order.
        
        Args:
            symbol: Trading pair
            order_id: Exchange order ID
            
        Returns:
            True if cancelled successfully
        """
        ...

    async def get_order(
        self,
        symbol: str,
        order_id: str,
    ) -> Order:
        """
        Get order status.
        
        Args:
            symbol: Trading pair
            order_id: Exchange order ID
            
        Returns:
            Order object with current status
        """
        ...

    async def get_open_orders(
        self,
        symbol: str | None = None,
    ) -> list[Order]:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of open Order objects
        """
        ...

    # ==========================================================================
    # Account Methods
    # ==========================================================================

    async def get_balance(self, asset: str) -> Decimal:
        """
        Get available balance for an asset.
        
        Args:
            asset: Asset symbol (e.g., "BTC", "USDT")
            
        Returns:
            Available balance
        """
        ...

    async def get_balances(self) -> dict[str, Decimal]:
        """
        Get all non-zero balances.
        
        Returns:
            Dict mapping asset to balance
        """
        ...

    # ==========================================================================
    # WebSocket Methods
    # ==========================================================================

    async def subscribe_candles(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Candle], None],
    ) -> None:
        """
        Subscribe to real-time candle updates.
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            callback: Function to call with each new candle
        """
        ...

    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Trade], None],
    ) -> None:
        """
        Subscribe to real-time trade updates.
        
        Args:
            symbol: Trading pair
            callback: Function to call with each trade
        """
        ...

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all WebSocket streams."""
        ...

    async def close(self) -> None:
        """Close all connections."""
        ...


class BaseExchange(ABC):
    """
    Abstract base class for exchange implementations.
    
    Provides common functionality and enforces the Exchange protocol.
    """

    name: str = "base"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize the exchange.
        
        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet
            **kwargs: Additional exchange-specific options
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._options = kwargs

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Candle]:
        """Fetch historical OHLCV candles."""
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current ticker data."""
        pass

    @abstractmethod
    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 100,
    ) -> dict[str, list]:
        """Fetch order book."""
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """Place a new order."""
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order status."""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders."""
        pass

    @abstractmethod
    async def get_balance(self, asset: str) -> Decimal:
        """Get available balance."""
        pass

    @abstractmethod
    async def get_balances(self) -> dict[str, Decimal]:
        """Get all non-zero balances."""
        pass

    async def subscribe_candles(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Candle], None],
    ) -> None:
        """Default implementation raises NotImplementedError."""
        raise NotImplementedError("WebSocket not supported by this exchange")

    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Trade], None],
    ) -> None:
        """Default implementation raises NotImplementedError."""
        raise NotImplementedError("WebSocket not supported by this exchange")

    async def unsubscribe_all(self) -> None:
        """Default implementation does nothing."""
        pass

    async def close(self) -> None:
        """Default implementation does nothing."""
        pass
