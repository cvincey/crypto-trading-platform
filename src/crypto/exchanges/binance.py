"""Binance exchange adapter implementation."""

import asyncio
import hashlib
import hmac
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable
from urllib.parse import urlencode

import httpx

from crypto.core.types import Candle, Order, OrderSide, OrderStatus, OrderType, Trade
from crypto.exchanges.base import BaseExchange
from crypto.exchanges.registry import exchange_registry

logger = logging.getLogger(__name__)

# Binance interval mapping
INTERVAL_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}

# Binance order status mapping
STATUS_MAP = {
    "NEW": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "PENDING_CANCEL": OrderStatus.OPEN,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
}


@exchange_registry.register("binance", description="Binance cryptocurrency exchange")
class BinanceExchange(BaseExchange):
    """
    Binance exchange adapter.
    
    Supports both testnet and production Binance APIs.
    """

    name = "binance"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        rate_limit: int = 1200,
        timeout: int = 30,
        **kwargs: Any,
    ):
        """
        Initialize Binance exchange.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True
            rate_limit: Max requests per minute
            timeout: Request timeout in seconds
        """
        super().__init__(api_key, api_secret, testnet, **kwargs)
        
        self.rate_limit = rate_limit
        self.timeout = timeout
        
        # API URLs
        if testnet:
            self.base_url = "https://testnet.binance.vision"
            self.ws_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com"
            self.ws_url = "wss://stream.binance.com:9443/ws"

        # HTTP client
        self._client: httpx.AsyncClient | None = None
        
        # WebSocket state
        self._ws_callbacks: dict[str, Callable] = {}
        self._ws_task: asyncio.Task | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def _sign_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Sign a request with HMAC-SHA256."""
        params["timestamp"] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API key."""
        return {"X-MBX-APIKEY": self.api_key}

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        """Make an API request."""
        client = await self._get_client()
        params = params or {}
        
        if signed:
            params = self._sign_request(params)
            headers = self._get_headers()
        else:
            headers = {}

        try:
            if method == "GET":
                response = await client.get(endpoint, params=params, headers=headers)
            elif method == "POST":
                response = await client.post(endpoint, params=params, headers=headers)
            elif method == "DELETE":
                response = await client.delete(endpoint, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Binance API error: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

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
        """Fetch historical OHLCV candles from Binance."""
        binance_interval = INTERVAL_MAP.get(interval, interval)
        
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": binance_interval,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
        }
        if limit:
            params["limit"] = min(limit, 1000)  # Binance max is 1000

        all_candles: list[Candle] = []
        current_start = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        while current_start < end_ms:
            params["startTime"] = current_start
            data = await self._request("GET", "/api/v3/klines", params)
            
            if not data:
                break

            for kline in data:
                candle = Candle(
                    symbol=symbol,
                    interval=interval,
                    open_time=datetime.fromtimestamp(kline[0] / 1000),
                    open=Decimal(str(kline[1])),
                    high=Decimal(str(kline[2])),
                    low=Decimal(str(kline[3])),
                    close=Decimal(str(kline[4])),
                    volume=Decimal(str(kline[5])),
                    close_time=datetime.fromtimestamp(kline[6] / 1000),
                    quote_volume=Decimal(str(kline[7])),
                    trades=int(kline[8]),
                )
                all_candles.append(candle)

            # Move to next batch
            if len(data) < 1000:
                break
            current_start = data[-1][6] + 1  # Close time + 1ms

        return all_candles

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current ticker data."""
        data = await self._request(
            "GET",
            "/api/v3/ticker/24hr",
            {"symbol": symbol},
        )
        return {
            "symbol": data["symbol"],
            "price": Decimal(data["lastPrice"]),
            "bid": Decimal(data["bidPrice"]),
            "ask": Decimal(data["askPrice"]),
            "volume": Decimal(data["volume"]),
            "quote_volume": Decimal(data["quoteVolume"]),
            "change_pct": Decimal(data["priceChangePercent"]),
        }

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int = 100,
    ) -> dict[str, list]:
        """Fetch order book."""
        data = await self._request(
            "GET",
            "/api/v3/depth",
            {"symbol": symbol, "limit": limit},
        )
        return {
            "bids": [
                {"price": Decimal(bid[0]), "quantity": Decimal(bid[1])}
                for bid in data["bids"]
            ],
            "asks": [
                {"price": Decimal(ask[0]), "quantity": Decimal(ask[1])}
                for ask in data["asks"]
            ],
        }

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
        """Place a new order on Binance."""
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "quantity": str(quantity),
        }

        if price and order_type in (OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT):
            params["price"] = str(price)
            params["timeInForce"] = "GTC"

        if stop_price and order_type in (
            OrderType.STOP_LOSS,
            OrderType.STOP_LOSS_LIMIT,
            OrderType.TAKE_PROFIT,
            OrderType.TAKE_PROFIT_LIMIT,
        ):
            params["stopPrice"] = str(stop_price)

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        data = await self._request("POST", "/api/v3/order", params, signed=True)

        return Order(
            id=str(data["orderId"]),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=STATUS_MAP.get(data["status"], OrderStatus.PENDING),
            filled_quantity=Decimal(data.get("executedQty", "0")),
            exchange_order_id=str(data["orderId"]),
            client_order_id=data.get("clientOrderId"),
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            await self._request(
                "DELETE",
                "/api/v3/order",
                {"symbol": symbol, "orderId": order_id},
                signed=True,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order status."""
        data = await self._request(
            "GET",
            "/api/v3/order",
            {"symbol": symbol, "orderId": order_id},
            signed=True,
        )

        return Order(
            id=str(data["orderId"]),
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["type"]),
            quantity=Decimal(data["origQty"]),
            price=Decimal(data["price"]) if data["price"] != "0.00000000" else None,
            status=STATUS_MAP.get(data["status"], OrderStatus.PENDING),
            filled_quantity=Decimal(data["executedQty"]),
            avg_fill_price=(
                Decimal(data["cummulativeQuoteQty"]) / Decimal(data["executedQty"])
                if Decimal(data["executedQty"]) > 0
                else None
            ),
            exchange_order_id=str(data["orderId"]),
            client_order_id=data.get("clientOrderId"),
        )

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._request(
            "GET",
            "/api/v3/openOrders",
            params,
            signed=True,
        )

        return [
            Order(
                id=str(order["orderId"]),
                symbol=order["symbol"],
                side=OrderSide(order["side"]),
                order_type=OrderType(order["type"]),
                quantity=Decimal(order["origQty"]),
                price=(
                    Decimal(order["price"])
                    if order["price"] != "0.00000000"
                    else None
                ),
                status=STATUS_MAP.get(order["status"], OrderStatus.OPEN),
                filled_quantity=Decimal(order["executedQty"]),
                exchange_order_id=str(order["orderId"]),
            )
            for order in data
        ]

    # ==========================================================================
    # Account Methods
    # ==========================================================================

    async def get_balance(self, asset: str) -> Decimal:
        """Get available balance for an asset."""
        data = await self._request("GET", "/api/v3/account", {}, signed=True)
        
        for balance in data["balances"]:
            if balance["asset"] == asset:
                return Decimal(balance["free"])
        
        return Decimal("0")

    async def get_balances(self) -> dict[str, Decimal]:
        """Get all non-zero balances."""
        data = await self._request("GET", "/api/v3/account", {}, signed=True)
        
        balances = {}
        for balance in data["balances"]:
            free = Decimal(balance["free"])
            locked = Decimal(balance["locked"])
            total = free + locked
            if total > 0:
                balances[balance["asset"]] = free
        
        return balances

    # ==========================================================================
    # WebSocket Methods
    # ==========================================================================

    async def subscribe_candles(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Candle], None],
    ) -> None:
        """Subscribe to real-time candle updates."""
        import websockets

        stream_name = f"{symbol.lower()}@kline_{interval}"
        
        async def _ws_handler():
            url = f"{self.ws_url}/{stream_name}"
            async with websockets.connect(url) as ws:
                async for message in ws:
                    import json
                    data = json.loads(message)
                    kline = data["k"]
                    candle = Candle(
                        symbol=kline["s"],
                        interval=kline["i"],
                        open_time=datetime.fromtimestamp(kline["t"] / 1000),
                        open=Decimal(kline["o"]),
                        high=Decimal(kline["h"]),
                        low=Decimal(kline["l"]),
                        close=Decimal(kline["c"]),
                        volume=Decimal(kline["v"]),
                        close_time=datetime.fromtimestamp(kline["T"] / 1000),
                    )
                    callback(candle)

        self._ws_callbacks[stream_name] = callback
        self._ws_task = asyncio.create_task(_ws_handler())

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all WebSocket streams."""
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        self._ws_callbacks.clear()

    async def close(self) -> None:
        """Close all connections."""
        await self.unsubscribe_all()
        if self._client:
            await self._client.aclose()
