"""Tests for core types."""

from datetime import datetime
from decimal import Decimal

import pytest

from crypto.core.types import (
    Candle,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    Trade,
)


class TestSignal:
    """Tests for Signal enum."""

    def test_signal_values(self):
        """Test signal enum values."""
        assert Signal.BUY.name == "BUY"
        assert Signal.SELL.name == "SELL"
        assert Signal.HOLD.name == "HOLD"

    def test_signal_str(self):
        """Test signal string representation."""
        assert str(Signal.BUY) == "BUY"
        assert str(Signal.SELL) == "SELL"


class TestOrderSide:
    """Tests for OrderSide enum."""

    def test_order_side_values(self):
        """Test order side values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"


class TestCandle:
    """Tests for Candle dataclass."""

    def test_candle_creation(self):
        """Test candle creation."""
        candle = Candle(
            symbol="BTCUSDT",
            interval="1h",
            open_time=datetime(2024, 1, 1, 12, 0),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )

        assert candle.symbol == "BTCUSDT"
        assert candle.interval == "1h"
        assert candle.open == Decimal("50000")
        assert candle.close == Decimal("50500")

    def test_candle_to_dict(self):
        """Test candle to_dict method."""
        candle = Candle(
            symbol="BTCUSDT",
            interval="1h",
            open_time=datetime(2024, 1, 1, 12, 0),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )

        data = candle.to_dict()
        assert data["symbol"] == "BTCUSDT"
        assert data["close"] == Decimal("50500")

    def test_candle_immutable(self):
        """Test that candles are immutable."""
        candle = Candle(
            symbol="BTCUSDT",
            interval="1h",
            open_time=datetime(2024, 1, 1, 12, 0),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )

        with pytest.raises(AttributeError):
            candle.symbol = "ETHUSDT"


class TestOrder:
    """Tests for Order dataclass."""

    def test_order_creation(self):
        """Test order creation."""
        order = Order(
            id="test123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order.id == "test123"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING

    def test_order_is_filled(self):
        """Test order is_filled property."""
        order = Order(
            id="test123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
        )

        assert order.is_filled is True

    def test_order_is_active(self):
        """Test order is_active property."""
        order = Order(
            id="test123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.OPEN,
        )

        assert order.is_active is True

        order.status = OrderStatus.FILLED
        assert order.is_active is False


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test position creation."""
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            entry_time=datetime.now(),
        )

        assert position.is_long is True
        assert position.is_short is False
        assert position.value == Decimal("50000")

    def test_position_update_pnl(self):
        """Test position PnL update."""
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            entry_time=datetime.now(),
        )

        position.update_pnl(Decimal("51000"))
        assert position.unrealized_pnl == Decimal("1000")

        position.update_pnl(Decimal("49000"))
        assert position.unrealized_pnl == Decimal("-1000")
