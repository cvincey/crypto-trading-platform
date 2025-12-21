"""Risk management for live trading."""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from crypto.core.types import OrderSide, Signal

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    max_position_size: Decimal = Decimal("0.1")  # 10% of capital
    max_single_trade: Decimal = Decimal("0.05")  # 5% of capital
    stop_loss_pct: Decimal = Decimal("0.02")     # 2% stop loss
    take_profit_pct: Decimal = Decimal("0.04")   # 4% take profit
    max_daily_loss: Decimal = Decimal("0.05")    # 5% max daily loss
    max_drawdown: Decimal = Decimal("0.20")      # 20% max drawdown
    max_open_positions: int = 5


@dataclass
class Position:
    """Current position state."""

    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    allowed: bool
    reason: str = ""
    adjusted_quantity: Decimal | None = None


class RiskManager:
    """
    Risk manager for live trading.
    
    Handles:
    - Position sizing
    - Stop loss / take profit
    - Daily loss limits
    - Drawdown limits
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        initial_capital: Decimal = Decimal("10000"),
    ):
        """
        Initialize risk manager.
        
        Args:
            limits: Risk limit configuration
            initial_capital: Starting capital
        """
        self.limits = limits or RiskLimits()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_pnl = Decimal("0")
        self.positions: dict[str, Position] = {}

    def update_capital(self, new_capital: Decimal) -> None:
        """Update current capital and track peak."""
        self.current_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

    def update_position(
        self,
        symbol: str,
        quantity: Decimal,
        entry_price: Decimal,
        current_price: Decimal,
    ) -> None:
        """Update position state."""
        if quantity == Decimal("0"):
            if symbol in self.positions:
                del self.positions[symbol]
            return

        unrealized_pnl = (current_price - entry_price) * quantity

        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=Decimal("0"),
        )

    def check_trade(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
    ) -> RiskCheckResult:
        """
        Check if a trade is allowed by risk limits.
        
        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            RiskCheckResult with approval status
        """
        trade_value = quantity * price

        # Check max single trade size
        max_trade_value = self.current_capital * self.limits.max_single_trade
        if trade_value > max_trade_value:
            adjusted_quantity = max_trade_value / price
            return RiskCheckResult(
                allowed=True,
                reason="Trade size adjusted to max single trade limit",
                adjusted_quantity=adjusted_quantity,
            )

        # Check daily loss limit
        if self.daily_pnl < -self.initial_capital * self.limits.max_daily_loss:
            return RiskCheckResult(
                allowed=False,
                reason="Daily loss limit reached",
            )

        # Check max drawdown
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown >= self.limits.max_drawdown:
            return RiskCheckResult(
                allowed=False,
                reason="Maximum drawdown limit reached",
            )

        # Check max open positions
        if side == OrderSide.BUY and len(self.positions) >= self.limits.max_open_positions:
            if symbol not in self.positions:
                return RiskCheckResult(
                    allowed=False,
                    reason="Maximum open positions reached",
                )

        # Check max position size
        existing_position = self.positions.get(symbol)
        if existing_position and side == OrderSide.BUY:
            total_position = existing_position.quantity + quantity
            total_value = total_position * price
            max_position_value = self.current_capital * self.limits.max_position_size
            
            if total_value > max_position_value:
                allowed_additional = (max_position_value - existing_position.quantity * price) / price
                if allowed_additional <= 0:
                    return RiskCheckResult(
                        allowed=False,
                        reason="Maximum position size reached",
                    )
                return RiskCheckResult(
                    allowed=True,
                    reason="Quantity adjusted to max position size",
                    adjusted_quantity=allowed_additional,
                )

        return RiskCheckResult(allowed=True)

    def calculate_position_size(
        self,
        symbol: str,
        price: Decimal,
        signal_strength: float = 1.0,
    ) -> Decimal:
        """
        Calculate optimal position size.
        
        Args:
            symbol: Trading pair
            price: Current price
            signal_strength: Signal strength (0 to 1)
            
        Returns:
            Position size in base asset
        """
        # Base position size (fraction of capital)
        base_size = self.current_capital * self.limits.max_single_trade

        # Adjust by signal strength
        adjusted_size = base_size * Decimal(str(signal_strength))

        # Convert to quantity
        quantity = adjusted_size / price

        # Apply position limits
        check = self.check_trade(symbol, OrderSide.BUY, quantity, price)
        if check.adjusted_quantity is not None:
            quantity = check.adjusted_quantity

        return quantity

    def calculate_stop_loss(
        self,
        entry_price: Decimal,
        side: OrderSide,
    ) -> Decimal:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            side: Position side (BUY = long, SELL = short)
            
        Returns:
            Stop loss price
        """
        if side == OrderSide.BUY:
            # Long position: stop below entry
            return entry_price * (Decimal("1") - self.limits.stop_loss_pct)
        else:
            # Short position: stop above entry
            return entry_price * (Decimal("1") + self.limits.stop_loss_pct)

    def calculate_take_profit(
        self,
        entry_price: Decimal,
        side: OrderSide,
    ) -> Decimal:
        """
        Calculate take profit price.
        
        Args:
            entry_price: Entry price
            side: Position side
            
        Returns:
            Take profit price
        """
        if side == OrderSide.BUY:
            return entry_price * (Decimal("1") + self.limits.take_profit_pct)
        else:
            return entry_price * (Decimal("1") - self.limits.take_profit_pct)

    def check_stop_loss(
        self,
        symbol: str,
        current_price: Decimal,
    ) -> bool:
        """
        Check if stop loss is triggered for a position.
        
        Args:
            symbol: Trading pair
            current_price: Current price
            
        Returns:
            True if stop loss triggered
        """
        position = self.positions.get(symbol)
        if not position:
            return False

        stop_price = self.calculate_stop_loss(position.entry_price, OrderSide.BUY)
        return current_price <= stop_price

    def check_take_profit(
        self,
        symbol: str,
        current_price: Decimal,
    ) -> bool:
        """
        Check if take profit is triggered for a position.
        
        Args:
            symbol: Trading pair
            current_price: Current price
            
        Returns:
            True if take profit triggered
        """
        position = self.positions.get(symbol)
        if not position:
            return False

        tp_price = self.calculate_take_profit(position.entry_price, OrderSide.BUY)
        return current_price >= tp_price

    def should_exit(
        self,
        symbol: str,
        current_price: Decimal,
    ) -> tuple[bool, str]:
        """
        Check if position should be exited.
        
        Args:
            symbol: Trading pair
            current_price: Current price
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if self.check_stop_loss(symbol, current_price):
            return True, "stop_loss"

        if self.check_take_profit(symbol, current_price):
            return True, "take_profit"

        return False, ""

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of each day)."""
        self.daily_pnl = Decimal("0")

    def get_risk_summary(self) -> dict[str, Any]:
        """Get current risk summary."""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        return {
            "current_capital": float(self.current_capital),
            "peak_capital": float(self.peak_capital),
            "drawdown_pct": float(drawdown * 100),
            "daily_pnl": float(self.daily_pnl),
            "open_positions": len(self.positions),
            "max_positions": self.limits.max_open_positions,
        }
