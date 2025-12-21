"""Portfolio management for backtesting."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from crypto.core.types import OrderSide, Signal


@dataclass
class Trade:
    """Record of an executed trade."""

    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal
    value: Decimal  # Total value of the trade

    @property
    def net_value(self) -> Decimal:
        """Value after commission."""
        if self.side == OrderSide.BUY:
            return self.value + self.commission
        return self.value - self.commission


@dataclass
class Position:
    """Current position in an asset."""

    symbol: str
    quantity: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")
    entry_time: datetime | None = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.quantity != Decimal("0")

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > Decimal("0")

    @property
    def value(self) -> Decimal:
        """Position value at entry price."""
        return abs(self.quantity) * self.avg_entry_price

    def update_unrealized_pnl(self, current_price: Decimal) -> None:
        """Update unrealized PnL based on current price."""
        if self.quantity > 0:
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity
        else:
            self.unrealized_pnl = Decimal("0")


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a point in time."""

    timestamp: datetime
    cash: Decimal
    position_value: Decimal
    total_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal


class Portfolio:
    """
    Portfolio manager for backtesting.
    
    Tracks:
    - Cash balance
    - Open positions
    - Trade history
    - Equity curve
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal("10000"),
        commission_rate: Decimal = Decimal("0.001"),
        slippage_rate: Decimal = Decimal("0.0005"),
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission as fraction (e.g., 0.001 = 0.1%)
            slippage_rate: Slippage as fraction
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[PortfolioState] = []

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()

    def get_position(self, symbol: str) -> Position:
        """Get or create position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def execute_signal(
        self,
        signal: Signal,
        symbol: str,
        price: Decimal,
        timestamp: datetime,
        position_size: Decimal | None = None,
    ) -> Trade | None:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal (BUY/SELL/HOLD)
            symbol: Trading pair
            price: Current price
            timestamp: Signal timestamp
            position_size: Optional position size (default: use all available)
            
        Returns:
            Trade object if executed, None if no trade
        """
        if signal == Signal.HOLD:
            return None

        position = self.get_position(symbol)

        if signal == Signal.BUY:
            return self._execute_buy(position, price, timestamp, position_size)
        elif signal == Signal.SELL:
            return self._execute_sell(position, price, timestamp)

        return None

    def _execute_buy(
        self,
        position: Position,
        price: Decimal,
        timestamp: datetime,
        position_size: Decimal | None = None,
    ) -> Trade | None:
        """Execute a buy order."""
        # Apply slippage (buy at slightly higher price)
        execution_price = price * (Decimal("1") + self.slippage_rate)

        # Calculate position size
        if position_size is None:
            # Use all available cash
            max_value = self.cash / (Decimal("1") + self.commission_rate)
            quantity = max_value / execution_price
        else:
            quantity = position_size

        if quantity <= Decimal("0"):
            return None

        # Calculate trade value and commission
        trade_value = quantity * execution_price
        commission = trade_value * self.commission_rate

        # Check if we have enough cash
        total_cost = trade_value + commission
        if total_cost > self.cash:
            # Adjust quantity to fit available cash
            max_value = self.cash / (Decimal("1") + self.commission_rate)
            quantity = max_value / execution_price
            trade_value = quantity * execution_price
            commission = trade_value * self.commission_rate
            total_cost = trade_value + commission

        if quantity <= Decimal("0"):
            return None

        # Execute trade
        self.cash -= total_cost

        # Update position
        if position.quantity > 0:
            # Add to existing position (average up/down)
            total_quantity = position.quantity + quantity
            total_cost_basis = (
                position.quantity * position.avg_entry_price + trade_value
            )
            position.avg_entry_price = total_cost_basis / total_quantity
            position.quantity = total_quantity
        else:
            # New position
            position.quantity = quantity
            position.avg_entry_price = execution_price
            position.entry_time = timestamp

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol=position.symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            value=trade_value,
        )
        self.trades.append(trade)

        return trade

    def _execute_sell(
        self,
        position: Position,
        price: Decimal,
        timestamp: datetime,
    ) -> Trade | None:
        """Execute a sell order (close position)."""
        if position.quantity <= Decimal("0"):
            return None

        # Apply slippage (sell at slightly lower price)
        execution_price = price * (Decimal("1") - self.slippage_rate)
        quantity = position.quantity

        # Calculate trade value and commission
        trade_value = quantity * execution_price
        commission = trade_value * self.commission_rate
        net_proceeds = trade_value - commission

        # Calculate realized PnL
        cost_basis = quantity * position.avg_entry_price
        realized_pnl = net_proceeds - cost_basis
        position.realized_pnl += realized_pnl

        # Execute trade
        self.cash += net_proceeds

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol=position.symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            value=trade_value,
        )
        self.trades.append(trade)

        # Close position
        position.quantity = Decimal("0")
        position.avg_entry_price = Decimal("0")
        position.unrealized_pnl = Decimal("0")
        position.entry_time = None

        return trade

    def update_equity(
        self,
        timestamp: datetime,
        prices: dict[str, Decimal],
    ) -> PortfolioState:
        """
        Update equity curve with current prices.
        
        Args:
            timestamp: Current timestamp
            prices: Dict mapping symbol to current price
            
        Returns:
            Current portfolio state
        """
        position_value = Decimal("0")
        unrealized_pnl = Decimal("0")
        realized_pnl = Decimal("0")

        for symbol, position in self.positions.items():
            if position.is_open and symbol in prices:
                current_price = prices[symbol]
                position.update_unrealized_pnl(current_price)
                position_value += position.quantity * current_price
                unrealized_pnl += position.unrealized_pnl
            realized_pnl += position.realized_pnl

        total_value = self.cash + position_value

        state = PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            position_value=position_value,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
        )
        self.equity_curve.append(state)

        return state

    def get_total_value(self, prices: dict[str, Decimal]) -> Decimal:
        """Get total portfolio value at current prices."""
        position_value = sum(
            position.quantity * prices.get(symbol, Decimal("0"))
            for symbol, position in self.positions.items()
            if position.is_open
        )
        return self.cash + position_value

    def get_total_commission(self) -> Decimal:
        """Get total commission paid."""
        return sum(trade.commission for trade in self.trades)

    def get_trade_count(self) -> int:
        """Get total number of trades."""
        return len(self.trades)

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        if not self.equity_curve:
            return {}

        final_value = self.equity_curve[-1].total_value
        total_return = final_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * Decimal("100")

        return {
            "initial_capital": float(self.initial_capital),
            "final_value": float(final_value),
            "total_return": float(total_return),
            "total_return_pct": float(total_return_pct),
            "total_trades": len(self.trades),
            "total_commission": float(self.get_total_commission()),
        }
