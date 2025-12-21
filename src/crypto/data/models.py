"""SQLAlchemy models for the crypto trading platform."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all models."""

    pass


class CandleModel(Base):
    """
    OHLCV candle data model.
    
    This table will be converted to a TimescaleDB hypertable
    for efficient time-series queries.
    """

    __tablename__ = "candles"

    # Primary key columns
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Candle identification
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # OHLCV data
    open: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)

    # Optional additional data
    close_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    quote_volume: Mapped[Decimal | None] = mapped_column(Numeric(30, 8))
    trades: Mapped[int | None] = mapped_column(BigInteger)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "open_time", name="uq_candle"),
        Index("ix_candles_symbol_interval_time", "symbol", "interval", "open_time"),
        Index("ix_candles_open_time", "open_time"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "interval": self.interval,
            "open_time": self.open_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_time": self.close_time,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
        }


class TradeModel(Base):
    """
    Executed trade records.
    
    Stores all trades made by the platform for analysis and auditing.
    """

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Trade identification
    trade_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    order_id: Mapped[str] = mapped_column(String(50), nullable=False)
    exchange: Mapped[str] = mapped_column(String(20), nullable=False)

    # Trade details
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    commission: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    commission_asset: Mapped[str] = mapped_column(String(10), nullable=False)

    # Strategy info
    strategy_name: Mapped[str | None] = mapped_column(String(50))

    # Timestamps
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        Index("ix_trades_symbol_time", "symbol", "executed_at"),
        Index("ix_trades_strategy", "strategy_name", "executed_at"),
    )


class OrderModel(Base):
    """Order records."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Order identification
    order_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    client_order_id: Mapped[str | None] = mapped_column(String(50))
    exchange_order_id: Mapped[str | None] = mapped_column(String(50))
    exchange: Mapped[str] = mapped_column(String(20), nullable=False)

    # Order details
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # Status
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    filled_quantity: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False, default=Decimal("0")
    )
    avg_fill_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # Strategy info
    strategy_name: Mapped[str | None] = mapped_column(String(50))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_orders_symbol_status", "symbol", "status"),
        Index("ix_orders_strategy", "strategy_name", "created_at"),
    )


class BacktestResultModel(Base):
    """
    Backtest result records.
    
    Stores results of backtests for comparison and analysis.
    """

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Backtest identification
    backtest_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    backtest_name: Mapped[str] = mapped_column(String(100), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)

    # Configuration
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    commission: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)

    # Performance metrics
    total_return: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    total_return_pct: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    sortino_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    max_drawdown: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    profit_factor: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    total_trades: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Full metrics JSON
    metrics_json: Mapped[dict | None] = mapped_column(JSONB)

    # Strategy parameters
    strategy_params: Mapped[dict | None] = mapped_column(JSONB)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        Index("ix_backtest_results_strategy", "strategy_name", "created_at"),
        Index("ix_backtest_results_symbol", "symbol", "created_at"),
    )


class FundingRateModel(Base):
    """
    Funding rate data for perpetual futures.
    
    Binance funding rates are paid every 8 hours.
    Positive rate = longs pay shorts.
    Negative rate = shorts pay longs.
    """

    __tablename__ = "funding_rates"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    funding_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Funding rate data
    funding_rate: Mapped[Decimal] = mapped_column(Numeric(20, 10), nullable=False)
    
    # Optional: mark price at funding time
    mark_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint("symbol", "funding_time", name="uq_funding_rate"),
        Index("ix_funding_rates_symbol_time", "symbol", "funding_time"),
        Index("ix_funding_rates_time", "funding_time"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "funding_time": self.funding_time,
            "funding_rate": self.funding_rate,
            "mark_price": self.mark_price,
        }


class OpenInterestModel(Base):
    """
    Open interest data for futures contracts.
    
    Open interest = total number of outstanding derivative contracts.
    Increasing OI with rising price = bullish confirmation.
    Decreasing OI with rising price = weak rally (distribution).
    """

    __tablename__ = "open_interest"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Open interest data
    open_interest: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    open_interest_value: Mapped[Decimal | None] = mapped_column(Numeric(30, 8))  # In quote currency

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_open_interest"),
        Index("ix_open_interest_symbol_time", "symbol", "timestamp"),
        Index("ix_open_interest_time", "timestamp"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open_interest": self.open_interest,
            "open_interest_value": self.open_interest_value,
        }


# SQL to convert candles table to TimescaleDB hypertable
HYPERTABLE_SQL = """
-- Create hypertable (run this after table creation)
SELECT create_hypertable('candles', 'open_time', 
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Add compression policy (compress chunks older than 1 month)
ALTER TABLE candles SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,interval'
);

SELECT add_compression_policy('candles', INTERVAL '1 month', if_not_exists => TRUE);

-- Add retention policy (optional - keep data for 2 years)
-- SELECT add_retention_policy('candles', INTERVAL '2 years', if_not_exists => TRUE);
"""

# SQL to convert funding_rates and open_interest tables to TimescaleDB hypertables
ALTERNATIVE_DATA_HYPERTABLE_SQL = """
-- Create hypertable for funding rates
SELECT create_hypertable('funding_rates', 'funding_time', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Compress funding rates (rarely updated)
ALTER TABLE funding_rates SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('funding_rates', INTERVAL '1 week', if_not_exists => TRUE);

-- Create hypertable for open interest
SELECT create_hypertable('open_interest', 'timestamp', 
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Compress open interest
ALTER TABLE open_interest SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('open_interest', INTERVAL '1 month', if_not_exists => TRUE);
"""
