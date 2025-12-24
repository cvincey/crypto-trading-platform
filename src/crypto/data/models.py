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


class FearGreedIndexModel(Base):
    """
    Crypto Fear & Greed Index data.
    
    Source: alternative.me/crypto/fear-and-greed-index/
    
    Values:
    - 0-24: Extreme Fear
    - 25-49: Fear
    - 50-74: Greed
    - 75-100: Extreme Greed
    
    Trading hypothesis: Extreme fear = buy opportunity, extreme greed = sell signal.
    """

    __tablename__ = "fear_greed_index"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Index data
    value: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 0-100
    value_classification: Mapped[str] = mapped_column(String(20), nullable=False)  # "Extreme Fear", "Fear", etc.

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint("timestamp", name="uq_fear_greed_timestamp"),
        Index("ix_fear_greed_time", "timestamp"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "value": self.value,
            "value_classification": self.value_classification,
        }


class MacroIndicatorModel(Base):
    """
    Macro economic indicators for correlation analysis.
    
    Stores DXY (Dollar Index), VIX, SPX, and other macro indicators
    that may correlate with crypto price movements.
    
    Trading hypothesis:
    - DXY up = BTC down (inverse correlation)
    - VIX spike = risk-off = crypto down
    - SPX correlation varies by market regime
    """

    __tablename__ = "macro_indicators"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    indicator: Mapped[str] = mapped_column(String(20), nullable=False)  # "DXY", "VIX", "SPX", etc.
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Indicator data
    value: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    open: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    high: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    low: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    close: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint("indicator", "timestamp", name="uq_macro_indicator"),
        Index("ix_macro_indicator_time", "indicator", "timestamp"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "indicator": self.indicator,
            "timestamp": self.timestamp,
            "value": self.value,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
        }


class LongShortRatioModel(Base):
    """
    Long/Short ratio data from Binance Futures.
    
    Source: Binance Futures API
    
    Represents the ratio of accounts or positions that are long vs short.
    
    Trading hypothesis:
    - Extreme long ratio (>70%) = crowded long = potential reversal down
    - Extreme short ratio (<30%) = crowded short = potential squeeze up
    """

    __tablename__ = "long_short_ratio"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ratio_type: Mapped[str] = mapped_column(String(20), nullable=False)  # "accounts" or "positions"

    # Ratio data
    long_ratio: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)  # 0.0 - 1.0
    short_ratio: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)  # 0.0 - 1.0
    long_short_ratio: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)  # long/short

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "ratio_type", name="uq_long_short_ratio"),
        Index("ix_long_short_ratio_symbol_time", "symbol", "timestamp"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "ratio_type": self.ratio_type,
            "long_ratio": self.long_ratio,
            "short_ratio": self.short_ratio,
            "long_short_ratio": self.long_short_ratio,
        }


class BTCDominanceModel(Base):
    """
    BTC market dominance data.
    
    Source: CoinGecko
    
    BTC dominance = BTC market cap / Total crypto market cap
    
    Trading hypothesis:
    - Rising BTC dominance = risk-off, capital rotating to BTC
    - Falling BTC dominance = risk-on, capital rotating to alts
    """

    __tablename__ = "btc_dominance"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Dominance data
    btc_dominance: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)  # Percentage
    eth_dominance: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    total_market_cap: Mapped[Decimal | None] = mapped_column(Numeric(30, 2))  # In USD
    btc_market_cap: Mapped[Decimal | None] = mapped_column(Numeric(30, 2))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint("timestamp", name="uq_btc_dominance_timestamp"),
        Index("ix_btc_dominance_time", "timestamp"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "btc_dominance": self.btc_dominance,
            "eth_dominance": self.eth_dominance,
            "total_market_cap": self.total_market_cap,
            "btc_market_cap": self.btc_market_cap,
        }


class ExchangeFlowModel(Base):
    """
    Exchange inflow/outflow data (on-chain).
    
    Source: CryptoQuant or Glassnode (requires API key)
    
    Tracks the flow of crypto into and out of exchanges.
    
    Trading hypothesis:
    - Large inflows = selling pressure (moving to exchange to sell)
    - Large outflows = accumulation (moving off exchange to hold)
    """

    __tablename__ = "exchange_flows"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)  # BTC, ETH, etc.
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exchange: Mapped[str | None] = mapped_column(String(50))  # Optional: specific exchange

    # Flow data
    inflow: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)  # Amount flowing in
    outflow: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)  # Amount flowing out
    netflow: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)  # inflow - outflow
    
    # Value in USD
    inflow_usd: Mapped[Decimal | None] = mapped_column(Numeric(30, 2))
    outflow_usd: Mapped[Decimal | None] = mapped_column(Numeric(30, 2))
    netflow_usd: Mapped[Decimal | None] = mapped_column(Numeric(30, 2))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "exchange", name="uq_exchange_flow"),
        Index("ix_exchange_flows_symbol_time", "symbol", "timestamp"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "exchange": self.exchange,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "netflow": self.netflow,
            "inflow_usd": self.inflow_usd,
            "outflow_usd": self.outflow_usd,
            "netflow_usd": self.netflow_usd,
        }


# SQL to create hypertables for new alternative data tables
NEW_ALTERNATIVE_DATA_HYPERTABLE_SQL = """
-- Create hypertable for Fear & Greed Index
SELECT create_hypertable('fear_greed_index', 'timestamp', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Create hypertable for macro indicators
SELECT create_hypertable('macro_indicators', 'timestamp', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Create hypertable for long/short ratio
SELECT create_hypertable('long_short_ratio', 'timestamp', 
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Create hypertable for BTC dominance
SELECT create_hypertable('btc_dominance', 'timestamp', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Create hypertable for exchange flows
SELECT create_hypertable('exchange_flows', 'timestamp', 
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Create hypertable for stablecoin supply
SELECT create_hypertable('stablecoin_supply', 'timestamp', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Compression policies
ALTER TABLE fear_greed_index SET (timescaledb.compress);
SELECT add_compression_policy('fear_greed_index', INTERVAL '1 week', if_not_exists => TRUE);

ALTER TABLE macro_indicators SET (timescaledb.compress, timescaledb.compress_segmentby = 'indicator');
SELECT add_compression_policy('macro_indicators', INTERVAL '1 week', if_not_exists => TRUE);

ALTER TABLE long_short_ratio SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
SELECT add_compression_policy('long_short_ratio', INTERVAL '1 week', if_not_exists => TRUE);

ALTER TABLE btc_dominance SET (timescaledb.compress);
SELECT add_compression_policy('btc_dominance', INTERVAL '1 week', if_not_exists => TRUE);

ALTER TABLE exchange_flows SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
SELECT add_compression_policy('exchange_flows', INTERVAL '1 week', if_not_exists => TRUE);

ALTER TABLE stablecoin_supply SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
SELECT add_compression_policy('stablecoin_supply', INTERVAL '1 week', if_not_exists => TRUE);
"""


class StablecoinSupplyModel(Base):
    """
    Stablecoin supply data for liquidity analysis.
    
    Source: CoinGecko
    
    Tracks USDT, USDC, DAI market caps as crypto liquidity proxies.
    
    Trading hypothesis:
    - Rising stablecoin supply = fresh capital entering = risk-on
    - Falling stablecoin supply = capital exiting = risk-off
    """

    __tablename__ = "stablecoin_supply"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)  # USDT, USDC, DAI
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Supply data
    market_cap: Mapped[Decimal] = mapped_column(Numeric(30, 2), nullable=False)
    supply_change_24h: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))  # Percentage

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_stablecoin_supply"),
        Index("ix_stablecoin_supply_symbol_time", "symbol", "timestamp"),
        Index("ix_stablecoin_supply_time", "timestamp"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "market_cap": self.market_cap,
            "supply_change_24h": self.supply_change_24h,
        }
