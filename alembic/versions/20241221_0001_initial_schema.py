"""Initial schema with candles table and hypertable

Revision ID: 0001
Revises: 
Create Date: 2024-12-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create candles table
    op.create_table(
        'candles',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('interval', sa.String(10), nullable=False),
        sa.Column('open_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('open', sa.Numeric(20, 8), nullable=False),
        sa.Column('high', sa.Numeric(20, 8), nullable=False),
        sa.Column('low', sa.Numeric(20, 8), nullable=False),
        sa.Column('close', sa.Numeric(20, 8), nullable=False),
        sa.Column('volume', sa.Numeric(30, 8), nullable=False),
        sa.Column('close_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('quote_volume', sa.Numeric(30, 8), nullable=True),
        sa.Column('trades', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'interval', 'open_time', name='uq_candle')
    )
    op.create_index('ix_candles_symbol_interval_time', 'candles', 
                    ['symbol', 'interval', 'open_time'])
    op.create_index('ix_candles_open_time', 'candles', ['open_time'])

    # Create trades table
    op.create_table(
        'trades',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('trade_id', sa.String(50), nullable=False),
        sa.Column('order_id', sa.String(50), nullable=False),
        sa.Column('exchange', sa.String(20), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('quantity', sa.Numeric(20, 8), nullable=False),
        sa.Column('price', sa.Numeric(20, 8), nullable=False),
        sa.Column('commission', sa.Numeric(20, 8), nullable=False),
        sa.Column('commission_asset', sa.String(10), nullable=False),
        sa.Column('strategy_name', sa.String(50), nullable=True),
        sa.Column('executed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('trade_id')
    )
    op.create_index('ix_trades_symbol_time', 'trades', ['symbol', 'executed_at'])
    op.create_index('ix_trades_strategy', 'trades', ['strategy_name', 'executed_at'])

    # Create orders table
    op.create_table(
        'orders',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('order_id', sa.String(50), nullable=False),
        sa.Column('client_order_id', sa.String(50), nullable=True),
        sa.Column('exchange_order_id', sa.String(50), nullable=True),
        sa.Column('exchange', sa.String(20), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('order_type', sa.String(20), nullable=False),
        sa.Column('quantity', sa.Numeric(20, 8), nullable=False),
        sa.Column('price', sa.Numeric(20, 8), nullable=True),
        sa.Column('stop_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('filled_quantity', sa.Numeric(20, 8), nullable=False, default=0),
        sa.Column('avg_fill_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('strategy_name', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('order_id')
    )
    op.create_index('ix_orders_symbol_status', 'orders', ['symbol', 'status'])
    op.create_index('ix_orders_strategy', 'orders', ['strategy_name', 'created_at'])

    # Create backtest_results table
    op.create_table(
        'backtest_results',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('backtest_id', sa.String(50), nullable=False),
        sa.Column('backtest_name', sa.String(100), nullable=False),
        sa.Column('strategy_name', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('interval', sa.String(10), nullable=False),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('initial_capital', sa.Numeric(20, 8), nullable=False),
        sa.Column('commission', sa.Numeric(10, 6), nullable=False),
        sa.Column('total_return', sa.Numeric(20, 8), nullable=False),
        sa.Column('total_return_pct', sa.Numeric(10, 4), nullable=False),
        sa.Column('sharpe_ratio', sa.Numeric(10, 4), nullable=True),
        sa.Column('sortino_ratio', sa.Numeric(10, 4), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(10, 4), nullable=True),
        sa.Column('win_rate', sa.Numeric(10, 4), nullable=True),
        sa.Column('profit_factor', sa.Numeric(10, 4), nullable=True),
        sa.Column('total_trades', sa.BigInteger(), nullable=False),
        sa.Column('metrics_json', sa.JSON(), nullable=True),
        sa.Column('strategy_params', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('backtest_id')
    )
    op.create_index('ix_backtest_results_strategy', 'backtest_results', 
                    ['strategy_name', 'created_at'])
    op.create_index('ix_backtest_results_symbol', 'backtest_results',
                    ['symbol', 'created_at'])

    # Note: Run this manually if using TimescaleDB:
    # SELECT create_hypertable('candles', 'open_time', 
    #     chunk_time_interval => INTERVAL '1 week',
    #     if_not_exists => TRUE
    # );


def downgrade() -> None:
    op.drop_table('backtest_results')
    op.drop_table('orders')
    op.drop_table('trades')
    op.drop_table('candles')
