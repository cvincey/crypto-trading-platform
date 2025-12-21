"""Add funding_rates and open_interest tables for alternative data

Revision ID: 0002
Revises: 0001
Create Date: 2024-12-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create funding_rates table
    op.create_table(
        'funding_rates',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('funding_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('funding_rate', sa.Numeric(20, 10), nullable=False),
        sa.Column('mark_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'funding_time', name='uq_funding_rate')
    )
    op.create_index('ix_funding_rates_symbol_time', 'funding_rates',
                    ['symbol', 'funding_time'])
    op.create_index('ix_funding_rates_time', 'funding_rates', ['funding_time'])

    # Create open_interest table
    op.create_table(
        'open_interest',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('open_interest', sa.Numeric(30, 8), nullable=False),
        sa.Column('open_interest_value', sa.Numeric(30, 8), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timestamp', name='uq_open_interest')
    )
    op.create_index('ix_open_interest_symbol_time', 'open_interest',
                    ['symbol', 'timestamp'])
    op.create_index('ix_open_interest_time', 'open_interest', ['timestamp'])

    # Note: Run this manually if using TimescaleDB:
    #
    # -- Create hypertable for funding rates
    # SELECT create_hypertable('funding_rates', 'funding_time', 
    #     chunk_time_interval => INTERVAL '1 month',
    #     if_not_exists => TRUE
    # );
    #
    # -- Create hypertable for open interest
    # SELECT create_hypertable('open_interest', 'timestamp', 
    #     chunk_time_interval => INTERVAL '1 week',
    #     if_not_exists => TRUE
    # );


def downgrade() -> None:
    op.drop_table('open_interest')
    op.drop_table('funding_rates')
