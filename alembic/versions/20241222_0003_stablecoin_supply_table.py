"""Add stablecoin_supply table for liquidity tracking

Revision ID: 0003
Revises: 0002
Create Date: 2024-12-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0003'
down_revision: Union[str, None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create stablecoin_supply table
    op.create_table(
        'stablecoin_supply',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('market_cap', sa.Numeric(30, 2), nullable=False),
        sa.Column('supply_change_24h', sa.Numeric(10, 6), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timestamp', name='uq_stablecoin_supply')
    )
    op.create_index('ix_stablecoin_supply_symbol_time', 'stablecoin_supply',
                    ['symbol', 'timestamp'])
    op.create_index('ix_stablecoin_supply_time', 'stablecoin_supply', ['timestamp'])

    # Note: Run this manually if using TimescaleDB:
    #
    # SELECT create_hypertable('stablecoin_supply', 'timestamp', 
    #     chunk_time_interval => INTERVAL '1 month',
    #     if_not_exists => TRUE
    # );
    #
    # ALTER TABLE stablecoin_supply SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
    # SELECT add_compression_policy('stablecoin_supply', INTERVAL '1 week', if_not_exists => TRUE);


def downgrade() -> None:
    op.drop_table('stablecoin_supply')
