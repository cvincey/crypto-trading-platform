"""Database connection and session management."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from crypto.config.settings import get_settings
from crypto.data.models import Base

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """
    Get the async database engine.
    
    Creates the engine on first call using settings.
    
    Returns:
        AsyncEngine instance
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database.async_url,
            echo=settings.logging_config.level == "DEBUG",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        logger.info(f"Database engine created: {settings.database.host}")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get the async session factory.
    
    Returns:
        Session factory for creating database sessions
    """
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session.
    
    Usage:
        async with get_async_session() as session:
            result = await session.execute(...)
    
    Yields:
        AsyncSession instance
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """
    Initialize the database.
    
    Creates all tables if they don't exist.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")


async def init_timescale_hypertable() -> None:
    """
    Convert candles table to TimescaleDB hypertable.
    
    Should be called after init_db if using TimescaleDB.
    """
    async with get_async_session() as session:
        # Check if TimescaleDB is available
        try:
            result = await session.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
            )
            if not result.scalar():
                logger.warning(
                    "TimescaleDB extension not found. "
                    "Install it for optimal time-series performance."
                )
                return
        except Exception as e:
            logger.warning(f"Could not check for TimescaleDB: {e}")
            return

        # Create hypertable
        try:
            await session.execute(
                text("""
                    SELECT create_hypertable('candles', 'open_time',
                        chunk_time_interval => INTERVAL '1 week',
                        if_not_exists => TRUE
                    )
                """)
            )
            logger.info("TimescaleDB hypertable created for candles")
        except Exception as e:
            logger.warning(f"Could not create hypertable: {e}")


async def close_db() -> None:
    """Close database connections."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")


async def check_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        True if connection is successful
    """
    try:
        async with get_async_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
