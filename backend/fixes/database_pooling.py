"""
Database connection pooling fixes
Fixes: Connection exhaustion, race conditions, SQLite threading issues
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import logging
import os

logger = logging.getLogger(__name__)


def create_engine_with_proper_pooling(database_url: str):
    """
    Create SQLAlchemy async engine with proper connection pooling.
    
    Fixes:
    - NullPool creating new connection per query
    - SQLite threading issues
    - Connection exhaustion at scale
    
    Args:
        database_url: Database connection string
    
    Returns:
        Configured async engine
    """
    
    # Detect database type
    is_postgresql = "postgresql" in database_url.lower()
    is_sqlite = "sqlite" in database_url.lower()
    
    if is_sqlite:
        logger.warning(
            "⚠️  Using SQLite: Not recommended for production. Consider PostgreSQL for:\n"
            "  - Thread-safe concurrent access\n"
            "  - Proper connection pooling\n"
            "  - WAL mode support"
        )
        # SQLite-specific settings for better concurrency
        engine = create_async_engine(
            database_url,
            connect_args={
                "timeout": 30,  # Connection timeout
                "check_same_thread": False,  # Allow multi-threaded access (with caution)
            },
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            echo=False,  # Set to True for debugging SQL
        )
    elif is_postgresql:
        logger.info("✓ Using PostgreSQL with optimized pooling")
        engine = create_async_engine(
            database_url,
            # PostgreSQL connection pooling settings
            poolclass=QueuePool,
            pool_size=20,  # Number of connections to keep open
            max_overflow=40,  # Additional connections allowed
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,  # Set to True for debugging
        )
    else:
        logger.error(f"❌ Unsupported database: {database_url}")
        raise ValueError(f"Unsupported database URL: {database_url}")
    
    logger.info(
        "✓ Engine created with proper pooling: "
        f"pool_size=20, max_overflow=40, pool_pre_ping=True"
    )
    
    return engine


def get_async_session_factory(engine):
    """Create an async session factory with proper configuration."""
    return sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


# RECOMMENDED SETTINGS BY DATABASE
"""
PostgreSQL (Recommended for production):
└─ pool_size: 20 (adjust based on concurrent users)
└─ max_overflow: 40 (allow temporary excess connections)
└─ pool_recycle: 3600 (recycle old connections hourly)
└─ pool_pre_ping: True (verify connection alive)

MySQL:
└─ pool_size: 15
└─ max_overflow: 25
└─ pool_recycle: 1800

SQLite (Development only):
└─ pool_size: 5
└─ max_overflow: 10
└─ check_same_thread: False (with caution)

Rule of thumb: pool_size ≈ num_concurrent_requests / 2
"""
