"""
============================================================
Database Connection Management — Neon PostgreSQL
============================================================

Features:
  • Neon-optimized connection pooling (pgBouncer compatible)
  • Async SQLAlchemy with asyncpg
  • Connection health checks
  • Graceful degradation on connection failure
  • Redis session cache (optional)

Configuration:
  • DATABASE_URL: Neon connection string
  • REDIS_URL: Redis connection string (optional)
  • DB_POOL_SIZE: Max connections (default: 20)
  • DB_MAX_OVERFLOW: Additional connections (default: 0)
"""

import os
import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

try:
    import redis.asyncio as redis
    HAS_REDIS_LIB = True
except Exception:
    redis = None
    HAS_REDIS_LIB = False

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, event
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# LOAD ENVIRONMENT
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logger = logging.getLogger("Database")

# ─────────────────────────────────────────────────────────────
# DATABASE URL CONFIGURATION
# ─────────────────────────────────────────────────────────────

DATABASE_URL_ENV = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

if DATABASE_URL_ENV:
    # Use Neon or external PostgreSQL
    if DATABASE_URL_ENV.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL_ENV.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif DATABASE_URL_ENV.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL_ENV.replace("postgres://", "postgresql+asyncpg://", 1)
    else:
        DATABASE_URL = DATABASE_URL_ENV
    
    # Clean up sslmode and channel_binding (asyncpg doesn't support in query string)
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    
    parsed = urlparse(DATABASE_URL)
    query_params = parse_qs(parsed.query)
    
    ssl_mode = query_params.pop("sslmode", [None])[0]
    channel_binding = query_params.pop("channel_binding", [None])[0]
    
    new_query = urlencode(query_params, doseq=True)
    DATABASE_URL = urlunparse(parsed._replace(query=new_query))
    
    connect_args = {}
    if ssl_mode and ssl_mode.lower() == "require":
        connect_args["ssl"] = "require"
    
    logger.info("✓ Using Neon/External PostgreSQL")
    IS_NEON = True
else:
    # Fallback to local PostgreSQL
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "sentinel_sigma")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    
    DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    connect_args = {}
    
    logger.warning("⚠️  Using local PostgreSQL (not recommended for production)")
    IS_NEON = False

# ─────────────────────────────────────────────────────────────
# CONNECTION POOLING CONFIGURATION
# ─────────────────────────────────────────────────────────────

DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "0"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# Connection timeout
if "timeout" not in connect_args:
    connect_args["timeout"] = 10

# ─────────────────────────────────────────────────────────────
# CREATE ASYNC ENGINE
# ─────────────────────────────────────────────────────────────

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args=connect_args,
    pool_pre_ping=True,  # Verify connections before use
)

# Create async session factory
async_session_factory = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# ─────────────────────────────────────────────────────────────
# SESSION DEPENDENCY
# ─────────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session.
    
    Usage:
        @app.get("/api/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"DB session error: {e}")
            raise
        finally:
            await session.close()


# ─────────────────────────────────────────────────────────────
# DATABASE INITIALIZATION & HEALTH CHECKS
# ─────────────────────────────────────────────────────────────

async def check_db_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connected, False otherwise
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1;"))
        logger.info("✓ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False


async def init_db():
    """
    Initialize database (create tables if they don't exist).
    
    Uses SQLAlchemy Base.metadata.create_all() pattern.
    For production, use Alembic migrations instead.
    """
    from database.models_v2 import Base
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
            # Individual column safety for newly added fields
            await conn.execute(text("ALTER TABLE chats ADD COLUMN IF NOT EXISTS engine VARCHAR"))
            await conn.execute(text("ALTER TABLE chats ADD COLUMN IF NOT EXISTS search_text TEXT"))
            
        logger.info("✓ Database tables initialized")
        return True
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        return False


async def check_db_health() -> dict:
    """
    Get detailed database health status.
    
    Returns:
        {
            connected: bool,
            pool_size: int,
            pool_checked_out: int,
            response_time_ms: float,
            is_neon: bool
        }
    """
    import time
    
    start = time.perf_counter()
    connected = await check_db_connection()
    response_time_ms = (time.perf_counter() - start) * 1000
    
    # Get pool info
    pool = engine.pool
    pool_info = {
        "connected": connected,
        "response_time_ms": round(response_time_ms, 2),
        "is_neon": IS_NEON,
    }
    
    # For QueuePool, add additional info
    if hasattr(pool, "size"):
        pool_info["pool_size"] = pool.size()
    if hasattr(pool, "checkedout"):
        pool_info["pool_checked_out"] = pool.checkedout()
    
    return pool_info


# ─────────────────────────────────────────────────────────────
# REDIS (OPTIONAL SESSION CACHE)
# ─────────────────────────────────────────────────────────────

redis_client: Optional[redis.Redis] = None

async def init_redis() -> bool:
    """Initialize Redis connection (optional)."""
    global redis_client
    
    if not HAS_REDIS_LIB:
        logger.warning("⚠️  Redis library not available (aioredis not installed)")
        return False
    
    REDIS_URL = os.getenv("REDIS_URL", "")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_DB = os.getenv("REDIS_DB", "0")
    
    try:
        if REDIS_URL:
            redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
        else:
            redis_client = await redis.Redis(
                host=REDIS_HOST,
                port=int(REDIS_PORT),
                db=int(REDIS_DB),
                decode_responses=True,
            )
        
        # Test connection
        await redis_client.ping()
        logger.info("✓ Redis connection successful")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Redis connection failed (cache disabled): {e}")
        redis_client = None
        return False


async def check_redis() -> bool:
    """Check if Redis is connected."""
    if redis_client is None:
        return False
    
    try:
        await redis_client.ping()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────

async def close_db():
    """Close database engine (call on app shutdown)."""
    await engine.dispose()
    logger.info("✓ Database engine closed")


async def close_redis():
    """Close Redis connection (call on app shutdown)."""
    if redis_client:
        await redis_client.close()
        logger.info("✓ Redis connection closed")


# ─────────────────────────────────────────────────────────────
# CONTEXT MANAGER FOR TRANSACTIONS
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def get_transactional_session():
    """
    Get a transactional session.
    
    Usage:
        async with get_transactional_session() as session:
            await my_crud_operation(session)
    """
    async with async_session_factory() as session:
        async with session.begin():
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
