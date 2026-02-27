import os
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load .env from backend root (one level up from database/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Database Configuration
# Prioritize full connection string if available (e.g. Neon, Render)
DATABASE_URL_ENV = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

if DATABASE_URL_ENV:
    # Ensure scheme is asyncpg compatible
    if DATABASE_URL_ENV.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL_ENV.replace("postgresql://", "postgresql+asyncpg://", 1)
    else:
        DATABASE_URL = DATABASE_URL_ENV
    
    # Robustly remove sslmode and channel_binding for asyncpg
    # asyncpg does not support 'sslmode' in the query string, it uses connect_args={"ssl": ...}
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    
    parsed = urlparse(DATABASE_URL)
    query_params = parse_qs(parsed.query)
    
    # Check if sslmode matches 'require' (ignoring case)
    ssl_mode = query_params.pop("sslmode", [None])[0]
    channel_binding = query_params.pop("channel_binding", [None])[0]
    
    # Reconstruct URL without these params
    new_query = urlencode(query_params, doseq=True)
    DATABASE_URL = urlunparse(parsed._replace(query=new_query))
            
    # Define connect_args for SSL if sslmode was present
    connect_args = {}
    if ssl_mode and ssl_mode.lower() == 'require':
        connect_args["ssl"] = "require"
else:
    # ... existing fallback ...
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "sentinel_sigma")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

    DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    connect_args = {}

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")

# SQLAlchemy Async Engine
import logging
_db_logger = logging.getLogger("Database")
_db_logger.info("Connecting to database...")
# SECURITY: Never log connection strings or credentials

# PostgreSQL Async Engine Configuration
from sqlalchemy.pool import NullPool

# Use NullPool for NeonDB to avoid connection issues with transaction poolers
# and ensure fresh connections are used.
engine = create_async_engine(
    DATABASE_URL, 
    echo=False, 
    future=True, 
    connect_args=connect_args, # Use the computed connect_args
    poolclass=NullPool 
)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Redis Client — prefer REDIS_URL (Render/Railway), fallback to host/port,
# gracefully degrade to in-memory LRU cache if Redis is unavailable.


class InMemoryRedisStub:
    """
    In-memory LRU fallback when Redis is unavailable.
    Supports setex/get/ping/delete — enough for session + metadata caching.
    NOT a full Redis replacement — no pub/sub, no persistence.
    """
    _MAX_KEYS = 512

    def __init__(self):
        from collections import OrderedDict
        self._store: 'OrderedDict[str, str]' = OrderedDict()
        self._is_stub = True

    async def ping(self):
        return True

    async def setex(self, key: str, ttl: int, value: str):
        if len(self._store) >= self._MAX_KEYS:
            self._store.popitem(last=False)  # evict oldest
        self._store[key] = value
        self._store.move_to_end(key)

    async def get(self, key: str):
        return self._store.get(key)

    async def delete(self, key: str):
        self._store.pop(key, None)

    async def keys(self, pattern: str = "*"):
        import fnmatch
        return [k for k in self._store if fnmatch.fnmatch(k, pattern)]


try:
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    else:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=int(REDIS_PORT),
            db=int(REDIS_DB),
            decode_responses=True,
        )
except Exception:
    redis_client = InMemoryRedisStub()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    # Only needed if using SQLAlchemy to create tables
    # For production, use Alembic migrations
    from .models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def check_redis():
    try:
        await redis_client.ping()
        if getattr(redis_client, '_is_stub', False):
            print("Redis unavailable — using in-memory LRU fallback.")
        else:
            print("Redis connection successful.")
        return True
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return False
