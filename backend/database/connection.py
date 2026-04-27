import os
try:
    import redis.asyncio as redis
    HAS_REDIS_LIB = True
except Exception:  # pragma: no cover - environment-dependent
    redis = None
    HAS_REDIS_LIB = False
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
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

# NOTE: For async engines (create_async_engine), SQLAlchemy does NOT support QueuePool
# Use NullPool for async PostgreSQL connections - connection pooling is handled by:
# 1. asyncpg's built-in connection pooling (backend default)
# 2. pgbouncer or similar proxy (Render deployment default)
# 3. PostgreSQL's connection pooling at the server level
from sqlalchemy.pool import NullPool

# PostgreSQL Async Engine Configuration
if "timeout" not in connect_args:
    connect_args["timeout"] = 10  # asyncpg connection timeout in seconds

engine = create_async_engine(
    DATABASE_URL, 
    echo=False, 
    future=True, 
    connect_args=connect_args,
    poolclass=NullPool,  # Required for async engines - connection pooling handled by pgbouncer/asyncpg
    pool_pre_ping=True,     # Verify connection health before using
    pool_recycle=3600,      # Recycle connections after 1 hour
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
    if HAS_REDIS_LIB and REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5, socket_connect_timeout=5)
    elif HAS_REDIS_LIB:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=int(REDIS_PORT),
            db=int(REDIS_DB),
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
    else:
        redis_client = InMemoryRedisStub()
except Exception:
    redis_client = InMemoryRedisStub()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """Idempotent database initialization."""
    try:
        from .models import Base
        async with engine.begin() as conn:
            # 1. Create all tables defined in models.py
            await conn.run_sync(Base.metadata.create_all)
            
            # 2. Individual column safety (for migrations that metadata.create_all skips)
            await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS clerk_user_id VARCHAR"))
            await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS provider VARCHAR"))
            await conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS image_b64 TEXT"))
            await conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS reasoning_json JSONB"))
            await conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS metadata_json JSONB"))
            await conn.execute(text("ALTER TABLE user_memory ADD COLUMN IF NOT EXISTS weight FLOAT DEFAULT 1.0"))
            await conn.execute(text("ALTER TABLE user_memory ADD COLUMN IF NOT EXISTS last_used TIMESTAMP DEFAULT NOW()"))
            await conn.execute(text("ALTER TABLE user_memory ADD COLUMN IF NOT EXISTS recency_score FLOAT DEFAULT 1.0"))

            # SQLite fallback path (IF NOT EXISTS unsupported on older sqlite builds)
            if conn.dialect.name == "sqlite":
                try:
                    await conn.execute(text("ALTER TABLE user_memory ADD COLUMN weight REAL DEFAULT 1.0"))
                except Exception:
                    pass
                try:
                    await conn.execute(text("ALTER TABLE user_memory ADD COLUMN last_used TEXT DEFAULT (datetime('now'))"))
                except Exception:
                    pass
                try:
                    await conn.execute(text("ALTER TABLE user_memory ADD COLUMN recency_score REAL DEFAULT 1.0"))
                except Exception:
                    pass
            
            # 3. Create indices for performance and isolation
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_memory_user_id ON user_memory(user_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_preferences_user_id ON user_preferences(user_id)"))
            
            _db_logger.info("✅ Database schema synchronized successfully")
    except Exception as e:
        _db_logger.warning(f"Database sync warning: {e}")
    except Exception as e:
        _db_logger.warning(f"Database init failed (non-fatal, will retry on first request): {e}")

async def check_redis():
    global redis_client
    try:
        await redis_client.ping()
        if getattr(redis_client, '_is_stub', False):
            _db_logger.info("Redis unavailable — using in-memory LRU fallback.")
        else:
            _db_logger.info("Redis connection successful.")
        return True
    except Exception as e:
        _db_logger.warning(f"Redis connection failed (non-fatal, using in-memory fallback): {e}")
        redis_client = InMemoryRedisStub()
        return False
