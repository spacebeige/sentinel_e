import os
import logging
import json

try:
    import redis.asyncio as redis
    HAS_REDIS_LIB = True
except ImportError:
    HAS_REDIS_LIB = False

logger = logging.getLogger("RedisClient")

class RedisClient:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL")
        self.client = None
        self.connected = False
        
    async def connect(self):
        if not HAS_REDIS_LIB:
            logger.warning("redis-py not installed. Caching DISABLED.")
            return

        if not self.redis_url:
            logger.warning("REDIS_URL not found. Caching DISABLED.")
            return

        try:
            logger.info(f"Connecting to Redis at {self.redis_url}...")
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            # Ping to verify
            await self.client.ping()
            self.connected = True
            logger.info("Redis Connected.")
        except Exception as e:
            logger.error(f"Redis Connection Failed: {e}. Caching disabled.")
            self.connected = False

    async def get(self, key):
        if not self.connected: 
            return None
        try:
            return await self.client.get(key)
        except Exception:
            return None

    async def set(self, key, value, ttl=None):
        if not self.connected: 
            return
        try:
            if ttl:
                await self.client.setex(key, ttl, value)
            else:
                await self.client.set(key, value)
        except Exception as e:
            logger.warning(f"Redis Set Failed: {e}")
            
    async def close(self):
        if self.client:
            await self.client.close()
