import os
import hashlib
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from google import genai
import redis.asyncio as redis

logger = logging.getLogger("VectorService")

class VectorService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.pc_api_key = os.getenv("PINECONE_API_KEY")
        self.pc_index_name = os.getenv("PINECONE_INDEX_NAME", "sentinel-e")
        self.google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.redis_url = os.getenv("REDIS_URL")

        if not self.pc_api_key or not self.google_api_key:
            logger.warning("PINECONE_API_KEY or GEMINI_API_KEY not set. Vector features will be limited.")
        
        try:
            self.pc = Pinecone(api_key=self.pc_api_key) if self.pc_api_key else None
            self.index = self.pc.Index(self.pc_index_name) if self.pc and self.pc_index_name else None
            
            # Using new google-genai SDK
            self.client = genai.Client(api_key=self.google_api_key) if self.google_api_key else None
            
            self.redis = redis.from_url(self.redis_url) if self.redis_url else None
        except Exception as e:
            logger.error(f"Failed to initialize VectorService: {e}")
            self.pc = self.index = self.client = self.redis = None

        self._initialized = True

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Gemini text-embedding-004."""
        if not self.client:
            return []
        
        text = text.strip()
        if not text:
            return []

        # Cache key
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = f"embed:gemini:{text_hash}"

        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        try:
            # google-genai SDK embedding call
            # Note: models/text-embedding-004 is current SOTA for Gemini embeddings
            response = await asyncio.to_thread(
                self.client.models.embed_content,
                model="text-embedding-004",
                contents=text
            )
            embedding = response.embeddings[0].values
            
            if self.redis:
                try:
                    await self.redis.set(cache_key, json.dumps(embedding), ex=3600*24)
                except Exception:
                    pass
            
            return embedding
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            return []

    async def upsert(self, namespace: str, items: List[Dict[str, Any]]):
        """Upsert items to Pinecone namespace."""
        if not self.index:
            return
        
        try:
            await asyncio.to_thread(
                self.index.upsert,
                vectors=items,
                namespace=namespace
            )
        except Exception as e:
            logger.error(f"Pinecone upsert failed in {namespace}: {e}")

    async def query(self, namespace: str, vector: List[float], top_k: int = 5, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query Pinecone namespace."""
        if not self.index or not vector:
            return []
        
        try:
            res = await asyncio.to_thread(
                self.index.query,
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=True
            )
            return [
                {
                    "id": m["id"],
                    "score": m["score"],
                    "metadata": m["metadata"]
                }
                for m in res.get("matches", [])
            ]
        except Exception as e:
            logger.error(f"Pinecone query failed in {namespace}: {e}")
            return []

def get_vector_service() -> VectorService:
    return VectorService()
