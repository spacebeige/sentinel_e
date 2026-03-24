"""
Session Cache Manager with LRU & TTL expiration
Fixes: Global state memory leak, unbounded session growth
"""

from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import threading
import logging

logger = logging.getLogger(__name__)


class SessionCacheManager:
    """Thread-safe LRU cache with TTL expiration for session storage."""
    
    def __init__(self, max_sessions: int = 500, ttl_minutes: int = 60):
        """
        Initialize cache with max size and TTL.
        
        Args:
            max_sessions: Maximum number of sessions to keep in memory
            ttl_minutes: Time-to-live for each session in minutes
        """
        self.max_sessions = max_sessions
        self.ttl = timedelta(minutes=ttl_minutes)
        self._cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get session, return None if expired or not found."""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            
            # Check if expired
            if datetime.utcnow() - timestamp > self.ttl:
                del self._cache[key]
                logger.debug(f"Session {key} expired (TTL exceeded)")
                return None
            
            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Set session with current timestamp."""
        with self._lock:
            # Remove if exists to re-add at end
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = (value, datetime.utcnow())
            
            # Enforce max size by removing oldest
            while len(self._cache) > self.max_sessions:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.info(f"Evicted oldest session {oldest_key} (cache full: {self.max_sessions})")
    
    def delete(self, key: str) -> bool:
        """Delete session, return True if existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        with self._lock:
            now = datetime.utcnow()
            expired_keys = [
                k for k, (_, timestamp) in self._cache.items()
                if now - timestamp > self.ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired sessions")
            
            return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_sessions,
                "ttl_minutes": int(self.ttl.total_seconds() / 60),
                "utilization_percent": (len(self._cache) / self.max_sessions * 100)
            }
    
    # ── Dict-like interface for compatibility ──────────────────────────
    def __setitem__(self, key: str, value: Any) -> None:
        """Support dict-style assignment: cache[key] = value"""
        self.set(key, value)
    
    def __getitem__(self, key: str) -> Any:
        """Support dict-style access: value = cache[key]"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Session {key} not found (or expired)")
        return value
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator: if key in cache"""
        return self.get(key) is not None
    
    def pop(self, key: str, default: Any = None) -> Any:
        """Remove and return session, with optional default."""
        with self._lock:
            if key in self._cache:
                value, _ = self._cache[key]
                del self._cache[key]
                return value
            return default
    
    def keys(self):
        """Return iterator of current session keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def __len__(self) -> int:
        """Return number of sessions in cache."""
        with self._lock:
            return len(self._cache)
