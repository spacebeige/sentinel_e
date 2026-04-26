import os
import uuid
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import redis.asyncio as redis

logger = logging.getLogger("DAGLogger")

class CallNode:
    def __init__(
        self,
        call_id: str,
        user_id: str,
        session_id: str,
        endpoint: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        status: str = "pending",
        depends_on: List[str] = None
    ):
        self.call_id = call_id
        self.user_id = user_id
        self.session_id = session_id
        self.endpoint = endpoint
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.status = status
        self.depends_on = depends_on or []
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self):
        return self.__dict__

class DAGLogger:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL")
        self.redis = redis.from_url(self.redis_url) if self.redis_url else None
        self._in_memory_store = {} # Fallback

    async def log_call(self, node: CallNode):
        key = f"dag:{node.session_id}"
        data = json.dumps(node.to_dict())
        
        if self.redis:
            await self.redis.hset(key, node.call_id, data)
            await self.redis.expire(key, 3600 * 24) # 24h
        else:
            if key not in self._in_memory_store:
                self._in_memory_store[key] = {}
            self._in_memory_store[key][node.call_id] = data

    async def get_session_graph(self, session_id: str) -> List[Dict[str, Any]]:
        key = f"dag:{session_id}"
        if self.redis:
            nodes = await self.redis.hgetall(key)
            return [json.loads(v) for v in nodes.values()]
        else:
            nodes = self._in_memory_store.get(key, {})
            return [json.loads(v) for v in nodes.values()]

    async def get_critical_path(self, session_id: str) -> List[str]:
        """Simple critical path calculation (longest latency chain)."""
        graph = await self.get_session_graph(session_id)
        if not graph:
            return []
        
        # Build adjacency list
        adj = {node['call_id']: [] for node in graph}
        latencies = {node['call_id']: node['latency_ms'] for node in graph}
        
        for node in graph:
            for dep in node['depends_on']:
                if dep in adj:
                    adj[dep].append(node['call_id'])
        
        # DP for longest path
        memo = {}
        
        def find_longest(u):
            if u in memo:
                return memo[u]
            
            max_lat = latencies[u]
            path = [u]
            
            for v in adj[u]:
                v_lat, v_path = find_longest(v)
                if latencies[u] + v_lat > max_lat:
                    max_lat = latencies[u] + v_lat
                    path = [u] + v_path
            
            memo[u] = (max_lat, path)
            return memo[u]
        
        # Start from nodes with no dependencies (roots)
        roots = [node['call_id'] for node in graph if not node['depends_on']]
        
        best_path = []
        best_lat = -1
        
        for root in roots:
            lat, path = find_longest(root)
            if lat > best_lat:
                best_lat = lat
                best_path = path
                
        return best_path

_logger = None
def get_dag_logger() -> DAGLogger:
    global _logger
    if _logger is None:
        _logger = DAGLogger()
    return _logger
