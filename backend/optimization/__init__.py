"""
Sentinel-E Optimization Layer
==============================
Production-grade modules for token optimization, caching, fallback routing,
cost governance, and observability. All modules are memory-safe (no torch,
no FAISS, no heavy embeddings). Designed for 512MB RAM constraint.
"""

from optimization.token_optimizer import get_token_optimizer
from optimization.response_cache import get_response_cache
from optimization.fallback_router import get_fallback_router
from optimization.cost_governor import get_cost_governor
from optimization.observability import get_observability_hub

__all__ = [
    "get_token_optimizer",
    "get_response_cache",
    "get_fallback_router",
    "get_cost_governor",
    "get_observability_hub",
]
