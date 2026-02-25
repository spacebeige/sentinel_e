"""
============================================================
Sentinel-E — Multi-Model Fallback Router
============================================================
Circuit breaker + cascading fallback + query-aware model selection.

Wraps the existing ProviderRouter with:
  - Circuit breaker pattern (per-provider)
  - Exponential backoff with jitter
  - Provider health monitoring
  - Complexity-aware model selection
  - Automatic tier escalation on low confidence
  - Error-class detection

Memory safety: stdlib only. No heavy deps.
============================================================
"""

import time
import asyncio
import logging
import random
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("FallbackRouter")


# ============================================================
# CIRCUIT BREAKER
# ============================================================

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Provider down — skip calls
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Per-provider circuit breaker.

    CLOSED → OPEN after failure_threshold consecutive failures.
    OPEN → HALF_OPEN after recovery_timeout seconds.
    HALF_OPEN → CLOSED on success, back to OPEN on failure.
    """
    provider_name: str
    failure_threshold: int = 3
    recovery_timeout: float = 60.0  # seconds
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0

    def can_execute(self) -> bool:
        """Check if the circuit allows a request."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit {self.provider_name}: OPEN → HALF_OPEN (testing recovery)")
                return True
            return False
        # HALF_OPEN: allow one test request
        return True

    def record_success(self):
        """Record a successful call."""
        self.total_requests += 1
        self.total_successes += 1
        self.failure_count = 0
        self.last_success_time = time.time()
        if self.state != CircuitState.CLOSED:
            logger.info(f"Circuit {self.provider_name}: → CLOSED (recovered)")
        self.state = CircuitState.CLOSED

    def record_failure(self, error_type: str = "unknown"):
        """Record a failed call."""
        self.total_requests += 1
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.provider_name}: HALF_OPEN → OPEN (recovery failed: {error_type})")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit {self.provider_name}: CLOSED → OPEN "
                f"({self.failure_count} consecutive failures: {error_type})"
            )

    @property
    def health(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "success_rate": round(
                self.total_successes / self.total_requests, 3
            ) if self.total_requests > 0 else 1.0,
            "last_failure": self.last_failure_time,
            "last_success": self.last_success_time,
        }


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

class ErrorClass(Enum):
    """Classified error types for routing decisions."""
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    AUTH_FAILURE = "auth_failure"
    SERVER_ERROR = "server_error"
    INVALID_RESPONSE = "invalid_response"
    QUOTA_EXCEEDED = "quota_exceeded"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


def classify_error(error_message: str) -> ErrorClass:
    """Classify an error string into an error class."""
    msg = error_message.lower()
    if any(k in msg for k in ["429", "rate limit", "too many requests"]):
        return ErrorClass.RATE_LIMIT
    if any(k in msg for k in ["timeout", "timed out", "deadline"]):
        return ErrorClass.TIMEOUT
    if any(k in msg for k in ["401", "403", "unauthorized", "forbidden", "api key"]):
        return ErrorClass.AUTH_FAILURE
    if any(k in msg for k in ["500", "502", "503", "504", "internal server"]):
        return ErrorClass.SERVER_ERROR
    if any(k in msg for k in ["quota", "exceeded", "billing"]):
        return ErrorClass.QUOTA_EXCEEDED
    if any(k in msg for k in ["connection", "network", "dns", "refused"]):
        return ErrorClass.NETWORK_ERROR
    if any(k in msg for k in ["invalid", "parse", "unexpected"]):
        return ErrorClass.INVALID_RESPONSE
    return ErrorClass.UNKNOWN


# ============================================================
# MODEL TIER DEFINITIONS
# ============================================================

@dataclass
class ModelTier:
    """A model in the fallback chain."""
    name: str
    model_id: str
    tier: str          # budget | standard | premium
    priority: int      # lower = preferred for routing
    cost_weight: float # relative cost (1.0 = baseline)


# Default tier chain: cheapest → most capable
DEFAULT_MODEL_TIERS: List[ModelTier] = [
    ModelTier(name="Groq LLaMA 8B", model_id="llama-3.1-8b", tier="budget", priority=1, cost_weight=0.3),
    ModelTier(name="Qwen 2.5 7B", model_id="qwen-2.5-7b", tier="standard", priority=2, cost_weight=0.5),
    ModelTier(name="Groq LLaMA 70B", model_id="llama-3.3-70b", tier="premium", priority=3, cost_weight=1.0),
]


# ============================================================
# FALLBACK ROUTER
# ============================================================

class FallbackRouter:
    """
    Multi-model fallback router with circuit breakers and
    complexity-aware model selection.

    Does NOT replace ProviderRouter — wraps it.

    Usage:
        router = FallbackRouter(provider_router)
        result = await router.route(
            prompt="...",
            complexity="simple",   # from DepthAssessment
            mode="standard",
        )
    """

    def __init__(
        self,
        provider_router,  # ProviderRouter instance
        model_tiers: Optional[List[ModelTier]] = None,
        max_retries: int = 2,
        confidence_threshold: float = 0.4,
    ):
        self.provider = provider_router
        self.tiers = model_tiers or DEFAULT_MODEL_TIERS
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold

        # Circuit breakers per model
        self.circuits: Dict[str, CircuitBreaker] = {
            t.model_id: CircuitBreaker(provider_name=t.name)
            for t in self.tiers
        }

        # Request log for analytics
        self._request_log: List[Dict[str, Any]] = []
        self._max_log_size = 200

    def _select_initial_model(self, complexity: str) -> ModelTier:
        """Select starting model based on query complexity."""
        if complexity == "simple":
            # Use cheapest model for simple queries
            return self.tiers[0]  # budget
        elif complexity == "complex":
            # Use most capable model for complex queries
            return self.tiers[-1]  # premium
        else:
            # Standard/moderate → mid-tier
            return self.tiers[min(1, len(self.tiers) - 1)]

    def _get_fallback_chain(self, start_model: ModelTier) -> List[ModelTier]:
        """
        Build fallback chain starting from the given model.
        For simple queries: budget → standard → premium
        For complex queries: premium → standard → budget
        """
        start_idx = self.tiers.index(start_model)
        # Try remaining tiers in order of capability
        chain = [start_model]
        for tier in self.tiers:
            if tier not in chain:
                chain.append(tier)
        return chain

    async def route(
        self,
        prompt: str,
        complexity: str = "moderate",
        mode: str = "standard",
        system_role: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        messages: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Route a request through the fallback chain.

        Returns a dict with:
          - response: LLMResponse (from provider_router)
          - model_used: str
          - fallbacks_triggered: int
          - circuit_states: Dict
          - error_classes: List[str]
        """
        start_model = self._select_initial_model(complexity)
        chain = self._get_fallback_chain(start_model)

        errors_seen = []
        fallbacks = 0

        for tier in chain:
            circuit = self.circuits.get(tier.model_id)
            if circuit and not circuit.can_execute():
                logger.info(f"Skipping {tier.name} — circuit OPEN")
                fallbacks += 1
                continue

            # Retry loop for this model
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self.provider.generate(
                        model_id=tier.model_id,
                        prompt=prompt,
                        system_role=system_role,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=messages,
                    )

                    if response.success:
                        if circuit:
                            circuit.record_success()

                        result = {
                            "response": response,
                            "model_used": tier.model_id,
                            "model_name": tier.name,
                            "tier": tier.tier,
                            "fallbacks_triggered": fallbacks,
                            "attempts": attempt + 1,
                            "error_classes": [e["class"] for e in errors_seen],
                        }
                        self._log_request(result)
                        return result

                    # Failed response — classify error
                    error_class = classify_error(response.error or "")
                    errors_seen.append({
                        "model": tier.model_id,
                        "class": error_class.value,
                        "error": response.error,
                        "attempt": attempt,
                    })

                    if circuit:
                        circuit.record_failure(error_class.value)

                    # Don't retry auth/quota errors — they won't self-resolve
                    if error_class in (ErrorClass.AUTH_FAILURE, ErrorClass.QUOTA_EXCEEDED):
                        logger.warning(f"{tier.name}: {error_class.value} — skipping retries")
                        break

                    # Backoff with jitter before retry
                    if attempt < self.max_retries:
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(delay)

                except asyncio.TimeoutError:
                    errors_seen.append({
                        "model": tier.model_id,
                        "class": ErrorClass.TIMEOUT.value,
                        "error": "async timeout",
                        "attempt": attempt,
                    })
                    if circuit:
                        circuit.record_failure(ErrorClass.TIMEOUT.value)
                    break  # Move to next model on timeout

                except Exception as e:
                    error_class = classify_error(str(e))
                    errors_seen.append({
                        "model": tier.model_id,
                        "class": error_class.value,
                        "error": str(e),
                        "attempt": attempt,
                    })
                    if circuit:
                        circuit.record_failure(error_class.value)
                    if attempt < self.max_retries:
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(delay)

            fallbacks += 1

        # All models exhausted
        logger.error(f"All models exhausted. Errors: {errors_seen}")
        return {
            "response": None,
            "model_used": None,
            "model_name": "none",
            "tier": "none",
            "fallbacks_triggered": fallbacks,
            "attempts": 0,
            "error_classes": [e["class"] for e in errors_seen],
            "all_failed": True,
            "errors": errors_seen,
        }

    async def escalate(
        self,
        prompt: str,
        current_tier: str,
        reason: str = "low_confidence",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Escalate to a higher-tier model when confidence is low.
        """
        current_idx = next(
            (i for i, t in enumerate(self.tiers) if t.tier == current_tier),
            0,
        )
        if current_idx >= len(self.tiers) - 1:
            logger.info(f"Already at highest tier — cannot escalate (reason: {reason})")
            return {"escalated": False, "reason": "at_max_tier"}

        higher_tier = self.tiers[current_idx + 1]
        logger.info(f"Escalating: {current_tier} → {higher_tier.tier} (reason: {reason})")

        result = await self.route(
            prompt=prompt,
            complexity="complex",  # Force premium path
            **kwargs,
        )
        result["escalated"] = True
        result["escalation_reason"] = reason
        return result

    def _log_request(self, result: Dict[str, Any]):
        """Log routing decision for analytics."""
        self._request_log.append({
            "model": result.get("model_used"),
            "tier": result.get("tier"),
            "fallbacks": result.get("fallbacks_triggered", 0),
            "timestamp": time.time(),
        })
        if len(self._request_log) > self._max_log_size:
            self._request_log = self._request_log[-self._max_log_size:]

    def get_health(self) -> Dict[str, Any]:
        """Get health status for all providers."""
        return {
            model_id: circuit.health
            for model_id, circuit in self.circuits.items()
        }

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self._request_log:
            return {"total_requests": 0}

        tier_counts = {}
        fallback_total = 0
        for entry in self._request_log:
            tier = entry.get("tier", "unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            fallback_total += entry.get("fallbacks", 0)

        return {
            "total_requests": len(self._request_log),
            "tier_distribution": tier_counts,
            "total_fallbacks": fallback_total,
            "avg_fallbacks": round(fallback_total / len(self._request_log), 2),
        }


# ── Module-level singleton ──
_fallback_router: Optional[FallbackRouter] = None


def get_fallback_router(provider_router=None) -> FallbackRouter:
    """Get or create the singleton FallbackRouter."""
    global _fallback_router
    if _fallback_router is None:
        if provider_router is None:
            from providers.provider_router import get_provider_router
            provider_router = get_provider_router()
        _fallback_router = FallbackRouter(provider_router)
    return _fallback_router
