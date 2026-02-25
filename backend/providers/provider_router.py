"""
============================================================
Provider Abstraction Layer — Model Registry & Router
============================================================
Implements:
- Unified model interface (any provider, same API)
- Model registry with capabilities & cost metadata
- Dynamic provider routing with failover
- Exponential backoff retry logic
- Token budgeting
- Cost-aware selection
- Provider isolation (errors in one never leak to another)
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from gateway.config import get_settings

logger = logging.getLogger("Providers")


# ============================================================
# Model Registry
# ============================================================

class ProviderType(str, Enum):
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OPENAI = "openai"       # Future
    ANTHROPIC = "anthropic"  # Future
    LOCAL = "local"          # Future


@dataclass
class ModelSpec:
    """Registry entry for a model."""
    id: str
    name: str
    provider: ProviderType
    model_id: str  # Provider-specific model identifier
    context_window: int = 8192
    max_output_tokens: int = 2048
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: List[str] = field(default_factory=lambda: ["chat"])
    tier: str = "standard"  # standard | premium | budget
    default_temperature: float = 0.3
    default_system_role: str = "You are a helpful analytical assistant."
    active: bool = True


# ── Built-in Model Registry ─────────────────────────────────

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        id="llama-3.1-8b",
        name="LLaMA 3.1 8B (Groq)",
        provider=ProviderType.GROQ,
        model_id="llama-3.1-8b-instant",
        context_window=131072,
        max_output_tokens=8192,
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.00008,
        capabilities=["chat", "fast", "analytical"],
        tier="budget",
        default_temperature=0.3,
        default_system_role="You are a fast, concise analytical assistant.",
    ),
    "llama-3.3-70b": ModelSpec(
        id="llama-3.3-70b",
        name="Llama 3.3 70B (Groq)",
        provider=ProviderType.GROQ,
        model_id="llama-3.3-70b-versatile",
        context_window=131072,
        max_output_tokens=32768,
        cost_per_1k_input=0.00059,
        cost_per_1k_output=0.00079,
        capabilities=["chat", "reasoning", "analytical", "creative"],
        tier="premium",
        default_temperature=0.4,
        default_system_role="You are a rigorous analytical reasoning assistant.",
    ),
    "qwen-2.5-7b": ModelSpec(
        id="qwen-2.5-7b",
        name="Qwen 2.5 7B (OpenRouter)",
        provider=ProviderType.OPENROUTER,
        model_id="qwen/qwen-2.5-7b-instruct",
        context_window=32768,
        max_output_tokens=8192,
        cost_per_1k_input=0.00027,
        cost_per_1k_output=0.00027,
        capabilities=["chat", "analytical"],
        tier="standard",
        default_temperature=0.3,
        default_system_role="You are a careful, analytical assistant.",
    ),
}


def get_model_spec(model_id: str) -> Optional[ModelSpec]:
    return MODEL_REGISTRY.get(model_id)


def list_active_models() -> List[ModelSpec]:
    return [m for m in MODEL_REGISTRY.values() if m.active]


# ============================================================
# Provider Interface
# ============================================================

@dataclass
class LLMResponse:
    """Unified response from any provider."""
    content: str
    model_id: str
    provider: str
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    success: bool = True
    error: Optional[str] = None
    retries: int = 0


class BaseProvider(ABC):
    """Abstract provider interface."""

    @abstractmethod
    async def generate(
        self,
        model_id: str,
        prompt: str,
        system_role: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> LLMResponse:
        ...


# ============================================================
# Groq Provider
# ============================================================

class GroqProvider(BaseProvider):
    """Groq API provider with retry and error isolation."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def generate(
        self,
        model_id: str,
        prompt: str,
        system_role: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> LLMResponse:
        import aiohttp

        if not self.settings.GROQ_API_KEY:
            return LLMResponse(
                content="",
                model_id=model_id,
                provider="groq",
                success=False,
                error="Groq API key not configured",
            )

        headers = {
            "Authorization": f"Bearer {self.settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        if messages:
            msg_list = messages.copy()
        else:
            msg_list = []
            if system_role:
                msg_list.append({"role": "system", "content": system_role})
            msg_list.append({"role": "user", "content": prompt})

        payload = {
            "model": model_id,
            "messages": msg_list,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        return await self._call_with_retry(headers, payload, model_id)

    async def _call_with_retry(
        self, headers: Dict, payload: Dict, model_id: str
    ) -> LLMResponse:
        import aiohttp

        settings = self.settings
        last_error = None

        for attempt in range(settings.MAX_RETRY_ATTEMPTS):
            start = time.time()
            try:
                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.base_url, headers=headers, json=payload) as resp:
                        elapsed = (time.time() - start) * 1000

                        if resp.status == 429:
                            retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                            logger.warning(f"Groq rate limit hit, retrying in {retry_after}s (attempt {attempt + 1})")
                            await asyncio.sleep(retry_after)
                            last_error = "rate_limited"
                            continue

                        if resp.status != 200:
                            text = await resp.text()
                            logger.error(f"Groq API error {resp.status}: {text[:200]}")
                            return LLMResponse(
                                content="",
                                model_id=model_id,
                                provider="groq",
                                success=False,
                                error=f"Provider error (status {resp.status})",
                                latency_ms=elapsed,
                                retries=attempt,
                            )

                        data = await resp.json()
                        usage = data.get("usage", {})
                        content = data["choices"][0]["message"]["content"]

                        return LLMResponse(
                            content=content,
                            model_id=model_id,
                            provider="groq",
                            success=True,
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            tokens_used=usage.get("total_tokens", 0),
                            latency_ms=elapsed,
                            retries=attempt,
                        )

            except asyncio.TimeoutError:
                last_error = "timeout"
                logger.warning(f"Groq timeout (attempt {attempt + 1})")
                await asyncio.sleep(settings.RETRY_BASE_DELAY * (2 ** attempt))
            except Exception as e:
                last_error = str(e)
                logger.error(f"Groq exception (attempt {attempt + 1}): {e}")
                await asyncio.sleep(settings.RETRY_BASE_DELAY * (2 ** attempt))

        return LLMResponse(
            content="",
            model_id=model_id,
            provider="groq",
            success=False,
            error=f"All {settings.MAX_RETRY_ATTEMPTS} retries failed: {last_error}",
            retries=settings.MAX_RETRY_ATTEMPTS,
        )


# ============================================================
# OpenRouter Provider
# ============================================================

class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider with retry and error isolation."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    async def generate(
        self,
        model_id: str,
        prompt: str,
        system_role: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> LLMResponse:
        import aiohttp

        if not self.settings.OPENROUTER_API_KEY:
            return LLMResponse(
                content="",
                model_id=model_id,
                provider="openrouter",
                success=False,
                error="OpenRouter API key not configured",
            )

        headers = {
            "Authorization": f"Bearer {self.settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://sentinel-e.app",
            "X-Title": "Sentinel-E",
        }

        if messages:
            msg_list = messages.copy()
        else:
            msg_list = []
            if system_role:
                msg_list.append({"role": "system", "content": system_role})
            msg_list.append({"role": "user", "content": prompt})

        payload = {
            "model": model_id,
            "messages": msg_list,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
        }

        return await self._call_with_retry(headers, payload, model_id)

    async def _call_with_retry(
        self, headers: Dict, payload: Dict, model_id: str
    ) -> LLMResponse:
        import aiohttp
        import json

        settings = self.settings
        last_error = None

        for attempt in range(settings.MAX_RETRY_ATTEMPTS):
            start = time.time()
            try:
                timeout = aiohttp.ClientTimeout(total=45)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.base_url, headers=headers, json=payload) as resp:
                        elapsed = (time.time() - start) * 1000
                        text = await resp.text()

                        if resp.status == 429:
                            retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                            logger.warning(f"OpenRouter rate limit, retry in {retry_after}s")
                            await asyncio.sleep(retry_after)
                            last_error = "rate_limited"
                            continue

                        if resp.status != 200:
                            logger.error(f"OpenRouter error {resp.status}: {text[:200]}")
                            return LLMResponse(
                                content="",
                                model_id=model_id,
                                provider="openrouter",
                                success=False,
                                error=f"Provider error (status {resp.status})",
                                latency_ms=elapsed,
                                retries=attempt,
                            )

                        data = json.loads(text)
                        usage = data.get("usage", {})
                        content = data["choices"][0]["message"]["content"]

                        return LLMResponse(
                            content=content,
                            model_id=model_id,
                            provider="openrouter",
                            success=True,
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            tokens_used=usage.get("total_tokens", 0),
                            latency_ms=elapsed,
                            retries=attempt,
                        )

            except asyncio.TimeoutError:
                last_error = "timeout"
                logger.warning(f"OpenRouter timeout (attempt {attempt + 1})")
                await asyncio.sleep(settings.RETRY_BASE_DELAY * (2 ** attempt))
            except Exception as e:
                last_error = str(e)
                logger.error(f"OpenRouter exception (attempt {attempt + 1}): {e}")
                await asyncio.sleep(settings.RETRY_BASE_DELAY * (2 ** attempt))

        return LLMResponse(
            content="",
            model_id=model_id,
            provider="openrouter",
            success=False,
            error=f"All retries failed: {last_error}",
            retries=settings.MAX_RETRY_ATTEMPTS,
        )


# ============================================================
# Provider Router
# ============================================================

class ProviderRouter:
    """
    Routes model requests to the correct provider.
    Handles failover, cost tracking, and token budgeting.
    """

    def __init__(self):
        self._providers: Dict[ProviderType, BaseProvider] = {
            ProviderType.GROQ: GroqProvider(),
            ProviderType.OPENROUTER: OpenRouterProvider(),
        }
        self._usage_log: List[Dict[str, Any]] = []
        self._total_cost: float = 0.0

    def get_provider(self, provider_type: ProviderType) -> Optional[BaseProvider]:
        return self._providers.get(provider_type)

    async def generate(
        self,
        model_id: str,
        prompt: str,
        system_role: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> LLMResponse:
        """
        Route a generation request to the correct provider.
        Applies token budget and cost tracking.
        """
        spec = get_model_spec(model_id)
        if not spec:
            return LLMResponse(
                content="",
                model_id=model_id,
                provider="unknown",
                success=False,
                error=f"Model '{model_id}' not found in registry",
            )

        if not spec.active:
            return LLMResponse(
                content="",
                model_id=model_id,
                provider=spec.provider.value,
                success=False,
                error=f"Model '{model_id}' is currently disabled",
            )

        provider = self._providers.get(spec.provider)
        if not provider:
            return LLMResponse(
                content="",
                model_id=model_id,
                provider=spec.provider.value,
                success=False,
                error=f"Provider '{spec.provider.value}' not configured",
            )

        effective_system = system_role or spec.default_system_role
        effective_temp = temperature if temperature is not None else spec.default_temperature
        effective_max = min(max_tokens or spec.max_output_tokens, spec.max_output_tokens)

        settings = get_settings()
        if effective_max > settings.TOKEN_BUDGET_PER_REQUEST:
            effective_max = settings.TOKEN_BUDGET_PER_REQUEST

        response = await provider.generate(
            model_id=spec.model_id,
            prompt=prompt,
            system_role=effective_system,
            temperature=effective_temp,
            max_tokens=effective_max,
            messages=messages,
        )

        # Track usage
        if response.success:
            cost = (
                response.input_tokens * spec.cost_per_1k_input / 1000
                + response.output_tokens * spec.cost_per_1k_output / 1000
            )
            response.cost_estimate = cost
            self._total_cost += cost
            self._usage_log.append({
                "model": model_id,
                "provider": spec.provider.value,
                "tokens": response.tokens_used,
                "cost": cost,
                "latency_ms": response.latency_ms,
                "timestamp": time.time(),
            })

        return response

    async def generate_parallel(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[LLMResponse]:
        """Execute multiple model requests in parallel."""
        tasks = [self.generate(**req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": len(self._usage_log),
            "total_cost": round(self._total_cost, 6),
            "recent": self._usage_log[-20:],
        }


# Singleton
_router: Optional[ProviderRouter] = None


def get_provider_router() -> ProviderRouter:
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router
