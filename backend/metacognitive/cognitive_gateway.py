"""
============================================================
API 1 — Cognitive Model Gateway
============================================================
Pure reasoning execution. Each model is a separate endpoint.
No cross-model contamination. No retrieval. No session mutation.
No persistence logic. No knowledge injection decisions.
No cutoff disclaimers allowed.

Models (configurable via environment):
  - Qwen3 Coder 480B A35B    → Code-heavy tasks
  - Qwen3 VL 30B A3B         → Image/vision tasks
  - Nemotron 3 Nano 30B A3B  → Baseline comparator
  - Llama 3.3 30B            → Conceptual reasoning
  - Kimi 2.5                 → Long-context reasoning
============================================================
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import aiohttp

from gateway.config import get_settings
from metacognitive.schemas import (
    CognitiveGatewayInput,
    CognitiveGatewayOutput,
    ModelRole,
)

logger = logging.getLogger("MCO-CognitiveGateway")


# ============================================================
# Model Configuration
# ============================================================

@dataclass
class CognitiveModelSpec:
    """Specification for a cognitive model endpoint."""
    name: str
    model_id: str           # Provider-specific model ID
    provider: str           # groq | openrouter | qwen | nvidia | kimi
    role: ModelRole         # Routing hint
    context_window: int = 131072
    max_output_tokens: int = 8192
    default_temperature: float = 0.3
    api_base_url: str = ""  # Provider base URL
    api_key_env: str = ""   # Environment variable name for this model's key
    active: bool = True     # Structural flag (can be toggled manually)
    enabled: bool = True    # Runtime flag (auto-set based on key availability)


# ── Model Registry ───────────────────────────────────────────
# These are the five models from the spec. Override via env vars.

COGNITIVE_MODEL_REGISTRY: Dict[str, CognitiveModelSpec] = {
    "qwen3-coder": CognitiveModelSpec(
        name="Qwen3 Coder 480B A35B",
        model_id="qwen/qwen3-coder-480b-a35b",
        provider="qwen",
        role=ModelRole.CODE,
        context_window=131072,
        max_output_tokens=16384,
        default_temperature=0.2,
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="QWEN3_CODER_API_KEY",
    ),
    "qwen3-vl": CognitiveModelSpec(
        name="Qwen3 VL 30B A3B Thinking",
        model_id="qwen/qwen3-vl-30b-a3b",
        provider="qwen",
        role=ModelRole.VISION,
        context_window=32768,
        max_output_tokens=8192,
        default_temperature=0.3,
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="QWEN3_VL_API_KEY",
    ),
    "nemotron-nano": CognitiveModelSpec(
        name="Nemotron 3 Nano 30B A3B",
        model_id="nvidia/nemotron-3-nano-30b-a3b",
        provider="nvidia",
        role=ModelRole.BASELINE,
        context_window=32768,
        max_output_tokens=4096,
        default_temperature=0.3,
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="NEMOTRON_API_KEY",
    ),
    "llama-3.3": CognitiveModelSpec(
        name="Llama 3.3 30B",
        model_id="meta-llama/llama-3.3-70b-instruct",
        provider="groq",
        role=ModelRole.CONCEPTUAL,
        context_window=131072,
        max_output_tokens=32768,
        default_temperature=0.4,
        api_base_url="https://api.groq.com/openai/v1/chat/completions",
        api_key_env="GROQ_API_KEY",
    ),
    "kimi-2.5": CognitiveModelSpec(
        name="Kimi 2.5",
        model_id="moonshotai/kimi-k2",
        provider="kimi",
        role=ModelRole.LONGCTX,
        context_window=262144,
        max_output_tokens=16384,
        default_temperature=0.3,
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="KIMI_API_KEY",
    ),
}


def _initialize_registry():
    """
    Runtime initialization: check env for each model's API key.
    If key is missing, mark model as disabled (not active).
    Runs once at import time. Non-blocking, never crashes.
    """
    for key, spec in COGNITIVE_MODEL_REGISTRY.items():
        if not spec.api_key_env:
            # No env var configured — model stays as-is
            continue
        api_key = os.getenv(spec.api_key_env, "")
        if api_key:
            spec.enabled = True
            logger.info(f"Model '{key}' ({spec.provider}): enabled")
        else:
            spec.enabled = False
            spec.active = False
            logger.warning(
                f"Model '{key}' ({spec.provider}): DISABLED — "
                f"{spec.api_key_env} not set"
            )


# Run at import time — safe, non-blocking
_initialize_registry()


def get_models_for_task(
    code_heavy: bool = False,
    image_heavy: bool = False,
    conceptual: bool = False,
    long_context: bool = False,
) -> List[str]:
    """
    Return ALL enabled models for every task.

    The arbitration engine (Step 7) handles scoring and selection.
    No model is excluded based on task type — all enabled models
    participate in every invocation for fair scoring.
    """
    return [
        k for k in COGNITIVE_MODEL_REGISTRY
        if COGNITIVE_MODEL_REGISTRY[k].active
        and COGNITIVE_MODEL_REGISTRY[k].enabled
    ]


# ============================================================
# Cognitive Model Gateway
# ============================================================

class CognitiveModelGateway:
    """
    API 1 — Pure reasoning execution.
    
    Restrictions enforced:
      ✗ No retrieval
      ✗ No session mutation
      ✗ No persistence logic
      ✗ No knowledge injection decisions
      ✗ No cutoff disclaimers
    
    All models receive identical stabilized context.
    Raw outputs are returned unmodified.
    """

    CUTOFF_DISCLAIMER_PATTERNS = [
        "my knowledge cutoff",
        "my training data",
        "as of my last update",
        "I don't have access to real-time",
        "I cannot browse",
        "my training only goes up to",
    ]

    def __init__(self):
        self.settings = get_settings()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Public Interface ─────────────────────────────────────

    def _resolve_api_key(self, spec: CognitiveModelSpec) -> Optional[str]:
        """
        Resolve the API key for a model from its dedicated env var.
        Provider isolation: each model uses its own key.
        Returns None if key not available.
        """
        if spec.api_key_env:
            key = os.getenv(spec.api_key_env, "")
            if key:
                return key
        # No dedicated key — model cannot be invoked
        return None

    async def invoke_model(
        self,
        model_key: str,
        gateway_input: CognitiveGatewayInput,
    ) -> CognitiveGatewayOutput:
        """
        Invoke a single model with stabilized context.
        Returns raw output — no modification before scoring.
        """
        spec = COGNITIVE_MODEL_REGISTRY.get(model_key)
        if not spec or not spec.active or not spec.enabled:
            return CognitiveGatewayOutput(
                model_name=model_key,
                raw_output="",
                success=False,
                error=f"Model '{model_key}' not found or disabled",
            )

        # Resolve per-model API key (provider isolation)
        api_key = self._resolve_api_key(spec)
        if not api_key:
            return CognitiveGatewayOutput(
                model_name=spec.name,
                raw_output="",
                success=False,
                error=f"{spec.api_key_env} not configured",
            )

        # Build messages from stabilized context
        messages = self._build_messages(gateway_input, spec)

        start = time.monotonic()
        try:
            if spec.provider == "groq":
                result = await self._call_groq(spec, messages, api_key)
            elif spec.provider in ("qwen", "nvidia", "kimi", "openrouter"):
                result = await self._call_openrouter_isolated(
                    spec, messages, api_key,
                )
            elif spec.provider == "openai":
                result = await self._call_openai(spec, messages)
            else:
                result = CognitiveGatewayOutput(
                    model_name=spec.name,
                    raw_output="",
                    success=False,
                    error=f"Unsupported provider: {spec.provider}",
                )
        except Exception as e:
            logger.error(f"Model invocation failed [{model_key}]: {e}")
            result = CognitiveGatewayOutput(
                model_name=spec.name,
                raw_output="",
                success=False,
                error=str(e),
            )

        result.latency_ms = (time.monotonic() - start) * 1000

        # Enforce no-cutoff-disclaimer rule (flag, don't modify)
        if result.success:
            self._flag_cutoff_disclaimers(result)

        return result

    async def invoke_parallel(
        self,
        model_keys: List[str],
        gateway_input: CognitiveGatewayInput,
    ) -> List[CognitiveGatewayOutput]:
        """
        Invoke multiple models in parallel with identical context.
        No cross-model contamination — each gets a fresh copy.
        """
        tasks = [
            self.invoke_model(key, gateway_input.model_copy(deep=True))
            for key in model_keys
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                outputs.append(CognitiveGatewayOutput(
                    model_name=model_keys[i],
                    raw_output="",
                    success=False,
                    error=str(result),
                ))
            else:
                outputs.append(result)
        return outputs

    # ── Message Building ─────────────────────────────────────

    def _build_messages(
        self,
        inp: CognitiveGatewayInput,
        spec: CognitiveModelSpec,
    ) -> List[Dict[str, str]]:
        """
        Build message array from stabilized context.
        Injects knowledge bundle and session summary as system context.
        """
        messages = []

        # System prompt — role-specific, no cutoff framing
        system_parts = [
            "You are a rigorous analytical reasoning assistant. "
            "Provide precise, well-structured answers grounded in the provided context. "
            "Never mention knowledge cutoffs or training data limitations."
        ]

        # Inject knowledge bundle
        if inp.knowledge_bundle:
            kb_text = "\n\n[KNOWLEDGE CONTEXT]\n"
            for block in inp.knowledge_bundle:
                kb_text += f"Source: {block.source}\n"
                kb_text += f"Content: {block.content}\n\n"
            system_parts.append(kb_text)

        # Inject session summary
        if inp.session_summary:
            summary_text = "\n\n[SESSION CONTEXT]\n"
            for key, val in inp.session_summary.items():
                summary_text += f"{key}: {val}\n"
            system_parts.append(summary_text)

        # Inject stabilized context
        if inp.stabilized_context:
            ctx_text = "\n\n[STABILIZED CONTEXT]\n"
            for key, val in inp.stabilized_context.items():
                if isinstance(val, (list, dict)):
                    import json
                    ctx_text += f"{key}: {json.dumps(val, default=str)}\n"
                else:
                    ctx_text += f"{key}: {val}\n"
            system_parts.append(ctx_text)

        messages.append({"role": "system", "content": "\n".join(system_parts)})
        messages.append({"role": "user", "content": inp.user_query})

        return messages

    # ── Provider Implementations ─────────────────────────────

    async def _call_groq(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
        api_key: str = "",
    ) -> CognitiveGatewayOutput:
        """Call Groq API with per-model isolated key."""
        if not api_key:
            return CognitiveGatewayOutput(
                model_name=spec.name, raw_output="",
                success=False, error=f"{spec.api_key_env} not configured",
            )

        url = spec.api_base_url or "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": spec.model_id,
            "messages": messages,
            "temperature": spec.default_temperature,
            "max_tokens": spec.max_output_tokens,
        }

        session = await self._get_session()
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                return CognitiveGatewayOutput(
                    model_name=spec.name, raw_output="",
                    success=False, error=f"Groq {resp.status}: {text[:500]}",
                )
            data = await resp.json()
            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            return CognitiveGatewayOutput(
                model_name=spec.name,
                raw_output=choice.get("message", {}).get("content", ""),
                tokens_used=usage.get("total_tokens", 0),
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                success=True,
            )

    async def _call_openrouter_isolated(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
        api_key: str = "",
    ) -> CognitiveGatewayOutput:
        """
        Call OpenRouter-compatible API with per-model isolated key.
        Used by qwen, nvidia, kimi, and generic openrouter providers.
        Provider isolation: each model uses its own key, never shared.
        """
        if not api_key:
            return CognitiveGatewayOutput(
                model_name=spec.name, raw_output="",
                success=False, error=f"{spec.api_key_env} not configured",
            )

        url = spec.api_base_url or "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sentinel-e",
            "X-Title": "Sentinel-E Meta-Cognitive",
        }
        payload = {
            "model": spec.model_id,
            "messages": messages,
            "temperature": spec.default_temperature,
            "max_tokens": spec.max_output_tokens,
        }

        session = await self._get_session()
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                return CognitiveGatewayOutput(
                    model_name=spec.name, raw_output="",
                    success=False, error=f"{spec.provider} {resp.status}: {text[:500]}",
                )
            data = await resp.json()
            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            return CognitiveGatewayOutput(
                model_name=spec.name,
                raw_output=choice.get("message", {}).get("content", ""),
                tokens_used=usage.get("total_tokens", 0),
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                success=True,
            )

    # Legacy alias — preserved for backward compatibility
    async def _call_openrouter(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
        api_key: str = "",
    ) -> CognitiveGatewayOutput:
        """Backward-compatible alias. Delegates to isolated method."""
        key = api_key or self.settings.OPENROUTER_API_KEY
        return await self._call_openrouter_isolated(spec, messages, key)

    async def _call_openai(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
    ) -> CognitiveGatewayOutput:
        """Call OpenAI-compatible API (future extension)."""
        return CognitiveGatewayOutput(
            model_name=spec.name, raw_output="",
            success=False, error="OpenAI provider not yet configured",
        )

    # ── Cutoff Disclaimer Enforcement ────────────────────────

    def _flag_cutoff_disclaimers(self, output: CognitiveGatewayOutput):
        """
        Flag outputs containing knowledge cutoff disclaimers.
        Per spec: No cutoff disclaimers allowed.
        We flag but do NOT modify raw output (rule: no modification before scoring).
        """
        text_lower = output.raw_output.lower()
        for pattern in self.CUTOFF_DISCLAIMER_PATTERNS:
            if pattern in text_lower:
                logger.warning(
                    f"Cutoff disclaimer detected in {output.model_name} output"
                )
                # Reduce confidence estimate as signal to arbitration
                if output.confidence_estimate is None:
                    output.confidence_estimate = 0.3
                else:
                    output.confidence_estimate *= 0.5
                break
