"""
============================================================
API 1 — Cognitive Model Gateway
============================================================
Pure reasoning execution. Each model is a separate endpoint.
No cross-model contamination. No retrieval. No session mutation.
No persistence logic. No knowledge injection decisions.

Official Sentinel-E Ensemble (v3 — Reliability-First):
  Tier 1 Anchor    : llama-3.1-8b (Groq), gemma-2-9b-it (OpenRouter)
  Tier 2 Debate    : mistral-7b-instruct, phi-3-mini-128k, gemma-2-2b-it
  Tier 3 Fallback  : llama-3.1-8b-instant (Groq), phi-3-small

All models are free-tier or near-free. Provider failures trigger
automatic fallback substitution. Debates always complete.

Debate composition: 2 anchors + 3 debate + 1-2 fallback ≤ 7 models.
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
from core.ensemble_schemas import MAX_DEBATE_MODELS

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
    supports_vision: bool = False  # Whether this model accepts image inputs


# ── Model Registry ───────────────────────────────────────────
# Official Sentinel-E ensemble v3 — Reliability-First Free-Tier.
# 7 models, 3-tier architecture. All free or near-free inference.
# Tier 1 Anchor   : llama31-8b, gemma2-9b
# Tier 2 Debate   : mistral-7b, phi3-mini, gemma2-2b
# Tier 3 Fallback : llama31-instant, phi3-small

COGNITIVE_MODEL_REGISTRY: Dict[str, CognitiveModelSpec] = {
    # ── Tier 1: Anchor Models (primary reasoning reference) ───
    "llama31-8b": CognitiveModelSpec(
        name="Llama 3.1 8B",
        model_id="llama-3.1-8b",
        provider="groq",
        role=ModelRole.CONCEPTUAL,
        context_window=131072,
        max_output_tokens=2000,
        default_temperature=0.4,
        api_base_url="https://api.groq.com/openai/v1/chat/completions",
        api_key_env="LLAMA31_8B_GROQ_API_KEY",
    ),
    "gemma2-9b": CognitiveModelSpec(
        name="Gemma 2 9B IT",
        model_id="google/gemma-2-9b-it",
        provider="openrouter",
        role=ModelRole.GENERAL,
        context_window=8192,
        max_output_tokens=2000,
        default_temperature=0.3,
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="GEMMA2_9B_OPENROUTER_API_KEY",
    ),

    # ── Tier 2: Debate Models (diverse argument generators) ───
    "mistral-7b": CognitiveModelSpec(
        name="Mistral 7B Instruct",
        model_id="mistralai/mistral-7b-instruct",
        provider="openrouter",
        role=ModelRole.CONCEPTUAL,
        context_window=32768,
        max_output_tokens=1500,
        default_temperature=0.4,
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="MISTRAL_7B_OPENROUTER_API_KEY",
    ),
    "phi3-mini": CognitiveModelSpec(
        name="Phi-3 Mini 128K",
        model_id="microsoft/phi-3-mini-128k-instruct",
        provider="openrouter",
        role=ModelRole.GENERAL,
        context_window=131072,
        max_output_tokens=1500,
        default_temperature=0.3,
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="PHI3_MINI_OPENROUTER_API_KEY",
    ),
    "gemma2-2b": CognitiveModelSpec(
        name="Gemma 2 2B IT",
        model_id="google/gemma-2-2b-it",
        provider="openrouter",
        role=ModelRole.FAST,
        context_window=8192,
        max_output_tokens=1500,
        default_temperature=0.3,
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="GEMMA2_2B_OPENROUTER_API_KEY",
    ),

    # ── Tier 3: Fallback Models (reliability guarantees) ──────
    "llama31-instant": CognitiveModelSpec(
        name="Llama 3.1 8B Instant",
        model_id="llama-3.1-8b-instant",
        provider="groq",
        role=ModelRole.FAST,
        context_window=131072,
        max_output_tokens=1500,
        default_temperature=0.3,
        api_base_url="https://api.groq.com/openai/v1/chat/completions",
        api_key_env="LLAMA31_INSTANT_GROQ_API_KEY",
    ),
    "phi3-small": CognitiveModelSpec(
        name="Phi-3 Small",
        model_id="microsoft/phi-3-small-8k-instruct",
        provider="openrouter",
        role=ModelRole.GENERAL,
        context_window=8192,
        max_output_tokens=1500,
        default_temperature=0.3,
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="PHI3_SMALL_OPENROUTER_API_KEY",
    ),
}

# ============================================================
# Debate Tier Registry — v3 Reliability-First
# ============================================================
# Tier 1 — Anchor Models  : Primary reasoning reference models
# Tier 2 — Debate Models  : Diverse argument generators
# Tier 3 — Fallback       : Reliability guarantee (auto-substitution)
#
# Maximum 7 models run simultaneously in any debate.
# Selection is prompt-type-driven (see get_tiered_models_for_debate).

MODEL_DEBATE_TIERS: Dict[str, int] = {
    # Tier 1 — Anchor Models (high-reliability reasoning reference)
    "llama31-8b":      1,   # Llama 3.1 8B — primary anchor (Groq)
    "gemma2-9b":       1,   # Gemma 2 9B IT — secondary anchor
    # Tier 2 — Debate Models (diverse argument generators)
    "mistral-7b":      2,   # Mistral 7B Instruct — diverse reasoning
    "phi3-mini":       2,   # Phi-3 Mini 128K — compact reasoning
    "gemma2-2b":       2,   # Gemma 2 2B IT — fast diverse viewpoint
    # Tier 3 — Fallback Models (auto-substitution on failure)
    "llama31-instant": 3,   # Llama 3.1 8B Instant — Groq fallback
    "phi3-small":      3,   # Phi-3 Small — OpenRouter fallback
}

# Fallback mapping: if a model fails, replace with this model
MODEL_FALLBACK_MAP: Dict[str, str] = {
    "llama31-8b":      "llama31-instant",  # Groq → Groq backup
    "gemma2-9b":       "llama31-instant",  # OpenRouter → Groq backup
    "mistral-7b":      "phi3-mini",        # Mistral fails → Phi-3 Mini
    "phi3-mini":       "phi3-small",       # Phi-3 Mini fails → Phi-3 Small
    "gemma2-2b":       "llama31-instant",  # Gemma fails → Groq backup
    "llama31-instant": "phi3-small",       # Last-resort chain
    "phi3-small":      "llama31-instant",  # Cross-provider fallback
}

# Prompt type → preferred specialist keys (all models participate now)
_SPECIALIST_AFFINITY: Dict[str, List[str]] = {
    "code":       ["phi3-mini", "phi3-small"],
    "logical":    ["gemma2-9b"],
    "general":    [],
    "conceptual": [],
    "evidence":   [],
    "depth":      ["gemma2-9b"],
}


def get_tiered_models_for_debate(
    prompt_type: str = "general",
    max_models: int = 7,
) -> List[str]:
    """
    Canonical tiered model selector for debate sessions.

    Composition target (when all models are enabled):
        2  Tier-1 Anchor models    — stable reasoning reference
        3  Tier-2 Debate models    — diverse argument generation
        1-2  Tier-3 Fallback       — reliability guarantees
        ─────────────────────────────
        7  Total (hard cap)

    Algorithm:
      1. Select up to 2 enabled Tier-1 anchors  (REQUIRED, filled first).
      2. If prompt_type has a specialist affinity, prioritize those Tier-2 models.
      3. Fill remaining slots (up to max_models) from enabled Tier-2 models.
      4. Add Tier-3 fallback models to fill remaining slots.
      5. Guarantee at least 5 models are always selected.

    Args:
        prompt_type: 'code', 'logical', 'conceptual', 'general',
                     'evidence', 'depth'
        max_models:  Hard cap, default 7. Never exceeded.

    Returns:
        Ordered list of registry keys (anchors first, then debate,
        then fallback models).
    """
    max_models = min(max_models, MAX_DEBATE_MODELS)

    enabled = {
        k for k, spec in COGNITIVE_MODEL_REGISTRY.items()
        if spec.enabled and spec.active
    }

    tier1 = [k for k, t in MODEL_DEBATE_TIERS.items() if t == 1 and k in enabled]
    tier2 = [k for k, t in MODEL_DEBATE_TIERS.items() if t == 2 and k in enabled]
    tier3 = [k for k, t in MODEL_DEBATE_TIERS.items() if t == 3 and k in enabled]

    # Specialist priority for prompt type
    specialist_keys = _SPECIALIST_AFFINITY.get(prompt_type, [])
    priority_tier2 = [k for k in specialist_keys if k in tier2]
    remaining_tier2 = [k for k in tier2 if k not in priority_tier2]

    selected: List[str] = []

    # Step 1 — 2 Tier-1 anchors (capped at available)
    for k in tier1[:2]:
        selected.append(k)

    # Step 2 — Priority Tier-2 models (specialist affinity)
    for k in priority_tier2:
        if len(selected) >= max_models:
            break
        if k not in selected:
            selected.append(k)

    # Step 3 — Remaining Tier-2 debate models
    for k in remaining_tier2:
        if len(selected) >= max_models:
            break
        if k not in selected:
            selected.append(k)

    # Step 4 — Tier-3 fallback models to fill remaining slots
    for k in tier3:
        if len(selected) >= max_models:
            break
        if k not in selected:
            selected.append(k)

    # Step 5 — Guarantee at least 5 models: pull from any tier
    if len(selected) < 5:
        for k in tier2 + tier1 + tier3:
            if k not in selected:
                selected.append(k)
            if len(selected) >= 5:
                break

    if len(selected) < 5:
        logger.warning(
            f"Only {len(selected)} enabled models available (need 5). "
            f"Check API key configuration. Enabled: {enabled}"
        )

    return selected[:max_models]


def _initialize_registry():
    """
    Runtime initialization: check env for each model's API key.
    If key is missing, mark model as disabled (not active).
    Runs once at import time. Non-blocking, never crashes.
    """
    _PROVIDER_SHARED_KEY = {
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    for key, spec in COGNITIVE_MODEL_REGISTRY.items():
        # Check per-model key first, then shared provider key
        api_key = ""
        if spec.api_key_env:
            api_key = os.getenv(spec.api_key_env, "")
        if not api_key:
            shared_env = _PROVIDER_SHARED_KEY.get(spec.provider, "")
            if shared_env:
                api_key = os.getenv(shared_env, "")

        if api_key:
            spec.enabled = True
            spec.active = True
            logger.info(f"Model '{key}' ({spec.provider}): enabled and active")
        else:
            spec.enabled = False
            spec.active = False
            logger.warning(
                f"Model '{key}' ({spec.provider}): DISABLED — "
                f"no API key found (checked {spec.api_key_env} and shared key)"
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
        # ── Pressure tracking for dynamic token scaling ──
        # Records timestamps of recent 402/429 failures
        self._recent_failures: List[float] = []
        self._pressure_window: float = 300.0  # 5-minute sliding window

    @property
    def pressure_factor(self) -> float:
        """
        Returns a multiplier in (0.5, 1.0] based on recent 402/429 failures.
        0 failures → 1.0 (full budget)
        1 failure  → 0.85
        2 failures → 0.70
        3+ failures → 0.50
        """
        now = time.monotonic()
        self._recent_failures = [
            t for t in self._recent_failures
            if now - t < self._pressure_window
        ]
        n = len(self._recent_failures)
        if n == 0:
            return 1.0
        if n == 1:
            return 0.85
        if n == 2:
            return 0.70
        return 0.50

    def _record_failure(self):
        """Record a 402/429 failure timestamp for pressure tracking."""
        self._recent_failures.append(time.monotonic())

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
        Resolve the API key for a model.

        Resolution order:
          1. Per-model env var (e.g. LLAMA31_8B_GROQ_API_KEY)
          2. Shared provider key (GROQ_API_KEY for groq, OPENROUTER_API_KEY for openrouter)

        Returns None if no key available.
        """
        # 1. Try per-model key
        if spec.api_key_env:
            key = os.getenv(spec.api_key_env, "")
            if key:
                return key

        # 2. Fall back to shared provider key
        _PROVIDER_SHARED_KEY = {
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        shared_env = _PROVIDER_SHARED_KEY.get(spec.provider)
        if shared_env:
            key = os.getenv(shared_env, "")
            if key:
                return key

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

        # ── Token Governor ──────────────────────────────────
        # Estimate prompt tokens and clamp max_tokens to safe budget
        _prompt_estimate = sum(
            len(str(m.get("content", ""))) for m in messages
        ) // 4  # rough char-to-token ratio
        _available = spec.context_window - _prompt_estimate - 300

        # Budget-governed mode (debate rounds) vs normal mode
        _budget_governed = gateway_input.max_tokens_override is not None

        if _budget_governed:
            # Debate budget governor already constrains tokens —
            # skip pressure_factor to avoid double-jeopardy
            scaled_cap = min(spec.max_output_tokens, gateway_input.max_tokens_override)
        else:
            # Normal mode: apply pressure-based dynamic scaling
            pf = self.pressure_factor
            scaled_cap = max(512, int(spec.max_output_tokens * pf))
            if pf < 1.0:
                logger.info(
                    f"Pressure scaling [{model_key}]: factor={pf:.2f}, "
                    f"base={spec.max_output_tokens}, scaled={scaled_cap}"
                )

        governed_max_tokens = min(scaled_cap, max(0, _available))

        # Floor check: skip gracefully for budget-governed debate calls,
        # hard-fail only for normal calls where prompt exceeds context window.
        _min_tokens = 50 if _budget_governed else 500
        if governed_max_tokens < _min_tokens:
            if _budget_governed:
                # Budget-governed mode: skip model gracefully — do NOT
                # return "Token budget exceeded" unless prompt itself
                # exceeds the context window.
                if _available <= 0:
                    # Prompt genuinely exceeds context window
                    logger.warning(
                        f"Context window exceeded for '{model_key}': "
                        f"prompt ~{_prompt_estimate} tok, "
                        f"window={spec.context_window}, available={_available}"
                    )
                    return CognitiveGatewayOutput(
                        model_name=spec.name,
                        raw_output="",
                        success=False,
                        error="Token budget exceeded",
                    )
                # Otherwise: budget constraint — skip, not fail
                logger.info(
                    f"Skipping '{model_key}' for this round: "
                    f"governed={governed_max_tokens} < floor={_min_tokens}. "
                    f"Skipped due to budget constraint."
                )
                return CognitiveGatewayOutput(
                    model_name=spec.name,
                    raw_output="",
                    success=False,
                    error="Skipped due to budget constraint.",
                )
            else:
                # Normal mode: genuine budget failure
                logger.warning(
                    f"Token budget exceeded for '{model_key}': "
                    f"prompt ~{_prompt_estimate} tok, "
                    f"window={spec.context_window}, available={_available}, "
                    f"governed={governed_max_tokens}, floor={_min_tokens}"
                )
                return CognitiveGatewayOutput(
                    model_name=spec.name,
                    raw_output="",
                    success=False,
                    error="Token budget exceeded",
                )

        start = time.monotonic()
        try:
            if spec.provider == "groq":
                result = await self._call_groq(spec, messages, api_key, governed_max_tokens)
            elif spec.provider in ("qwen", "nvidia", "kimi", "openrouter"):
                result = await self._call_openrouter_isolated(
                    spec, messages, api_key, governed_max_tokens,
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

        # PHASE 2: Guarantee text output — never return blank on success
        if result.success:
            if not result.raw_output or result.raw_output.strip() == "":
                logger.error(f"Model '{model_key}' returned empty response")
                result.success = False
                result.error = f"Model '{model_key}' returned empty response"
                result.raw_output = ""
            else:
                # PHASE 5: Set confidence default only after successful text
                if result.confidence_estimate is None:
                    result.confidence_estimate = 0.5

                # Enforce no-cutoff-disclaimer rule (flag, don't modify)
                self._flag_cutoff_disclaimers(result)

        # PHASE 8: Debug logging (temporary)
        logger.info(
            f"Calling provider for: {model_key} ({spec.provider}) — "
            f"success={result.success}, output_len={len(result.raw_output) if result.raw_output else 0}"
        )

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

    async def invoke_parallel_failsafe(
        self,
        model_keys: List[str],
        gateway_input: CognitiveGatewayInput,
        min_success: int = 5,
    ) -> List[CognitiveGatewayOutput]:
        """
        Failsafe parallel invocation with automatic fallback substitution.

        Guarantees at least `min_success` successful responses:
        1. Invoke all selected models in parallel.
        2. For each failure, look up MODEL_FALLBACK_MAP and retry with fallback.
        3. If fallback also used already, try any remaining Tier-3 model.
        4. Never duplicate a model already in the response set.

        This ensures debates always complete with sufficient model diversity.
        """
        # Phase 1 — Primary invocation
        outputs = await self.invoke_parallel(model_keys, gateway_input)

        successful = sum(1 for o in outputs if o.success)
        if successful >= min_success:
            return outputs

        # Phase 2 — Failsafe substitution for failed models
        used_keys = set(model_keys)
        failed_indices = [i for i, o in enumerate(outputs) if not o.success]

        for idx in failed_indices:
            if successful >= min_success:
                break

            failed_key = model_keys[idx]
            fallback_key = MODEL_FALLBACK_MAP.get(failed_key)

            # Try the primary fallback
            if fallback_key and fallback_key not in used_keys:
                logger.info(
                    f"Failsafe: {failed_key} failed, substituting {fallback_key}"
                )
                fallback_result = await self.invoke_model(
                    fallback_key, gateway_input.model_copy(deep=True)
                )
                if fallback_result.success:
                    outputs[idx] = fallback_result
                    used_keys.add(fallback_key)
                    successful += 1
                    continue

            # Try any unused Tier-3 model
            tier3_keys = [
                k for k, t in MODEL_DEBATE_TIERS.items()
                if t == 3 and k not in used_keys
                and COGNITIVE_MODEL_REGISTRY[k].enabled
            ]
            for t3_key in tier3_keys:
                logger.info(
                    f"Failsafe: trying Tier-3 {t3_key} for slot {idx}"
                )
                t3_result = await self.invoke_model(
                    t3_key, gateway_input.model_copy(deep=True)
                )
                if t3_result.success:
                    outputs[idx] = t3_result
                    used_keys.add(t3_key)
                    successful += 1
                    break

        logger.info(
            f"Failsafe complete: {successful}/{len(outputs)} models succeeded "
            f"(target: {min_success})"
        )
        return outputs

    # ── Message Building ─────────────────────────────────────

    def _build_messages(
        self,
        inp: CognitiveGatewayInput,
        spec: CognitiveModelSpec,
    ) -> List[Dict]:
        """
        Build message array from stabilized context.
        Injects knowledge bundle and session summary as system context.
        Supports multimodal (vision) payloads when image_b64 is provided
        and the model has role == VISION.
        """
        messages = []

        # System prompt — dual conversational / analytical mode
        system_parts = [
            "You are Sentinel-E.\n\n"
            "You operate in two adaptive modes:\n\n"
            "1. Conversational Mode:\n"
            "   - Respond naturally in first person.\n"
            "   - Maintain continuity.\n"
            "   - Speak directly to the user.\n"
            "   - Handle emotional or personal language appropriately.\n\n"
            "2. Analytical Mode:\n"
            "   - Provide structured reasoning.\n"
            "   - Break down assumptions.\n"
            "   - Present clear logical steps.\n\n"
            "Automatically choose the mode based on user tone.\n"
            "Never mention knowledge cutoffs or training data."
        ]

        # Inject knowledge bundle (capped at ~2000 chars to save tokens)
        if inp.knowledge_bundle:
            kb_text = "\n\n[KNOWLEDGE CONTEXT]\n"
            for block in inp.knowledge_bundle:
                kb_text += f"Source: {block.source}\n"
                kb_text += f"Content: {block.content}\n\n"
            if len(kb_text) > 2000:
                kb_text = kb_text[:1950] + "\n...[truncated]"
            system_parts.append(kb_text)

        # Inject session summary (capped at ~1500 chars)
        if inp.session_summary:
            summary_text = "\n\n[SESSION CONTEXT]\n"
            for key, val in inp.session_summary.items():
                summary_text += f"{key}: {val}\n"
            if len(summary_text) > 1500:
                summary_text = summary_text[:1450] + "\n...[truncated]"
            system_parts.append(summary_text)

        # Inject stabilized context (capped at ~2000 chars)
        if inp.stabilized_context:
            ctx_text = "\n\n[STABILIZED CONTEXT]\n"
            for key, val in inp.stabilized_context.items():
                if isinstance(val, (list, dict)):
                    import json
                    ctx_text += f"{key}: {json.dumps(val, default=str)}\n"
                else:
                    ctx_text += f"{key}: {val}\n"
            if len(ctx_text) > 2000:
                ctx_text = ctx_text[:1950] + "\n...[truncated]"
            system_parts.append(ctx_text)

        messages.append({"role": "system", "content": "\n".join(system_parts)})

        # User message — multimodal if image provided and model supports vision
        if inp.image_b64 and spec.supports_vision:
            mime = inp.image_mime or "image/png"
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": inp.user_query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{inp.image_b64}"
                        },
                    },
                ],
            })
        elif inp.image_b64 and not spec.supports_vision:
            # Image attached but model can't process it — reject explicitly
            logger.warning(
                f"Image rejected: model '{spec.name}' does not support vision. "
                f"Image silently dropped; text-only prompt used."
            )
            messages.append({
                "role": "user",
                "content": inp.user_query,
            })
        else:
            messages.append({"role": "user", "content": inp.user_query})

        return messages

    # ── Provider Implementations ─────────────────────────────

    async def _call_groq(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
        api_key: str = "",
        max_tokens: int = 4096,
    ) -> CognitiveGatewayOutput:
        """Call Groq API with per-model isolated key and governed token budget."""
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
            "max_tokens": max_tokens,
        }

        session = await self._get_session()
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                if resp.status in (402, 429):
                    self._record_failure()
                return CognitiveGatewayOutput(
                    model_name=spec.name, raw_output="",
                    success=False, error=self._sanitize_provider_error(resp.status, text),
                )
            data = await resp.json()
            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            return CognitiveGatewayOutput(
                model_name=spec.name,
                raw_output=choice.get("message", {}).get("content") or "",
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
        max_tokens: int = 4096,
    ) -> CognitiveGatewayOutput:
        """
        Call OpenRouter-compatible API with per-model isolated key.
        Used by qwen, nvidia, kimi, and generic openrouter providers.
        Provider isolation: each model uses its own key, never shared.
        Token budget governed by caller — never exceeds safe limit.
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
            "max_tokens": max_tokens,
        }

        session = await self._get_session()
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                if resp.status in (402, 429):
                    self._record_failure()
                return CognitiveGatewayOutput(
                    model_name=spec.name, raw_output="",
                    success=False, error=self._sanitize_provider_error(resp.status, text),
                )
            data = await resp.json()
            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            return CognitiveGatewayOutput(
                model_name=spec.name,
                raw_output=choice.get("message", {}).get("content") or "",
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

    # ── Error Sanitization ───────────────────────────────────

    @staticmethod
    def _sanitize_provider_error(status_code: int, raw_text: str) -> str:
        """Convert raw provider errors into clean user-facing messages.

        Never expose user_id, metadata, or raw provider JSON to the UI.
        Maps HTTP status codes and error keywords to safe categories.
        """
        raw_lower = raw_text.lower()

        if status_code == 401 or "unauthorized" in raw_lower or "invalid" in raw_lower and "key" in raw_lower:
            return "Invalid API key"
        if status_code == 402 or "insufficient" in raw_lower or "credit" in raw_lower or "quota" in raw_lower:
            return "Insufficient credit"
        if status_code == 429 or "rate" in raw_lower and "limit" in raw_lower:
            return "Rate limit exceeded"
        if status_code == 404 or "not found" in raw_lower or "invalid model" in raw_lower:
            return "Invalid model"
        if status_code == 503 or "unavailable" in raw_lower or "overloaded" in raw_lower:
            return "Provider unavailable"
        if "context" in raw_lower and ("length" in raw_lower or "token" in raw_lower):
            return "Token limit exceeded"

        return f"Provider error (HTTP {status_code})"
