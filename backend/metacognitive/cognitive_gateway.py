"""
============================================================
API 1 — Cognitive Model Gateway
============================================================
Pure reasoning execution. Each model is a separate endpoint.
No cross-model contamination. No retrieval. No session mutation.
No persistence logic. No knowledge injection decisions.

Official Sentinel-E Ensemble (v5 — Decommissioned models replaced):
  Analysis     : llama-3.3-70b-versatile (Groq)
  Critique A   : qwen/qwen3-32b (Groq)
  Critique B   : meta-llama/llama-4-scout-17b-16e-instruct (Groq)
  Critique C   : qwen2.5-vl-7b-instruct (DashScope)
  Synthesis    : gemini-2.0-flash (Google)
  Verification : llama-3.1-8b-instant (Groq)

Providers: Groq, Gemini, Qwen/DashScope. No OpenRouter.
Pipeline: Analysis → 3 Critiques (parallel) → Synthesis → Verification.
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

# ── Claude Usage Tracker ──────────────────────────────────────
# Track total tokens used by Claude to help user monitor $5 budget
# Claude Sonnet 4: ~$3/M input tokens, ~$15/M output tokens
_claude_usage = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_calls": 0,
    "estimated_cost_usd": 0.0,
}

def get_claude_usage() -> dict:
    """Return current Claude API usage statistics."""
    return dict(_claude_usage)

def _track_claude_usage(input_tokens: int, output_tokens: int):
    """Track Claude API usage for cost monitoring."""
    _claude_usage["total_input_tokens"] += input_tokens
    _claude_usage["total_output_tokens"] += output_tokens
    _claude_usage["total_calls"] += 1
    # Claude Sonnet 4 pricing: $3/M input, $15/M output
    cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
    _claude_usage["estimated_cost_usd"] = round(_claude_usage["estimated_cost_usd"] + cost, 6)
    logger.info(
        f"Claude usage: call #{_claude_usage['total_calls']}, "
        f"this call: {input_tokens}in/{output_tokens}out, "
        f"total est. cost: ${_claude_usage['estimated_cost_usd']:.4f}"
    )


# ============================================================
# Model Configuration
# ============================================================

@dataclass
class CognitiveModelSpec:
    """Specification for a cognitive model endpoint."""
    name: str
    model_id: str           # Provider-specific model ID
    provider: str           # groq | gemini | qwen | local
    role: ModelRole         # Routing hint
    model_type: str = "external"  # "external" (API) or "internal" (local inference)
    context_window: int = 131072
    max_output_tokens: int = 8192
    default_temperature: float = 0.3
    api_base_url: str = ""  # Provider base URL
    api_key_env: str = ""   # Environment variable name for this model's key
    active: bool = True     # Structural flag (can be toggled manually)
    enabled: bool = True    # Runtime flag (auto-set based on key availability)
    supports_vision: bool = False  # Whether this model accepts image inputs
    disable_reason: str = None  # Reason why model is disabled
    synthesis_only: bool = False  # If True, model is excluded from Phase 1 and debate


# ── Model Registry ───────────────────────────────────────────
# Official Sentinel-E ensemble v4 — No OpenRouter.
# 6 models, 3 providers (Groq, Gemini, Qwen/DashScope).
# Pipeline: Analysis → 3 Critiques (parallel) → Synthesis → Verification.

COGNITIVE_MODEL_REGISTRY: Dict[str, CognitiveModelSpec] = {
    # ── Analysis (primary deep-reasoning anchor) ──────────────
    "llama33-70b": CognitiveModelSpec(
        name="Llama 3.3 70B",
        model_id="llama-3.3-70b-versatile",
        provider="groq",
        role=ModelRole.CONCEPTUAL,
        context_window=131072,
        max_output_tokens=2000,
        default_temperature=0.4,
        api_base_url="https://api.groq.com/openai/v1/chat/completions",
        api_key_env="LLAMA31_8B_GROQ_API_KEY",
    ),

    # ── Critique A (diverse argument generator) ───────────────
    # Replaced: mixtral-8x7b-32768 decommissioned by Groq
    "mixtral-8x7b": CognitiveModelSpec(
        name="Qwen3 32B",
        model_id="qwen/qwen3-32b",
        provider="groq",
        role=ModelRole.CONCEPTUAL,
        context_window=32768,
        max_output_tokens=1500,
        default_temperature=0.4,
        api_base_url="https://api.groq.com/openai/v1/chat/completions",
        api_key_env="GROQ_API_KEY",
    ),

    # ── Critique B (alternative viewpoint) ────────────────────
    # Llama 4 Scout 17B — critique B (Groq)
    "llama4-scout": CognitiveModelSpec(
        name="Llama 4 Scout 17B",
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
        provider="groq",
        role=ModelRole.GENERAL,
        context_window=131072,
        max_output_tokens=1500,
        default_temperature=0.3,
        api_base_url="https://api.groq.com/openai/v1/chat/completions",
        api_key_env="GROQ_API_KEY",
    ),

    # ── Critique C (alternative perspectives via Qwen) ────────
    "qwen-2.5-vl": CognitiveModelSpec(
        name="Qwen 2.5 VL 7B",
        model_id="qwen/qwen-2.5-vl-7b-instruct",
        provider="qwen",
        role=ModelRole.VISION,
        context_window=32768,
        max_output_tokens=1500,
        default_temperature=0.3,
        api_base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="QWEN_API_KEY",
        supports_vision=True,
    ),

    # ── Synthesis (merges critiques into final answer) ────────
    "gemini-flash": CognitiveModelSpec(
        name="Gemini Flash 2.0",
        model_id="gemini-2.0-flash",
        provider="gemini",
        role=ModelRole.GENERAL,
        context_window=1048576,
        max_output_tokens=2000,
        default_temperature=0.3,
        api_base_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        api_key_env="GEMINI_API_KEY",
        supports_vision=True,
    ),

    # ── Verification (fast sanity check) ──────────────────────
    "llama31-8b": CognitiveModelSpec(
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

    # ── Mistral Large 675B (NVIDIA) ───────────────────────────
    "mistral-large-675b": CognitiveModelSpec(
        name="Mistral Large 3 675B",
        model_id="mistralai/mistral-large-3-675b-instruct-2512",
        provider="nvidia",
        role=ModelRole.CONCEPTUAL,
        context_window=131072,
        max_output_tokens=4000,
        default_temperature=0.15,
        api_base_url="https://integrate.api.nvidia.com/v1/chat/completions",
        api_key_env="MISTRAL_LARGE_NVIDIA_API_KEY",
    ),

    # ── Kimi K2 Thinking (NVIDIA) ─────────────────────────────
    "kimi-k2-thinking": CognitiveModelSpec(
        name="Kimi K2 Thinking",
        model_id="moonshotai/kimi-k2-thinking",
        provider="nvidia",
        role=ModelRole.CONCEPTUAL,
        context_window=131072,
        max_output_tokens=4000,
        default_temperature=0.2,
        api_base_url="https://integrate.api.nvidia.com/v1/chat/completions",
        api_key_env="KIMI_K2_NVIDIA_API_KEY",
    ),

    # ── Claude Sonnet 4.6 (Anthropic — synthesis only) ────────
    "claude-sonnet-4.6": CognitiveModelSpec(
        name="Claude Sonnet 4.6",
        model_id="claude-sonnet-4-20250514",
        provider="anthropic",
        role=ModelRole.GENERAL,
        context_window=200000,
        max_output_tokens=500,
        default_temperature=0.3,
        api_base_url="https://api.anthropic.com/v1/messages",
        api_key_env="ANTHROPIC_API_KEY",
        supports_vision=True,
        synthesis_only=True,
    ),
}

# ============================================================
# Debate Pipeline Registry — v4 (No OpenRouter)
# ============================================================
# Analysis → Critique (3 parallel) → Synthesis → Verification
# All Groq + Gemini + Qwen. No tiered fallback needed.

MODEL_DEBATE_TIERS: Dict[str, int] = {
    # Tier 1 — Analysis (primary reasoning)
    "llama33-70b":     1,   # Llama 3.3 70B — analysis anchor (Groq)
    "mistral-large-675b": 1,  # Mistral Large 675B — analysis anchor (NVIDIA)
    # Tier 2 — Critique (diverse argument generators)
    "mixtral-8x7b":    2,   # Mixtral 8x7B — critique A (Groq)
    "llama4-scout":     2,   # Llama 4 Scout 17B — critique B (Groq)
    "qwen-2.5-vl":     2,   # Qwen 2.5 VL — critique C (DashScope)
    "kimi-k2-thinking": 2,  # Kimi K2 Thinking — critique D (NVIDIA)
    # Tier 3 — Synthesis + Verification
    "gemini-flash":    3,   # Gemini Flash 2.0 — synthesis (Google)
    "llama31-8b":      3,   # Llama 3.1 8B — verification (Groq)
    # Note: claude-sonnet-4.6 excluded from debate — synthesis pipeline only
}

# Fallback mapping: if a model fails, replace with this model
MODEL_FALLBACK_MAP: Dict[str, str] = {
    "llama33-70b":         "mistral-large-675b",  # Groq 70B → NVIDIA Mistral
    "mistral-large-675b":  "llama33-70b",         # NVIDIA Mistral → Groq 70B
    "mixtral-8x7b":        "llama4-scout",         # Mixtral → Llama 4 Scout
    "llama4-scout":        "kimi-k2-thinking",     # Llama 4 Scout → Kimi K2
    "qwen-2.5-vl":         "kimi-k2-thinking",     # Qwen → Kimi K2
    "kimi-k2-thinking":    "llama4-scout",         # Kimi K2 → Llama 4 Scout
    "gemini-flash":        "llama33-70b",          # Gemini → Llama 70B
    "llama31-8b":          "mixtral-8x7b",         # Llama 8B → Mixtral
    # Note: claude-sonnet-4.6 has no fallback — synthesis pipeline only
}

# Prompt type → preferred specialist keys
_SPECIALIST_AFFINITY: Dict[str, List[str]] = {
    "code":       ["mixtral-8x7b", "kimi-k2-thinking"],
    "logical":    ["llama33-70b", "mistral-large-675b"],
    "general":    ["mistral-large-675b"],
    "conceptual": ["kimi-k2-thinking"],
    "evidence":   ["mistral-large-675b"],
    "depth":      ["llama33-70b", "mistral-large-675b"],
    "vision":     ["qwen-2.5-vl"],
}


def get_tiered_models_for_debate(
    prompt_type: str = "general",
    max_models: int = 7,
) -> List[str]:
    """
    Canonical tiered model selector for debate sessions.

    Composition target (v4 — all models enabled):
        1  Tier-1 Analysis model    — deep reasoning anchor
        3  Tier-2 Critique models   — diverse argument generation
        2  Tier-3 Synthesis/Verify  — convergence + sanity check
        ─────────────────────────────
        6  Total (hard cap)

    Algorithm:
      1. Select Tier-1 analysis model (REQUIRED).
      2. If prompt_type has a specialist affinity, prioritize those models.
      3. Fill remaining slots from enabled Tier-2 models.
      4. Add Tier-3 models to fill remaining slots.
      5. Guarantee at least 3 models are always selected.

    Args:
        prompt_type: 'code', 'logical', 'conceptual', 'general',
                     'evidence', 'depth', 'vision'
        max_models:  Hard cap, default 7. Never exceeded.

    Returns:
        Ordered list of registry keys (analysis first, then critiques,
        then synthesis/verification).
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

    # Step 1 — Tier-1 analysis model (capped at available)
    for k in tier1[:1]:
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

    # Step 5 — Guarantee at least 3 models: pull from any tier
    if len(selected) < 3:
        for k in tier2 + tier1 + tier3:
            if k not in selected:
                selected.append(k)
            if len(selected) >= 3:
                break

    if len(selected) < 3:
        logger.warning(
            f"Only {len(selected)} enabled models available (need 3). "
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
        "gemini": "GEMINI_API_KEY",
        "qwen": "QWEN_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
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
            # Claude starts INACTIVE by default — user must toggle it ON
            # This prevents accidental API spend on the $5 Anthropic plan
            if spec.synthesis_only:
                spec.active = False
                logger.info(f"Model '{key}' ({spec.provider}): enabled but inactive (synthesis-only — toggle to activate)")
            else:
                spec.active = True
                logger.info(f"Model '{key}' ({spec.provider}): enabled and active")
            spec.disable_reason = None
        else:
            spec.enabled = False
            spec.active = False
            spec.disable_reason = f"API key missing: checked {spec.api_key_env or '[none]'} and provider key { _PROVIDER_SHARED_KEY.get(spec.provider, '[none]') }"
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
            timeout = aiohttp.ClientTimeout(total=30)
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
          1. Per-model env var (e.g. LLAMA31_8B_GROQ_API_KEY, KIMI_K2_NVIDIA_API_KEY)
          2. Per-model fallback key (e.g. KIMI_K2_NVIDIA_API_KEY_FALLBACK for resilience)
          3. Shared provider key (GROQ_API_KEY for groq, GEMINI_API_KEY for gemini, etc.)

        Returns None if no key available.
        """
        # 1. Try per-model key
        if spec.api_key_env:
            key = os.getenv(spec.api_key_env, "")
            if key:
                return key
        
        # 2. Try per-model fallback key (for resilience)
        if spec.api_key_env:
            fallback_env = f"{spec.api_key_env}_FALLBACK"
            key = os.getenv(fallback_env, "")
            if key:
                logger.info(f"Using fallback API key for '{spec.name}' ({fallback_env})")
                return key

        # 3. Fall back to shared provider key
        _PROVIDER_SHARED_KEY = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "qwen": "QWEN_API_KEY",
            "nvidia": "NVIDIA_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        shared_env = _PROVIDER_SHARED_KEY.get(spec.provider)
        if shared_env:
            key = os.getenv(shared_env, "")
            if key:
                logger.info(f"Using shared provider key for '{spec.name}' ({shared_env})")
                return key

        logger.warning(f"API key missing for model '{spec.name}' ({spec.provider}). Checked: {spec.api_key_env}, {spec.api_key_env}_FALLBACK, {shared_env}")
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
            logger.error(f"Model '{model_key}' not found or disabled. Provider: {getattr(spec, 'provider', None)}")
            return CognitiveGatewayOutput(
                model_name=model_key,
                raw_output="",
                success=False,
                error=f"Model '{model_key}' not found or disabled",
            )

        # Resolve per-model API key (provider isolation)
        api_key = self._resolve_api_key(spec)
        if not api_key:
            logger.error(f"API key not configured for model '{spec.name}' ({spec.provider})")
            return CognitiveGatewayOutput(
                model_name=spec.name,
                raw_output="",
                success=False,
                error=f"{spec.api_key_env} not configured",
            )

        # Build messages from stabilized context
        messages = self._build_messages(gateway_input, spec)

        # ── Message Validation ──────────────────────────────
        if not messages or not any(m.get("role") == "user" for m in messages):
            logger.error(f"Message validation failed for '{model_key}': no user message")
            return CognitiveGatewayOutput(
                model_name=spec.name, raw_output="",
                success=False, error="Message validation failed: missing user message",
            )
        for msg in messages:
            if not msg.get("content", "").strip():
                logger.error(f"Message validation failed for '{model_key}': empty content in role={msg.get('role')}")
                return CognitiveGatewayOutput(
                    model_name=spec.name, raw_output="",
                    success=False, error=f"Message validation failed: empty {msg.get('role')} content",
                )

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

        logger.info(
            f"Token governor [{model_key}]: prompt_est={_prompt_estimate}, "
            f"ctx_window={spec.context_window}, available={_available}, "
            f"scaled_cap={scaled_cap}, governed={governed_max_tokens}, "
            f"budget_governed={_budget_governed}"
        )

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

        # ── Token Clamping ───────────────────────────────────
        # Ensure max_tokens does not exceed provider-specific hard limits
        _PROVIDER_MAX_OUTPUT = {"groq": 8192, "gemini": 8192, "qwen": 8192, "openai": 4096, "nvidia": 4096, "anthropic": 500}
        _provider_cap = _PROVIDER_MAX_OUTPUT.get(spec.provider, 4096)
        governed_max_tokens = max(1, min(governed_max_tokens, _provider_cap))

        start = time.monotonic()
        try:
            if spec.provider == "groq":
                result = await self._call_groq(spec, messages, api_key, governed_max_tokens)
            elif spec.provider == "gemini":
                result = await self._call_gemini(spec, messages, api_key, governed_max_tokens)
            elif spec.provider == "qwen":
                result = await self._call_qwen(spec, messages, api_key, governed_max_tokens)
            elif spec.provider == "nvidia":
                result = await self._call_nvidia(spec, messages, api_key, governed_max_tokens)
            elif spec.provider == "anthropic":
                result = await self._call_anthropic(spec, messages, api_key, governed_max_tokens)
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
                logger.error(f"Model '{model_key}' returned empty response. Provider: {spec.provider}")
                result.success = False
                result.error = f"Model '{model_key}' returned empty response"
                result.raw_output = ""
            else:
                # PHASE 5: Set confidence default only after successful text
                if result.confidence_estimate is None:
                    result.confidence_estimate = 0.5

                # Enforce no-cutoff-disclaimer rule (flag, don't modify)
                self._flag_cutoff_disclaimers(result)

        # PHASE 8: Debug logging (permanent)
        logger.info(
            f"Model invocation: key={model_key}, provider={spec.provider}, success={result.success}, response_len={len(result.raw_output) if result.raw_output else 0}"
        )

        # PHASE 9: Record API key tokenization (for shared API key quota tracking)
        if result.success and spec.provider == "nvidia":
            from optimization.cost_governor import get_cost_governor
            api_key_env = spec.api_key_env
            # Resolve which API key was actually used
            if not api_key:  # If no per-model key, it came from shared NVIDIA_API_KEY
                governor = get_cost_governor()
                governor.record_api_key_usage(
                    "NVIDIA_API_KEY",
                    model_key,
                    result.input_tokens,
                    result.output_tokens,
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
        min_success: int = 3,
    ) -> List[CognitiveGatewayOutput]:
        """
        Failsafe parallel invocation with automatic fallback substitution.

        Guarantees at least `min_success` successful responses:
        1. Invoke all selected models in parallel.
        2. For each failure, look up MODEL_FALLBACK_MAP and retry in parallel.
        3. Never duplicate a model already in the response set.
        """
        # Phase 1 — Primary invocation (all parallel)
        outputs = await self.invoke_parallel(model_keys, gateway_input)

        successful = sum(1 for o in outputs if o.success)
        if successful >= min_success:
            return outputs

        # Phase 2 — Collect all fallback tasks and run in parallel
        used_keys = set(model_keys)
        failed_indices = [i for i, o in enumerate(outputs) if not o.success]

        fallback_tasks = []
        fallback_index_map = []

        for idx in failed_indices:
            failed_key = model_keys[idx]
            fallback_key = MODEL_FALLBACK_MAP.get(failed_key)

            if fallback_key and fallback_key not in used_keys:
                used_keys.add(fallback_key)
                fallback_tasks.append(
                    self.invoke_model(fallback_key, gateway_input.model_copy(deep=True))
                )
                fallback_index_map.append(idx)
                logger.info(f"Failsafe: {failed_key} failed, queuing {fallback_key}")

        if fallback_tasks:
            fallback_results = await asyncio.gather(*fallback_tasks, return_exceptions=True)
            for i, result in enumerate(fallback_results):
                idx = fallback_index_map[i]
                if not isinstance(result, Exception) and result.success:
                    outputs[idx] = result
                    successful += 1

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
        # If synthesis mode, use the intelligence aggregation prompt instead
        synthesis_prompt = (inp.stabilized_context or {}).pop("synthesis_system_prompt", None)
        if synthesis_prompt:
            system_parts = [synthesis_prompt]
        else:
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
                logger.error(
                    f"[GROQ ERROR] model={spec.model_id} "
                    f"status={resp.status} body={text[:500]}"
                )
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

    async def _call_gemini(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
        api_key: str = "",
        max_tokens: int = 4096,
    ) -> CognitiveGatewayOutput:
        """Call Google Gemini REST API with robust structured output handling."""
        if not api_key:
            return CognitiveGatewayOutput(
                model_name=spec.name, raw_output="",
                success=False, error=f"{spec.api_key_env} not configured",
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{spec.model_id}:generateContent"
        # Convert messages to Gemini format with vision support
        contents = []
        system_text = ""
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            else:
                parts = []
                content = msg["content"]
                if isinstance(content, list):
                    # Multimodal content (vision/PDF)
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            url_str = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                            if url_str.startswith("data:"):
                                header, _, b64_data = url_str.partition(",")
                                mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                                parts.append({"inline_data": {"mime_type": mime_type, "data": b64_data}})
                            else:
                                parts.append({"text": f"[Image: {url_str}]"})
                else:
                    parts.append({"text": str(content)})
                if parts:
                    contents.append({"role": "user", "parts": parts})
        if system_text and contents:
            # Prepend system text as first text part in the first message
            first_parts = contents[0]["parts"]
            if first_parts and "text" in first_parts[0]:
                first_parts[0]["text"] = system_text + "\n\n" + first_parts[0]["text"]
            else:
                first_parts.insert(0, {"text": system_text})

        # Ensure at least one content part exists
        if not contents:
            return CognitiveGatewayOutput(
                model_name=spec.name, raw_output="",
                success=False, error="No user content provided for Gemini",
            )

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": spec.default_temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "text/plain",
            },
        }

        session = await self._get_session()
        async with session.post(url, json=payload, headers={"Content-Type": "application/json", "x-goog-api-key": api_key}) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Gemini API error [{resp.status}] for {spec.model_id}: {text[:500]}")
                if resp.status in (402, 429):
                    self._record_failure()
                return CognitiveGatewayOutput(
                    model_name=spec.name, raw_output="",
                    success=False, error=self._sanitize_provider_error(resp.status, text),
                )
            data = await resp.json()
            candidates = data.get("candidates", [])

            # Handle blocked or empty candidates
            if not candidates:
                block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                return CognitiveGatewayOutput(
                    model_name=spec.name, raw_output="",
                    success=False, error=f"Gemini returned no candidates (block: {block_reason})",
                )

            parts = candidates[0].get("content", {}).get("parts", [])
            text = parts[0].get("text", "") if parts else ""

            # Handle finish reason SAFETY or empty text
            finish_reason = candidates[0].get("finishReason", "")
            if not text.strip():
                return CognitiveGatewayOutput(
                    model_name=spec.name, raw_output="",
                    success=False,
                    error=f"Gemini returned empty response (finishReason: {finish_reason})",
                )

            usage = data.get("usageMetadata", {})
            return CognitiveGatewayOutput(
                model_name=spec.name,
                raw_output=text,
                tokens_used=usage.get("totalTokenCount", 0),
                input_tokens=usage.get("promptTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0),
                success=True,
            )

    async def _call_qwen(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
        api_key: str = "",
        max_tokens: int = 4096,
    ) -> CognitiveGatewayOutput:
        """Call Qwen via OpenRouter (OpenAI-compatible API)."""
        if not api_key:
            return CognitiveGatewayOutput(
                model_name=spec.name, raw_output="",
                success=False, error=f"{spec.api_key_env} not configured",
            )

        url = spec.api_base_url or "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Sanitize messages: pass multimodal content for vision model, flatten for text-only
        sanitized_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                if spec.supports_vision:
                    # Qwen VL supports OpenAI vision format — pass through
                    sanitized_messages.append({"role": msg["role"], "content": content})
                else:
                    text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    sanitized_messages.append({"role": msg["role"], "content": " ".join(text_parts) if text_parts else str(content)})
            else:
                sanitized_messages.append({"role": msg["role"], "content": content})

        payload = {
            "model": spec.model_id,
            "messages": sanitized_messages,
            "temperature": spec.default_temperature,
            "max_tokens": max_tokens,
        }

        session = await self._get_session()
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Qwen API error [{resp.status}] for {spec.model_id}: {text[:500]}")
                if resp.status in (402, 429):
                    self._record_failure()
                return CognitiveGatewayOutput(
                    model_name=spec.name, raw_output="",
                    success=False, error=self._sanitize_provider_error(resp.status, text),
                )
            data = await resp.json()
            choice = data.get("choices", [{}])[0]
            usage = data.get("usage", {})
            raw_content = choice.get("message", {}).get("content") or ""

            # Handle empty or whitespace-only responses
            if not raw_content.strip():
                return CognitiveGatewayOutput(
                    model_name=spec.name, raw_output="",
                    success=False, error=f"Qwen returned empty response",
                )

            return CognitiveGatewayOutput(
                model_name=spec.name,
                raw_output=raw_content,
                tokens_used=usage.get("total_tokens", 0),
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                success=True,
            )

    # ── NVIDIA Provider (Mistral Large 675B, Kimi K2) ────────

    async def _call_nvidia(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
        api_key: str = "",
        max_tokens: int = 4096,
    ) -> CognitiveGatewayOutput:
        """Call NVIDIA API (OpenAI-compatible) for Mistral Large 675B and Kimi K2 Thinking."""
        if not api_key:
            return CognitiveGatewayOutput(
                model_name=spec.name, raw_output="",
                success=False, error=f"{spec.api_key_env} not configured",
            )

        url = spec.api_base_url or "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Sanitize messages to plain text
        sanitized_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(text_parts) if text_parts else str(content)
            sanitized_messages.append({"role": msg["role"], "content": content})

        # Cap max_tokens to 1000 for NVIDIA models
        max_tokens = min(max_tokens, 1000)

        payload = {
            "model": spec.model_id,
            "messages": sanitized_messages,
            "temperature": spec.default_temperature,
            "top_p": 1.00 if "mistral" in spec.model_id else 0.24,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # Add frequency/presence penalty for Mistral
        if "mistral" in spec.model_id:
            payload["frequency_penalty"] = 0.00
            payload["presence_penalty"] = 0.00

        session = await self._get_session()
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"NVIDIA API error [{resp.status}] for {spec.model_id}: {text[:500]}")
                    if resp.status in (402, 429):
                        self._record_failure()
                    return CognitiveGatewayOutput(
                        model_name=spec.name, raw_output="",
                        success=False, error=self._sanitize_provider_error(resp.status, text),
                    )
                data = await resp.json()
                choice = data.get("choices", [{}])[0]
                usage = data.get("usage", {})
                raw_content = choice.get("message", {}).get("content") or ""

                if not raw_content.strip():
                    return CognitiveGatewayOutput(
                        model_name=spec.name, raw_output="",
                        success=False, error=f"NVIDIA model returned empty response",
                    )

                return CognitiveGatewayOutput(
                    model_name=spec.name,
                    raw_output=raw_content,
                    tokens_used=usage.get("total_tokens", 0),
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    success=True,
                )
        except Exception as e:
            return CognitiveGatewayOutput(
                model_name=spec.name, raw_output="",
                success=False, error=f"NVIDIA API error: {str(e)}",
            )

    # ── Anthropic Provider (Claude Sonnet 4.6 — synthesis only) ──

    async def _call_anthropic(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
        api_key: str = "",
        max_tokens: int = 500,
    ) -> CognitiveGatewayOutput:
        """Call Anthropic API for Claude Sonnet 4.6 with vision/PDF support."""
        if not api_key:
            return CognitiveGatewayOutput(
                model_name=spec.name, raw_output="",
                success=False, error=f"{spec.api_key_env} not configured",
            )

        url = spec.api_base_url or "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Convert OpenAI-format messages to Anthropic format with vision/PDF support
        system_text = ""
        anthropic_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if msg["role"] == "system":
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    system_text = " ".join(text_parts) if text_parts else str(content)
                else:
                    system_text = content
            else:
                # Handle multimodal content (vision/PDF)
                if isinstance(content, list):
                    anthropic_content = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            anthropic_content.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {})
                            url_str = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                            if url_str.startswith("data:"):
                                # Parse data URI: data:<media_type>;base64,<data>
                                header, _, b64_data = url_str.partition(",")
                                media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                                if "pdf" in media_type:
                                    anthropic_content.append({
                                        "type": "document",
                                        "source": {"type": "base64", "media_type": media_type, "data": b64_data},
                                    })
                                else:
                                    anthropic_content.append({
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": media_type, "data": b64_data},
                                    })
                            else:
                                anthropic_content.append({
                                    "type": "image",
                                    "source": {"type": "url", "url": url_str},
                                })
                    if anthropic_content:
                        anthropic_messages.append({"role": msg["role"], "content": anthropic_content})
                else:
                    anthropic_messages.append({"role": msg["role"], "content": content})

        # ── Input token cap: ~500 tokens (≈2000 chars) ──
        # Truncate system text and message content to stay within budget
        _INPUT_CHAR_CAP = 2000  # ~500 tokens at 4 chars/token
        if system_text and len(system_text) > _INPUT_CHAR_CAP:
            system_text = system_text[:_INPUT_CHAR_CAP - 20] + "\n...[truncated]"
        for amsg in anthropic_messages:
            c = amsg.get("content", "")
            if isinstance(c, str) and len(c) > _INPUT_CHAR_CAP:
                amsg["content"] = c[:_INPUT_CHAR_CAP - 20] + "\n...[truncated]"

        # ── Output token cap: 500 tokens ──
        max_tokens = min(max_tokens, 500)

        payload = {
            "model": spec.model_id,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": spec.default_temperature,
        }
        if system_text:
            payload["system"] = system_text

        session = await self._get_session()
        try:
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
                # Anthropic response: { content: [{ type: "text", text: "..." }] }
                content_blocks = data.get("content", [])
                raw_content = ""
                for block in content_blocks:
                    if block.get("type") == "text":
                        raw_content += block.get("text", "")

                if not raw_content.strip():
                    return CognitiveGatewayOutput(
                        model_name=spec.name, raw_output="",
                        success=False, error="Claude returned empty response",
                    )

                usage = data.get("usage", {})
                input_tok = usage.get("input_tokens", 0)
                output_tok = usage.get("output_tokens", 0)
                _track_claude_usage(input_tok, output_tok)
                return CognitiveGatewayOutput(
                    model_name=spec.name,
                    raw_output=raw_content,
                    tokens_used=input_tok + output_tok,
                    input_tokens=input_tok,
                    output_tokens=output_tok,
                    success=True,
                )
        except Exception as e:
            return CognitiveGatewayOutput(
                model_name=spec.name, raw_output="",
                success=False, error=f"Anthropic API error: {str(e)}",
            )

    # ── Legacy OpenRouter methods — DISABLED ────────────────
    # OpenRouter credits exhausted (2026-03-09).
    # These methods are preserved for backward compatibility but
    # should never be called in production.

    async def _call_openrouter_isolated(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
        api_key: str = "",
        max_tokens: int = 4096,
    ) -> CognitiveGatewayOutput:
        """DISABLED — OpenRouter credits exhausted. Always returns error."""
        return CognitiveGatewayOutput(
            model_name=spec.name, raw_output="",
            success=False,
            error="OpenRouter provider disabled — credits exhausted. Use Groq/Gemini/Qwen.",
        )

    async def _call_openrouter(
        self,
        spec: CognitiveModelSpec,
        messages: List[Dict[str, str]],
        api_key: str = "",
    ) -> CognitiveGatewayOutput:
        """DISABLED — OpenRouter credits exhausted. Always returns error."""
        return await self._call_openrouter_isolated(spec, messages, api_key)

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

        if status_code == 401 or "unauthorized" in raw_lower or ("invalid" in raw_lower and "key" in raw_lower):
            return "Invalid API key"
        if status_code == 402 or "insufficient" in raw_lower or "credit" in raw_lower or "quota" in raw_lower:
            return "Insufficient credit"
        if status_code == 429 or ("rate" in raw_lower and "limit" in raw_lower):
            return "Rate limit exceeded"
        if (
            status_code == 404
            or "not found" in raw_lower
            or "invalid model" in raw_lower
            or "does not exist" in raw_lower
            or "decommissioned" in raw_lower
            or "deprecated" in raw_lower
            or "model_decommissioned" in raw_lower
            or "model_not_found" in raw_lower
            or "model_not_active" in raw_lower
            or "not active" in raw_lower
        ):
            return "Model decommissioned or not found"
        if status_code == 503 or "unavailable" in raw_lower or "overloaded" in raw_lower:
            return "Provider unavailable"
        if "context" in raw_lower and ("length" in raw_lower or "token" in raw_lower):
            return "Token limit exceeded"

        return f"Provider error (HTTP {status_code})"
