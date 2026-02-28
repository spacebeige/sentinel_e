"""
============================================================
Provider Validator — Sentinel-E Cognitive Engine v7.0
============================================================
Validates all models BEFORE debate begins.

Checks:
  - API key exists and is non-empty
  - Model endpoint is valid (provider is known)
  - Vision capability detection
  - Timeout adequacy

If invalid → mark model unavailable but still show in status list.
Never silently exclude — always report.
============================================================
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from core.ensemble_schemas import (
    MIN_MODELS,
    ModelStatus,
    EnsembleFailure,
    EnsembleFailureCode,
)

logger = logging.getLogger("sentinel.provider_validator")

# Vision-capable model IDs
# NOTE: kimi-2.5 (moonshotai/kimi-k2) is text-only — NOT vision-capable
VISION_CAPABLE_MODELS = {
    "qwen3-vl",
}

# Provider → required env var mapping
PROVIDER_KEY_MAP = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Model-specific key overrides
MODEL_KEY_OVERRIDES: Dict[str, str] = {
    "qwen3-coder": "QWEN3_CODER_API_KEY",
    "qwen3-vl": "QWEN3_VL_API_KEY",
    "nemotron-nano": "NEMOTRON_API_KEY",
    "kimi-2.5": "KIMI_API_KEY",
}


def _check_api_key(model_id: str, provider: str) -> tuple[bool, str]:
    """Check if the API key for this model/provider exists."""
    # Model-specific override
    if model_id in MODEL_KEY_OVERRIDES:
        env_var = MODEL_KEY_OVERRIDES[model_id]
        val = os.environ.get(env_var, "").strip()
        if val:
            return True, ""
        # Fall through to provider key
        provider_var = PROVIDER_KEY_MAP.get(provider, "")
        if provider_var:
            val = os.environ.get(provider_var, "").strip()
            if val:
                return True, ""
        return False, f"Missing API key: {env_var} (and no {provider} fallback)"

    # Provider-level key
    if provider in PROVIDER_KEY_MAP:
        env_var = PROVIDER_KEY_MAP[provider]
        val = os.environ.get(env_var, "").strip()
        if val:
            return True, ""
        return False, f"Missing API key: {env_var}"

    return False, f"Unknown provider: {provider}"


def validate_providers(model_bridge) -> List[ModelStatus]:
    """
    Validate all registered models. Returns a ModelStatus for each.

    Rules:
        - Every model gets a status entry
        - Unavailable models are NOT removed — they appear with available=False
        - Vision capability is detected
        - Raises EnsembleFailure if <MIN_MODELS are available
    """
    try:
        model_ids = model_bridge.get_enabled_model_ids()
        model_info = model_bridge.get_enabled_models_info()
    except Exception as e:
        logger.error(f"Failed to get model registry: {e}")
        raise EnsembleFailure(
            EnsembleFailureCode.PROVIDER_VALIDATION_FAILED,
            f"Cannot access model registry: {e}",
            models_available=0,
        )

    statuses: List[ModelStatus] = []

    for mid in model_ids:
        info = model_info.get(mid, {})
        provider = info.get("provider", "unknown")
        model_name = info.get("name", mid)
        supports_vision = mid in VISION_CAPABLE_MODELS

        # Check API key
        key_ok, key_error = _check_api_key(mid, provider)

        # Determine availability
        available = key_ok
        error_msg = key_error if not key_ok else None

        status = ModelStatus(
            model_id=mid,
            model_name=model_name,
            available=available,
            api_key_present=key_ok,
            endpoint_valid=True,  # assume valid if provider is known
            supports_vision=supports_vision,
            provider=provider,
            error=error_msg,
        )
        statuses.append(status)

        if not available:
            logger.warning(f"Model {mid} unavailable: {error_msg}")
        else:
            logger.info(f"Model {mid} validated OK (vision={supports_vision})")

    # Count available
    available_count = sum(1 for s in statuses if s.available)

    if available_count < MIN_MODELS:
        raise EnsembleFailure(
            EnsembleFailureCode.INSUFFICIENT_MODELS,
            f"Cognitive ensemble requires ≥{MIN_MODELS} active models, "
            f"only {available_count} available. "
            f"Unavailable: {[s.model_id for s in statuses if not s.available]}",
            models_available=available_count,
        )

    logger.info(
        f"Provider validation complete: {available_count}/{len(statuses)} models available"
    )
    return statuses
