"""
============================================================
Standard Mode Routes — Sentinel-E v2
============================================================
POST /chat/{model_id}  — Individual model routing with retry + fallback.

Each request is routed directly to the specified model via the
CognitiveModelGateway.  On rate-limit (429) or service error (503)
the gateway retries up to MAX_RETRIES times with exponential back-off,
then falls back to the Tier-1 anchor model before returning an error.

Response shape:
  {
    "model_id":      str,
    "model_name":    str,
    "provider":      str,
    "response":      str,
    "latency_ms":    float,
    "tokens_used":   int,
    "retried":       bool,
    "fallback_used": bool,
    "fallback_model": str | null
  }
============================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from metacognitive.cognitive_gateway import (
    COGNITIVE_MODEL_REGISTRY,
    CognitiveModelGateway,
)
from metacognitive.schemas import CognitiveGatewayInput, QueryMode

logger = logging.getLogger("ChatRoutes")

router = APIRouter(prefix="/chat", tags=["Standard Mode"])

# ── Retry / Fallback Configuration ───────────────────────────
MAX_RETRIES: int = 2              # Up to 2 retry attempts for 429/503
RETRY_BASE_DELAY: float = 1.0    # Initial back-off in seconds
RETRY_MAX_DELAY: float = 8.0     # Maximum back-off cap
FALLBACK_MODEL: str = "llama31-8b"  # Tier-1 anchor used as fallback

# HTTP error codes that trigger retry
RETRYABLE_ERRORS = {"429", "503", "rate limit", "service unavailable", "overloaded"}

# Singleton gateway (re-uses shared HTTP session across requests)
_gateway: Optional[CognitiveModelGateway] = None


def _get_gateway() -> CognitiveModelGateway:
    global _gateway
    if _gateway is None:
        _gateway = CognitiveModelGateway()
    return _gateway


# ── Request / Response Models ─────────────────────────────────


class ChatRequest(BaseModel):
    """Direct-model chat request."""
    query: str = Field(..., description="User query to send to the model")
    chat_id: Optional[str] = Field(None, description="Session / chat identifier")
    system_role: Optional[str] = Field(
        None,
        description="Optional system-level instruction override"
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Optional token cap override (default: registry max)"
    )


class ChatResponse(BaseModel):
    """Response from direct model invocation."""
    model_id: str
    model_name: str
    provider: str
    response: str
    latency_ms: float
    tokens_used: int
    retried: bool = False
    fallback_used: bool = False
    fallback_model: Optional[str] = None
    error: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────


def _is_retryable(error_msg: str) -> bool:
    """Return True if the error message indicates a transient failure."""
    msg_lower = error_msg.lower()
    return any(trigger in msg_lower for trigger in RETRYABLE_ERRORS)


async def _invoke_with_retry(
    gateway: CognitiveModelGateway,
    model_key: str,
    gateway_input: CognitiveGatewayInput,
    max_retries: int = MAX_RETRIES,
) -> tuple:
    """
    Invoke a model with exponential back-off retry on transient errors.

    Returns:
        (output, retried: bool)
    """
    retried = False
    delay = RETRY_BASE_DELAY

    for attempt in range(max_retries + 1):
        output = await gateway.invoke_model(model_key, gateway_input)
        if output.success:
            return output, retried
        # Retryable? (429 rate-limit or 503 service error)
        if attempt < max_retries and output.error and _is_retryable(output.error):
            retried = True
            logger.warning(
                f"[ChatRoutes] Transient error for '{model_key}' "
                f"(attempt {attempt + 1}/{max_retries + 1}): {output.error}. "
                f"Retrying in {delay:.1f}s…"
            )
            await asyncio.sleep(min(delay, RETRY_MAX_DELAY))
            delay *= 2  # exponential back-off
        else:
            # Non-retryable or exhausted retries
            return output, retried

    # Should not reach here
    return output, retried  # type: ignore[return-value]


# ── Route ─────────────────────────────────────────────────────


@router.post("/{model_id}", response_model=ChatResponse)
async def chat_with_model(
    model_id: str,
    req: ChatRequest,
) -> ChatResponse:
    """
    Route a query to a specific model by its registry key.

    Behaviour:
      1. Validate model exists in COGNITIVE_MODEL_REGISTRY.
      2. Invoke via CognitiveModelGateway with retry on 429/503.
      3. If still failing after retries, fallback to FALLBACK_MODEL (llama31-8b).
      4. Return structured response including latency, tokens, and flag metadata.

    Path parameter:
      model_id — canonical registry key (e.g. "llama31-8b", "gemma2-9b")

    Example:
      POST /chat/gemma2-9b
      {"query": "Explain quantum entanglement in one paragraph."}
    """
    gateway = _get_gateway()

    # ── 1. Validate model ──────────────────────────────────────
    spec = COGNITIVE_MODEL_REGISTRY.get(model_id)
    if spec is None:
        available = sorted(COGNITIVE_MODEL_REGISTRY.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available: {available}",
        )
    if not spec.active:
        raise HTTPException(
            status_code=409,
            detail=f"Model '{model_id}' is structurally disabled (active=False).",
        )
    if not spec.enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{model_id}' is disabled — API key '{spec.api_key_env}' "
                f"is not configured."
            ),
        )

    # ── 2. Build gateway input ────────────────────────────────
    gateway_input = CognitiveGatewayInput(
        user_query=req.query,
        mode=QueryMode.RAW,
        max_tokens_override=req.max_tokens,
    )

    # ── 3. Invoke with retry ──────────────────────────────────
    start = time.monotonic()
    output, retried = await _invoke_with_retry(gateway, model_id, gateway_input)
    elapsed_ms = (time.monotonic() - start) * 1000

    # ── 4. Fallback to Tier-1 anchor if still failing ─────────
    fallback_used = False
    fallback_model_id: Optional[str] = None

    if not output.success and model_id != FALLBACK_MODEL:
        fallback_spec = COGNITIVE_MODEL_REGISTRY.get(FALLBACK_MODEL)
        if fallback_spec and fallback_spec.active and fallback_spec.enabled:
            logger.warning(
                f"[ChatRoutes] '{model_id}' failed after retries — "
                f"falling back to '{FALLBACK_MODEL}'"
            )
            fallback_input = CognitiveGatewayInput(
                user_query=req.query,
                mode=QueryMode.RAW,
                max_tokens_override=req.max_tokens,
            )
            fb_start = time.monotonic()
            fb_output, _ = await _invoke_with_retry(
                gateway, FALLBACK_MODEL, fallback_input, max_retries=1
            )
            elapsed_ms = (time.monotonic() - fb_start) * 1000

            if fb_output.success:
                output = fb_output
                fallback_used = True
                fallback_model_id = FALLBACK_MODEL
            else:
                logger.error(
                    f"[ChatRoutes] Fallback model '{FALLBACK_MODEL}' also failed: "
                    f"{fb_output.error}"
                )

    # ── 5. Build response ─────────────────────────────────────
    if not output.success:
        # All attempts (including fallback) exhausted
        raise HTTPException(
            status_code=502,
            detail={
                "model_id": model_id,
                "model_name": spec.name,
                "error": output.error or "Model invocation failed",
                "retried": retried,
                "fallback_attempted": fallback_model_id is not None,
            },
        )

    # Use most recent spec for response metadata
    resolved_spec = (
        COGNITIVE_MODEL_REGISTRY.get(FALLBACK_MODEL, spec)
        if fallback_used
        else spec
    )

    return ChatResponse(
        model_id=FALLBACK_MODEL if fallback_used else model_id,
        model_name=output.model_name,
        provider=resolved_spec.provider,
        response=output.raw_output,
        latency_ms=round(elapsed_ms, 2),
        tokens_used=output.tokens_used,
        retried=retried,
        fallback_used=fallback_used,
        fallback_model=fallback_model_id,
    )


@router.get("/models/available")
async def list_available_models() -> Dict[str, Any]:
    """
    Return all models currently enabled in the registry, with tier info.

    Used by the frontend to populate the model selector.
    """
    from metacognitive.cognitive_gateway import MODEL_DEBATE_TIERS

    models = []
    for key, spec in COGNITIVE_MODEL_REGISTRY.items():
        models.append({
            "id": key,
            "name": spec.name,
            "provider": spec.provider,
            "role": spec.role.value,
            "tier": MODEL_DEBATE_TIERS.get(key, 2),
            "enabled": spec.enabled and spec.active,
            "context_window": spec.context_window,
            "max_output_tokens": spec.max_output_tokens,
        })

    # Sort: enabled first, then by tier, then name
    models.sort(key=lambda m: (not m["enabled"], m["tier"], m["name"]))

    return {
        "models": models,
        "total": len(models),
        "enabled_count": sum(1 for m in models if m["enabled"]),
    }
