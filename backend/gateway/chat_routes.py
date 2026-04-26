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
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from metacognitive.cognitive_gateway import (
    COGNITIVE_MODEL_REGISTRY,
    CognitiveModelGateway,
    MODEL_FALLBACK_MAP,
)
from metacognitive.schemas import CognitiveGatewayInput, QueryMode
from utils.output_sanitizer import sanitize_output
from gateway.auth import get_optional_user
from database.connection import get_db
from database.crud import create_chat, get_chat, add_message, get_chat_messages
from core.context_builder import get_context_builder

logger = logging.getLogger("ChatRoutes")

router = APIRouter(prefix="/chat", tags=["Standard Mode"])

# ── Retry / Fallback Configuration ───────────────────────────
MAX_RETRIES: int = 2              # Up to 2 retry attempts for 429/503
RETRY_BASE_DELAY: float = 1.0    # Initial back-off in seconds
RETRY_MAX_DELAY: float = 8.0     # Maximum back-off cap
FALLBACK_MODEL: str = "llama33-70b"  # Tier-1 anchor used as fallback

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
    image_b64: Optional[str] = Field(
        None,
        description="Optional base64 encoded image payload"
    )
    image_mime: Optional[str] = Field(
        None,
        description="Optional image mime type"
    )


class ChatResponse(BaseModel):
    """Response from direct model invocation."""
    chat_id: Optional[str] = None
    model_id: str
    model_name: str
    provider: str
    response: str
    formatted_output: str = ""
    priority_answer: str = ""
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
    db: AsyncSession = Depends(get_db),
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> ChatResponse:
    """
    Route a query to a specific model by its registry key.

    Behaviour:
      1. Validate model exists in COGNITIVE_MODEL_REGISTRY.
      2. Invoke via CognitiveModelGateway with retry on 429/503.
      3. If still failing after retries, fallback to FALLBACK_MODEL (llama31-8b).
      4. Return structured response including latency, tokens, and flag metadata.

    Path parameter:
      model_id — canonical registry key (e.g. "llama33-70b", "llama4-scout")

    Example:
      POST /chat/llama4-scout
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

    user_id = (user or {}).get("user_id") or "anonymous"

    # Resolve or create chat for persistence continuity
    chat = None
    if req.chat_id:
        try:
            chat = await get_chat(db, UUID(req.chat_id), user_id=user_id)
        except Exception:
            chat = None

    if chat is None:
        chat_name = (req.query.strip()[:60] or "Direct model chat")
        chat = await create_chat(db, chat_name, "standard", user_id=user_id)

    # Persist user turn (including images, if present)
    await add_message(
        db,
        chat.id,
        "user",
        req.query,
        image_b64=req.image_b64,
        image_mime=req.image_mime,
    )

    # Build prioritized, token-trimmed context before model execution
    contextual_query = req.query
    context_meta: Dict[str, Any] = {"context_applied": False}
    try:
        recent = await get_chat_messages(db, chat.id, user_id=user_id)
        recent_payload = [
            {"role": m.role, "content": m.content}
            for m in (recent[-12:] if recent else [])
            if m and m.content
        ]
        builder = get_context_builder(max_tokens=req.max_tokens or 2048, model=model_id)
        built = await builder.build_context(
            db=db,
            user_id=user_id,
            query=req.query,
            recent_messages=recent_payload,
            semantic_search_results=None,
        )
        built_context = (built or {}).get("context", "")
        if built_context and isinstance(built_context, str) and built_context.strip():
            contextual_query = f"{built_context}\n\n[CURRENT USER QUERY]\n{req.query}"
            context_meta = {
                "context_applied": True,
                "token_usage": built.get("token_usage", {}),
                "model": built.get("model"),
                "available_tokens": built.get("available_tokens"),
            }
    except Exception as ctx_err:
        logger.warning("[ChatRoutes] Context build skipped: %s", ctx_err)
        context_meta = {"context_applied": False, "error": str(ctx_err)[:200]}

    # ── 2. Build gateway input ────────────────────────────────
    gateway_input = CognitiveGatewayInput(
        user_query=contextual_query,
        mode=QueryMode.RAW,
        max_tokens_override=req.max_tokens,
    )

    # ── 3. Invoke with retry ──────────────────────────────────
    start = time.monotonic()
    output, retried = await _invoke_with_retry(gateway, model_id, gateway_input)
    elapsed_ms = (time.monotonic() - start) * 1000

    # ── 4. Per-model fallback if still failing ──────────────────
    fallback_used = False
    fallback_model_id: Optional[str] = None

    if not output.success:
        # Use per-model fallback from MODEL_FALLBACK_MAP instead of universal fallback
        fb_key = MODEL_FALLBACK_MAP.get(model_id)
        if fb_key and fb_key != model_id:
            fallback_spec = COGNITIVE_MODEL_REGISTRY.get(fb_key)
            if fallback_spec and fallback_spec.active and fallback_spec.enabled:
                logger.warning(
                    f"[ChatRoutes] '{model_id}' failed after retries — "
                    f"falling back to '{fb_key}'"
                )
                fallback_input = CognitiveGatewayInput(
                    user_query=req.query,
                    mode=QueryMode.RAW,
                    max_tokens_override=req.max_tokens,
                )
                fb_start = time.monotonic()
                fb_output, _ = await _invoke_with_retry(
                    gateway, fb_key, fallback_input, max_retries=1
                )
                elapsed_ms = (time.monotonic() - fb_start) * 1000

                if fb_output.success:
                    output = fb_output
                    fallback_used = True
                    fallback_model_id = fb_key
                else:
                    logger.error(
                        f"[ChatRoutes] Fallback model '{fb_key}' also failed: "
                        f"{fb_output.error}"
                    )

    # ── 5. Build response ─────────────────────────────────────
    if not output.success:
        # All attempts (including fallback) exhausted
        logger.error(
            f"[ChatRoutes] All attempts failed for '{model_id}': "
            f"{output.error}, retried={retried}, "
            f"fallback_attempted={fallback_model_id is not None}"
        )
        raise HTTPException(
            status_code=502,
            detail="Provider unavailable. Please try again or select a different model.",
        )

    # Use most recent spec for response metadata
    resolved_spec = (
        COGNITIVE_MODEL_REGISTRY.get(fallback_model_id, spec)
        if fallback_used
        else spec
    )

    # Sanitize output to remove internal reasoning tags
    sanitized_output = sanitize_output(output.raw_output)

    # Persist assistant turn with reasoning metadata
    assistant_reasoning = {
        "mode": "standard",
        "model_id": fallback_model_id if fallback_used else model_id,
        "model_name": output.model_name,
        "provider": resolved_spec.provider,
        "retried": retried,
        "fallback_used": fallback_used,
        "fallback_model": fallback_model_id,
        "context_builder": context_meta,
    }
    await add_message(
        db,
        chat.id,
        "assistant",
        sanitized_output,
        reasoning_json=assistant_reasoning,
    )
    
    return ChatResponse(
        chat_id=str(chat.id),
        model_id=fallback_model_id if fallback_used else model_id,
        model_name=output.model_name,
        provider=resolved_spec.provider,
        response=sanitized_output,
        formatted_output=sanitized_output,
        priority_answer=sanitized_output,
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
            "model_type": getattr(spec, "model_type", "external"),
            "role": spec.role.value,
            "tier": MODEL_DEBATE_TIERS.get(key, 2),
            "enabled": spec.enabled and spec.active,
            "active": spec.active,
            "disable_reason": getattr(spec, "disable_reason", None),
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
