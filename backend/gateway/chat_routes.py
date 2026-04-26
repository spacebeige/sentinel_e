from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from metacognitive.cognitive_gateway import (
    COGNITIVE_MODEL_REGISTRY,
    CognitiveModelGateway,
    MODEL_FALLBACK_MAP,
)
from metacognitive.schemas import CognitiveGatewayInput, QueryMode
from utils.output_sanitizer import sanitize_output
from gateway.auth import get_user_id
from database.connection import get_db
from database.crud import create_chat, get_chat, add_message, get_chat_messages
from core.context_builder import get_context_builder
from utils.api_response import api_response, api_error, api_success

logger = logging.getLogger("ChatRoutes")

router = APIRouter(prefix="/chat", tags=["Standard Mode"])

# ── Retry / Fallback Configuration ───────────────────────────
MAX_RETRIES: int = 2
RETRY_BASE_DELAY: float = 1.0
RETRY_MAX_DELAY: float = 8.0
FALLBACK_MODEL: str = "llama33-70b"
RETRYABLE_ERRORS = {"429", "503", "rate limit", "service unavailable", "overloaded"}

_gateway: Optional[CognitiveModelGateway] = None

def _get_gateway() -> CognitiveModelGateway:
    global _gateway
    if _gateway is None:
        _gateway = CognitiveModelGateway()
    return _gateway

class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    system_role: Optional[str] = None
    max_tokens: Optional[int] = None
    image_b64: Optional[str] = None
    image_mime: Optional[str] = None

async def _invoke_with_retry(
    gateway: CognitiveModelGateway,
    model_key: str,
    gateway_input: CognitiveGatewayInput,
    max_retries: int = MAX_RETRIES,
) -> tuple:
    retried = False
    delay = RETRY_BASE_DELAY
    for attempt in range(max_retries + 1):
        output = await gateway.invoke_model(model_key, gateway_input)
        if output.success:
            return output, retried
        if attempt < max_retries and output.error and any(t in output.error.lower() for t in RETRYABLE_ERRORS):
            retried = True
            await asyncio.sleep(min(delay, RETRY_MAX_DELAY))
            delay *= 2
        else:
            return output, retried
    return output, retried

@router.post("/{model_id}")
async def chat_with_model(
    model_id: str,
    req: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        user_id = await get_user_id(request)
        if not user_id:
            return api_error("Authentication required", status_code=401)

        gateway = _get_gateway()
        spec = COGNITIVE_MODEL_REGISTRY.get(model_id)
        if spec is None or not spec.active or not spec.enabled:
            return api_error(f"Model {model_id} is unavailable", status_code=503)

        # Resolve or create chat
        chat = None
        if req.chat_id:
            try:
                chat = await get_chat(db, UUID(req.chat_id), user_id=user_id)
            except Exception:
                chat = None
        
        if chat is None:
            chat_name = req.query.strip()[:60] or "Direct model chat"
            chat = await create_chat(db, chat_name, "standard", user_id=user_id)

        # Persist user turn
        await add_message(
            db, chat.id, user_id, "user", req.query, 
            image_b64=req.image_b64
        )

        # Build context
        builder = get_context_builder()
        context_bundle = await builder.build_context(db, user_id, chat.id, req.query)
        system_instructions = context_bundle.get("system_instructions", "")
        
        contextual_query = f"{system_instructions}\n\n[USER QUERY]\n{req.query}"

        # Invoke model
        gateway_input = CognitiveGatewayInput(
            user_query=contextual_query,
            mode=QueryMode.RAW,
            max_tokens_override=req.max_tokens,
        )
        
        start = time.monotonic()
        output, retried = await _invoke_with_retry(gateway, model_id, gateway_input)
        elapsed_ms = (time.monotonic() - start) * 1000

        if not output.success:
            return api_error(f"Model invocation failed: {output.error}", status_code=502)

        # Sanitize and Persist
        sanitized = sanitize_output(output.raw_output)
        
        assistant_reasoning = {
            "model_id": model_id,
            "retried": retried,
            "latency_ms": elapsed_ms
        }
        
        await add_message(
            db, chat.id, user_id, "assistant", sanitized, 
            reasoning_json=assistant_reasoning
        )

        return api_success({
            "chat_id": str(chat.id),
            "model_id": model_id,
            "response": sanitized,
            "latency_ms": round(elapsed_ms, 2),
            "tokens_used": output.tokens_used
        })

    except Exception as e:
        logger.error(f"Error in direct chat: {e}")
        return api_error(str(e))

@router.get("/models/available")
async def list_available_models():
    try:
        from metacognitive.cognitive_gateway import MODEL_DEBATE_TIERS
        models = []
        for key, spec in COGNITIVE_MODEL_REGISTRY.items():
            if spec.enabled and spec.active:
                models.append({
                    "id": key,
                    "name": spec.name,
                    "provider": spec.provider,
                    "tier": MODEL_DEBATE_TIERS.get(key, 2)
                })
        return api_success({"models": models})
    except Exception as e:
        return api_error(str(e))
