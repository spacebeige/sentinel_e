from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from metacognitive.cognitive_gateway import (
    COGNITIVE_MODEL_REGISTRY,
    CognitiveModelGateway,
    MODEL_FALLBACK_MAP,
)
from metacognitive.schemas import CognitiveGatewayInput, QueryMode
from utils.output_sanitizer import sanitize_output
from gateway.auth_v2 import get_optional_user
from database.connection import get_db
from database.crud import create_chat, get_chat, add_message, get_chat_messages
from core.context_builder import get_context_builder
from core.document_cognition import build_document_cognition
from utils.api_response import api_response, api_error, api_success

try:
    from core.orchestration_run import create_orchestration_run, CognitivePhase
    from core.runtime_event_bus import create_run_bus
    _COGNITIVE_RUNTIME_ENABLED = True
except ImportError:
    create_orchestration_run = None
    CognitivePhase = None
    create_run_bus = None
    _COGNITIVE_RUNTIME_ENABLED = False

try:
    from memory.behavioral_memory import BehavioralMemoryManager
except ImportError:
    BehavioralMemoryManager = None

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
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    try:
        user = await get_optional_user(request)
        user_id = user.get("id") if user else None
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

        orch_run = None
        orch_bus = None
        if _COGNITIVE_RUNTIME_ENABLED:
            try:
                orch_run = create_orchestration_run(
                    chat_id=str(chat.id),
                    user_id=user_id,
                    query_preview=req.query[:80],
                    execution_path="single_model",
                )
                orch_bus = create_run_bus(orch_run.orchestration_run_id)
                orch_run.transition_to(CognitivePhase.OBSERVE, {"source": "chat.direct"})
                orch_run.transition_to(CognitivePhase.ROUTE, {
                    "path": "single_model",
                    "selected_model": model_id,
                    "reason": "direct_model_endpoint",
                })
                orch_run.record_routing_decision({
                    "path": "single_model",
                    "selected_model": model_id,
                    "reason": "direct_model_endpoint",
                    "query_complexity": "direct_invocation",
                })
            except Exception:
                orch_run = None
                orch_bus = None

        # Persist user turn
        await add_message(
            db, chat.id, user_id, "user", req.query, 
            image_b64=req.image_b64,
            image_mime=req.image_mime,
        )

        document_cognition: Dict[str, Any] = {"available": False}
        if req.image_b64:
            try:
                document_cognition = await build_document_cognition(req.image_b64, req.image_mime)
                if orch_run and document_cognition.get("semantic_context"):
                    orch_run.transition_to(CognitivePhase.VERIFY, {"source": "document_cognition", "mime": req.image_mime})
                    orch_run.record_memory_retrieval(
                        "working",
                        "document_cognition",
                        document_cognition.get("semantic_context", ""),
                        0.88,
                    )
            except Exception as doc_err:
                logger.warning("Document cognition skipped for direct chat: %s", doc_err)
                document_cognition = {"available": False, "error": str(doc_err)[:200]}

        # Build context
        builder = get_context_builder()
        context_bundle = await builder.build_context(db, user_id, chat.id, req.query)
        system_instructions = context_bundle.get("system_instructions", "")
        if orch_run and system_instructions:
            try:
                orch_run.transition_to(CognitivePhase.RETRIEVE_MEMORY, {"layer": "episodic", "key": "context_builder"})
                orch_run.record_memory_retrieval("episodic", "context_builder", system_instructions[:200], 0.9)
            except Exception:
                pass
        
        contextual_query = f"{system_instructions}\n\n[USER QUERY]\n{req.query}"
        if document_cognition.get("semantic_context"):
            contextual_query = f"[DOCUMENT COGNITION]\n{document_cognition['semantic_context']}\n\n{contextual_query}"

        # Invoke model
        gateway_input = CognitiveGatewayInput(
            user_query=contextual_query,
            mode=QueryMode.RAW,
            max_tokens_override=req.max_tokens,
        )
        if orch_run:
            try:
                orch_run.transition_to(CognitivePhase.SPAWN_AGENTS, {"path": "single_model"})
            except Exception:
                pass
        
        start = time.monotonic()
        output, retried = await _invoke_with_retry(gateway, model_id, gateway_input)
        elapsed_ms = (time.monotonic() - start) * 1000

        if not output.success:
            if orch_run:
                try:
                    orch_run.mark_failed(output.error or "direct_model_failed", "DIRECT_MODEL_FAILED")
                    if orch_bus:
                        orch_bus.close()
                except Exception:
                    pass
            return api_error(f"Model invocation failed: {output.error}", status_code=502)

        # Sanitize and Persist
        sanitized = sanitize_output(output.raw_output)
        omega_metadata = {
            "version": "8.0.0-cognitive-runtime",
            "mode": "single_model",
            "sub_mode": None,
            "selected_model": model_id,
            "model_name": output.model_name,
            "provider": getattr(spec, "provider", "unknown"),
            "confidence": 0.78,
            "latency_ms": round(elapsed_ms, 2),
            "retried": retried,
            "document_cognition": {
                key: value
                for key, value in document_cognition.items()
                if key != "semantic_context"
            },
        }
        if orch_run:
            try:
                orch_run.record_provider_call(
                    model_id=model_id,
                    model_name=output.model_name,
                    provider=getattr(spec, "provider", "unknown"),
                    latency_ms=elapsed_ms,
                    succeeded=True,
                    input_tokens=int(getattr(output, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(output, "output_tokens", 0) or 0),
                )
                orch_run.record_confidence_snapshot("single_model", 0.78, method="direct_model")
                orch_run.transition_to(CognitivePhase.SYNTHESIZE, {"method": "single_model"})
                orch_run.record_synthesis_start("single_model", 1)
                orch_run.record_synthesis_complete(len(sanitized))
                orch_run.transition_to(CognitivePhase.STORE_SNAPSHOT)
                orch_run.mark_completed()
                omega_metadata["orchestration_run"] = orch_run.to_frontend_dict()
                if orch_bus:
                    orch_bus.close()
            except Exception:
                pass
        
        assistant_reasoning = {
            "model_id": model_id,
            "retried": retried,
            "latency_ms": elapsed_ms,
            **omega_metadata,
        }
        
        await add_message(
            db, chat.id, user_id, "assistant", sanitized, 
            reasoning_json=assistant_reasoning
        )

        if BehavioralMemoryManager and user_id:
            background_tasks.add_task(
                BehavioralMemoryManager.update_profile_async,
                db=db,
                user_id=user_id,
                interaction_metrics={
                    "model": model_id,
                    "query_complexity": "complex" if len(req.query) > 100 else "simple",
                    "latency_ms": elapsed_ms,
                }
            )

        return api_success({
            "chat_id": str(chat.id),
            "mode": "single_model",
            "sub_mode": None,
            "model_id": model_id,
            "formatted_output": sanitized,
            "data": {"priority_answer": sanitized},
            "confidence": 0.78,
            "boundary_result": {"risk_level": "LOW", "severity_score": 10},
            "omega_metadata": omega_metadata,
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
