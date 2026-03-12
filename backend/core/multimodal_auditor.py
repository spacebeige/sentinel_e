"""
============================================================
Sentinel-E Multimodal Capability Auditor & Router
============================================================
8-Phase pipeline that inspects incoming requests, determines
required capabilities, audits model availability, performs
auto-recovery, and routes to the correct model pipeline.

Phases:
  1. Input Inspection — classify request type
  2. Capability Check — match type to capable models
  3. Model Availability Audit — verify API keys & endpoints
  4. Auto Recovery — fallback if primary models are OFF
  5. Model Routing — build the execution pipeline
  6. Integration Check — verify subsystems (Redis, RAG, etc.)
  7. Execution — invoke the pipeline through CognitiveOrchestrator
  8. Output Structure — return structured audit + response

No silent skips.  Always report disabled models.
Always attempt fallback.  Never run multimodal on text-only models.
Minimum 3 models in every reasoning pipeline.
============================================================
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from metacognitive.cognitive_gateway import (
    COGNITIVE_MODEL_REGISTRY,
    CognitiveModelSpec,
    _initialize_registry,
)

logger = logging.getLogger("MultimodalAuditor")

# ============================================================
# Constants
# ============================================================

MINIMUM_PIPELINE_MODELS = 3

# MIME types recognised as images
_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# MIME types recognised as PDF documents
_PDF_MIMES = {"application/pdf"}

# Base64 header patterns
_B64_IMAGE_PREFIX = re.compile(
    r"^data:(image/(?:jpeg|png|gif|webp));base64,", re.IGNORECASE
)
_B64_PDF_PREFIX = re.compile(
    r"^data:application/pdf;base64,", re.IGNORECASE
)

# URL pattern (simplified)
_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


# ============================================================
# Enums & Data Structures
# ============================================================

class InputType(str, Enum):
    TEXT_ONLY = "TEXT_ONLY"
    IMAGE_INPUT = "IMAGE_INPUT"
    PDF_DOCUMENT = "PDF_DOCUMENT"
    MULTIMODAL = "MULTIMODAL"
    DOCUMENT_ANALYSIS = "DOCUMENT_ANALYSIS"


class AuditStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DEGRADED = "DEGRADED"


@dataclass
class ModelAuditEntry:
    """Availability record for a single model."""
    registry_key: str
    name: str
    provider: str
    model_id: str
    enabled: bool
    active: bool
    api_key_present: bool
    api_key_env: str
    supports_vision: bool
    disabled_reason: Optional[str] = None


@dataclass
class DisabledModelReport:
    """Why a model is OFF."""
    registry_key: str
    provider: str
    required_env_var: str
    reason: str


@dataclass
class InputInspection:
    """Phase 1 result."""
    input_type: InputType
    has_text: bool = True
    has_image: bool = False
    has_pdf: bool = False
    has_base64_content: bool = False
    has_urls: bool = False
    detected_urls: List[str] = field(default_factory=list)
    image_mime: Optional[str] = None
    multimodal_required: bool = False


@dataclass
class CapabilityCheck:
    """Phase 2 result — which models CAN handle this request."""
    required_capabilities: List[str] = field(default_factory=list)
    capable_models: List[str] = field(default_factory=list)
    preferred_models: List[str] = field(default_factory=list)
    fallback_models: List[str] = field(default_factory=list)


@dataclass
class ModelPipeline:
    """Phase 5 result — the execution pipeline."""
    analysis_model: Optional[str] = None
    critique_models: List[str] = field(default_factory=list)
    synthesis_model: Optional[str] = None
    verification_model: Optional[str] = None

    @property
    def all_models(self) -> List[str]:
        models = []
        if self.analysis_model:
            models.append(self.analysis_model)
        models.extend(self.critique_models)
        if self.synthesis_model:
            models.append(self.synthesis_model)
        if self.verification_model:
            models.append(self.verification_model)
        return models

    @property
    def model_count(self) -> int:
        return len(self.all_models)


@dataclass
class SubsystemStatus:
    """Phase 6 — integration check."""
    model_registry_loaded: bool = False
    redis_active: bool = False
    retrieval_active: bool = False
    graphrag_available: bool = False
    evidence_extraction: bool = False
    all_healthy: bool = False


@dataclass
class SystemAuditReport:
    """Phase 8 — final structured output."""
    # Phase 1
    input_type: InputType = InputType.TEXT_ONLY
    required_capabilities: List[str] = field(default_factory=list)
    multimodal_required: bool = False

    # Phase 3
    selected_models: List[str] = field(default_factory=list)
    disabled_models: List[DisabledModelReport] = field(default_factory=list)
    fallback_models: List[str] = field(default_factory=list)

    # Phase 5
    pipeline: Optional[ModelPipeline] = None

    # Phase 6
    subsystems: Optional[SubsystemStatus] = None

    # Phase 7
    execution_status: AuditStatus = AuditStatus.SUCCESS
    execution_error: Optional[str] = None

    # Timing
    audit_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for API response."""
        return {
            "SYSTEM_AUDIT": {
                "input_type": self.input_type.value,
                "required_capabilities": self.required_capabilities,
                "multimodal_required": self.multimodal_required,
                "selected_models": self.selected_models,
                "disabled_models": [
                    {
                        "MODEL_DISABLED_REASON": d.reason,
                        "provider": d.provider,
                        "required_env_var": d.required_env_var,
                    }
                    for d in self.disabled_models
                ],
                "fallback_models": self.fallback_models,
            },
            "MODEL_PIPELINE": {
                "analysis_model": self.pipeline.analysis_model if self.pipeline else None,
                "critique_models": self.pipeline.critique_models if self.pipeline else [],
                "synthesis_model": self.pipeline.synthesis_model if self.pipeline else None,
                "verification_model": self.pipeline.verification_model if self.pipeline else None,
            },
            "EXECUTION_STATUS": self.execution_status.value,
            "SUBSYSTEMS": {
                "model_registry_loaded": self.subsystems.model_registry_loaded if self.subsystems else False,
                "redis_active": self.subsystems.redis_active if self.subsystems else False,
                "retrieval_active": self.subsystems.retrieval_active if self.subsystems else False,
                "graphrag_available": self.subsystems.graphrag_available if self.subsystems else False,
                "evidence_extraction": self.subsystems.evidence_extraction if self.subsystems else False,
            },
            "audit_latency_ms": round(self.audit_latency_ms, 2),
        }


# ============================================================
# Capability Matrix
# ============================================================

# Maps InputType → list of registry keys that can handle it.
# Order = preference (first = best).

_VISION_MODELS = ["gemini-flash", "qwen-2.5-vl"]
_TEXT_MODELS = list(COGNITIVE_MODEL_REGISTRY.keys())

CAPABILITY_MATRIX: Dict[InputType, List[str]] = {
    InputType.TEXT_ONLY: _TEXT_MODELS,
    InputType.IMAGE_INPUT: ["gemini-flash", "qwen-2.5-vl"],
    InputType.PDF_DOCUMENT: ["gemini-flash", "qwen-2.5-vl"],
    InputType.MULTIMODAL: ["gemini-flash", "qwen-2.5-vl"],
    InputType.DOCUMENT_ANALYSIS: ["gemini-flash", "qwen-2.5-vl"],
}

# Provider → shared env var (mirrors _initialize_registry)
_PROVIDER_SHARED_KEY: Dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "qwen": "QWEN_API_KEY",
}

# Routing table: InputType → role assignments
_ROUTING_TABLE: Dict[InputType, Dict[str, List[str]]] = {
    InputType.TEXT_ONLY: {
        "analysis": ["llama33-70b"],
        "critique": ["mixtral-8x7b", "llama4-scout", "qwen-2.5-vl"],
        "synthesis": ["gemini-flash"],
        "verification": ["llama31-8b"],
    },
    InputType.IMAGE_INPUT: {
        "analysis": ["gemini-flash"],
        "critique": ["qwen-2.5-vl", "mixtral-8x7b", "llama4-scout"],
        "synthesis": ["llama33-70b"],
        "verification": ["llama31-8b"],
    },
    InputType.PDF_DOCUMENT: {
        "analysis": ["gemini-flash"],
        "critique": ["qwen-2.5-vl", "mixtral-8x7b", "llama4-scout"],
        "synthesis": ["llama33-70b"],
        "verification": ["llama31-8b"],
    },
    InputType.MULTIMODAL: {
        "analysis": ["gemini-flash"],
        "critique": ["qwen-2.5-vl", "llama33-70b", "mixtral-8x7b"],
        "synthesis": ["llama4-scout"],
        "verification": ["llama31-8b"],
    },
    InputType.DOCUMENT_ANALYSIS: {
        "analysis": ["gemini-flash"],
        "critique": ["qwen-2.5-vl", "mixtral-8x7b", "llama4-scout"],
        "synthesis": ["llama33-70b"],
        "verification": ["llama31-8b"],
    },
}


# ============================================================
# Phase Implementations
# ============================================================

def phase1_inspect_input(
    query: str,
    image_b64: Optional[str] = None,
    image_mime: Optional[str] = None,
    file_mime: Optional[str] = None,
) -> InputInspection:
    """
    PHASE 1 — Input Inspection.

    Classify the incoming request as one of:
      TEXT_ONLY, IMAGE_INPUT, PDF_DOCUMENT, MULTIMODAL, DOCUMENT_ANALYSIS

    Checks:
      1. Text query present
      2. Image attachments (via image_b64 / image_mime)
      3. PDF documents (via file_mime)
      4. Embedded base64 content in query text
      5. URLs in query text
    """
    result = InputInspection(input_type=InputType.TEXT_ONLY)
    result.has_text = bool(query and query.strip())

    # ── Image attachment ─────────────────────────────────────
    if image_b64:
        result.has_image = True
        result.image_mime = image_mime
        result.has_base64_content = True

    # ── File MIME check ──────────────────────────────────────
    if file_mime:
        if file_mime in _PDF_MIMES:
            result.has_pdf = True
        elif file_mime in _IMAGE_MIMES:
            result.has_image = True
            result.image_mime = file_mime

    # ── Embedded base64 in query text ────────────────────────
    if query:
        if _B64_IMAGE_PREFIX.search(query):
            result.has_base64_content = True
            result.has_image = True
        if _B64_PDF_PREFIX.search(query):
            result.has_base64_content = True
            result.has_pdf = True

        # URL detection
        urls = _URL_PATTERN.findall(query)
        if urls:
            result.has_urls = True
            result.detected_urls = urls[:10]  # cap to prevent abuse

    # ── Classify ─────────────────────────────────────────────
    if result.has_image and result.has_text:
        result.input_type = InputType.MULTIMODAL
    elif result.has_image and not result.has_text:
        result.input_type = InputType.IMAGE_INPUT
    elif result.has_pdf:
        result.input_type = InputType.PDF_DOCUMENT
        # PDFs with text are document analysis
        if result.has_text:
            result.input_type = InputType.DOCUMENT_ANALYSIS
    else:
        result.input_type = InputType.TEXT_ONLY

    result.multimodal_required = result.input_type in (
        InputType.IMAGE_INPUT,
        InputType.PDF_DOCUMENT,
        InputType.MULTIMODAL,
        InputType.DOCUMENT_ANALYSIS,
    )

    return result


def phase2_capability_check(
    inspection: InputInspection,
) -> CapabilityCheck:
    """
    PHASE 2 — Capability Check.

    Match the input type to models that support the required capability.
    Returns preferred models (enabled + capable) and fallback options.
    """
    input_type = inspection.input_type
    required_caps: List[str] = []

    if inspection.has_image:
        required_caps.append("vision")
    if inspection.has_pdf:
        required_caps.append("document_parsing")
    if inspection.has_text:
        required_caps.append("text_reasoning")

    # Which models can handle this input type?
    capable_keys = CAPABILITY_MATRIX.get(input_type, _TEXT_MODELS)

    # Filter to enabled models
    preferred = [
        k for k in capable_keys
        if k in COGNITIVE_MODEL_REGISTRY
        and COGNITIVE_MODEL_REGISTRY[k].enabled
        and COGNITIVE_MODEL_REGISTRY[k].active
    ]

    # Fallback = all enabled text models (for text-extraction degradation)
    fallback = [
        k for k, spec in COGNITIVE_MODEL_REGISTRY.items()
        if spec.enabled and spec.active and k not in preferred
    ]

    return CapabilityCheck(
        required_capabilities=required_caps,
        capable_models=capable_keys,
        preferred_models=preferred,
        fallback_models=fallback,
    )


def phase3_model_availability_audit() -> Tuple[List[ModelAuditEntry], List[DisabledModelReport]]:
    """
    PHASE 3 — Model Availability Audit.

    For every model in the registry, check:
      1. Registry entry exists
      2. Model is enabled
      3. API key present (per-model or shared provider key)
      4. Provider endpoint configured
      5. Model marked active
    """
    entries: List[ModelAuditEntry] = []
    disabled_reports: List[DisabledModelReport] = []

    for key, spec in COGNITIVE_MODEL_REGISTRY.items():
        # Check API key (per-model, then shared)
        api_key = ""
        if spec.api_key_env:
            api_key = os.getenv(spec.api_key_env, "")
        if not api_key:
            shared_env = _PROVIDER_SHARED_KEY.get(spec.provider, "")
            if shared_env:
                api_key = os.getenv(shared_env, "")

        key_present = bool(api_key)

        entry = ModelAuditEntry(
            registry_key=key,
            name=spec.name,
            provider=spec.provider,
            model_id=spec.model_id,
            enabled=spec.enabled,
            active=spec.active,
            api_key_present=key_present,
            api_key_env=spec.api_key_env,
            supports_vision=spec.supports_vision,
        )

        if not spec.enabled or not spec.active:
            reason_parts = []
            if not key_present:
                reason_parts.append(f"missing {spec.api_key_env}")
            if not spec.active:
                reason_parts.append("explicitly deactivated")

            reason = " — ".join(reason_parts) if reason_parts else "disabled"
            entry.disabled_reason = reason

            disabled_reports.append(DisabledModelReport(
                registry_key=key,
                provider=spec.provider,
                required_env_var=spec.api_key_env or _PROVIDER_SHARED_KEY.get(spec.provider, ""),
                reason=f"{spec.name} disabled — {reason}",
            ))

        entries.append(entry)

    return entries, disabled_reports


def phase4_auto_recovery(
    inspection: InputInspection,
    capability: CapabilityCheck,
) -> CapabilityCheck:
    """
    PHASE 4 — Auto Recovery.

    If the requested capability requires Gemini/Qwen but they're OFF:
      1. Re-check environment variables (in case of late loading)
      2. Re-check registry activation flags
      3. Attempt fallback routing:
         - PDF → text extraction + llama-3.3-70b
         - IMAGE → reject with capability message OR fallback to Qwen VL

    Returns updated CapabilityCheck with recovery applied.
    """
    if not inspection.multimodal_required:
        return capability

    # If we already have vision-capable models, no recovery needed
    vision_available = any(
        COGNITIVE_MODEL_REGISTRY[k].supports_vision
        for k in capability.preferred_models
        if k in COGNITIVE_MODEL_REGISTRY
    )

    if vision_available:
        return capability

    # ── Recovery attempt: re-scan environment ────────────────
    logger.warning("No vision models available — attempting auto-recovery")

    # Re-initialise the registry (picks up any late-loaded env vars)
    _initialize_registry()

    # Re-check vision models
    recovered_preferred = []
    for key in ["gemini-flash", "qwen-2.5-vl"]:
        spec = COGNITIVE_MODEL_REGISTRY.get(key)
        if spec and spec.enabled and spec.active:
            recovered_preferred.append(key)
            logger.info(f"Auto-recovery: {key} is now available")

    if recovered_preferred:
        capability.preferred_models = recovered_preferred + capability.preferred_models
        return capability

    # ── Fallback routing ─────────────────────────────────────
    logger.warning("Auto-recovery failed — applying fallback routing")

    if inspection.input_type in (InputType.PDF_DOCUMENT, InputType.DOCUMENT_ANALYSIS):
        # PDF fallback: extract text, process with text models
        logger.info(
            "PDF fallback: will attempt text extraction → llama-3.3-70b"
        )
        capability.fallback_models = ["llama33-70b", "mixtral-8x7b", "llama31-8b"]
        capability.preferred_models = capability.fallback_models

    elif inspection.input_type in (InputType.IMAGE_INPUT, InputType.MULTIMODAL):
        # Image fallback: check if Qwen VL is at least structurally present
        qwen_spec = COGNITIVE_MODEL_REGISTRY.get("qwen-2.5-vl")
        if qwen_spec and not qwen_spec.enabled:
            logger.warning(
                "IMAGE capability required but no vision model available. "
                "Qwen VL exists but is disabled (missing QWEN_API_KEY)."
            )
        # Use text-only models as degraded fallback
        capability.fallback_models = ["llama33-70b", "mixtral-8x7b", "llama31-8b"]
        capability.preferred_models = capability.fallback_models

    return capability


def phase5_build_pipeline(
    inspection: InputInspection,
    capability: CapabilityCheck,
) -> ModelPipeline:
    """
    PHASE 5 — Model Routing / Pipeline Construction.

    Build the execution pipeline based on input type + available models.
    Guarantees minimum 3 models participate.
    """
    input_type = inspection.input_type
    routing = _ROUTING_TABLE.get(input_type, _ROUTING_TABLE[InputType.TEXT_ONLY])

    enabled_set = set(capability.preferred_models + capability.fallback_models)

    pipeline = ModelPipeline()

    # Select analysis model (first available from routing table)
    for key in routing["analysis"]:
        if key in enabled_set:
            pipeline.analysis_model = key
            break
    if not pipeline.analysis_model:
        # Absolute fallback: any enabled model
        for key in enabled_set:
            pipeline.analysis_model = key
            break

    # Select critique models (up to 3)
    for key in routing["critique"]:
        if key in enabled_set and key != pipeline.analysis_model:
            pipeline.critique_models.append(key)
        if len(pipeline.critique_models) >= 3:
            break
    # Fill critique slots from other enabled models
    if len(pipeline.critique_models) < 2:
        for key in enabled_set:
            if key not in pipeline.critique_models and key != pipeline.analysis_model:
                pipeline.critique_models.append(key)
            if len(pipeline.critique_models) >= 3:
                break

    # Select synthesis model
    for key in routing.get("synthesis", []):
        if key in enabled_set and key not in pipeline.all_models:
            pipeline.synthesis_model = key
            break

    # Select verification model
    for key in routing.get("verification", []):
        if key in enabled_set and key not in pipeline.all_models:
            pipeline.verification_model = key
            break
    # If verification slot empty, pick any remaining
    if not pipeline.verification_model:
        for key in enabled_set:
            if key not in pipeline.all_models:
                pipeline.verification_model = key
                break

    # ── Validation: minimum 3 models ─────────────────────────
    if pipeline.model_count < MINIMUM_PIPELINE_MODELS:
        logger.warning(
            f"Pipeline has only {pipeline.model_count} models "
            f"(minimum {MINIMUM_PIPELINE_MODELS}). "
            f"Available: {enabled_set}"
        )

    return pipeline


async def phase6_integration_check(
    redis_client=None,
    cognitive_rag=None,
) -> SubsystemStatus:
    """
    PHASE 6 — Integration Check.

    Verify subsystems before execution:
      - Model registry loaded
      - Redis session active
      - Retrieval services active
      - GraphRAG available
      - Evidence extraction enabled
    """
    status = SubsystemStatus()

    # Model registry
    status.model_registry_loaded = len(COGNITIVE_MODEL_REGISTRY) > 0

    # Redis
    if redis_client:
        try:
            await redis_client.ping()
            status.redis_active = True
        except Exception:
            status.redis_active = False
    else:
        status.redis_active = False

    # Retrieval / RAG
    status.retrieval_active = cognitive_rag is not None

    # GraphRAG (check if module is importable)
    try:
        from retrieval.cognitive_rag import CognitiveRAG  # noqa: F401
        status.graphrag_available = True
    except ImportError:
        status.graphrag_available = False

    # Evidence extraction (always available in v5)
    status.evidence_extraction = True

    status.all_healthy = all([
        status.model_registry_loaded,
        status.redis_active,
        status.retrieval_active,
    ])

    if not status.all_healthy:
        failed = []
        if not status.model_registry_loaded:
            failed.append("model_registry")
        if not status.redis_active:
            failed.append("redis")
        if not status.retrieval_active:
            failed.append("retrieval")
        logger.warning(
            f"Integration check: degraded — failed subsystems: {failed}"
        )

    return status


# ============================================================
# Main Auditor Class
# ============================================================

class MultimodalAuditor:
    """
    Sentinel-E Multimodal Capability Auditor.

    Runs the full 8-phase audit pipeline:
      1. Input Inspection
      2. Capability Check
      3. Model Availability Audit
      4. Auto Recovery
      5. Pipeline Construction
      6. Integration Check
      7. Execution (delegates to CognitiveOrchestrator)
      8. Output Assembly

    Usage:
        auditor = MultimodalAuditor(
            cognitive_engine=cognitive_orchestrator_engine,
            redis_client=redis_client,
            cognitive_rag=cognitive_rag,
        )
        result = await auditor.audit_and_route(
            query="Analyze this image",
            image_b64="...",
            image_mime="image/png",
        )
    """

    def __init__(
        self,
        cognitive_engine=None,
        redis_client=None,
        cognitive_rag=None,
    ):
        self._engine = cognitive_engine
        self._redis = redis_client
        self._rag = cognitive_rag

    async def audit_and_route(
        self,
        query: str,
        chat_id: str = "",
        rounds: int = 3,
        history: Optional[List[Dict[str, str]]] = None,
        image_b64: Optional[str] = None,
        image_mime: Optional[str] = None,
        file_mime: Optional[str] = None,
        execute: bool = True,
    ) -> Dict[str, Any]:
        """
        Full audit pipeline.  Returns structured audit report + model response.

        Args:
            query: User's text query.
            chat_id: Chat session ID.
            rounds: Debate rounds (minimum 3).
            history: Conversation history.
            image_b64: Base64-encoded image attachment.
            image_mime: MIME type of image attachment.
            file_mime: MIME type of uploaded file (PDF, etc.).
            execute: If True, run the model pipeline.  If False, audit only.

        Returns:
            Dict with keys: SYSTEM_AUDIT, MODEL_PIPELINE,
            EXECUTION_STATUS, FINAL_RESPONSE
        """
        start = time.monotonic()

        # ── PHASE 1: Input Inspection ────────────────────────
        inspection = phase1_inspect_input(
            query=query,
            image_b64=image_b64,
            image_mime=image_mime,
            file_mime=file_mime,
        )
        logger.info(
            f"Phase 1 — Input: type={inspection.input_type.value}, "
            f"multimodal={inspection.multimodal_required}, "
            f"urls={len(inspection.detected_urls)}"
        )

        # ── PHASE 2: Capability Check ───────────────────────
        capability = phase2_capability_check(inspection)
        logger.info(
            f"Phase 2 — Capabilities: {capability.required_capabilities}, "
            f"preferred={capability.preferred_models}, "
            f"fallback={capability.fallback_models}"
        )

        # ── PHASE 3: Model Availability Audit ────────────────
        audit_entries, disabled_reports = phase3_model_availability_audit()
        for dr in disabled_reports:
            logger.warning(f"Phase 3 — {dr.reason}")

        # ── PHASE 4: Auto Recovery ───────────────────────────
        capability = phase4_auto_recovery(inspection, capability)
        logger.info(
            f"Phase 4 — Post-recovery preferred: {capability.preferred_models}"
        )

        # ── PHASE 5: Pipeline Construction ────────────────────
        pipeline = phase5_build_pipeline(inspection, capability)
        logger.info(
            f"Phase 5 — Pipeline: analysis={pipeline.analysis_model}, "
            f"critique={pipeline.critique_models}, "
            f"synthesis={pipeline.synthesis_model}, "
            f"verification={pipeline.verification_model} "
            f"(total={pipeline.model_count})"
        )

        # ── PHASE 6: Integration Check ───────────────────────
        subsystems = await phase6_integration_check(
            redis_client=self._redis,
            cognitive_rag=self._rag,
        )
        logger.info(
            f"Phase 6 — Subsystems: registry={subsystems.model_registry_loaded}, "
            f"redis={subsystems.redis_active}, "
            f"retrieval={subsystems.retrieval_active}, "
            f"graphrag={subsystems.graphrag_available}"
        )

        # Build audit report
        report = SystemAuditReport(
            input_type=inspection.input_type,
            required_capabilities=capability.required_capabilities,
            multimodal_required=inspection.multimodal_required,
            selected_models=pipeline.all_models,
            disabled_models=disabled_reports,
            fallback_models=capability.fallback_models,
            pipeline=pipeline,
            subsystems=subsystems,
        )

        # ── PHASE 7: Execution ───────────────────────────────
        final_response = None

        if execute and self._engine:
            if pipeline.model_count < MINIMUM_PIPELINE_MODELS:
                report.execution_status = AuditStatus.FAILURE
                report.execution_error = (
                    f"Insufficient models for pipeline: "
                    f"{pipeline.model_count} < {MINIMUM_PIPELINE_MODELS}"
                )
                logger.error(report.execution_error)
            else:
                try:
                    logger.info("Phase 7 — Executing model pipeline")

                    # Determine effective image args:
                    # Only pass images if analysis model supports vision
                    effective_image_b64 = None
                    effective_image_mime = None
                    if inspection.multimodal_required and pipeline.analysis_model:
                        analysis_spec = COGNITIVE_MODEL_REGISTRY.get(pipeline.analysis_model)
                        if analysis_spec and analysis_spec.supports_vision:
                            effective_image_b64 = image_b64
                            effective_image_mime = image_mime
                        else:
                            logger.warning(
                                f"Analysis model {pipeline.analysis_model} does not support vision. "
                                f"Image will be forwarded to vision-capable critique models only."
                            )
                            # Still pass images — the gateway's _build_messages()
                            # handles per-model vision filtering
                            effective_image_b64 = image_b64
                            effective_image_mime = image_mime

                    ensemble_response = await self._engine.process(
                        query=query,
                        chat_id=chat_id,
                        rounds=max(rounds, 3),
                        history=history,
                        image_b64=effective_image_b64,
                        image_mime=effective_image_mime,
                    )

                    report.execution_status = AuditStatus.SUCCESS
                    final_response = ensemble_response

                except Exception as exc:
                    report.execution_status = AuditStatus.FAILURE
                    report.execution_error = str(exc)
                    logger.error(f"Phase 7 — Execution failed: {exc}")
        elif not execute:
            report.execution_status = AuditStatus.SUCCESS
            logger.info("Phase 7 — Audit-only mode (no execution)")
        else:
            report.execution_status = AuditStatus.FAILURE
            report.execution_error = "CognitiveCoreEngine not available"

        # ── PHASE 8: Output Assembly ─────────────────────────
        report.audit_latency_ms = (time.monotonic() - start) * 1000

        result = report.to_dict()
        result["FINAL_RESPONSE"] = None

        if final_response is not None:
            try:
                payload = final_response.to_frontend_payload()
                result["FINAL_RESPONSE"] = {
                    "formatted_output": final_response.formatted_output,
                    "confidence": final_response.confidence.final_confidence,
                    "models_executed": final_response.models_executed,
                    "models_succeeded": final_response.models_succeeded,
                    "debate_rounds": final_response.debate_result.total_rounds,
                    "payload": payload,
                }
            except Exception as e:
                logger.warning(f"Phase 8 — Response serialisation failed: {e}")
                result["FINAL_RESPONSE"] = {
                    "formatted_output": str(final_response),
                    "error": str(e),
                }

        logger.info(
            f"Audit complete: status={report.execution_status.value}, "
            f"latency={report.audit_latency_ms:.1f}ms, "
            f"models={pipeline.model_count}"
        )

        return result


# ============================================================
# Convenience: standalone audit (no execution)
# ============================================================

async def audit_request(
    query: str,
    image_b64: Optional[str] = None,
    image_mime: Optional[str] = None,
    file_mime: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quick audit without execution.  Useful for diagnostics.

    Returns the full SYSTEM_AUDIT and MODEL_PIPELINE without
    invoking any models.
    """
    auditor = MultimodalAuditor()
    return await auditor.audit_and_route(
        query=query,
        image_b64=image_b64,
        image_mime=image_mime,
        file_mime=file_mime,
        execute=False,
    )
