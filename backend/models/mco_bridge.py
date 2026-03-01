"""
MCO Model Bridge — Unified model invocation adapter (v5.1)

Drop-in replacement for CloudModelClient that routes ALL model calls
through the canonical CognitiveModelGateway.

This eliminates dual routing:
  OLD: OmegaKernel → CloudModelClient → direct aiohttp → provider
  NEW: OmegaKernel → MCOModelBridge → CognitiveModelGateway → provider

The bridge preserves the call_groq / call_llama70b / call_qwenvl interface
so that DebateOrchestrator, AggregationEngine, and other consumers
continue to work without modification to their core logic.

v5.1: Also supports dynamic invocation of ANY registered model via
      call_model(legacy_id, prompt, system_role) and get_enabled_model_ids().

All calls flow through one gateway, one registry, one validation layer.
"""

import logging
from typing import Optional, List, Dict, Any

from metacognitive.cognitive_gateway import CognitiveModelGateway, COGNITIVE_MODEL_REGISTRY
from metacognitive.schemas import CognitiveGatewayInput

logger = logging.getLogger("MCOModelBridge")

# ── Registry key mapping ─────────────────────────────────────
# Maps legacy caller IDs to canonical COGNITIVE_MODEL_REGISTRY keys
LEGACY_TO_REGISTRY: Dict[str, str] = {
    "groq": "groq-small",           # LLaMA 3.1 8B (fast)
    "llama70b": "llama-3.3",        # LLaMA 3.3 70B (primary reasoning)
    "qwen": "qwen-vl-2.5",         # Qwen 2.5 7B (methodical)
    "qwen3-coder": "qwen3-coder",  # Qwen3 235B A22B
    "qwen3-vl": "qwen3-vl",        # Qwen 2.5 VL 32B
    "nemotron": "nemotron-nano",    # Nemotron 70B Instruct
    "kimi": "kimi-2.5",            # Kimi 2.5
}

# Reverse map: registry key → legacy ID
REGISTRY_TO_LEGACY: Dict[str, str] = {v: k for k, v in LEGACY_TO_REGISTRY.items()}


class MCOModelBridge:
    """
    Bridge adapter: presents CloudModelClient interface,
    routes through CognitiveModelGateway.

    Supports:
    1. Legacy interface: call_groq(), call_llama70b(), call_qwenvl()
    2. Dynamic interface: call_model(legacy_id, prompt, system_role)
    3. Registry queries: get_enabled_model_ids(), get_enabled_models_info()

    Usage:
        bridge = MCOModelBridge(gateway)
        text = await bridge.call_groq("What is X?", "You are an analyst.")
        text = await bridge.call_model("nemotron", "Analyze this.", "Be rigorous.")
        ids = bridge.get_enabled_model_ids()  # ["groq", "llama70b", ..., "kimi"]
    """

    def __init__(self, gateway: CognitiveModelGateway):
        self.gateway = gateway

    def _resolve_registry_key(self, legacy_or_registry_id: str) -> Optional[str]:
        """
        Resolve a legacy ID or registry key to a canonical registry key.
        Accepts: "groq", "groq-small", "llama70b", "llama-3.3", etc.
        """
        # Direct registry key
        if legacy_or_registry_id in COGNITIVE_MODEL_REGISTRY:
            return legacy_or_registry_id
        # Legacy ID → registry key
        if legacy_or_registry_id in LEGACY_TO_REGISTRY:
            return LEGACY_TO_REGISTRY[legacy_or_registry_id]
        return None

    ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

    async def _invoke(
        self,
        registry_key: str,
        prompt: str,
        system_role: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        image_b64: Optional[str] = None,
        image_mime: Optional[str] = None,
    ) -> str:
        """
        Core invocation: route through CognitiveModelGateway.invoke_model().

        Strict validation:
        - Model must exist and be enabled
        - API key must be present (enforced by gateway)
        - Vision-capable check: rejects images for non-vision models
        - MIME validation: only allows jpeg, png, gif, webp

        Returns raw text on success, or error string matching the legacy
        format that consumers already handle (e.g. "Groq Error ...").
        """
        spec = COGNITIVE_MODEL_REGISTRY.get(registry_key)
        if not spec:
            return f"Error: Model '{registry_key}' not found in registry"
        if not spec.enabled:
            return f"Error: Model '{registry_key}' is disabled (missing API key or explicitly disabled)"

        # Strict multimodal validation
        if image_b64:
            if not spec.supports_vision:
                return (
                    f"Error: Model '{spec.name}' does not support vision/image input. "
                    f"Use a vision-capable model (e.g., qwen3-vl, qwen-vl-2.5)."
                )
            if image_mime and image_mime not in self.ALLOWED_IMAGE_MIMES:
                return (
                    f"Error: Invalid image MIME type '{image_mime}'. "
                    f"Allowed types: {', '.join(sorted(self.ALLOWED_IMAGE_MIMES))}"
                )
            if not image_mime:
                return "Error: Image provided without MIME type. Include image_mime (e.g., 'image/png')."

        try:
            gw_input = CognitiveGatewayInput(
                user_query=prompt,
                stabilized_context={
                    "system_role": system_role,
                },
                knowledge_bundle=[],
                session_summary={},
                image_b64=image_b64,
                image_mime=image_mime,
            )

            result = await self.gateway.invoke_model(registry_key, gw_input)

            if result.success and result.raw_output and result.raw_output.strip():
                return result.raw_output
            else:
                error_msg = result.error or "Empty response"
                logger.warning(f"MCOModelBridge: {registry_key} failed — {error_msg}")
                return f"{spec.name} Error: {error_msg}"
        except Exception as e:
            logger.error(f"MCOModelBridge: {registry_key} invocation crashed — {e}")
            return f"Error: Model invocation failed for '{registry_key}': {str(e)}"

    # ── Dynamic Model Interface ───────────────────────────────

    async def call_model(
        self,
        model_id: str,
        prompt: str,
        system_role: str = "You are a rigorous analytical reasoning assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        image_b64: Optional[str] = None,
        image_mime: Optional[str] = None,
    ) -> str:
        """
        Invoke ANY registered model by its legacy ID or registry key.
        This is the universal entry point for dynamic model invocation.
        Supports optional image_b64 for vision-capable models.
        """
        registry_key = self._resolve_registry_key(model_id)
        if not registry_key:
            return f"Model '{model_id}' not found in registry"
        return await self._invoke(registry_key, prompt, system_role, temperature, max_tokens, image_b64, image_mime)

    def get_enabled_model_ids(self) -> List[str]:
        """
        Return legacy IDs of all enabled models.
        Used by engines that iterate models dynamically.
        """
        ids = []
        for key, spec in COGNITIVE_MODEL_REGISTRY.items():
            if spec.enabled and spec.active:
                legacy_id = REGISTRY_TO_LEGACY.get(key, key)
                ids.append(legacy_id)
        return ids

    def get_enabled_models_info(self) -> List[Dict[str, Any]]:
        """
        Return info dicts for all enabled models.
        Includes id (legacy_id alias), legacy_id, registry_key, name, provider, role,
        supports_vision, supports_debate.
        """
        models = []
        for key, spec in COGNITIVE_MODEL_REGISTRY.items():
            if spec.enabled and spec.active:
                legacy_id = REGISTRY_TO_LEGACY.get(key, key)
                role_val = spec.role.value if hasattr(spec.role, 'value') else str(spec.role)
                models.append({
                    "id": legacy_id,               # canonical ID used by orchestrator
                    "legacy_id": legacy_id,
                    "registry_key": key,
                    "name": spec.name,
                    "provider": spec.provider,
                    "role": role_val,
                    "supports_vision": spec.supports_vision,
                    "supports_debate": True,        # all enabled models participate in debate
                })
        return models

    # ── CloudModelClient-compatible interface ─────────────────

    async def call_groq(
        self, prompt: str, system_role: str = "You are a fast, concise analytical assistant."
    ) -> str:
        """Route legacy Groq calls through gateway → groq-small."""
        return await self._invoke(
            LEGACY_TO_REGISTRY["groq"], prompt, system_role
        )

    async def call_llama70b(
        self, prompt: str,
        system_role: str = "You are a rigorous analytical reasoning assistant.",
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> str:
        """Route legacy Llama 3.3 70B calls through gateway → llama-3.3."""
        return await self._invoke(
            LEGACY_TO_REGISTRY["llama70b"], prompt, system_role,
            temperature=temperature, max_tokens=max_tokens,
        )

    async def call_qwenvl(
        self, prompt: str,
        system_role: str = "You are a careful, analytical assistant.",
    ) -> str:
        """Route legacy Qwen calls through gateway → qwen-vl-2.5."""
        return await self._invoke(
            LEGACY_TO_REGISTRY["qwen"], prompt, system_role
        )
