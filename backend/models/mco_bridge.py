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
    "qwen3-coder": "qwen3-coder",  # Qwen3 Coder 480B A35B
    "qwen3-vl": "qwen3-vl",        # Qwen3 VL 30B A3B
    "nemotron": "nemotron-nano",    # Nemotron 3 Nano 30B A3B
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

    async def _invoke(
        self,
        registry_key: str,
        prompt: str,
        system_role: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Core invocation: route through CognitiveModelGateway.invoke_model().

        Returns raw text on success, or error string matching the legacy
        format that consumers already handle (e.g. "Groq Error ...").
        """
        spec = COGNITIVE_MODEL_REGISTRY.get(registry_key)
        if not spec or not spec.enabled:
            return f"Model '{registry_key}' not available (disabled or missing key)"

        gw_input = CognitiveGatewayInput(
            user_query=prompt,
            stabilized_context={
                "system_role": system_role,
            },
            knowledge_bundle=[],
            session_summary={},
        )

        result = await self.gateway.invoke_model(registry_key, gw_input)

        if result.success and result.raw_output and result.raw_output.strip():
            return result.raw_output
        else:
            error_msg = result.error or "Empty response"
            logger.warning(f"MCOModelBridge: {registry_key} failed — {error_msg}")
            return f"{spec.name} Error: {error_msg}"

    # ── Dynamic Model Interface ───────────────────────────────

    async def call_model(
        self,
        model_id: str,
        prompt: str,
        system_role: str = "You are a rigorous analytical reasoning assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Invoke ANY registered model by its legacy ID or registry key.
        This is the universal entry point for dynamic model invocation.
        """
        registry_key = self._resolve_registry_key(model_id)
        if not registry_key:
            return f"Model '{model_id}' not found in registry"
        return await self._invoke(registry_key, prompt, system_role, temperature, max_tokens)

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
        Includes legacy_id, registry_key, name, provider, role.
        """
        models = []
        for key, spec in COGNITIVE_MODEL_REGISTRY.items():
            if spec.enabled and spec.active:
                models.append({
                    "legacy_id": REGISTRY_TO_LEGACY.get(key, key),
                    "registry_key": key,
                    "name": spec.name,
                    "provider": spec.provider,
                    "role": spec.role.value if hasattr(spec.role, 'value') else str(spec.role),
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
