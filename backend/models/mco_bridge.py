"""
MCO Model Bridge — Unified model invocation adapter

Drop-in replacement for CloudModelClient that routes ALL model calls
through the canonical CognitiveModelGateway.

This eliminates dual routing:
  OLD: OmegaKernel → CloudModelClient → direct aiohttp → provider
  NEW: OmegaKernel → MCOModelBridge → CognitiveModelGateway → provider

The bridge preserves the call_groq / call_llama70b / call_qwenvl interface
so that DebateOrchestrator, AggregationEngine, and other consumers
continue to work without modification to their core logic.

All calls flow through one gateway, one registry, one validation layer.
"""

import logging
from typing import Optional

from metacognitive.cognitive_gateway import CognitiveModelGateway, COGNITIVE_MODEL_REGISTRY
from metacognitive.schemas import CognitiveGatewayInput

logger = logging.getLogger("MCOModelBridge")

# ── Registry key mapping ─────────────────────────────────────
# Maps legacy caller IDs to canonical COGNITIVE_MODEL_REGISTRY keys
LEGACY_TO_REGISTRY = {
    "groq": "groq-small",       # LLaMA 3.1 8B (fast)
    "llama70b": "llama-3.3",    # LLaMA 3.3 70B (primary reasoning)
    "qwen": "qwen-vl-2.5",     # Qwen 2.5 7B (methodical)
}


class MCOModelBridge:
    """
    Bridge adapter: presents CloudModelClient interface,
    routes through CognitiveModelGateway.

    Usage:
        bridge = MCOModelBridge(gateway)
        text = await bridge.call_groq("What is X?", "You are an analyst.")
        text = await bridge.call_llama70b("Explain Y.", "You are rigorous.")
        text = await bridge.call_qwenvl("Describe Z.", "You are careful.")
    """

    def __init__(self, gateway: CognitiveModelGateway):
        self.gateway = gateway

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
