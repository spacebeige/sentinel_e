"""
============================================================
Central Model Registry Abstraction — Sentinel-E v5.1
============================================================
Single source of truth for ALL model capabilities.

Every engine (Debate, Aggregation, BlindAudit, Forensic, Analytics)
MUST iterate this registry dynamically — never hardcode model names.

Each provider defines:
  - id:                      Canonical registry key
  - legacy_id:               Backward-compat key for older engines
  - tier:                    budget | standard | premium
  - supports_debate:         Whether model participates in debate mode
  - supports_research:       Whether model participates in research mode
  - structured_output_capable: Whether model can produce JSON/structured output
  - token_cost_profile:      Cost category for governor integration
  - role:                    Routing hint (code, vision, conceptual, etc.)

Derived from COGNITIVE_MODEL_REGISTRY at runtime — never duplicated.
============================================================
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from metacognitive.cognitive_gateway import COGNITIVE_MODEL_REGISTRY, CognitiveModelSpec
from metacognitive.schemas import ModelRole

logger = logging.getLogger("ModelRegistry")


class ModelTier(str, Enum):
    BUDGET = "budget"
    STANDARD = "standard"
    PREMIUM = "premium"


class CostProfile(str, Enum):
    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ModelCapability:
    """Full capability descriptor for a registered model."""
    id: str                           # Canonical registry key (e.g. "qwen3-coder")
    legacy_id: str                    # Legacy ID (e.g. "qwen3-coder" or "groq")
    name: str                         # Human-readable name
    provider: str                     # groq | gemini | qwen
    role: ModelRole                   # Routing hint
    tier: ModelTier                   # budget | standard | premium
    supports_debate: bool             # Participates in debate rounds
    supports_research: bool           # Participates in research/glass/evidence
    supports_aggregation: bool        # Participates in standard mode aggregation
    structured_output_capable: bool   # Can produce structured JSON output
    cost_profile: CostProfile         # Cost tier for governor
    context_window: int               # Max context window
    max_output_tokens: int            # Max output tokens
    enabled: bool                     # Runtime availability (key present)
    active: bool                      # Structural availability


# ── Legacy ID Mapping ────────────────────────────────────────
# Maps canonical registry keys → legacy IDs used by older engines
_LEGACY_ID_MAP: Dict[str, str] = {
    "llama33-70b": "llama31",
    "llama31-8b": "llama31-fast",
}

# ── Tier Assignment ──────────────────────────────────────────
_TIER_MAP: Dict[str, ModelTier] = {
    # Analysis
    "llama33-70b": ModelTier.STANDARD,
    # Critique
    "mixtral-8x7b": ModelTier.BUDGET,
    "gemma-7b": ModelTier.BUDGET,
    "qwen-2.5-vl": ModelTier.BUDGET,
    # Synthesis + Verification
    "gemini-flash": ModelTier.STANDARD,
    "llama31-8b": ModelTier.BUDGET,
}

# ── Cost Profile ─────────────────────────────────────────────
_COST_MAP: Dict[str, CostProfile] = {
    # Analysis
    "llama33-70b": CostProfile.FREE,
    # Critique
    "mixtral-8x7b": CostProfile.FREE,
    "gemma-7b": CostProfile.FREE,
    "qwen-2.5-vl": CostProfile.FREE,
    # Synthesis + Verification
    "gemini-flash": CostProfile.FREE,
    "llama31-8b": CostProfile.FREE,
}


def _build_capability(key: str, spec: CognitiveModelSpec) -> ModelCapability:
    """Build a ModelCapability from a CognitiveModelSpec."""
    return ModelCapability(
        id=key,
        legacy_id=_LEGACY_ID_MAP.get(key, key),
        name=spec.name,
        provider=spec.provider,
        role=spec.role,
        tier=_TIER_MAP.get(key, ModelTier.STANDARD),
        supports_debate=True,          # All models participate in debate
        supports_research=True,        # All models participate in research
        supports_aggregation=True,     # All models participate in aggregation
        structured_output_capable=True,  # All models can attempt structured output
        cost_profile=_COST_MAP.get(key, CostProfile.FREE),
        context_window=spec.context_window,
        max_output_tokens=spec.max_output_tokens,
        enabled=spec.enabled,
        active=spec.active,
    )


# ============================================================
# PUBLIC API
# ============================================================

def get_all_models() -> Dict[str, ModelCapability]:
    """
    Return all registered models with their capabilities.
    Derived from COGNITIVE_MODEL_REGISTRY at call time.
    """
    return {
        key: _build_capability(key, spec)
        for key, spec in COGNITIVE_MODEL_REGISTRY.items()
    }


def get_enabled_models() -> Dict[str, ModelCapability]:
    """Return only models that are enabled and active (API key present)."""
    return {
        key: cap for key, cap in get_all_models().items()
        if cap.enabled and cap.active
    }


def get_debate_models() -> Dict[str, ModelCapability]:
    """Return models eligible for debate mode."""
    return {
        key: cap for key, cap in get_enabled_models().items()
        if cap.supports_debate
    }


def get_aggregation_models() -> Dict[str, ModelCapability]:
    """Return models eligible for standard aggregation."""
    return {
        key: cap for key, cap in get_enabled_models().items()
        if cap.supports_aggregation
    }


def get_research_models() -> Dict[str, ModelCapability]:
    """Return models eligible for research/glass/evidence."""
    return {
        key: cap for key, cap in get_enabled_models().items()
        if cap.supports_research
    }


def get_model_by_legacy_id(legacy_id: str) -> Optional[ModelCapability]:
    """Look up a model by its legacy ID."""
    for key, cap in get_all_models().items():
        if cap.legacy_id == legacy_id:
            return cap
    return None


def get_legacy_id(registry_key: str) -> str:
    """Get the legacy ID for a registry key."""
    return _LEGACY_ID_MAP.get(registry_key, registry_key)


def get_registry_key(legacy_id: str) -> Optional[str]:
    """Reverse lookup: legacy ID → registry key."""
    for key, lid in _LEGACY_ID_MAP.items():
        if lid == legacy_id:
            return key
    # Check if it's already a registry key
    if legacy_id in COGNITIVE_MODEL_REGISTRY:
        return legacy_id
    return None


def get_model_names_list() -> List[str]:
    """Return human-readable names of all enabled models."""
    return [cap.name for cap in get_enabled_models().values()]


def get_model_ids_list() -> List[str]:
    """Return legacy IDs of all enabled models."""
    return [cap.legacy_id for cap in get_enabled_models().values()]


def get_debug_table() -> List[Dict[str, Any]]:
    """
    Return diagnostic table for audit:
    Model | Registered | Enabled | Tier | Debate | Research | Aggregation
    """
    table = []
    for key, cap in get_all_models().items():
        table.append({
            "model": cap.name,
            "registry_key": key,
            "legacy_id": cap.legacy_id,
            "registered": True,
            "enabled": cap.enabled,
            "active": cap.active,
            "provider": cap.provider,
            "role": cap.role.value,
            "tier": cap.tier.value,
            "supports_debate": cap.supports_debate,
            "supports_research": cap.supports_research,
            "supports_aggregation": cap.supports_aggregation,
            "cost_profile": cap.cost_profile.value,
        })
    return table
