"""
============================================================
Query Router — Sentinel-E v6.1
============================================================
Centralized routing controller that decides execution path
based on query characteristics, selected model, and mode.

Execution Paths:
  1. SINGLE_MODEL   — Direct model invocation (no debate)
  2. STANDARD       — Single best-model response via MCO
  3. DEBATE_EXTERNAL — Multi-model debate (external APIs only)
  4. DEBATE_INTERNAL — Multi-model debate (local models only)
  5. DEBATE_HYBRID   — Multi-model debate (mixed internal+external)

Query Complexity Detection:
  - Trivial queries (< 3 words, greetings) → STANDARD (skip debate)
  - Analytical queries → DEBATE path
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger("QueryRouter")


class ExecutionPath(str, Enum):
    """Possible execution paths for a query."""
    SINGLE_MODEL = "single_model"
    STANDARD = "standard"
    DEBATE_EXTERNAL = "debate_external"
    DEBATE_INTERNAL = "debate_internal"
    DEBATE_HYBRID = "debate_hybrid"


@dataclass
class RoutingDecision:
    """Result of the routing analysis."""
    path: ExecutionPath
    reason: str
    selected_model: Optional[str] = None
    skip_debate: bool = False
    model_filter: Optional[str] = None  # "external", "internal", or None (all)
    query_complexity: str = "unknown"  # "trivial", "simple", "analytical"


# ── Greeting / Trivial Patterns ──────────────────────────────
_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|howdy|yo|sup|greetings|good\s+(morning|afternoon|evening|night)|"
    r"what'?s?\s+up|how\s+are\s+you|thanks?|thank\s+you|bye|goodbye|ok|okay|sure|yes|no|"
    r"test|ping|help)\s*[!?.]*\s*$",
    re.IGNORECASE,
)

# Minimum word count for debate consideration
MIN_WORDS_FOR_DEBATE = 3


def classify_query_complexity(query: str) -> str:
    """
    Classify a query's complexity level.

    Returns:
        "trivial"    — Greetings, single words, < 3 words
        "simple"     — Short factual questions
        "analytical" — Requires reasoning, analysis, or comparison
    """
    stripped = query.strip()

    if not stripped:
        return "trivial"

    # Check for greeting patterns
    if _GREETING_PATTERNS.match(stripped):
        return "trivial"

    word_count = len(stripped.split())

    if word_count < MIN_WORDS_FOR_DEBATE:
        return "trivial"

    # Simple heuristic: analytical queries tend to contain certain markers
    analytical_markers = [
        "analyze", "compare", "evaluate", "explain why", "debate",
        "what are the pros", "what are the cons", "trade-off",
        "implications", "consequences", "versus", " vs ",
        "should i", "which is better", "difference between",
        "how does", "why does", "what causes", "impact of",
        "advantages", "disadvantages", "risks of", "benefits of",
        "critically", "in depth", "comprehensive",
    ]
    lower = stripped.lower()
    if any(marker in lower for marker in analytical_markers):
        return "analytical"

    # Medium-length queries default to simple
    if word_count < 10:
        return "simple"

    # Longer queries are treated as analytical
    return "analytical"


def route_query(
    query: str,
    mode: str = "standard",
    sub_mode: Optional[str] = None,
    selected_model: Optional[str] = None,
    model_registry: Optional[dict] = None,
) -> RoutingDecision:
    """
    Determine the execution path for a query.

    Args:
        query: The user's input text
        mode: Operating mode (standard, experimental, research, etc.)
        sub_mode: Sub-mode (debate, glass, evidence, stress)
        selected_model: If user explicitly selected a specific model
        model_registry: The COGNITIVE_MODEL_REGISTRY dict

    Returns:
        RoutingDecision with path, reason, and configuration.
    """
    complexity = classify_query_complexity(query)

    # ── Path 1: Single Model Selection ───────────────────────
    if selected_model:
        return RoutingDecision(
            path=ExecutionPath.SINGLE_MODEL,
            reason=f"User selected specific model: {selected_model}",
            selected_model=selected_model,
            skip_debate=True,
            query_complexity=complexity,
        )

    # ── Path 2: Trivial Query → Standard (skip debate) ──────
    if complexity == "trivial":
        return RoutingDecision(
            path=ExecutionPath.STANDARD,
            reason=f"Trivial query detected ('{query[:30]}...' — skipping debate)",
            skip_debate=True,
            query_complexity=complexity,
        )

    # ── Path 3: Explicit Debate Request ──────────────────────
    if sub_mode == "debate" or mode == "experimental":
        # Determine model filter based on available models
        model_filter = _determine_model_filter(model_registry)

        if model_filter == "external":
            path = ExecutionPath.DEBATE_EXTERNAL
        elif model_filter == "internal":
            path = ExecutionPath.DEBATE_INTERNAL
        else:
            path = ExecutionPath.DEBATE_HYBRID

        return RoutingDecision(
            path=path,
            reason=f"Debate mode requested (mode={mode}, sub_mode={sub_mode})",
            skip_debate=False,
            model_filter=model_filter,
            query_complexity=complexity,
        )

    # ── Path 4: Analytical Query → Debate ────────────────────
    if complexity == "analytical" and mode in ("research", "experimental"):
        model_filter = _determine_model_filter(model_registry)
        return RoutingDecision(
            path=ExecutionPath.DEBATE_EXTERNAL if model_filter == "external" else ExecutionPath.DEBATE_HYBRID,
            reason=f"Analytical query in {mode} mode — debate recommended",
            skip_debate=False,
            model_filter=model_filter,
            query_complexity=complexity,
        )

    # ── Path 5: Default → Standard ───────────────────────────
    return RoutingDecision(
        path=ExecutionPath.STANDARD,
        reason=f"Standard routing (complexity={complexity}, mode={mode})",
        skip_debate=complexity != "analytical",
        query_complexity=complexity,
    )


def _determine_model_filter(model_registry: Optional[dict]) -> Optional[str]:
    """
    Determine whether to use external, internal, or hybrid models.

    Returns "external", "internal", or None (hybrid).
    """
    if not model_registry:
        return "external"  # Default to external when registry unavailable

    has_external = False
    has_internal = False

    for spec in model_registry.values():
        if not getattr(spec, "enabled", False):
            continue
        model_type = getattr(spec, "model_type", "external")
        if model_type == "internal":
            has_internal = True
        else:
            has_external = True

    if has_external and has_internal:
        return None  # hybrid
    if has_internal:
        return "internal"
    return "external"
