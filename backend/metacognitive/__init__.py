"""
============================================================
Distributed Meta-Cognitive Orchestrator
============================================================
A distributed cognitive research architecture with three
physically and logically separated API contracts:

  API 1 — Cognitive Model Gateway   (pure reasoning)
  API 2 — Knowledge & Retrieval     (live data acquisition)
  API 3 — Session & Persistence     (structured state)

Orchestrator coordinates all three, enforces stability,
grounding, and arbitration without modifying model weights.
============================================================
"""

from metacognitive.orchestrator import MetaCognitiveOrchestrator
from metacognitive.schemas import OperatingMode

__all__ = ["MetaCognitiveOrchestrator", "OperatingMode"]
