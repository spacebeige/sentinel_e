# Sentinel-E Ensemble Cognitive Engine v6.0 — Execution Contract

## Architecture Summary

Sentinel-E has been redesigned from a mode-based routing system into a fully
ensemble-driven cognitive engine. **Every request** executes the full multi-model
pipeline. There is no single-model fallback. There is no silent bypass.

---

## Hard Failure Constants

| Constant             | Value | Meaning                                    |
|----------------------|-------|--------------------------------------------|
| `MIN_MODELS`         | 3     | Minimum models that must respond            |
| `MIN_DEBATE_ROUNDS`  | 3     | Minimum structured debate rounds            |
| `MIN_ANALYTICS_OUTPUTS` | 2  | Minimum analytics outputs required          |

Violation of any constant raises `EnsembleFailure` — the request fails loudly.

---

## 10-Phase Execution Pipeline

```
                    ┌─────────────────────────────────────────┐
                    │         CognitiveOrchestrator           │
                    │            process(query)               │
                    └─────────────┬───────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │  Phase 1: VALIDATE MODELS   │                             │
    │  Assert ≥3 models available │                             │
    │  via MCOModelBridge         │                             │
    └─────────────────────────────┼─────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │  Phase 2: PARALLEL EXECUTION                              │
    │  asyncio.gather() all models simultaneously               │
    │  Each gets: system_prompt + structured_output_schema      │
    │  Timeout: graceful per-model failure (not hard fail)      │
    └─────────────────────────────┼─────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │  Phase 3: PARSE STRUCTURED OUTPUTS                        │
    │  Extract: position, reasoning, assumptions,               │
    │           vulnerabilities, confidence, stance_vector      │
    │  Assert ≥3 valid outputs (hard fail otherwise)            │
    └─────────────────────────────┼─────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │  Phase 4: STRUCTURED DEBATE (≥3 rounds)                   │
    │  Round 1: Independent positions                           │
    │  Round 2+: Full transcript injection → rebuttals          │
    │  Each round produces per-model structured output          │
    └─────────────────────────────┼─────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │  Phase 5: AGREEMENT MATRIX                                │
    │  Pairwise similarity: trigram Jaccard + stance cosine      │
    │  Cluster detection (threshold = 0.6)                      │
    │  Output: N×N matrix + cluster groups                      │
    └─────────────────────────────┼─────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │  Phase 6: ENSEMBLE METRICS                                │
    │  disagreement_entropy — Shannon entropy of agreements     │
    │  contradiction_density — fraction of low-agreement pairs  │
    │  stability_index — round-over-round variance              │
    │  consensus_velocity — convergence rate                    │
    │  fragility_score — sensitivity to model removal           │
    └─────────────────────────────┼─────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │  Phase 7: CONFIDENCE CALIBRATION                          │
    │  NOT from model self-confidence                           │
    │  Sources: agreement mean, entropy, stability,             │
    │           debate convergence, contradiction density        │
    │  Method: weighted harmonic combination                    │
    └─────────────────────────────┼─────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │  Phase 8: TACTICAL MAP                                    │
    │  Cross-model finding extraction                           │
    │  Evidence model vs. dissenting model tagging              │
    │  Confidence per finding                                   │
    │  Assert non-empty (hard fail if empty)                    │
    └─────────────────────────────┼─────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │  Phase 9: SYNTHESIS                                       │
    │  Majority-weighted consensus generation                   │
    │  Disagreement surfacing (not suppression)                 │
    │  Full reasoning chain inclusion                           │
    └─────────────────────────────┼─────────────────────────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │  Phase 10: SESSION UPDATE                                 │
    │  EnsembleSessionEngine.update(response)                   │
    │  Tracks: message_count, confidence_history,               │
    │          topic_clusters, boundary_hits, volatility         │
    └─────────────────────────────┼─────────────────────────────┘
                                  │
                    ┌─────────────┴───────────────────────────┐
                    │   EnsembleResponse.to_frontend_payload() │
                    │   → Single structured JSON payload       │
                    └─────────────────────────────────────────┘
```

---

## Request Flow (Backend)

```
Client POST /run/ensemble
      │
      ▼
  main.py: run_sentinel()
      │
      ├── Input validation (max length)
      ├── Prompt firewall (analyze + sanitize)
      ├── Chat resolution (create/retrieve)
      ├── Session & memory (get/create kernel + memory)
      ├── Conversation history (last N messages)
      ├── Frontend context injection (follow-up resolution)
      ├── Memory context injection (build_prompt_context)
      ├── Cognitive RAG (external evidence retrieval)
      ├── Observability tracing (spans, cache check)
      ├── Cost governance (budget check)
      ├── Token optimization (compression, deduplication)
      │
      ▼
  ══════════════════════════════════════════════
  ENSEMBLE PATH (cognitive_orchestrator_engine)
  ══════════════════════════════════════════════
      │
      ├── cognitive_orchestrator_engine.process(query, chat_id, rounds)
      │     → 10-phase pipeline (see above)
      │     → Returns: EnsembleResponse
      │
      ├── RAG citation injection
      ├── omega_metadata construction (version 6.0.0-ensemble)
      ├── Database persistence
      ├── Memory update
      ├── Redis session cache
      ├── Response cache storage
      ├── Cost governor usage recording
      ├── Observability finalization
      │
      ▼
  Return response_payload {
    chat_id, mode: "ensemble", sub_mode: "full_debate",
    formatted_output, confidence,
    omega_metadata: {
      version: "6.0.0-ensemble",
      ensemble_metrics, debate_result, agreement_matrix,
      tactical_map, confidence_graph, model_stances,
      session_analytics, reasoning_trace, boundary_result,
      rag_result (if applicable)
    }
  }
```

---

## Frontend Rendering Pipeline

```
ChatEngine.js
  └── POST /run/ensemble (FormData)
        │
        ▼
  FigmaChatShell.js
  └── Extract omegaMetadata from response
  └── Check: meta.ensemble_metrics? → hasStructuredData = true
  └── resolvedSubMode = 'ensemble'
        │
        ▼
  StructuredOutput.js
  └── resolveRenderMode(result) → mode: 'ensemble'
  └── effectiveMode = 'ensemble'
        │
        ▼
  EnsembleView.js
  └── Renders:
      ├── Ensemble Metrics Dashboard (5 metrics)
      ├── Confidence Calibration (bar + components + boundary)
      ├── Tactical Map (findings + evidence/dissenting models)
      ├── Debate Rounds (collapsible, per-model positions)
      │   └── ModelPositionCard (position, reasoning, assumptions,
      │       vulnerabilities, stance vector — all expandable)
      ├── Agreement Matrix Heatmap (N×N + clusters)
      └── Session Intelligence (message count, topics, avg confidence)
```

---

## File Inventory

### New Backend Files (6):
| File | Purpose | Key Export |
|------|---------|------------|
| `backend/core/ensemble_schemas.py` | All Pydantic data contracts | `EnsembleResponse`, `StructuredModelOutput`, `EnsembleMetrics`, etc. |
| `backend/core/agreement_matrix.py` | Pairwise model similarity | `AgreementMatrixEngine.compute()` |
| `backend/core/confidence_calibrator.py` | Entropy-weighted confidence | `ConfidenceCalibrator.calibrate()` |
| `backend/core/structured_debate_engine.py` | 3+ round structured debate | `StructuredDebateEngine.run_debate()` |
| `backend/core/ensemble_session.py` | Session intelligence tracking | `EnsembleSessionEngine.update()` |
| `backend/core/cognitive_orchestrator.py` | **Single entry point** | `CognitiveOrchestrator.process()` |

### Modified Backend Files (1):
| File | Changes |
|------|---------|
| `backend/main.py` | Import + global + lifespan init for `CognitiveOrchestrator`. Ensemble path before legacy kernel. New `/run/ensemble` FormData endpoint. |

### New Frontend Files (1):
| File | Purpose |
|------|---------|
| `frontend/src/components/structured/EnsembleView.js` | Full ensemble visualization (metrics, debate rounds, agreement matrix, tactical map, confidence graph, session analytics) |

### Modified Frontend Files (4):
| File | Changes |
|------|---------|
| `frontend/src/engines/modeController.js` | `resolveRenderMode()` detects `ensemble` mode. `getDefaultPipelineSteps()` has 10-step ensemble pipeline. |
| `frontend/src/components/structured/StructuredOutput.js` | Added `ensemble` case → renders `EnsembleView`. |
| `frontend/src/figma_shell/FigmaChatShell.js` | `hasStructuredData` includes `ensemble_metrics`. Sub-mode resolved to `'ensemble'` when ensemble detected. |
| `frontend/src/components/ChatEngine.js` | All requests route to `/run/ensemble`. Kill switch still goes to dedicated endpoint. |

---

## Data Contract: omega_metadata (v6.0.0-ensemble)

```json
{
  "version": "6.0.0-ensemble",
  "mode": "ensemble",
  "sub_mode": "full_debate",
  "confidence": 0.73,
  "ensemble_metrics": {
    "disagreement_entropy": 0.42,
    "contradiction_density": 0.15,
    "stability_index": 0.81,
    "consensus_velocity": 0.23,
    "fragility_score": 0.31
  },
  "debate_result": [
    {
      "round": 1,
      "model_outputs": [
        {
          "model_id": "llama-3.3",
          "position": "...",
          "reasoning": "...",
          "assumptions": ["..."],
          "vulnerabilities": ["..."],
          "confidence": 0.8,
          "stance_vector": {"certainty": 0.8, "scope": 0.6, "novelty": 0.3}
        }
      ]
    }
  ],
  "agreement_matrix": {
    "matrix": [[1.0, 0.7, 0.4], [0.7, 1.0, 0.5], [0.4, 0.5, 1.0]],
    "model_ids": ["llama-3.3", "qwen3-coder", "nemotron-nano"],
    "clusters": [["llama-3.3", "qwen3-coder"]]
  },
  "tactical_map": [
    {
      "finding": "Key consensus finding text",
      "evidence_models": ["llama-3.3", "qwen3-coder"],
      "dissenting_models": ["nemotron-nano"],
      "confidence": 0.85,
      "category": "consensus"
    }
  ],
  "confidence_graph": {
    "final_confidence": 0.73,
    "components": {
      "agreement_mean": 0.65,
      "entropy_penalty": 0.12,
      "stability_bonus": 0.08,
      "debate_convergence": 0.15
    },
    "calibration_method": "entropy_weighted_harmonic"
  },
  "model_stances": {
    "llama-3.3": {"position": "...", "confidence": 0.8, "stance_vector": {...}},
    "qwen3-coder": {"position": "...", "confidence": 0.75, "stance_vector": {...}}
  },
  "session_analytics": {
    "message_count": 5,
    "avg_confidence": 0.71,
    "topic_clusters": ["machine learning", "optimization"]
  },
  "reasoning_trace": {
    "engine": "CognitiveOrchestrator",
    "pipeline": "ensemble_v6",
    "models_used": 5,
    "debate_rounds": 3
  },
  "boundary_result": {
    "risk_level": "LOW",
    "severity_score": 27,
    "explanation": "Ensemble confidence from 5 models, 3 debate rounds"
  }
}
```

---

## Enforcement Rules

1. **No Single-Model Fallback**: If <3 models respond → `EnsembleFailure(INSUFFICIENT_MODELS)`
2. **No Silent Bypass**: Every request goes through all 10 phases
3. **Minimum 3 Debate Rounds**: `max(rounds, 3)` enforced at endpoint + orchestrator
4. **Structured Output Required**: Each model must produce position/reasoning/assumptions/vulnerabilities/confidence/stance_vector
5. **Tactical Map Non-Empty**: Hard fail if zero findings extracted
6. **Confidence From Metrics**: Never from model self-reported confidence
7. **Full Metadata Exposure**: Frontend receives complete ensemble_metrics, debate_rounds, agreement_matrix, tactical_map, confidence_graph
8. **Legacy Fallback**: If `CognitiveOrchestrator` init fails → falls back to legacy `OmegaCognitiveKernel` (logged as error)
