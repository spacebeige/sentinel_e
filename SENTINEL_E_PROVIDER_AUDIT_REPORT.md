# Sentinel-E Provider Integration Audit Report

**Date**: 2025-07-17  
**Scope**: Full-stack 8-phase provider integration audit  
**System**: Sentinel-E v5.0 — Enterprise Multi-Model AI Orchestration  
**Status**: ✅ All integration gaps resolved

---

## Executive Summary

Sentinel-E had **7 models** registered in its authoritative `COGNITIVE_MODEL_REGISTRY` (in `metacognitive/cognitive_gateway.py`) but only **3 legacy models** were wired end-to-end through the execution pipeline. Four newer models (Qwen3 Coder, Qwen3 VL, Nemotron Nano, Kimi K2.5) were:

- ✅ Registered in the gateway
- ✅ Have API dispatch logic (`_call_groq`, `_call_openrouter_isolated`)
- ❌ Not mapped in the MCO bridge adapter
- ❌ Not dispatched by any downstream engine
- ❌ Not scored in debate prompts
- ❌ Not included in glass/evidence/stress forensic passes

**Root Cause**: The `MCOModelBridge` adapter only translated 3 legacy IDs (`groq`, `llama70b`, `qwen`) while 4 new models had no legacy mapping. All downstream engines consumed models through this 3-model bottleneck.

---

## 1. Model Registry — Canonical Source of Truth

| Registry Key | Model | Provider | API | Role | Tier |
|:---|:---|:---|:---|:---|:---|
| `groq-small` | LLaMA 3.1 8B Instant | Groq | Native Groq | FAST | free |
| `llama-3.3` | LLaMA 3.3 70B Versatile | Groq | Native Groq | CONCEPTUAL | free |
| `qwen-vl-2.5` | Qwen 2.5 VL 7B | OpenRouter | OpenRouter | GENERAL | free |
| `qwen3-coder` | Qwen3 Coder 480B A35B | OpenRouter | OpenRouter | CODE | free |
| `qwen3-vl` | Qwen3 VL 30B A3B | OpenRouter | OpenRouter | VISION | free |
| `nemotron-nano` | Nemotron 3 Nano 30B v1 | OpenRouter | OpenRouter | BASELINE | free |
| `kimi-2.5` | Kimi K2 (Preview) | OpenRouter | OpenRouter | LONGCTX | free |

**Source**: `backend/metacognitive/cognitive_gateway.py` → `COGNITIVE_MODEL_REGISTRY`

---

## 2. Integration Gap Analysis (Pre-Fix)

### 2.1 Bridge Layer (MCOModelBridge)

| Legacy ID | Registry Key | Pre-Fix Status |
|:---|:---|:---|
| `groq` | `groq-small` | ✅ Mapped |
| `llama70b` | `llama-3.3` | ✅ Mapped |
| `qwen` | `qwen-vl-2.5` | ✅ Mapped |
| `qwen3-coder` | `qwen3-coder` | ❌ Missing |
| `qwen3-vl` | `qwen3-vl` | ❌ Missing |
| `nemotron` | `nemotron-nano` | ❌ Missing |
| `kimi` | `kimi-2.5` | ❌ Missing |

### 2.2 Engine Integration Matrix (Pre-Fix)

| Engine | File | Models Used | Gap |
|:---|:---|:---|:---|
| AggregationEngine | `engines/aggregation_engine.py` | 3 hardcoded | if/elif `call_groq`/`call_llama70b`/`call_qwenvl` |
| BlindAuditEngine | `engines/blind_audit_engine.py` | 3 hardcoded | Fixed model name dict, fixed audit pairs |
| ForensicEvidenceEngine | `engines/forensic_evidence_engine.py` | 3 hardcoded | 3 dispatch patterns (claims, verification, citations) |
| DebateOrchestrator | `core/debate_orchestrator.py` | Dynamic ✅ | BUT judge/analysis prompts hardcoded "Groq, Llama 3.3 70B, Qwen" |
| StressEngine | `engines/stress_engine.py` | Dynamic ✅ | Uses `_model_callers` dict from bridge — OK |
| OmegaKernel | `core/omega_kernel.py` | N/A | Pipeline step labels said "3 Models" |
| DynamicAnalyticsEngine | `engines/dynamic_analytics.py` | Provider-agnostic ✅ | Accepts `List[str]` — no fix needed |
| MetaCognitiveOrchestrator | `metacognitive/orchestrator.py` | Dynamic ✅ | Uses `get_models_for_task()` — OK |
| CostGovernor | `core/cost_governor.py` | All 7 ✅ | Already had entries for all models |

### 2.3 API Gateway (main.py)

| Issue | Location | Pre-Fix |
|:---|:---|:---|
| Analytics ingestion | Lines 587-590 | Looked for `groq_response`, `llama70b_response`, `qwen_response` keys — these keys DON'T EXIST in `AggregationResult.to_dict()` output |

---

## 3. Contract Violations Found

### 3.1 AggregationResult Schema vs Consumer Expectation

**Producer** (`AggregationResult.to_dict()`):
```python
{
    "model_outputs": [
        {"output": "...", "model_id": "groq", "error": null},
        {"output": "...", "model_id": "llama70b", "error": null},
        ...
    ],
    "models_succeeded": 7,
    "models_failed": 0,
    "timestamp": "..."
}
```

**Consumer** (`main.py` analytics ingestion — OLD):
```python
for key in ["groq_response", "llama70b_response", "qwen_response"]:
    if key in agg:
        model_outputs.append(str(agg[key]))
```

**Verdict**: Complete contract mismatch. Analytics was NEVER receiving model outputs. Zero data flowing to confidence computation. This means `analytics.confidence` was always computed from empty inputs.

### 3.2 Debate Prompt Hardcoding

**ANALYSIS_SYSTEM** and **JUDGE_SCORING_SYSTEM** templates contained literal model names ("Groq", "Llama 3.3 70B", "Qwen") which became stale when new debaters were added dynamically by `_build_debate_models()`.

---

## 4. Fixes Applied

### 4.1 New File: Central Model Registry Abstraction

**File**: `backend/core/model_registry.py`

Provides a clean API over `COGNITIVE_MODEL_REGISTRY`:
- `get_all_models()` → List[ModelInfo]
- `get_enabled_models()` → List[ModelInfo] (API key present)
- `get_debate_models()` → List[ModelInfo]
- `get_model_by_id(id)` / `get_model_by_legacy_id(legacy_id)`
- `ModelInfo` dataclass with: id, legacy_id, name, provider, role, tier, supports_debate, supports_research, structured_output_capable, token_cost_profile, enabled

### 4.2 MCOModelBridge — Extended to 7 Models

**File**: `backend/models/mco_bridge.py`

- `LEGACY_TO_REGISTRY` expanded: `groq→groq-small`, `llama70b→llama-3.3`, `qwen→qwen-vl-2.5`, `qwen3-coder→qwen3-coder`, `qwen3-vl→qwen3-vl`, `nemotron→nemotron-nano`, `kimi→kimi-2.5`
- New methods:
  - `call_model(legacy_id, prompt, system_role)` → unified dispatch
  - `get_enabled_model_ids()` → List[str] of legacy IDs with valid API keys
  - `get_enabled_models_info()` → List[dict] with `{id, name, role}` for each enabled model

### 4.3 AggregationEngine — Dynamic Model Dispatch

**File**: `backend/engines/aggregation_engine.py`

- **Before**: if/elif chain calling `call_groq`, `call_llama70b`, `call_qwenvl`
- **After**: `self.client.get_enabled_models_info()` → dynamic task list → `self.client.call_model(model_id, query, system_role)` per model
- Error detection: Provider-agnostic pattern matching (`"error"`, `"api key missing"`, `"exception"`)

### 4.4 BlindAuditEngine — Dynamic Glass Mode

**File**: `backend/engines/blind_audit_engine.py`

- `_get_dynamic_model_names()`: Builds model name map from MCOModelBridge
- `MODEL_NAMES` expanded to 7 entries
- `run_blind_audit()`: Dynamically gets model IDs, creates circular audit pairs `[(0→1), (1→2), ..., (n-1→0)]`
- `_call_model()`: Uses `self.client.call_model()` instead of if/elif

### 4.5 ForensicEvidenceEngine — Dynamic Evidence Mode

**File**: `backend/engines/forensic_evidence_engine.py`

- `_call_model_dynamic(model_id, prompt)`: Generic dispatch
- `_get_model_ids()`: Returns enabled model IDs from bridge
- Phase 1 (Independent Claims): Dynamic model iteration
- Phase 2 (Verification): Dynamic verification groups (each model verified by all others)
- Phase 5 (Citations): Dynamic citation extraction across all models

### 4.6 OmegaKernel — Dynamic Pipeline Labels

**File**: `backend/core/omega_kernel.py`

- Pipeline step labels now show actual model count: `f"Running {aggregation_result.models_succeeded} Models in Parallel"` instead of hardcoded "3"

### 4.7 DebateOrchestrator — Dynamic Prompts

**File**: `backend/core/debate_orchestrator.py`

- `ANALYSIS_SYSTEM` template: `{debater_names}` placeholder instead of literal "Groq, Llama 3.3 70B, Qwen"
- `JUDGE_SCORING_SYSTEM` template: `{debater_names}` and `{scoring_entries}` placeholders
- `_run_analysis()`: Builds `debater_names` from `DEBATE_MODELS` at call time
- `_run_judge_scoring()`: Builds `debater_names` and `scoring_entries` dynamically

### 4.8 main.py — Analytics Ingestion Fix

**File**: `backend/main.py`

- **Before**: Iterated over non-existent keys `groq_response`, `llama70b_response`, `qwen_response`
- **After**: Iterates over `agg.get("model_outputs", [])`, extracts `m.get("output", "")`, skips entries with errors

---

## 5. Data Flow — Post-Fix

```
User Query
    │
    ▼
main.py (FastAPI Gateway)
    │
    ▼
OmegaKernel._run_standard_pipeline()  or  _run_research_pipeline()
    │                                           │
    ▼                                           ▼
AggregationEngine                        DebateOrchestrator (DEBATE)
    │                                    BlindAuditEngine (GLASS)
    │                                    ForensicEvidenceEngine (EVIDENCE)
    │                                    StressEngine (STRESS)
    │
    ▼
MCOModelBridge.call_model(legacy_id)
    │
    ▼
CognitiveModelGateway._dispatch_to_provider()
    │
    ├─ provider="groq" → _call_groq()       [groq-small, llama-3.3]
    └─ provider="qwen"|"nvidia"|"kimi"|"openrouter" → _call_openrouter_isolated()
                                              [qwen-vl-2.5, qwen3-coder, qwen3-vl,
                                               nemotron-nano, kimi-2.5]
```

**Key invariant**: Every model in `COGNITIVE_MODEL_REGISTRY` with a valid API key is now dispatched by every engine mode.

---

## 6. Provider Adapter Layer Design

The system uses a two-tier adapter pattern:

### Tier 1: CognitiveModelGateway (Low-level)
- Owns API keys and HTTP dispatch
- Provider-specific code isolated in `_call_groq()` and `_call_openrouter_isolated()`
- Handles retries, timeouts, error formatting
- Returns raw text responses

### Tier 2: MCOModelBridge (High-level)
- Presents a uniform `call_model(legacy_id, prompt, system_role)` interface
- Maps legacy IDs to registry keys via `LEGACY_TO_REGISTRY`
- Provides discovery: `get_enabled_model_ids()`, `get_enabled_models_info()`
- Backwards-compatible: Still exposes `call_groq()`, `call_llama70b()`, `call_qwenvl()` for any remaining legacy call sites

### Adding a New Model

1. Add entry to `COGNITIVE_MODEL_REGISTRY` in `metacognitive/cognitive_gateway.py`
2. Add legacy mapping in `MCOModelBridge.LEGACY_TO_REGISTRY` in `models/mco_bridge.py`
3. Add cost entry in `CostGovernor.MODEL_COSTS` in `core/cost_governor.py`
4. Set API key in environment (`.env` or deployment config)
5. **No other changes needed** — all engines discover models dynamically

---

## 7. Dynamic Model Registry Design

### Central Abstraction: `backend/core/model_registry.py`

```python
@dataclass
class ModelInfo:
    id: str                         # Registry key (e.g., "groq-small")
    legacy_id: str                  # Bridge key (e.g., "groq")
    name: str                       # Human-readable (e.g., "LLaMA 3.1 8B")
    provider: str                   # "groq" | "qwen" | "nvidia" | "kimi"
    role: str                       # "FAST" | "CONCEPTUAL" | "CODE" | etc.
    tier: str                       # "free" | "paid"
    supports_debate: bool           # Can participate in adversarial debate
    supports_research: bool         # Can be used in research sub-modes
    structured_output_capable: bool # Supports JSON mode
    token_cost_profile: dict        # {"input": float, "output": float}
    enabled: bool                   # API key present at startup
```

**Discovery APIs**:
- `get_all_models()` — All 7 registered models
- `get_enabled_models()` — Only those with valid API keys
- `get_debate_models()` — Subset eligible for debate rounds
- `get_model_by_id("groq-small")` — Lookup by registry key
- `get_model_by_legacy_id("groq")` — Lookup by legacy ID

---

## 8. Safety & Deployment Validation Checklist

### Pre-Deployment

- [ ] **API Keys**: Verify all 7 model API keys are set in environment:
  - `GROQ_API_KEY` (groq-small + llama-3.3)
  - `OPENROUTER_API_KEY` (qwen-vl-2.5)
  - `QWEN3_CODER_API_KEY` (qwen3-coder)
  - `QWEN3_VL_API_KEY` (qwen3-vl)
  - `NEMOTRON_API_KEY` (nemotron-nano)
  - `KIMI_API_KEY` (kimi-2.5)
- [ ] **Rate Limits**: Verify OpenRouter rate limits accommodate 5 concurrent calls
- [ ] **Cost Governor**: Confirm all 7 models have entries in `CostGovernor.MODEL_COSTS`
- [ ] **Graceful Degradation**: Test with 1, 3, 5, and 7 models enabled (some keys missing)

### Functional Validation

- [ ] **STANDARD mode**: Send query, verify all enabled models respond in `model_outputs`
- [ ] **DEBATE mode**: Verify all enabled models appear as debaters in rounds
- [ ] **GLASS mode**: Verify circular audit pairs include all enabled models
- [ ] **EVIDENCE mode**: Verify all models participate in 5-phase pipeline
- [ ] **Analytics**: Verify `analytics.confidence` is non-null (was broken before fix)
- [ ] **Pipeline labels**: Verify OmegaKernel trace shows actual model count

### Regression Checks

- [ ] 3-model subset (only GROQ_API_KEY + OPENROUTER_API_KEY set) behaves identically to pre-fix
- [ ] Error handling: Remove one API key, verify engine degrades gracefully without crash
- [ ] Debate judge scoring: Verify scoring format includes all debater names dynamically
- [ ] Analytics confidence: Compare pre-fix (always empty) vs post-fix (populated) values

### Performance

- [ ] Measure latency with 7 models vs 3 models in STANDARD mode
- [ ] Monitor OpenRouter rate limit headers for 5 concurrent model calls
- [ ] Verify async parallelism in AggregationEngine scales to 7 tasks

---

## 9. Files Modified Summary

| # | File | Action | Lines Changed |
|:---|:---|:---|:---|
| 1 | `backend/core/model_registry.py` | **Created** | ~120 lines |
| 2 | `backend/models/mco_bridge.py` | Modified | +40 lines |
| 3 | `backend/engines/aggregation_engine.py` | Modified | ~30 lines |
| 4 | `backend/engines/blind_audit_engine.py` | Modified | ~50 lines |
| 5 | `backend/engines/forensic_evidence_engine.py` | Modified | ~45 lines |
| 6 | `backend/core/omega_kernel.py` | Modified | ~5 lines |
| 7 | `backend/core/debate_orchestrator.py` | Modified | ~25 lines |
| 8 | `backend/main.py` | Modified | ~5 lines |

**Total**: 1 new file, 7 modified files, ~320 lines changed

---

## 10. Architecture Diagram — Model Dispatch

```
┌──────────────────────────────────────────────────────────────┐
│                    COGNITIVE_MODEL_REGISTRY                   │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │groq-small│llama-3.3 │qwen-vl   │qwen3-cod │qwen3-vl  │   │
│  │ 8B FAST  │70B CONC  │7B GEN    │480B CODE │30B VIS   │   │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘   │
│  ┌────┴─────┬────┴─────┐                                     │
│  │nemotron  │kimi-2.5  │                                     │
│  │30B BASE  │K2 LONG   │                                     │
│  └────┬─────┴────┬─────┘                                     │
└───────┼──────────┼───────────────────────────────────────────┘
        │          │
        ▼          ▼
┌──────────────────────────────────────┐
│       CognitiveModelGateway          │
│  _call_groq()  _call_openrouter()    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│          MCOModelBridge              │
│  call_model(legacy_id, prompt)       │
│  get_enabled_model_ids()             │
│  get_enabled_models_info()           │
└──────────────┬───────────────────────┘
               │
     ┌─────────┼─────────┬──────────┐
     ▼         ▼         ▼          ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Aggregat.│ │Debate  │ │Blind   │ │Forensic│
│Engine   │ │Orchest.│ │Audit   │ │Evidence│
│STANDARD │ │DEBATE  │ │GLASS   │ │EVIDENCE│
└─────────┘ └────────┘ └────────┘ └────────┘
```

---

*End of Audit Report*
