# 📚 QUICK REFERENCE: GOVERNANCE FILES INVENTORY
## All Reusable Components from Sentinel-E

**Purpose:** Quick lookup table for all governance files  
**Last Updated:** April 4, 2026

---

## 🔍 COMPLETE FILE INVENTORY WITH REUSABILITY SCORING

### **TIER 1: CORE ORCHESTRATION (100% Reusable)**

#### **Mode Control & Routing**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/engines/mode_controller.py` | Central mode routing logic | ⭐⭐⭐ 100% | 120 | `healthcare/mode_controller.py` | Add healthcare trigger words |
| `backend/core/mode_config.py` | Mode configuration schema | ⭐⭐⭐ 100% | 80 | `healthcare/mode_config.py` | Minimal |
| `backend/engines/__init__.py` | Engine registry initialization | ⭐⭐ 75% | 60 | `healthcare/engines/__init__.py` | Extend with healthcare engines |

**Total LOC to Copy:** 260 lines | **Effort:** 1.5 hours

---

### **TIER 2: CONFIDENCE & AGGREGATION (98% Reusable)**

#### **Confidence Computation**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/core/confidence_engine.py` | Confidence computation + calibration | ⭐⭐⭐ 98% | 350 | `healthcare/confidence/engine.py` | Add 5 healthcare-specific penalty components |
| `backend/core/cache_engine.py` | Result caching | ⭐⭐⭐ 100% | 200 | `healthcare/cache_engine.py` | Copy as-is |
| `backend/analysis/consensus_engine.py` | Multi-model consensus | ⭐⭐⭐ 95% | 180 | `healthcare/analysis/consensus_engine.py` | Add healthcare consensus metrics |

**Total LOC to Copy:** 730 lines | **Effort:** 2.5 hours

#### **Standard Mode Aggregation**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/engines/aggregation_engine.py` | Parallel model aggregation | ⭐⭐⭐ 100% | 400 | `healthcare/aggregation_engine.py` | Copy as-is |
| `backend/core/ensemble_schemas.py` | Ensemble data structures | ⭐⭐⭐ 98% | 500 | `healthcare/schemas/ensemble.py` | Add healthcare-specific structures |

**Total LOC to Copy:** 900 lines | **Effort:** 2 hours

---

### **TIER 3: DEBATE MODE (95% Reusable)**

#### **Core Debate Engines**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/core/structured_debate_engine.py` | Multi-round debate with position tracking | ⭐⭐⭐ 95% | 800 | `healthcare/modes/debate/engine.py` | Replace 3 system prompts with healthcare versions |
| `backend/core/debate_orchestrator.py` | Debate orchestration + model coordination | ⭐⭐⭐ 90% | 600 | `healthcare/modes/debate/orchestrator.py` | Adapt model role definitions |
| `backend/sentinel/debate_engine.py` | Lower-level debate mechanics | ⭐⭐ 85% | 400 | `healthcare/sentinel/debate_engine.py` | Minimal changes |
| `backend/core/agreement_matrix.py` | Model agreement computation | ⭐⭐⭐ 100% | 150 | `healthcare/analysis/agreement.py` | Copy as-is |

**Total LOC to Copy:** 1950 lines | **Effort:** 4-5 hours

---

### **TIER 4: SYNTHESIS MODE (98% Reusable)**

#### **Collaborative Refinement**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/core/synthesis_engine.py` | Iterative refinement with peer review | ⭐⭐⭐ 98% | 300 | `healthcare/modes/synthesis/engine.py` | Replace system prompt only |
| `backend/core/anchor_pass.py` | Anchor model selection + execution | ⭐⭐⭐ 100% | 150 | `healthcare/modes/synthesis/anchor.py` | Copy as-is |
| `backend/core/multipass_reasoning.py` | Multi-pass reasoning pipeline | ⭐⭐⭐ 95% | 250 | `healthcare/modes/synthesis/multipass.py` | Minimal changes |

**Total LOC to Copy:** 700 lines | **Effort:** 1.5 hours

---

### **TIER 5: GLASS MODE & AUDIT (97% Reusable)**

#### **Forensic Transparency & Audit**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/core/glass_pipeline.py` | Reasoning transparency + trust scoring | ⭐⭐⭐ 97% | 400 | `healthcare/modes/glass/pipeline.py` | Replace metric dimensions (8 new healthcare metrics) |
| `backend/engines/blind_audit_engine.py` | Blind audit execution | ⭐⭐⭐ 95% | 300 | `healthcare/modes/glass/blind_audit.py` | Add healthcare audit rules |
| `backend/engines/forensic_evidence_engine.py` | Forensic evidence verification | ⭐⭐⭐ 90% | 600 | `healthcare/modes/evidence/engine.py` | Adapt for clinical evidence triangulation |
| `backend/core/evidence_engine.py` | Core evidence extraction | ⭐⭐⭐ 95% | 350 | `healthcare/modes/evidence/core.py` | Minimal changes |
| `backend/core/evidence_debate_pipeline.py` | Evidence + debate integration | ⭐⭐⭐ 90% | 280 | `healthcare/modes/evidence/debate_pipeline.py` | Wire into healthcare debate |

**Total LOC to Copy:** 1930 lines | **Effort:** 4-5 hours

---

### **TIER 6: POLICY & SAFETY (75% Reusable)**

#### **Boundary & Policy Enforcement**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/risk_boundaries.py` | Risk boundary detection | ⭐⭐ 75% | 600 | `healthcare/governance/safety_policies.py` | Complete refactor to healthcare contraindications |
| `backend/core/stress_engine.py` | Stress testing + fragility detection | ⭐⭐⭐ 85% | 450 | `healthcare/governance/stress_test.py` | Add healthcare edge cases |
| `backend/core/boundary_detector.py` | Topic/domain boundary detection | ⭐⭐ 70% | 280 | `healthcare/governance/boundary.py` | Adapt to clinical domain |
| `backend/sentinel/shadow_engine.py` | Shadow mode execution | ⭐ 65% | 350 | `healthcare/governance/shadow.py` | Significant refactor needed |

**Total LOC to Copy:** 1680 lines | **Effort:** 6-8 hours (most adaptation needed)

---

### **TIER 7: ORCHESTRATION & EXECUTION (80% Reusable)**

#### **Master Orchestrators**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/core/cognitive_orchestrator.py` | 10-phase master orchestrator | ⭐⭐ 80% | 1200 | `healthcare/orchestrator.py` | Insert healthcare phases, add safety gate |
| `backend/core/orchestration.py` | Higher-level orchestration | ⭐⭐ 75% | 400 | `healthcare/orchestration_layer.py` | Significant clinical context injection |
| `backend/sentinel/sentinel_sigma_v4.py` | Multi-round orchestration v4 | ⭐⭐ 85% | 800 | `healthcare/sigma_orchestrator.py` | Add biomechanics analysis loop |
| `backend/core/model_execution_graph.py` | Model execution DAG | ⭐⭐⭐ 90% | 350 | `healthcare/execution_graph.py` | Minimal changes |

**Total LOC to Copy:** 2750 lines | **Effort:** 6-8 hours

---

### **TIER 8: MEMORY & LEARNING (95% Reusable)**

#### **Persistent Memory & Learning**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/memory/memory_engine.py` | 3-tier memory system | ⭐⭐⭐ 95% | 600 | `healthcare/memory/engine.py` | Add patient memory tier |
| `backend/core/knowledge_learner.py` | Learning from feedback | ⭐⭐⭐ 95% | 450 | `healthcare/memory/learner.py` | Add clinical outcome tracking |
| `backend/core/session_intelligence.py` | Session state management | ⭐⭐⭐ 98% | 300 | `healthcare/memory/session.py` | Minimal changes |
| `backend/memory/knowledge_memory.py` | Knowledge storage | ⭐⭐⭐ 100% | 280 | `healthcare/memory/knowledge.py` | Copy as-is |
| `backend/core/drift_tracker.py` | Model drift detection | ⭐⭐⭐ 95% | 200 | `healthcare/memory/drift.py` | Minimal changes |

**Total LOC to Copy:** 1830 lines | **Effort:** 3-4 hours

---

### **TIER 9: MODEL REGISTRY & EXECUTION (85% Reusable)**

#### **Model Management**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/metacognitive/cognitive_gateway.py` | Global model registry | ⭐⭐⭐ 90% | 400 | `healthcare/models/registry.py` | Add healthcare-specific models |
| `backend/models/mco_bridge.py` | Model provider abstraction | ⭐⭐⭐ 95% | 300 | `healthcare/models/bridge.py` | Minimal changes |
| `backend/core/model_registry.py` | Dynamic model registration | ⭐⭐⭐ 100% | 200 | `healthcare/models/local_registry.py` | Copy as-is |
| `backend/models/local_engine.py` | Local model execution | ⭐⭐ 80% | 250 | `healthcare/models/local_engine.py` | Add healthcare model specs |

**Total LOC to Copy:** 1150 lines | **Effort:** 2-3 hours

---

### **TIER 10: DATA & PERSISTENCE (100% Reusable)**

#### **Database & Caching**

| File | Purpose | Reusability | LOC | Target Use | Adaptation |
|------|---------|-------------|-----|------------|-----------|
| `backend/database/connection.py` | Database connection pooling | ⭐⭐⭐ 100% | 150 | Use as-is | None |
| `backend/database/crud.py` | CRUD operations | ⭐⭐⭐ 100% | 300 | Extend with healthcare tables | Add patient/exercise tables |
| `backend/core/evidence_cache.py` | Evidence caching | ⭐⭐⭐ 100% | 200 | Use as-is | None |

**Total LOC to Copy:** 650 lines | **Effort:** 1-2 hours

---

## 🎯 REUSABILITY SUMMARY BY LAYER

```
LAYER                          AVERAGE REUSABILITY    EFFORT
──────────────────────────────────────────────────────────────
1. Mode Control                ~98%                   1.5h
2. Confidence & Aggregation    ~97%                   4.5h
3. Debate Mode                 ~92%                   4.5h
4. Synthesis Mode              ~97%                   1.5h
5. Glass & Audit               ~93%                   4.5h
6. Policy & Safety             ~75% ⚠️                6h
7. Orchestration               ~82%                   7h
8. Memory & Learning           ~96%                   3.5h
9. Model Registry              ~91%                   2.5h
10. Data & Persistence         ~100%                  1.5h
──────────────────────────────────────────────────────────────
OVERALL AVERAGE REUSABILITY:   ~90.8%
TOTAL EFFORT:                  ~36.5 hours (MVP + 50% buffer = ~55h)
```

---

## 🗺️ COMPLETE FILE DEPENDENCY MAP

```
┌─────────────────────────────────────────────────┐
│ Entry Points (API Routes)                       │
├─────────────────────────────────────────────────┤
│ main.py                                         │
│ ├─ routes/healthcare/exercise_assessment.py    │
│ ├─ routes/healthcare/protocol_synthesis.py     │
│ ├─ routes/healthcare/audit_transparency.py     │
│ ├─ routes/healthcare/evidence_verification.py │
│ └─ routes/healthcare/posture_feedback.py       │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│ Orchestrator Layer                              │
├─────────────────────────────────────────────────┤
│ healthcare/orchestrator.py                      │
│ (from: core/cognitive_orchestrator.py)          │
│ ├─ 10 phases                                    │
│ ├─ Phase 8: Safety Policy Gate ⚠️              │
│ └─ Phase 9: Confidence Computation             │
└─────────┬───────────────────────────────────────┘
          │
          ├──────────────────────────────────────┐
          │                                      │
          ▼                                      ▼
    ┌─────────────────┐           ┌──────────────────────┐
    │ Mode Resolution │           │ Model Selection      │
    ├─────────────────┤           ├──────────────────────┤
    │ mode_controller │           │ model_registry.py    │
    │ mode_config.py  │           │ (3 models defined)   │
    └────────┬────────┘           └──────┬───────────────┘
             │                           │
             ├───────────┬───────────┬───┤
             │           │           │   │
             ▼           ▼           ▼   ▼
        ┌───────┐   ┌──────┐   ┌────┐ ┌──────────┐
        │DEBATE │   │SYNTH │   │GLASS   │EVIDENCE │
        │ENGINE │   │ENGINE│   │PIPELINE   │ENGINE │
        └───┬───┘   └──┬───┘   └─┬──┘ └───┬──────┘
            │          │        │        │
            └──────┬───┴────┬───┴────────┘
                   │        │
                   ▼        ▼
            ┌─────────────────────────┐
            │ Support Engines         │
            ├─────────────────────────┤
            │ • Confidence Engine     │
            │ • Biomechanics Engine   │
            │ • Memory Engine         │
            │ • Stress Engine         │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────────┐
            │ Policy & Safety Layer ⚠️     │
            ├─────────────────────────────┤
            │ Contraindication checks     │
            │ ROM limits                  │
            │ Pain thresholds             │
            │ Post-op windows             │
            │ Confidence gates            │
            └────────────┬────────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │ Database & Memory Layer  │
            ├──────────────────────────┤
            │ • Patient DB             │
            │ • Session Cache          │
            │ • Clinical Memory        │
            │ • Outcome Tracking       │
            └──────────────────────────┘
```

---

## 🔄 FILE COPY ORDER (Recommended)

### **Phase 1: Foundation (Week 1) — 15 hours**
1. Core mode controllers (`mode_controller.py`, `mode_config.py`)
2. Confidence engine (`confidence_engine.py`)
3. Model registry (`cognitive_gateway.py`)
4. System prompts (create healthcare versions)

### **Phase 2: Reasoning Engines (Week 2) — 12 hours**
5. Structured debate engine (`structured_debate_engine.py`)
6. Synthesis engine (`synthesis_engine.py`)
7. Aggregation engine (`aggregation_engine.py`)

### **Phase 3: Governance (Week 2-3) — 10 hours**
8. Glass pipeline (`glass_pipeline.py`)
9. Evidence engines (`evidence_engine.py`, `forensic_evidence_engine.py`)
10. Safety policies (`risk_boundaries.py`)

### **Phase 4: Orchestration (Week 3) — 8 hours**
11. Master orchestrator (`cognitive_orchestrator.py`)
12. Memory engine (`memory_engine.py`)

### **Phase 5: Integration (Week 4) — 10 hours**
13. API routes (healthcare-specific)
14. Database schema (healthcare tables)
15. Testing & validation

---

## 🚀 AUTONOMOUS FINANCE (CHIT FUND) SYSTEM

**Same architecture applies:**

```
Core Governance Files (unchanged):
├─ mode_controller.py ✓
├─ confidence_engine.py ✓
├─ debate_engine.py ✓
├─ synthesis_engine.py ✓
├─ glass_pipeline.py ✓
├─ evidence_engine.py ✓
└─ memory_engine.py ✓

Finance-Specific Adaptations:
├─ Model Registry: (Financial optimizer, Governance validator, Member advocate)
├─ System Prompts: (Financial reasoning, DAO governance, Member fairness)
├─ Policy Engine: (Risk limits, Member voting rules, Fund allocation rules)
├─ Memory: (Member history, Fund performance, Voting records)
└─ API Routes: (Fund assessment, Collective voting, Outcome tracking)
```

**Reusability for Finance:** ~85% (same core, different domain logic)

---

## 📊 COMPREHENSIVE MAPPING TABLE

| Component | Source File | Target File | Reusability | Priority | Effort |
|-----------|-------------|------------|-------------|----------|--------|
| Mode Routing | `modes_controller.py` | `healthcare/mode_controller.py` | 100% | P1 | 1h |
| Debate Engine | `structured_debate_engine.py` | `healthcare/debate/engine.py` | 95% | P2 | 3h |
| Synthesis | `synthesis_engine.py` | `healthcare/synthesis/engine.py` | 98% | P2 | 1.5h |
| Glass Audit | `glass_pipeline.py` | `healthcare/glass/pipeline.py` | 97% | P3 | 2h |
| Evidence | `evidence_engine.py` | `healthcare/evidence/engine.py` | 95% | P3 | 2h |
| Confidence | `confidence_engine.py` | `healthcare/confidence/engine.py` | 98% | P1 | 1.5h |
| Safety Policies | `risk_boundaries.py` | `healthcare/governance/policies.py` | 75% | P3 | 6h |
| Orchestrator | `cognitive_orchestrator.py` | `healthcare/orchestrator.py` | 80% | P2 | 5h |
| Memory | `memory/memory_engine.py` | `healthcare/memory/engine.py` | 95% | P4 | 2.5h |
| Model Registry | `cognitive_gateway.py` | `healthcare/models/registry.py` | 90% | P1 | 1.5h |
| Database | `database/crud.py` | Extend existing | 100% | P4 | 1h |

---

## ⚡ MINIMUM VIABLE HEALTHCARE SYSTEM (48 Hours)

**Essential files to get started:**

```
MUST HAVE (12 files):
✓ mode_controller.py
✓ confidence_engine.py  
✓ structured_debate_engine.py
✓ synthesis_engine.py
✓ cognitive_orchestrator.py
✓ risk_boundaries.py (safety policies)
✓ glass_pipeline.py
✓ evidence_engine.py
✓ memory/memory_engine.py
✓ model_registry files
✓ Healthcare API routes
✓ Database schema extension

NICE TO HAVE (add later):
- Stress testing engine
- Advanced biomechanics
- Real-time posture feedback
- Outcome learning
```

**MVP Timeline: 48 hours**
- Day 1-2: Copy core files + create prompts
- Day 3-4: Wire orchestrator + modes
- Day 5: Add safety policies
- Day 6: Create API routes
- Days 7-9: Testing + deployment

---

## 🎓 LEARNING PATH

**To understand the full governance system:**

1. **Start:** `backend/core/mode_controller.py` (routing logic)
2. **Then:** `backend/core/structured_debate_engine.py` (debate mechanics)
3. **Then:** `backend/core/synthesis_engine.py` (collaborative reasoning)
4. **Then:** `backend/core/glass_pipeline.py` (transparency)
5. **Then:** `backend/risk_boundaries.py` (policy enforcement)
6. **Then:** `backend/core/cognitive_orchestrator.py` (full pipeline)

**Estimated Learning Time: 6-8 hours**

---

## 🏥 HEALTHCARE-SPECIFIC POLICIES TO IMPLEMENT

After copying base files, add these healthcare rules:

```python
# 1. Contraindication Matrix (20+ conditions)
condition_matrix = {
    "osteoporosis": ["high_impact", "compression", "torsion"],
    "stenosis": ["spinal_flexion", "extension", "high_load"],
    "fusion": ["rotation", "shear", "dynamic", "in_first_month"],
    # ... 20+ more
}

# 2. Post-Op Windows (by surgery type)
post_op_windows = {
    "fusion": {"passive_only": 6, "protected_active": 14, "full": 30},
    "discectomy": {"passive_only": 3, "protected": 21, "full": 42},
    # ... more
}

# 3. ROM Requirements (by exercise)
rom_requirements = {
    "lumbar_flexion_30": {"lumbar_flexion": 40, "hip_flexion": 90},
    # ... hundreds of exercises
}

# 4. Pain Thresholds
pain_thresholds = {
    "dynamic_exercise": {"max_pain": 6, "max_pain_elderly": 4},
    # ... more
}

# 5. Age-Specific Rules
age_rules = {
    ">75": {"min_confidence": 0.75, "max_intensity": "moderate"},
    # ... more age groups
}
```

---

## 📞 SUPPORT & RESOURCES

- **Sentinel-E Documentation**: See `GOVERNANCE_ARCHITECTURE_MAPPING.md`
- **Implementation Guide**: See `GOVERNANCE_FILE_IMPLEMENTATION_GUIDE.md`
- **Architecture Diagrams**: See `HEALTHCARE_GOVERNANCE_ARCHITECTURE.md`
- **Model Registry**: See `/memories/repo/sentinel-e-model-registry.md`

---

**Total Reusable Code: ~10,000+ LOC**  
**Total Implementation Effort: 36-55 hours**  
**MVP Timeline: 48 hours**  
**Production Timeline: 2-3 weeks**

**Ready to start? Begin with Phase 1 core files!**
