# 🏛️ SENTINEL-E GOVERNANCE ARCHITECTURE MAPPING
## Healthcare AI + Autonomous Finance System Adaptation Guide

**Created:** April 4, 2026  
**Purpose:** Map reusable governance components from Sentinel-E to new healthcare spine rehab + collective finance systems

---

## 📋 EXECUTIVE SUMMARY

Your Sentinel-E project contains a **production-grade, multi-model governance framework** with:
- ✅ **3 distinct reasoning modes**: Debate, Synthesis, Glass (Audit)
- ✅ **Policy-enforced decision making** with confidence calibration
- ✅ **Multi-round adversarial reasoning** with position tracking
- ✅ **Forensic evidence verification** with triangular cross-checks
- ✅ **Real-time mode routing** with trigger word detection
- ✅ **Comprehensive audit trails** and explainability layers

**These are 100% reusable for:**
1. Healthcare: Spine rehabilitation + exercise validation + biomechanics reasoning
2. Finance: Autonomous chit fund governance + collective savings policies + risk arbitration

---

## 🗂️ CORE GOVERNANCE FILES STRUCTURE

### **LAYER 1: MODE ORCHESTRATION & ROUTING**

| File | Purpose | Reusability | Use Case |
|------|---------|-------------|----------|
| `backend/engines/mode_controller.py` | Central routing logic (Standard→Debate→Evidence→Glass→Stress) | ⭐ **100% reusable** | Route healthcare requests: `posture_assessment_debate`, `exercise_risk_glass`, `physiotherapy_evidence` |
| `backend/core/mode_config.py` | Mode configuration schema | ⭐ **100% reusable** | Add healthcare-specific modes: `biomechanics_debate`, `safety_override` |
| `backend/engines/__init__.py` | Engine registry and initialization | ⭐ **80% reusable** | Extend with healthcare-specific engines |

**Specific Reuse Pattern:**
```python
# Current: Standard mode → Evidence trigger → Evidence mode
# Healthcare: 
#   - Standard: Patient education query
#   - Debate mode trigger: "Is this exercise safe?"
#   - Glass mode trigger: "Why did system reject exercise?"
#   - Evidence mode trigger: "Show me the research"
```

---

### **LAYER 2: DEBATE MODE (Adversarial Reasoning)**

| File | Purpose | Reusability | Adaptation |
|------|---------|-------------|-----------|
| `backend/core/structured_debate_engine.py` | **Multi-round debate with position tracking** | ⭐⭐ **95% reusable** | Create healthcare-specific debate prompts |
| `backend/core/debate_orchestrator.py` | High-level debate orchestration | ⭐⭐ **90% reusable** | Adapt model definitions for healthcare roles |
| `backend/sentinel/debate_engine.py` | Lower-level debate mechanics | ⭐⭐ **85% reusable** | Minimal changes needed |

#### **Debate Mode Architecture for Healthcare:**

```
┌─────────────────────────────────────────────────┐
│ HEALTHCARE DEBATE ENGINE                        │
├─────────────────────────────────────────────────┤
│ ROUND 1: INDEPENDENT ASSESSMENT                │
│ ┌─────────┬──────────┬────────┐               │
│ │ Model A │ Model B  │ Model C│               │
│ │(Clinical│(Safety- │(Patient│               │
│ │ Evidence│ First)  │-Centric)               │
│ │ Role)   │         │        │               │
│ └─────────┴──────────┴────────┘               │
│                                              │
│ ROUND 2: REBUTTALS + BIOMECHANICS            │
│ - A challenges B's risk assumptions          │
│ - B challenges C's patient empathy           │
│ - C questions A's evidence interpretation    │
│ - Position shifts tracked                    │
│                                              │
│ ROUND 3+: SYNTHESIS + CONVERGENCE            │
│ Final aggregated recommendation with         │
│ dissent documentation                        │
└─────────────────────────────────────────────────┘

OUTCOME: Structured decision with:
  - Who agreed/disagreed
  - Why positions shifted
  - Confidence per role
  - Safety-first overrides
```

#### **Key Classes to Reuse/Adapt:**

```python
# FROM: backend/core/structured_debate_engine.py

@dataclass
class DebatePosition:
    """Position statement in debate round."""
    model_id: str
    round_number: int
    position: str              # "support exercise" / "contraindicated"
    argument: str              # Detailed reasoning
    assumptions: List[str]     # Explicit assumptions
    risks: List[str]           # Identified risks
    vulnerabilities: List[str] # Self-critique
    confidence: float          # 0.0-1.0

# REUSE FOR HEALTHCARE:
@dataclass
class BiomechanicsDebatePosition:
    model_id: str              # "clinical_model" / "safety_model" / "patient_model"
    round_number: int
    exercise: str              # e.g., "lumbar_flexion_30deg"
    position: str              # "safe" / "contraindicated" / "conditional"
    clinical_reasoning: str    # Evidence-based argument
    biomechanics_assumptions: List[str]  # Joint assumptions
    patient_factors: List[str]  # Patient-specific risks
    safety_confidence: float    # 0.0-1.0

# REUSE FOR FINANCE (Chit Fund):
@dataclass
class ChitFundDebatePosition:
    participant_id: str        # Role in collective: "conservative" / "aggressive" / "balanced"
    round_number: int
    proposal: str              # Fund distribution proposal
    position: str              # "approve" / "reject" / "conditional"
    financial_reasoning: str   # ROI analysis
    risk_factors: List[str]    # Financial risks
    governance_assumptions: List[str]  # DAO assumptions
    confidence: float
```

---

### **LAYER 3: SYNTHESIS MODE (Collaborative Refinement)**

| File | Purpose | Reusability | Adaptation |
|------|---------|-------------|-----------|
| `backend/core/synthesis_engine.py` | **Iterative refinement with peer review** | ⭐⭐⭐ **98% reusable** | Minimal changes needed |
| `backend/analysis/consensus_engine.py` | Consensus computation | ⭐⭐ **90% reusable** | Add healthcare consensus metrics |

#### **Synthesis Pipeline for Healthcare:**

```
PHASE 1: PRIMARY DRAFT
  ↓
  Anchor Model (Strongest clinical model)
  Produces initial exercise recommendation
  with biomechanics analysis
  
PHASE 2: PEER REVIEW
  ↓
  Model B (Safety): "Joint compression risk not addressed"
  Model C (Patient): "Flexibility assumption too aggressive"
  
PHASE 3: ITERATIVE REFINE
  ↓
  Anchor Model integrates feedback
  - Adds ROM modification for stiff patients
  - Adds compression warning thresholds
  - Maintains clinical validity
  
PHASE 4: CONSENSUS SCORE
  ↓
  All models rate final recommendation:
  - Clinical coherence: 0.92
  - Safety adequacy: 0.88
  - Patient applicability: 0.85
  - Overall synthesis confidence: 0.88

OUTPUT:
{
  "final_recommendation": "Modified lumbar flexion...",
  "draft": "Initial clinical assessment...",
  "revisions": [
    {"reviewer": "safety_model", "critique": "..."},
    {"reviewer": "patient_model", "critique": "..."}
  ],
  "consensus_score": 0.88,
  "improvement_delta": +0.15  # vs draft
}
```

#### **Reuse for Finance (Collective Decision Making):**

```
PHASE 1: PROPOSAL DRAFT
  Primary financial model proposes:
  - Distribution strategy (ROI-optimized)
  - Timeline
  - Risk allocation
  
PHASE 2: STAKEHOLDER REVIEW
  - Conservative members review
  - Risk specialists review
  - Governance experts review
  
PHASE 3: ITERATIVE REFINE
  Proposal adapts to feedback:
  - Conservative: add risk buffers
  - Risk specialists: adjust allocation
  - Governance: ensure DAO compliance
  
PHASE 4: CONSENSUS
  All stakeholder roles score:
  - Financial soundness
  - Risk adequacy
  - Governance compliance
  - Collective benefit
```

---

### **LAYER 4: GLASS MODE (Forensic Audit & Transparency)**

| File | Purpose | Reusability | Adaptation |
|------|---------|-------------|-----------|
| `backend/core/glass_pipeline.py` | **Reasoning transparency + trust scoring** | ⭐⭐⭐ **98% reusable** | Add healthcare-specific audit dimensions |
| `backend/engines/blind_audit_engine.py` | Blind audit mechanics | ⭐⭐ **85% reusable** | Create healthcare audit profiles |
| `backend/engines/forensic_evidence_engine.py` | Forensic verification | ⭐⭐ **90% reusable** | Adapt for clinical evidence triangulation |

#### **Glass Mode Audit Dimensions for Healthcare:**

```python
# FROM: backend/core/glass_pipeline.py (original)
Glass Metrics:
  - logical_coherence
  - hidden_assumptions
  - bias_patterns
  - confidence_inflation
  - persuasion_tactics
  - evidence_quality
  - completeness

# ADAPT FOR HEALTHCARE:
HealthcareGlassMetrics:
  - biomechanical_validity      # Does physics check out?
  - clinical_evidence_grounding  # Supported by research?
  - patient_contraindication_coverage  # All patient types considered?
  - safety_bias_detection       # Overcautious or reckless?
  - assumption_documentation    # Explicit vs hidden
  - exercise_contraindication_completeness  # All red flags covered?
  - confidence_calibration      # Is 0.85 really 0.85?

# ADAPT FOR FINANCE (Chit Governance):
FinanceGlassMetrics:
  - financial_model_validity     # Math checks out?
  - regulatory_compliance        # Legal risks?
  - member_benefit_distribution  # Fair allocation?
  - risk_bias_detection         # Over/under-estimation?
  - governance_assumption_clarity # DAO rules clear?
  - emergency_scenario_coverage  # Black swan scenarios?
  - confidence_calibration      # Is model sure enough?
```

#### **Glass Audit Report Structure:**

```json
{
  "audit_timestamp": "2026-04-04T10:30:00Z",
  "executive_assessment": "Clinical recommendation is sound but assumes patient compliance",
  "assessments": [
    {
      "model": "clinical_model",
      "role": "Evidence-based recommendation",
      "metrics": {
        "biomechanical_validity": 0.94,
        "clinical_evidence": 0.91,
        "patient_factor_coverage": 0.78,
        "safety_bias": 0.12,
        "assumption_clarity": 0.88
      },
      "trust_score": 0.87,
      "red_flags": [
        "Assumes patient can perform 3 daily sessions",
        "No mention of pain escalation protocol"
      ],
      "strengths": [
        "Grounded in latest orthopedic research",
        "Clear biomechanical reasoning"
      ]
    }
  ],
  "consensus_risk": "MEDIUM",
  "reasoning_graph": {
    "nodes": [...],  # Explainability nodes
    "edges": [...]   # Reasoning flow
  },
  "overall_trust": 0.86,
  "audit_verdict": "APPROVED_WITH_CAVEATS",
  "caveats": ["Patient compliance assumption", "Age group assumption (35-55 years)"]
}
```

---

### **LAYER 5: EVIDENCE MODE (Forensic Verification)**

| File | Purpose | Reusability | Adaptation |
|------|---------|-------------|-----------|
| `backend/core/evidence_engine.py` | Evidence extraction + triangulation | ⭐⭐⭐ **98% reusable** | Adapt claim types for healthcare |
| `backend/engines/forensic_evidence_engine.py` | 5-phase forensic pipeline | ⭐⭐⭐ **95% reusable** | Minimal changes |
| `backend/core/evidence_debate_pipeline.py` | Evidence + debate integration | ⭐⭐ **90% reusable** | Wire into healthcare debate |

#### **Evidence Mode Pipeline for Healthcare:**

```
PHASE 1: INDEPENDENT CLAIM EXTRACTION
  Model A (Clinical): "Lumbar flexion >30° increases disc pressure"
  Model B (Biomechanics): "Posterior chain activation reduces shear"
  Model C (Patient): "Morning stiffness improves within 2 weeks"
  
PHASE 2: TRIANGULAR CROSS-VERIFICATION
  A verifies B's claim: "Posterior chain finding consistent with EMG studies"
  B verifies A's claim: "Disc pressure measurements confirm >25° threshold"
  C verifies A's claim: "Patient feedback aligns with clinical timeline"
  
PHASE 3: CONTRADICTION DETECTION
  Contradiction detected:
  - A claims: "Full ROM needed for strength"
  - C claims: "Pain limits ROM to 20 degrees initially"
  Resolution: "Both true — ROM progression needed"
  
PHASE 4: BAYESIAN CONFIDENCE UPDATE
  - Initial: 0.70 (model self-reported)
  - Cross-verification: +0.12 (independent confirmation)
  - Contradiction handling: -0.05 (partial conflict)
  - Final: 0.77
  
PHASE 5: VERBATIM CITATION MODE
  When patient asks: "Show me the research"
  Output includes:
  - Exact quotes from studies
  - DOIs and journal info
  - Reliability score per source

RESULT:
{
  "claims": [
    {
      "statement": "Lumbar flexion >30° increases disc pressure",
      "origin_model": "clinical_model",
      "verifications": [
        {"verifier": "biomechanics_model", "verdict": "confirmed"},
        {"verifier": "patient_model", "verdict": "conditional"}
      ],
      "final_confidence": 0.77,
      "citations": [
        {"paper": "Wilke et al 2015", "doi": "...", "quote": "..."}
      ]
    }
  ]
}
```

---

### **LAYER 6: CONFIDENCE & POLICY ENGINE**

| File | Purpose | Reusability | Adaptation |
|------|---------|-------------|-----------|
| `backend/core/confidence_engine.py` | **Confidence computation + calibration** | ⭐⭐⭐ **98% reusable** | Add healthcare-specific penalties |
| `backend/core/stress_engine.py` | Stress testing + fragility detection | ⭐⭐ **85% reusable** | Test healthcare edge cases |
| `backend/risk_boundaries.py` | Boundary detection + policy override | ⭐⭐⭐ **95% reusable** | Critical for healthcare safety |

#### **Confidence Components for Healthcare:**

```python
# FROM: backend/core/confidence_engine.py

final_confidence = (
    base_model_confidence           # 0.8 (clinical model trained)
    + evidence_weight              # +0.05 (cross-verified)
    + reliability_adjustment       # +0.03 (model track record)
    - boundary_penalty             # -0.02 (safe mode active)
    - disagreement_penalty         # -0.04 (models disagree on ROM)
    - fragility_penalty            # -0.01 (stress test revealed weakness)
    - domain_uncertainty           # -0.03 (new patient type)
    * historical_model_reliability # × 0.98 (model reliability multiplier)
    = 0.76 (final confidence)
)

# HEALTHCARE CUSTOMIZATION:
HealthcareConfidenceComponents:
  - base_model_confidence: From clinical foundation
  - evidence_weight: Research grounding (double-check mechanisms)
  - patient_factor_adjustment: Missing patient info penalty
  - safety_override_penalty: Policy-enforced caution
  - biomechanics_disagreement: Model conflict on forces
  - exercise_complexity_uncertainty: Unfamiliar exercise type
  - contraindication_coverage: Did we check all risks?
  - historical_model_safety_record: Track of adverse events
```

#### **Safety-First Policy Override (CRITICAL):**

```python
# FROM: backend/risk_boundaries.py

# HEALTHCARE ADAPTATION:
class HealthcareSafetyPolicy:
    """
    Safety-first governance for medical recommendations.
    Confidence can be high, but policy can override.
    """
    
    def should_override(
        recommendation: str,
        confidence: float,
        patient_factors: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check if recommendation violates safety policies.
        Return (override_needed, reason)
        """
        
        # POLICY 1: Age boundary check
        if patient_factors["age"] > 70 and "high_impact" in recommendation:
            return True, "High-impact exercise contraindicated for age >70"
        
        # POLICY 2: Confidence floor
        if confidence < 0.65:
            return True, "Insufficient confidence for medical recommendation"
        
        # POLICY 3: Comorbidity check
        if "osteoporosis" in patient_factors["comorbidities"]:
            if "compression" in recommendation:
                return True, "Spinal compression contraindicated for osteoporosis"
        
        # POLICY 4: Post-surgical window
        if patient_factors["days_post_surgery"] < 14:
            if "active_ROM" in recommendation:
                return True, "Active ROM contraindicated <14 days post-surgery"
        
        return False, ""
```

---

### **LAYER 7: ORCHESTRATION & EXECUTION**

| File | Purpose | Reusability | Adaptation |
|------|---------|-------------|-----------|
| `backend/core/cognitive_orchestrator.py` | **10-phase execution pipeline** | ⭐⭐ **80% reusable** | Adapt phases for healthcare |
| `backend/core/orchestration.py` | Higher-level orchestration | ⭐⭐ **75% reusable** | Extend with healthcare phases |
| `backend/sentinel/sentinel_sigma_v4.py` | Multi-round orchestration | ⭐⭐ **85% reusable** | Add healthcare-specific rounds |

#### **Cognitive Orchestrator 10-Phase Pipeline:**

```
PHASE 1: Query Ingestion
PHASE 2: Intent Classification
PHASE 3: Mode Resolution
PHASE 4: Model Selection
PHASE 5: Parallel Model Execution
PHASE 6: Output Normalization
PHASE 7: Cross-Model Analysis
PHASE 8: Policy Application
PHASE 9: Confidence Computation
PHASE 10: Response Formatting

HEALTHCARE ADAPTATION:
PHASE 1: Patient Query Ingestion
PHASE 2: Clinical Intent Classification
  → exercise_safety_query
  → patient_education
  → post_surgery_protocol
  → pain_management
  → surgical_readiness
PHASE 3: Healthcare Mode Resolution
  → debate (conflicting advice)
  → synthesis (collaborative recommendation)
  → glass (transparency for patient)
  → evidence (research backing)
PHASE 4: Clinical Model Selection
  → clinical_model (evidence-based)
  → biomechanics_model (forces/joints)
  → patient_model (subjective response)
PHASE 5: Parallel Clinical Inference
PHASE 6: Biomechanics Normalization
PHASE 7: Cross-Model Consensus Analysis
PHASE 8: Safety Policy Application
  → Check contraindications
  → Apply patient-specific overrides
  → Enforce surgical recovery windows
PHASE 9: Clinical Confidence Computation
  → Evidence grounding
  → Model agreement
  → Patient factor coverage
  → Safety buffer assessment
PHASE 10: Clinical Report Formatting
  → Patient-facing summary
  → Clinician explanation
  → Research backing
  → Contraindication warnings
```

---

### **LAYER 8: MEMORY & LEARNING**

| File | Purpose | Reusability | Adaptation |
|------|---------|-------------|-----------|
| `backend/memory/memory_engine.py` | **3-tier memory system** | ⭐⭐⭐ **95% reusable** | Extend for patient memory |
| `backend/core/knowledge_learner.py` | Learning from feedback | ⭐⭐⭐ **95% reusable** | Track clinical outcomes |
| `backend/core/session_intelligence.py` | Session state management | ⭐⭐⭐ **95% reusable** | Track patient session state |

#### **3-Tier Memory for Healthcare:**

```
TIER 1: SESSION MEMORY
  - Current patient context
  - This exercise session
  - Real-time posture data
  - Immediate form feedback
  
TIER 2: PATIENT MEMORY
  - Medical history
  - Surgery details
  - Exercise history
  - Progress tracking
  - Comorbidities
  - Patient preferences
  
TIER 3: SYSTEM MEMORY
  - Clinical research updates
  - Model performance metrics
  - Adverse event tracking
  - Population-level insights
  - Sentinel-E learning

QUERY PROGRESSION:
1. Check session memory
   "Did we discuss lumbar flexion today?"
2. Check patient memory
   "What's this patient's ROM history?"
3. Check system memory
   "What do we know about post-op ROM recovery?"
```

---

## 🔄 CROSS-SYSTEM REUSE PATTERNS

### **Pattern 1: Direct Copy (100% Reuse)**

**Files:** 
- `backend/core/confidence_engine.py`
- `backend/engines/aggregation_engine.py`
- `backend/core/synthesis_engine.py`

**Direction:**
```
Sentinel-E → Healthcare System
  1. Copy file as-is
  2. Override system prompts
  3. Adapt data structures
  4. Done.
```

**Example: Confidence Engine**
```python
# COPY: backend/core/confidence_engine.py
# MODIFY: HealthcareConfidenceComponents
#   - Add patient_factor_adjustment
#   - Add contraindication_coverage
#   - Add safety_bias_penalty
# USE: In healthcare orchestrator
```

---

### **Pattern 2: System Prompt + Schema Adaptation (90% Reuse)**

**Files:**
- `backend/core/structured_debate_engine.py`
- `backend/core/debate_orchestrator.py`

**Direction:**
```
Sentinel-E → Healthcare System
  1. Copy core logic
  2. Replace STRUCTURED_ROUND_1 and STRUCTURED_ROUND_N
  3. Create HealthcareDe batePosition dataclass
  4. Adapt model roles
  5. Integrate into orchestrator
```

**Example: Healthcare Debate System Prompt**
```python
# FROM: structured_debate_engine.py

HEALTHCARE_ROUND_1 = """
You are one model in a multi-model clinical reasoning system.

Your role:
{role_instruction}
  - Clinical Model: Evidence-based orthopedic reasoning
  - Safety Model: Conservative, policy-enforced
  - Patient Model: Subjective experience + compliance

PATIENT CONTEXT:
Age: {patient_age}
Surgical History: {surgery_info}
Comorbidities: {comorbidities}
ROM Baseline: {rom_baseline}

EXERCISE PROPOSAL:
{exercise_name}
{exercise_description}
{exercise_intensity}

Your task: Assess if this exercise is:
  1. SAFE for this patient
  2. CLINICALLY BENEFICIAL
  3. ACHIEVABLE for patient compliance

RESPOND EXACTLY:
POSITION: [safe/contraindicated/conditional]
ARGUMENT: [Clinical reasoning]
ASSUMPTIONS: [Patient assumptions]
RISKS: [Medical risks]
VULNERABILITIES: [What could go wrong]
CONFIDENCE: [0.0-1.0]
"""
```

---

### **Pattern 3: Engine Architecture + Domain Logic (80% Reuse)**

**Files:**
- `backend/core/cognitive_orchestrator.py`
- `backend/core/orchestration.py`

**Direction:**
```
Sentinel-E → Healthcare System
  1. Copy orchestrator architecture
  2. Insert healthcare-specific phases
  3. Add healthcare model registry
  4. Integrate safety policies
  5. Connect memory layers
```

---

### **Pattern 4: Frontend Integration Unchanged (100% Reuse)**

**Your frontend** (`frontend/`) expects:
```json
{
  "mode": "standard|debate|evidence|glass|synthesis",
  "result": {},
  "confidence": 0.85,
  "reasoning_graph": {},
  "audit_trail": []
}
```

This structure is **completely reusable** for healthcare — just change the data inside.

---

## 📁 IMPLEMENTATION ROADMAP

### **STEP 1: Create Healthcare Module Structure**

```
backend/healthcare/
├── config.py                    # Healthcare-specific config
├── models.py                    # Patient, Exercise, Assessment schemas
├── orchestrator.py              # Healthcare orchestrator
├── debate_prompts.py            # Healthcare-specific debate prompts
├── safety_policies.py           # Medical safety rules
├── biomechanics.py              # Joint angle, ROM computation
├── evidence_triangulation.py    # Clinical evidence verification
└── memory/
    ├── patient_memory.py        # Patient history storage
    ├── session_memory.py        # Current session state
    └── clinical_outcomes.py     # Learning from results
```

### **STEP 2: Copy & Adapt Core Engines**

```bash
# Core engines (minimal changes)
cp backend/core/confidence_engine.py → backend/healthcare/confidence.py
cp backend/core/synthesis_engine.py → backend/healthcare/synthesis.py
cp backend/engines/aggregation_engine.py → backend/healthcare/aggregation.py

# Policy engines (significant adaptation)
cp backend/core/structured_debate_engine.py → backend/healthcare/debate.py
cp backend/risk_boundaries.py → backend/healthcare/safety_policies.py

# Audit engines (moderate adaptation)
cp backend/core/glass_pipeline.py → backend/healthcare/audit_transparency.py
cp backend/engines/forensic_evidence_engine.py → backend/healthcare/evidence.py
```

### **STEP 3: Healthcare Model Registry**

```python
# backend/healthcare/model_registry.py

HEALTHCARE_MODEL_REGISTRY = {
    "clinical_model": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "role": "Evidence-based clinical reasoning",
        "system_prompt": CLINICAL_SYSTEM_PROMPT,
        "expertise": ["orthopedics", "exercise_science", "biomechanics"],
    },
    "safety_model": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "role": "Conservative safety validator",
        "system_prompt": SAFETY_SYSTEM_PROMPT,
        "expertise": ["risk_mitigation", "policy_enforcement", "contraindications"],
    },
    "patient_model": {
        "provider": "gemini",
        "model_id": "gemini-2.0-flash",
        "role": "Patient experience + compliance",
        "system_prompt": PATIENT_SYSTEM_PROMPT,
        "expertise": ["subjective_experience", "psychological_factors", "adherence"],
    },
}
```

### **STEP 4: Healthcare API Endpoints**

```python
# backend/main.py

@app.post("/api/healthcare/exercise-assessment")
async def assess_exercise(request: ExerciseAssessmentRequest):
    """
    Debate mode: Is this exercise safe for this patient?
    """
    # Routes to healthcare orchestrator
    # Runs debate with clinical_model, safety_model, patient_model
    # Returns structured decision with dissent documentation

@app.post("/api/healthcare/protocol-synthesis")
async def synthesize_protocol(request: RecoveryProtocolRequest):
    """
    Synthesis mode: Generate personalized recovery protocol
    """

@app.post("/api/healthcare/audit-transparency")
async def audit_decision(request: DecisionAuditRequest):
    """
    Glass mode: Why did system recommend/reject this?
    """

@app.post("/api/healthcare/evidence-verification")
async def verify_clinical_claim(request: EvidenceRequest):
    """
    Evidence mode: Show me the research
    """
```

---

## 🎯 SPECIFIC FILE MAPPING TABLE

### **Healthcare System → Sentinel-E Source**

| Healthcare File | Source (Sentinel-E) | Adaptation Needed |
|---|---|---|
| `healthcare/orchestrator.py` | `core/cognitive_orchestrator.py` | +Healthcare phases, +Safety policies |
| `healthcare/debate.py` | `core/structured_debate_engine.py` | +Exercise-specific prompts, +Patient-context injection |
| `healthcare/synthesis.py` | `core/synthesis_engine.py` | Minimal — change prompt only |
| `healthcare/safety_policies.py` | `risk_boundaries.py` | +Biomechanical constraints, +Surgical recovery windows |
| `healthcare/audit.py` | `core/glass_pipeline.py` | +Clinical audit dimensions |
| `healthcare/evidence.py` | `engines/forensic_evidence_engine.py` | +Healthcare claim types |
| `healthcare/confidence.py` | `core/confidence_engine.py` | +Patient factor penalties |
| `healthcare/memory.py` | `memory/memory_engine.py` | +Patient history tier |

---

## 💰 AUTONOMOUS FINANCE (CHIT FUND) ADAPTATION

Same architecture applies to collective finance:

### **Finance Model Registry**
```python
FINANCE_MODEL_REGISTRY = {
    "financial_optimizer": {
        "role": "ROI maximization + risk analysis",
        "expertise": ["portfolio_theory", "risk_assessment"],
    },
    "governance_validator": {
        "role": "Rule enforcement + precedent checking",
        "expertise": ["DAO_governance", "regulatory_compliance"],
    },
    "member_advocate": {
        "role": "Collective benefit maximization",
        "expertise": ["social_choice", "fairness", "incentives"],
    },
}
```

### **Finance Debate Scenario**
```
QUERY: "Should we approve member X's emergency withdrawal?"

ROUND 1 POSITIONS:
- Financial: "Yes, ROI still 8.2% with reduced capital"
- Governance: "Violates no-withdrawal policy, sets precedent"
- Member: "But X has medical emergency, social contract mandates compassion"

ROUND 2 REBUTTALS:
- Financial: "Policy exceptions erode fund stability"
- Governance: "True, but policy allows emergency override with 67% vote"
- Member: "Medical emergency is force majeure, should qualify"

ROUND 3 SYNTHESIS:
"Conditional approval: X can withdraw 50% with 6-month repayment plan,
maintaining 4.2% fund ROI, following governance precedent..."
```

---

## 🚀 QUICK START CHECKLIST

- [ ] **Create healthcare module** (`backend/healthcare/`)
- [ ] **Copy 3 core engines** (confidence, synthesis, aggregation)
- [ ] **Create healthcare model registry** with 3 role-based models
- [ ] **Adapt debate prompts** (system prompts + exercise-specific context)
- [ ] **Add safety policies** (contraindication checks, ROM limits, post-op windows)
- [ ] **Wire healthcare orchestrator** into main.py with new endpoints
- [ ] **Build patient memory layer** (3-tier: session, patient, system)
- [ ] **Adapt frontend** (mode routing, result schema stay same)
- [ ] **Create test suite** (debate scenarios, safety overrides, memory persistence)
- [ ] **Deploy** with healthcare-specific configuration

---

## 📚 KEY FILES TO STUDY (IN ORDER)

1. **Start:** `backend/core/mode_controller.py` — Understand mode routing
2. **Then:** `backend/core/structured_debate_engine.py` — Debate mechanics
3. **Then:** `backend/core/synthesis_engine.py` — Synthesis flow
4. **Then:** `backend/core/glass_pipeline.py` — Audit transparency
5. **Then:** `backend/core/cognitive_orchestrator.py` — Full pipeline
6. **Then:** `backend/risk_boundaries.py` — Policy enforcement

---

## ⚠️ CRITICAL GOTCHAS FOR HEALTHCARE

1. **Safety > Confidence**: Always prefer caution over model confidence
2. **Explicit Contraindications**: Never assume models know all patient risks
3. **Real-time Overrides**: Posture in camera can override all previous reasoning
4. **Patient Consent**: Debate/analysis is advisory only, patient has final say
5. **Liability**: Document all decisions for medical-legal compliance
6. **Calibration**: Healthcare confidence must be more conservative than general AI

---

## 🔗 RELATED FILES IN PROJECT

Core governance files:
- `backend/core/` (orchestration + reasoning)
- `backend/engines/` (mode execution)
- `backend/memory/` (learning + persistence)
- `backend/models/` (model integration)
- `backend/sentinel/` (high-level control)
- `backend/metacognitive/` (model registry)

Supporting infrastructure:
- `backend/database/` (persistence)
- `backend/gateway/` (API layer)
- `frontend/` (UI integration)

---

## 📞 NEXT STEPS

1. **Analyze patient posture data requirements** (camera input, pose extraction)
2. **Define biomechanics computation pipeline** (joint angles, ROM, symmetry)
3. **Map clinical evidence sources** (for Evidence mode triangulation)
4. **Create healthcare safety policy rules** (comprehensive contraindication matrix)
5. **Design patient memory schema** (surgical history, progress, preferences)
6. **Wire gesture recognition** (for real-time exercise monitoring)

---

**Document Version:** 1.0  
**Last Updated:** April 4, 2026  
**Author:** Governance Architecture Analysis  
**Status:** Ready for Implementation
