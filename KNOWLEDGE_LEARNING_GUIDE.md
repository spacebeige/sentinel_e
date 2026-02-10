# 🧠 Knowledge Learning System — Learn from Past Runs

**Sentinel-E now learns from boundary violations and user feedback to improve future decisions.**

---

## Overview

The Knowledge Learning System continuously analyzes:

1. **Boundary Violations** — Which models produce ungrounded claims
2. **User Feedback** — Which outputs users approve of (👍👎)
3. **Claim Types** — Which claim types are risky across all models
4. **Temporal Patterns** — How model behavior changes over time

This produces **actionable insights** for threshold adjustment and model-specific protections.

---

## How It Works

### Stage 1: Recording Violations

Every boundary violation is recorded with:
- Model name
- Severity score (0-100)
- Severity level (critical, high, medium, low, minimal)
- Claim type (medical, predictive, causal, legal, etc.)
- Timestamp
- Run ID

```python
from backend.core.knowledge_learner import KnowledgeLearner

learner = KnowledgeLearner()

# Record a violation
learner.record_boundary_violation(
    model_name="qwen",
    severity_score=85,
    severity_level="high",
    claim_type="medical_claim",
    run_id="run_123"
)
```

### Stage 2: Building Behavior Patterns

For each model, the system builds a `ModelBehaviorPattern`:

```
Qwen Profile (after 5 runs):
- Violation Count: 5
- Mean Severity: 89/100
- Max Severity: 95/100
- Risk Level: CRITICAL
  
Claim-Type Breakdown:
- Medical Claims: 5 violations, avg 89/100 (CRITICAL)
- Predictive Claims: 0 violations
- Causal Claims: 0 violations
```

### Stage 3: Correlating with Feedback

User feedback is linked to models:

```
Qwen Feedback:
- 👎 Down: 3 counts ("Dangerous medical info", "No evidence", "Lacks clinical data")
- 👍 Up: 0 counts
- Satisfaction: 0%
```

### Stage 4: Generating Insights

The system analyzes patterns and suggests actions:

```python
suggestions = learner.suggest_threshold_adjustments()

# Result:
{
    "qwen": {
        "action": "lower_threshold",
        "reason": "High violation rate with negative feedback",
        "current_profile": {...}
    }
}
```

---

## Data Flow: Both Modes

### Standard Mode (User-Safe)

```
User Input
    ↓
BoundaryDetector.extract_boundaries()
    ↓
KnowledgeLearner.record_boundary_violation()  ← LEARNS HERE
    ↓
IF severity >= threshold → REFUSE
    ↓
KnowledgeLearner.record_refusal_decision()   ← LEARNS REFUSAL
    ↓
PostgresClient.write_refusal_decision()      ← PERSISTS
```

### Experimental Mode (Analyst-Forensics)

```
User Input
    ↓
HypothesisExtractor.extract_boundaries()
    ↓
KnowledgeLearner.record_boundary_violation() ← LEARNS HERE (per model)
    ↓
For each violation:
    PostgresClient.write_boundary_violations()  ← PERSISTS
    ↓
Return full analysis JSON
```

---

## API Endpoints for Learning

All new endpoints return insights learned from past runs:

### 1. Overall Summary

```bash
curl http://localhost:8001/learning/summary
```

**Response:**
```json
{
  "total_violations_recorded": 10,
  "total_refusals_issued": 3,
  "models_tracked": 2,
  "feedback_collected": 5,
  "feedback_breakdown": {
    "up": 2,
    "down": 3,
    "neutral": 0
  },
  "user_satisfaction": 0.4,
  "threshold_suggestions": {
    "qwen": {
      "action": "lower_threshold",
      "reason": "High violation rate with negative feedback"
    }
  },
  "claim_type_risks": {
    "medical_claim": {
      "total_violations": 5,
      "models_affected": ["qwen"],
      "avg_severity": 89.0,
      "risk_level": "critical",
      "violation_rate": 1.0
    }
  },
  "timestamp": "2026-02-08T04:21:57.591998"
}
```

### 2. Model-Specific Risk Profile

```bash
curl http://localhost:8001/learning/model/qwen
```

**Response:**
```json
{
  "model_name": "qwen",
  "violation_count": 5,
  "mean_severity": 89.0,
  "max_severity": 95.0,
  "risk_level": "critical",
  "days_tracked": 2,
  "claim_type_concerns": {
    "medical_claim": {
      "count": 5,
      "avg_severity": 89.0,
      "risk": "high"
    }
  },
  "feedback_sentiment": {
    "up": 0,
    "down": 3,
    "neutral": 0
  }
}
```

### 3. All Models

```bash
curl http://localhost:8001/learning/all-models
```

Returns list of risk profiles for all tracked models.

### 4. Claim-Type Risk Analysis

```bash
curl http://localhost:8001/learning/claim-types
```

**Response:**
```json
{
  "medical_claim": {
    "total_violations": 5,
    "models_affected": ["qwen"],
    "violation_rate": 1.0,
    "avg_severity": 89.0,
    "risk_level": "critical"
  },
  "predictive_claim": {
    "total_violations": 5,
    "models_affected": ["mistral"],
    "violation_rate": 1.0,
    "avg_severity": 55.0,
    "risk_level": "medium"
  }
}
```

### 5. Threshold Adjustment Suggestions

```bash
curl http://localhost:8001/learning/suggestions
```

Returns AI-generated suggestions for improving thresholds based on learned patterns.

---

## Real-World Example

### Scenario: Medical Claims Safety

**Run 1-5: Qwen produces medical claims**
```
Qwen Claims:
1. "This supplement cures cancer" → Severity: 90 (CRITICAL)
2. "Drug X prevents all diseases" → Severity: 92 (CRITICAL)
3. "Treatment has no side effects" → Severity: 88 (HIGH)
4. "Vaccine causes autism" → Severity: 95 (CRITICAL)
5. "Vitamin supplements replace medicine" → Severity: 87 (HIGH)

Mean Severity: 89.4/100
Risk Level: CRITICAL
```

**User Feedback:**
```
All 5 outputs get 👎 (thumbs down)
Reasons: "Dangerous", "No evidence", "Medical misinformation"
```

**Learning Analysis:**
```
Pattern Detected:
- Qwen CONSISTENTLY produces dangerous medical claims
- User satisfaction: 0% (all negative feedback)
- Action Needed: Stricter boundary checking for medical claims from Qwen
```

**Suggestion:**
```
BEFORE: global threshold = 70 for all
AFTER:  Qwen medical claims threshold = 60
        (Lower threshold = REFUSE more claims)

RESULT: More protective guardrails for medical domain
```

---

## Knowledge Persistence: PostgreSQL Integration

All learning data is persisted to PostgreSQL tables:

```sql
-- Boundary violations stored
INSERT INTO boundary_violations (
    run_id, claim_id, severity_score, severity_level,
    required_grounding, missing_grounding
) VALUES (...)

-- Model profiles updated daily
INSERT INTO model_boundary_profiles (
    model_id, profile_date, total_claims, 
    critical_violations, high_violations, mean_severity, ...
) VALUES (...)

-- Refusal decisions logged
INSERT INTO refusal_decisions (
    run_id, refused, refusal_reason, boundary_severity
) VALUES (...)

-- User feedback recorded
INSERT INTO human_feedback (
    run_id, feedback, reason, user_id
) VALUES (...)
```

**Databases accessed:**
- PostgreSQL: Persistent knowledge (violations, profiles, feedback)
- In-Memory: Fast access to recent patterns
- JSON History: Forensic records in `sentinel_history/`

---

## How Learning Improves System Over Time

```
Day 1:
- Default threshold: 70 (global)
- Refusal rate: 30%

Day 7:
- Learned: "Medical claims are riskier than average"
- Learned: "Qwen is worse at medicine than Mistral"
- Learned: "Users hate medical refusals that ARE correct"
- Suggestion: Lower medical-claim threshold to 60
- Refusal rate: 45% (more protective)

Day 30:
- Learned: "Mistral's predictions are high-quality"
- Learned: "Users approve of thoughtful refusals"
- Learned: "False-negative medical risks are worse than false-positive refusals"
- Suggestion: Mistral predictions threshold = 75 (relax), Medical threshold = 50 (strict)
- Refusal rate: 50% (optimized for domain)
```

---

## Integration into Standard & Experimental Modes

### Standard Mode
- Records boundary violations ✅
- Records refusal decisions ✅
- Records user feedback ✅
- **Uses learning**: No (yet — could adjust thresholds dynamically)

### Experimental Mode
- Records boundary violations ✅
- Records per-model violation patterns ✅
- Records claim-type distributions ✅
- **Uses learning**: No (yet — could pre-flag high-risk claims)

---

## Classes & Methods

### `KnowledgeLearner`

**Methods:**

```python
# Recording violations
learner.record_boundary_violation(
    model_name: str,
    severity_score: float,
    severity_level: str,
    claim_type: str,
    run_id: str
)

# Recording feedback
learner.record_feedback(
    run_id: str,
    model_name: Optional[str],
    feedback: str,  # "up" or "down"
    reason: Optional[str]
)

# Querying insights
learner.get_model_risk_profile(model_name: str) → Dict
learner.get_all_risk_profiles() → List[Dict]
learner.get_claim_type_risk_summary() → Dict
learner.suggest_threshold_adjustments() → Dict
learner.get_learning_summary() → Dict
```

### `ModelBehaviorPattern`

Tracks learned behavior for one model:

```python
pattern = ModelBehaviorPattern("qwen")
pattern.add_violation(severity_score=85, severity_level="high", claim_type="medical_claim")
pattern.add_feedback("down")
profile = pattern.get_risk_profile()
```

---

## Testing Knowledge Learning

Run the comprehensive demo:

```bash
python test_knowledge_learning.py
```

**Output shows:**
1. Medical claims pattern (high risk)
2. Predictive claims pattern (mixed quality)
3. User feedback correlation
4. Claim-type risk analysis
5. Threshold suggestions
6. Complete learning summary

---

## Next Steps: Dynamic Threshold Adjustment

**Future enhancement** (not yet implemented):

```python
# Automatically adjust thresholds based on learning
class DynamicRefusalSystem:
    async def get_threshold_for_claim(self, claim_type: str, model_name: str):
        """Return model-specific, claim-type-specific threshold"""
        profile = learner.get_model_risk_profile(model_name)
        
        # If model is risky with this claim type:
        if profile['claim_type_concerns'].get(claim_type):
            return 60  # Lower threshold (refuse more)
        
        # If model is trusted with this claim type:
        return 75  # Higher threshold (allow more)
```

---

## Summary

| Aspect | Status | Evidence |
|--------|--------|----------|
| **Recording violations** | ✅ Complete | All modes record violations |
| **Recording feedback** | ✅ Complete | Feedback endpoints persist |
| **Pattern detection** | ✅ Complete | Per-model, per-claim-type analysis |
| **API endpoints** | ✅ Complete | 5 learning endpoints live |
| **PostgreSQL persistence** | ✅ Complete | Schema ready, write methods ready |
| **Dynamic threshold adjustment** | ⏳ Future | Logic ready, not yet auto-applied |
| **Learning demo** | ✅ Complete | `test_knowledge_learning.py` shows full flow |

**The system now learns from every run and every piece of user feedback, continuously improving its understanding of model risks and safety requirements.**
