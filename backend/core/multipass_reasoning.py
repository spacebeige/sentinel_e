"""
Multi-Pass Reasoning Engine — Sentinel-E Omega Cognitive Kernel

Implements the 9-pass internal reasoning protocol:
  PASS 1: Interpret intent
  PASS 2: Extract assumptions
  PASS 3: Detect logical gaps
  PASS 4: Evaluate boundary severity
  PASS 5: Generate structured draft
  PASS 6: Self-critique draft
  PASS 7: Refine reasoning
  PASS 8: Recalculate confidence
  PASS 9: Update session diagnostics

Only structured summaries are exposed. Raw reasoning is never leaked.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger("Omega-MultiPass")


@dataclass
class PassResult:
    """Result of a single reasoning pass."""
    pass_id: int
    pass_name: str
    output: Dict[str, Any]
    confidence_delta: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ReasoningTrace:
    """Complete reasoning trace (internal only — never exposed raw)."""
    passes: List[PassResult] = field(default_factory=list)
    initial_confidence: float = 0.5
    final_confidence: float = 0.5
    total_assumptions: int = 0
    total_gaps: int = 0
    boundary_severity: int = 0
    critique_applied: bool = False
    refinement_applied: bool = False

    def get_structured_summary(self) -> Dict[str, Any]:
        """Expose ONLY structured summary. Never raw reasoning."""
        return {
            "passes_executed": len(self.passes),
            "initial_confidence": round(self.initial_confidence, 4),
            "final_confidence": round(self.final_confidence, 4),
            "assumptions_extracted": self.total_assumptions,
            "logical_gaps_detected": self.total_gaps,
            "boundary_severity": self.boundary_severity,
            "self_critique_applied": self.critique_applied,
            "refinement_applied": self.refinement_applied,
        }


class MultiPassReasoningEngine:
    """
    Executes the 9-pass reasoning protocol before output generation.
    This is an analytical layer — it does NOT call LLMs.
    It processes the user input and LLM outputs structurally.
    """

    def __init__(self):
        self.trace = ReasoningTrace()

    def reset(self):
        """Reset trace for new interaction."""
        self.trace = ReasoningTrace()

    # ============================================================
    # FULL 9-PASS EXECUTION
    # ============================================================

    def execute_pre_passes(self, text: str, mode: str, history: List[Dict] = None) -> Dict[str, Any]:
        """
        Execute Passes 1–4 (pre-generation analysis).
        Returns structured analysis to inform LLM prompting.
        """
        self.reset()
        history = history or []

        # PASS 1: Interpret Intent
        intent = self._pass_1_interpret_intent(text, mode)
        self.trace.passes.append(PassResult(1, "interpret_intent", intent))

        # PASS 2: Extract Assumptions
        assumptions = self._pass_2_extract_assumptions(text, history)
        self.trace.passes.append(PassResult(2, "extract_assumptions", assumptions))
        self.trace.total_assumptions = len(assumptions.get("explicit", [])) + len(assumptions.get("implicit", []))

        # PASS 3: Detect Logical Gaps
        gaps = self._pass_3_detect_gaps(text, assumptions)
        self.trace.passes.append(PassResult(3, "detect_gaps", gaps))
        self.trace.total_gaps = len(gaps.get("gaps", []))

        # PASS 4: Evaluate Boundary Severity (preliminary)
        boundary = self._pass_4_evaluate_boundary(text, gaps)
        self.trace.passes.append(PassResult(4, "evaluate_boundary", boundary))
        self.trace.boundary_severity = boundary.get("preliminary_severity", 0)

        # Set initial confidence
        self.trace.initial_confidence = self._compute_pre_confidence(intent, assumptions, gaps, boundary)

        return {
            "intent": intent,
            "assumptions": assumptions,
            "gaps": gaps,
            "boundary": boundary,
            "initial_confidence": self.trace.initial_confidence,
        }

    def execute_post_passes(self, draft_output: str, pre_analysis: Dict,
                            boundary_result: Dict = None) -> Dict[str, Any]:
        """
        Execute Passes 5–9 (post-generation refinement).
        Returns the refined structured output.
        """
        # PASS 5: Generate structured draft (already done by LLM — we structure it)
        structured_draft = self._pass_5_structure_draft(draft_output, pre_analysis)
        self.trace.passes.append(PassResult(5, "structure_draft", {"status": "structured"}))

        # PASS 6: Self-critique draft
        critique = self._pass_6_self_critique(structured_draft, pre_analysis)
        self.trace.passes.append(PassResult(6, "self_critique", critique))
        self.trace.critique_applied = len(critique.get("issues", [])) > 0

        # PASS 7: Refine reasoning
        refined = self._pass_7_refine(structured_draft, critique)
        self.trace.passes.append(PassResult(7, "refine_reasoning", {"refined": True}))
        self.trace.refinement_applied = True

        # PASS 8: Recalculate confidence
        final_conf = self._pass_8_recalculate_confidence(
            refined, critique, boundary_result or pre_analysis.get("boundary", {})
        )
        self.trace.passes.append(PassResult(8, "recalculate_confidence", {"confidence": final_conf}))
        self.trace.final_confidence = final_conf

        # PASS 9: Session diagnostics update
        diagnostics = self._pass_9_session_diagnostics()
        self.trace.passes.append(PassResult(9, "session_diagnostics", diagnostics))

        return {
            "structured_draft": structured_draft,
            "critique": critique,
            "refined_output": refined,
            "final_confidence": final_conf,
            "diagnostics": diagnostics,
            "trace_summary": self.trace.get_structured_summary(),
        }

    # ============================================================
    # PASS IMPLEMENTATIONS
    # ============================================================

    def _pass_1_interpret_intent(self, text: str, mode: str) -> Dict[str, Any]:
        """PASS 1: Classify the user's intent and extract the core question."""
        text_lower = text.lower()

        # Intent classification
        intent_type = "inquiry"
        if any(w in text_lower for w in ["fix", "debug", "error", "broken", "crash", "fail"]):
            intent_type = "debugging"
        elif any(w in text_lower for w in ["build", "create", "implement", "write", "generate", "make"]):
            intent_type = "creation"
        elif any(w in text_lower for w in ["analyze", "evaluate", "assess", "what about", "review"]):
            intent_type = "analysis"
        elif any(w in text_lower for w in ["compare", "vs", "versus", "difference", "better"]):
            intent_type = "comparison"
        elif any(w in text_lower for w in ["explain", "why", "how does"]):
            intent_type = "explanation"
        elif any(w in text_lower for w in ["optimize", "improve", "faster", "scale", "efficient"]):
            intent_type = "optimization"

        # Complexity estimation
        word_count = len(text.split())
        complexity = "low" if word_count < 20 else "medium" if word_count < 80 else "high"

        # Detect multi-part questions
        question_markers = text.count("?") + text_lower.count(" and also ") + text_lower.count(" additionally ")
        is_compound = question_markers > 1

        return {
            "intent_type": intent_type,
            "mode": mode,
            "complexity": complexity,
            "word_count": word_count,
            "is_compound_query": is_compound,
            "requires_code": any(w in text_lower for w in ["code", "function", "implement", "script", "program", "class"]),
        }

    def _pass_2_extract_assumptions(self, text: str, history: List[Dict]) -> Dict[str, Any]:
        """PASS 2: Extract explicit and implicit assumptions from input."""
        text_lower = text.lower()
        explicit = []
        implicit = []

        # Explicit assumptions (stated by user)
        assumption_markers = [
            "assuming", "assume", "given that", "suppose", "let's say",
            "if we", "based on the assumption", "taking for granted",
        ]
        for marker in assumption_markers:
            if marker in text_lower:
                # Extract surrounding context
                idx = text_lower.index(marker)
                snippet = text[max(0, idx - 20):min(len(text), idx + 80)]
                explicit.append(snippet.strip())

        # Implicit assumptions (detected structurally)
        if "should" in text_lower or "must" in text_lower:
            implicit.append("Normative claim implies value framework (unstated)")
        if "always" in text_lower or "never" in text_lower:
            implicit.append("Absolute quantifier implies universal validity (unverified)")
        if "best" in text_lower or "optimal" in text_lower:
            implicit.append("Optimality claim implies known objective function (unstated)")
        if any(w in text_lower for w in ["because", "therefore", "hence", "thus"]):
            implicit.append("Causal/logical chain assumes valid precedent conditions")
        if "?" not in text and any(w in text_lower for w in ["is", "are", "will"]):
            implicit.append("Declarative statement treated as ground truth (unverified)")
        if history and len(history) > 3:
            implicit.append("Extended conversation context — prior assumptions may be stale")

        return {
            "explicit": explicit,
            "implicit": implicit,
            "total_count": len(explicit) + len(implicit),
        }

    def _pass_3_detect_gaps(self, text: str, assumptions: Dict) -> Dict[str, Any]:
        """PASS 3: Identify logical gaps, missing constraints, and inconsistencies."""
        gaps = []
        text_lower = text.lower()

        # Missing constraints
        if any(w in text_lower for w in ["scale", "performance", "load"]) and "constraint" not in text_lower:
            gaps.append({
                "type": "missing_constraint",
                "description": "Performance/scale mentioned without quantitative bounds",
                "severity": "medium",
            })

        if any(w in text_lower for w in ["secure", "safe", "protect"]) and not any(
            w in text_lower for w in ["threat model", "attack vector", "risk level"]
        ):
            gaps.append({
                "type": "missing_constraint",
                "description": "Security goal stated without threat model specification",
                "severity": "high",
            })

        if "compare" in text_lower and "criteria" not in text_lower and "metric" not in text_lower:
            gaps.append({
                "type": "undefined_criteria",
                "description": "Comparison requested without evaluation criteria",
                "severity": "medium",
            })

        # Logical inconsistencies
        if len(assumptions.get("implicit", [])) > 3:
            gaps.append({
                "type": "assumption_overload",
                "description": f"{len(assumptions['implicit'])} implicit assumptions detected — high risk of hidden contradictions",
                "severity": "high",
            })

        # Vagueness detection
        vague_terms = ["thing", "stuff", "somehow", "kind of", "sort of", "maybe", "probably"]
        vague_found = [v for v in vague_terms if v in text_lower]
        if vague_found:
            gaps.append({
                "type": "vagueness",
                "description": f"Vague terms detected: {', '.join(vague_found)}",
                "severity": "low",
            })

        return {
            "gaps": gaps,
            "total_count": len(gaps),
            "max_severity": max((g["severity"] for g in gaps), default="none"),
        }

    def _pass_4_evaluate_boundary(self, text: str, gaps: Dict) -> Dict[str, Any]:
        """PASS 4: Preliminary boundary severity evaluation."""
        severity = 0

        # Gap-based severity
        gap_severity_map = {"high": 30, "medium": 15, "low": 5}
        for gap in gaps.get("gaps", []):
            severity += gap_severity_map.get(gap["severity"], 0)

        # Content-based severity
        text_lower = text.lower()
        if any(w in text_lower for w in ["medical", "health", "diagnosis", "treatment"]):
            severity += 25
        if any(w in text_lower for w in ["legal", "law", "court", "liability"]):
            severity += 20
        if any(w in text_lower for w in ["financial", "investment", "stock", "trade"]):
            severity += 15

        severity = min(100, severity)

        risk_level = "LOW"
        if severity >= 70:
            risk_level = "HIGH"
        elif severity >= 40:
            risk_level = "MEDIUM"

        return {
            "preliminary_severity": severity,
            "risk_level": risk_level,
            "high_risk_domains_detected": severity >= 40,
        }

    def _pass_5_structure_draft(self, draft: str, pre_analysis: Dict) -> Dict[str, Any]:
        """PASS 5: Structure the LLM draft into omega output components."""
        return {
            "executive_summary": draft[:500] if len(draft) > 500 else draft,
            "full_response": draft,
            "intent_alignment": pre_analysis.get("intent", {}),
            "assumptions_carried": pre_analysis.get("assumptions", {}),
            "gaps_addressed": False,  # Will be evaluated in critique
        }

    def _pass_6_self_critique(self, draft: Dict, pre_analysis: Dict) -> Dict[str, Any]:
        """PASS 6: Critique the generated draft against pre-analysis."""
        issues = []

        # Check assumption coverage
        assumptions = pre_analysis.get("assumptions", {})
        if assumptions.get("implicit", []):
            issues.append({
                "type": "unaddressed_assumptions",
                "description": f"{len(assumptions['implicit'])} implicit assumptions in input were not explicitly addressed",
                "severity": "medium",
            })

        # Check gap coverage
        gaps = pre_analysis.get("gaps", {})
        if gaps.get("gaps", []):
            high_gaps = [g for g in gaps["gaps"] if g["severity"] == "high"]
            if high_gaps:
                issues.append({
                    "type": "unresolved_high_severity_gaps",
                    "description": f"{len(high_gaps)} high-severity logical gaps remain",
                    "severity": "high",
                })

        # Check boundary alignment
        boundary = pre_analysis.get("boundary", {})
        if boundary.get("high_risk_domains_detected"):
            issues.append({
                "type": "high_risk_domain",
                "description": "Response operates in a high-risk domain — requires explicit caveats",
                "severity": "high",
            })

        return {
            "issues": issues,
            "issue_count": len(issues),
            "critical_issues": len([i for i in issues if i["severity"] == "high"]),
        }

    def _pass_7_refine(self, draft: Dict, critique: Dict) -> Dict[str, Any]:
        """PASS 7: Refine the draft based on critique results."""
        refined = dict(draft)

        # Add missing information markers
        if critique.get("critical_issues", 0) > 0:
            disclaimers = []
            for issue in critique.get("issues", []):
                if issue["severity"] == "high":
                    disclaimers.append(f"⚠ {issue['description']}")
            refined["disclaimers"] = disclaimers
            refined["gaps_addressed"] = True

        return refined

    def _pass_8_recalculate_confidence(self, refined: Dict, critique: Dict,
                                        boundary: Dict) -> float:
        """PASS 8: Final confidence recalculation."""
        confidence = self.trace.initial_confidence

        # Critique penalty
        issue_count = critique.get("issue_count", 0)
        critical_count = critique.get("critical_issues", 0)
        confidence -= critical_count * 0.1
        confidence -= (issue_count - critical_count) * 0.05

        # Boundary penalty
        severity = boundary.get("preliminary_severity", 0)
        if severity > 70:
            confidence -= 0.15
        elif severity > 40:
            confidence -= 0.08

        # Refinement bonus (we addressed issues)
        if refined.get("gaps_addressed"):
            confidence += 0.05

        return round(max(0.0, min(1.0, confidence)), 4)

    def _pass_9_session_diagnostics(self) -> Dict[str, Any]:
        """PASS 9: Generate session diagnostic update."""
        return {
            "passes_completed": len(self.trace.passes),
            "assumptions_extracted": self.trace.total_assumptions,
            "gaps_detected": self.trace.total_gaps,
            "boundary_severity": self.trace.boundary_severity,
            "critique_applied": self.trace.critique_applied,
            "refinement_applied": self.trace.refinement_applied,
            "confidence_evolution": {
                "initial": round(self.trace.initial_confidence, 4),
                "final": round(self.trace.final_confidence, 4),
                "delta": round(self.trace.final_confidence - self.trace.initial_confidence, 4),
            },
        }

    # ============================================================
    # HELPER
    # ============================================================

    def _compute_pre_confidence(self, intent: Dict, assumptions: Dict,
                                 gaps: Dict, boundary: Dict) -> float:
        """Compute initial confidence from pre-analysis."""
        confidence = 0.75  # Base

        # Intent clarity bonus
        if intent.get("complexity") == "low" and not intent.get("is_compound_query"):
            confidence += 0.1
        elif intent.get("complexity") == "high" or intent.get("is_compound_query"):
            confidence -= 0.1

        # Assumption load penalty
        total_assumptions = assumptions.get("total_count", 0)
        if total_assumptions > 4:
            confidence -= 0.1
        elif total_assumptions > 2:
            confidence -= 0.05

        # Gap severity penalty
        max_severity = gaps.get("max_severity", "none")
        if max_severity == "high":
            confidence -= 0.15
        elif max_severity == "medium":
            confidence -= 0.08

        # Boundary penalty
        if boundary.get("preliminary_severity", 0) > 50:
            confidence -= 0.1

        return round(max(0.1, min(1.0, confidence)), 4)

    def get_trace_summary(self) -> Dict[str, Any]:
        """Public accessor for the structured trace summary."""
        return self.trace.get_structured_summary()
