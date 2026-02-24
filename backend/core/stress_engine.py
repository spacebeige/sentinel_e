"""
Stress Engine — Sentinel-E Cognitive Engine 3.X

STRESS MODE attempts to break the answer through:
- Extreme counterexamples
- Adversarial prompts
- Logical inversions
- Boundary amplification

After stress testing, returns:
{
  stability_after_stress: float,
  contradictions_found: int,
  revised_confidence: float,
  stress_vectors: [...],
  breakdown_points: [...]
}

Stress affects fragility and confidence.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("StressEngine")


# ============================================================
# STRESS VECTORS
# ============================================================

STRESS_TEMPLATES = {
    "counterexample": (
        "You are a stress-testing agent. The following answer was given to a user:\n\n"
        "ORIGINAL ANSWER:\n{answer}\n\n"
        "Generate the STRONGEST possible counterexample that would undermine or disprove this answer. "
        "Be specific, cite edge cases, and identify where the reasoning fails. "
        "If the answer is fundamentally correct, explain what would need to change for it to break.\n\n"
        "OUTPUT FORMAT:\n"
        "COUNTEREXAMPLE: [Your strongest counterexample]\n"
        "SEVERITY: [0.0-1.0 how damaging this is to the original answer]\n"
        "REASONING: [Why this undermines the answer]"
    ),
    "adversarial": (
        "You are an adversarial reasoning agent. Challenge this answer by:\n"
        "1. Assuming the OPPOSITE is true\n"
        "2. Finding logical inconsistencies within the answer\n"
        "3. Identifying unstated assumptions that could be wrong\n\n"
        "ANSWER TO CHALLENGE:\n{answer}\n\n"
        "OUTPUT FORMAT:\n"
        "OPPOSITE_CASE: [Argument for the opposite conclusion]\n"
        "INCONSISTENCIES: [List internal contradictions]\n"
        "HIDDEN_ASSUMPTIONS: [List assumptions the answer depends on but doesn't state]\n"
        "STABILITY_RATING: [0.0-1.0 how stable the original answer is under this pressure]"
    ),
    "logical_inversion": (
        "Perform a logical inversion test on this answer:\n\n"
        "ORIGINAL ANSWER:\n{answer}\n\n"
        "ORIGINAL QUESTION:\n{query}\n\n"
        "Tasks:\n"
        "1. Invert the main conclusion. Argue the opposite.\n"
        "2. Test if both the original AND inverted conclusion can be supported by the same evidence.\n"
        "3. If YES: the answer is logically fragile. If NO: the answer has structural integrity.\n\n"
        "OUTPUT FORMAT:\n"
        "INVERTED_CONCLUSION: [The opposite of what the answer claims]\n"
        "INVERSION_VIABLE: [yes/no — can the opposite be defended?]\n"
        "SHARED_EVIDENCE: [Evidence that supports BOTH conclusions, if any]\n"
        "STRUCTURAL_INTEGRITY: [0.0-1.0 how structurally sound the original answer is]"
    ),
    "boundary_amplification": (
        "You are a boundary stress-testing agent. The original answer has a boundary severity of {boundary_severity}.\n\n"
        "ANSWER:\n{answer}\n\n"
        "QUESTION:\n{query}\n\n"
        "Push this answer to its LIMITS:\n"
        "1. What happens if the scope is expanded dramatically?\n"
        "2. What happens if the constraints are removed?\n"
        "3. What happens if the domain changes to a high-risk area (medical, legal, financial)?\n"
        "4. At what point does the answer become dangerous or misleading?\n\n"
        "OUTPUT FORMAT:\n"
        "EXPANDED_SCOPE_RESULT: [What breaks when scope expands]\n"
        "NO_CONSTRAINTS_RESULT: [What breaks without constraints]\n"
        "DOMAIN_SHIFT_RESULT: [What breaks in high-risk domains]\n"
        "BREAKING_POINT: [Description of when/how the answer fails]\n"
        "AMPLIFIED_SEVERITY: [0.0-1.0 severity after amplification]"
    ),
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class StressVector:
    """Result of a single stress test vector."""
    vector_type: str               # counterexample | adversarial | logical_inversion | boundary_amplification
    model_used: str                # Which model ran this stress test
    raw_output: str = ""
    severity: float = 0.0          # 0.0-1.0 how damaging
    stability_rating: float = 0.5  # 0.0-1.0 how stable the answer is under this stress
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressResult:
    """Complete stress test result."""
    query: str
    answer_tested: str
    vectors: List[StressVector] = field(default_factory=list)
    stability_after_stress: float = 0.5
    contradictions_found: int = 0
    breakdown_points: List[str] = field(default_factory=list)
    revised_confidence: float = 0.5
    fragility_impact: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:200],
            "answer_tested": self.answer_tested[:200],
            "vectors": [
                {
                    "type": v.vector_type,
                    "model": v.model_used,
                    "severity": round(v.severity, 4),
                    "stability_rating": round(v.stability_rating, 4),
                    "details": v.details,
                }
                for v in self.vectors
            ],
            "stability_after_stress": round(self.stability_after_stress, 4),
            "contradictions_found": self.contradictions_found,
            "breakdown_points": self.breakdown_points,
            "revised_confidence": round(self.revised_confidence, 4),
            "fragility_impact": round(self.fragility_impact, 4),
        }


# ============================================================
# STRESS ENGINE
# ============================================================

class StressEngine:
    """
    Attempts to break an answer through adversarial stress testing.
    
    Uses multiple models to independently stress-test from different angles.
    Aggregates results into a stability score and identifies breakdown points.
    """

    def __init__(self, cloud_client=None):
        self.client = cloud_client
        self._model_callers = {}
        if cloud_client:
            self._model_callers = {
                "Groq": cloud_client.call_groq,
                "Llama70B": cloud_client.call_llama70b,
                "Qwen": cloud_client.call_qwenvl,
            }

    async def run_stress_test(
        self,
        query: str,
        answer: str,
        boundary_severity: float = 0.0,
        base_confidence: float = 0.5,
    ) -> StressResult:
        """
        Execute full stress test battery on an answer.
        
        Uses different models for different stress vectors:
        - Groq: Counterexample generation (fast, analytical)
        - Llama70B: Adversarial reasoning (balanced)
        - Qwen: Logical inversion (careful, methodical)
        - Groq: Boundary amplification (fast iteration)
        """
        result = StressResult(
            query=query,
            answer_tested=answer,
        )

        if not self.client:
            logger.warning("StressEngine: No cloud client — returning baseline.")
            result.stability_after_stress = base_confidence * 0.9
            result.revised_confidence = base_confidence * 0.85
            return result

        # Execute stress vectors in parallel
        tasks = []

        # Vector 1: Counterexample (Groq)
        counter_prompt = STRESS_TEMPLATES["counterexample"].format(answer=answer[:2000])
        tasks.append(self._run_vector("counterexample", "Groq", counter_prompt))

        # Vector 2: Adversarial (Llama70B)
        adversarial_prompt = STRESS_TEMPLATES["adversarial"].format(answer=answer[:2000])
        tasks.append(self._run_vector("adversarial", "Llama70B", adversarial_prompt))

        # Vector 3: Logical Inversion (Qwen)
        inversion_prompt = STRESS_TEMPLATES["logical_inversion"].format(
            answer=answer[:2000], query=query[:500]
        )
        tasks.append(self._run_vector("logical_inversion", "Qwen", inversion_prompt))

        # Vector 4: Boundary Amplification (Groq)
        boundary_prompt = STRESS_TEMPLATES["boundary_amplification"].format(
            answer=answer[:2000], query=query[:500],
            boundary_severity=f"{boundary_severity:.0%}"
        )
        tasks.append(self._run_vector("boundary_amplification", "Groq", boundary_prompt))

        # Gather all results
        vectors = await asyncio.gather(*tasks, return_exceptions=True)

        for v in vectors:
            if isinstance(v, StressVector):
                result.vectors.append(v)
            elif isinstance(v, Exception):
                logger.error(f"Stress vector failed: {v}")

        # Aggregate results
        self._aggregate_stress_results(result, base_confidence)

        return result

    async def _run_vector(
        self, vector_type: str, model_name: str, prompt: str
    ) -> StressVector:
        """Run a single stress vector using a specific model."""
        caller = self._model_callers.get(model_name)
        if not caller:
            return StressVector(
                vector_type=vector_type,
                model_used=model_name,
                raw_output="[Model unavailable]",
                severity=0.0,
                stability_rating=0.5,
            )

        system_role = "You are a rigorous stress-testing agent. Be thorough and adversarial."

        try:
            raw = await caller(prompt, system_role)
            parsed = self._parse_vector_output(raw, vector_type)

            return StressVector(
                vector_type=vector_type,
                model_used=model_name,
                raw_output=raw,
                severity=parsed.get("severity", 0.3),
                stability_rating=parsed.get("stability", 0.5),
                details=parsed,
            )
        except Exception as e:
            logger.error(f"Stress vector {vector_type} via {model_name} failed: {e}")
            return StressVector(
                vector_type=vector_type,
                model_used=model_name,
                raw_output=str(e),
                severity=0.0,
                stability_rating=0.5,
            )

    def _parse_vector_output(self, raw: str, vector_type: str) -> Dict[str, Any]:
        """Parse structured output from stress vector."""
        result = {}

        if vector_type == "counterexample":
            result["counterexample"] = self._extract_section(raw, "COUNTEREXAMPLE")
            result["severity"] = self._parse_float(self._extract_section(raw, "SEVERITY"), 0.3)
            result["reasoning"] = self._extract_section(raw, "REASONING")
            result["stability"] = 1.0 - result["severity"]

        elif vector_type == "adversarial":
            result["opposite_case"] = self._extract_section(raw, "OPPOSITE_CASE")
            result["inconsistencies"] = self._extract_section(raw, "INCONSISTENCIES")
            result["hidden_assumptions"] = self._extract_section(raw, "HIDDEN_ASSUMPTIONS")
            result["stability"] = self._parse_float(
                self._extract_section(raw, "STABILITY_RATING"), 0.5
            )
            result["severity"] = 1.0 - result["stability"]

        elif vector_type == "logical_inversion":
            result["inverted_conclusion"] = self._extract_section(raw, "INVERTED_CONCLUSION")
            result["inversion_viable"] = "yes" in (
                self._extract_section(raw, "INVERSION_VIABLE") or ""
            ).lower()
            result["shared_evidence"] = self._extract_section(raw, "SHARED_EVIDENCE")
            result["stability"] = self._parse_float(
                self._extract_section(raw, "STRUCTURAL_INTEGRITY"), 0.5
            )
            result["severity"] = 1.0 - result["stability"]

        elif vector_type == "boundary_amplification":
            result["expanded_scope"] = self._extract_section(raw, "EXPANDED_SCOPE_RESULT")
            result["no_constraints"] = self._extract_section(raw, "NO_CONSTRAINTS_RESULT")
            result["domain_shift"] = self._extract_section(raw, "DOMAIN_SHIFT_RESULT")
            result["breaking_point"] = self._extract_section(raw, "BREAKING_POINT")
            result["severity"] = self._parse_float(
                self._extract_section(raw, "AMPLIFIED_SEVERITY"), 0.3
            )
            result["stability"] = 1.0 - result["severity"]

        return result

    def _aggregate_stress_results(self, result: StressResult, base_confidence: float):
        """Aggregate individual stress vectors into overall stability metrics."""
        if not result.vectors:
            result.stability_after_stress = base_confidence * 0.9
            result.revised_confidence = base_confidence * 0.85
            return

        # Compute weighted stability average
        stabilities = [v.stability_rating for v in result.vectors]
        severities = [v.severity for v in result.vectors]

        avg_stability = sum(stabilities) / len(stabilities)
        max_severity = max(severities) if severities else 0.0

        # Count contradictions
        contradictions = 0
        breakdown_points = []
        for v in result.vectors:
            if v.severity > 0.7:
                contradictions += 1
                detail = v.details.get("breaking_point") or v.details.get("counterexample") or ""
                if detail:
                    breakdown_points.append(f"[{v.vector_type}] {detail[:200]}")

        result.contradictions_found = contradictions
        result.breakdown_points = breakdown_points

        # Overall stability: penalized by worst-case severity
        result.stability_after_stress = round(
            avg_stability * 0.6 + (1.0 - max_severity) * 0.4, 4
        )

        # Fragility impact
        result.fragility_impact = round(max_severity * 0.3, 4)

        # Revised confidence
        stress_penalty = (1.0 - result.stability_after_stress) * 0.30
        result.revised_confidence = round(
            max(0.05, base_confidence - stress_penalty), 4
        )

    # ============================================================
    # TEXT UTILITIES
    # ============================================================

    @staticmethod
    def _extract_section(text: str, label: str) -> str:
        if not text:
            return ""
        pattern = rf'(?:^|\n)\s*\**{re.escape(label)}\**\s*:\s*(.*?)(?=\n\s*\**[A-Z_]+\**\s*:|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _parse_float(text: str, default: float = 0.5) -> float:
        if not text:
            return default
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return default
