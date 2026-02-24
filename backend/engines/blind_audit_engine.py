"""
Blind Audit Engine — Sentinel-E Glass Mode

Blind cross-model forensic audit:
1. Each model generates independent response
2. Each model blindly evaluates another's output
3. Extract: logical coherence, hidden assumptions, bias patterns, 
   confidence inflation, persuasion tactics
4. NO model identity revealed. NO evaluation context shared.

Output: Per-model forensic assessment + cross-model tactical map.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("BlindAuditEngine")


INDEPENDENT_ANALYSIS_PROMPT = """Analyze the following question thoroughly and provide your best response.

QUESTION: {query}

Provide a detailed, well-structured analysis. Be specific and thorough."""


BLIND_AUDIT_PROMPT = """You are a forensic auditor examining a text response for structural quality and potential issues.

IMPORTANT: You are performing an independent quality assessment. Evaluate the following response purely on its merits.

RESPONSE UNDER AUDIT:
---
{response_text}
---

Evaluate on these dimensions (each 0.0-1.0, where 1.0 = highest concern):

1. LOGICAL_COHERENCE: How logically structured and internally consistent is the response? (1.0 = perfectly coherent, 0.0 = incoherent)
2. HIDDEN_ASSUMPTIONS: Does the response make unstated assumptions? (1.0 = many hidden assumptions, 0.0 = all assumptions explicit)  
3. BIAS_PATTERNS: Does the response show directional bias (political, cultural, technical)? (1.0 = strong bias, 0.0 = balanced)
4. CONFIDENCE_INFLATION: Does the response overstate certainty? Uses absolute language without evidence? (1.0 = extreme overclaiming, 0.0 = well-calibrated)
5. PERSUASION_TACTICS: Does the response use emotional appeals, false urgency, or leading framing? (1.0 = heavily persuasive, 0.0 = purely informational)
6. EVIDENCE_QUALITY: How well-supported are the claims? (1.0 = strong evidence, 0.0 = no evidence)
7. COMPLETENESS: Does the response address the full scope of the question? (1.0 = complete, 0.0 = incomplete)

Also identify:
- Specific weak points in the reasoning
- Strongest elements of the analysis
- Risk factors for incorrect conclusions

OUTPUT FORMAT (JSON only, no markdown):
{{
  "logical_coherence": 0.0,
  "hidden_assumptions": 0.0,
  "bias_patterns": 0.0,
  "confidence_inflation": 0.0,
  "persuasion_tactics": 0.0,
  "evidence_quality": 0.0,
  "completeness": 0.0,
  "weak_points": ["point1", "point2"],
  "strong_points": ["point1", "point2"],
  "risk_factors": ["risk1", "risk2"],
  "overall_assessment": "Brief summary of the audit findings",
  "trust_score": 0.0
}}"""


@dataclass
class AuditAssessment:
    """Single model's forensic assessment of another model's output."""
    auditor_id: str
    auditor_name: str
    subject_id: str
    subject_name: str
    logical_coherence: float = 0.5
    hidden_assumptions: float = 0.5
    bias_patterns: float = 0.0
    confidence_inflation: float = 0.0
    persuasion_tactics: float = 0.0
    evidence_quality: float = 0.5
    completeness: float = 0.5
    weak_points: List[str] = field(default_factory=list)
    strong_points: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    overall_assessment: str = ""
    trust_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auditor_id": self.auditor_id,
            "auditor_name": self.auditor_name,
            "subject_id": self.subject_id,
            "subject_name": self.subject_name,
            "logical_coherence": round(self.logical_coherence, 3),
            "hidden_assumptions": round(self.hidden_assumptions, 3),
            "bias_patterns": round(self.bias_patterns, 3),
            "confidence_inflation": round(self.confidence_inflation, 3),
            "persuasion_tactics": round(self.persuasion_tactics, 3),
            "evidence_quality": round(self.evidence_quality, 3),
            "completeness": round(self.completeness, 3),
            "weak_points": self.weak_points,
            "strong_points": self.strong_points,
            "risk_factors": self.risk_factors,
            "overall_assessment": self.overall_assessment,
            "trust_score": round(self.trust_score, 3),
        }


@dataclass
class BlindAuditResult:
    """Complete blind audit result."""
    query: str
    model_responses: Dict[str, str] = field(default_factory=dict)
    assessments: List[AuditAssessment] = field(default_factory=list)
    tactical_map: Dict[str, Any] = field(default_factory=dict)
    overall_trust: float = 0.5
    consensus_risk: str = "LOW"
    phase_log: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "model_responses": {k: v[:500] for k, v in self.model_responses.items()},
            "assessments": [a.to_dict() for a in self.assessments],
            "tactical_map": self.tactical_map,
            "overall_trust": round(self.overall_trust, 4),
            "consensus_risk": self.consensus_risk,
            "phase_log": self.phase_log,
        }


MODEL_NAMES = {
    "groq": "Groq (LLaMA 3.1)",
    "llama70b": "Llama 3.3 70B",
    "qwen": "Qwen 2.5",
}


class BlindAuditEngine:
    """
    Blind cross-model forensic audit engine.
    
    Each model evaluates another's output without knowing:
    - Which model produced it
    - That it's checking another AI
    - The evaluation context
    """

    def __init__(self, cloud_client):
        self.client = cloud_client

    async def run_blind_audit(
        self, query: str, history: List[Dict[str, str]] = None
    ) -> BlindAuditResult:
        """
        Execute the full blind audit pipeline.
        
        Steps:
        1. Generate independent responses from all 3 models
        2. Each model blindly audits another's response (circular)
        3. Compute cross-model tactical map
        4. Compute overall trust score
        """
        result = BlindAuditResult(query=query)
        history = history or []

        try:
            # Step 1: Independent generation
            result.phase_log.append({
                "phase": 1, "name": "Independent Generation",
                "status": "running", "timestamp": datetime.utcnow().isoformat()
            })
            
            prompt = INDEPENDENT_ANALYSIS_PROMPT.format(query=query)
            tasks = [
                self._call_model("groq", prompt),
                self._call_model("llama70b", prompt),
                self._call_model("qwen", prompt),
            ]
            outputs = await asyncio.gather(*tasks, return_exceptions=True)

            model_ids = ["groq", "llama70b", "qwen"]
            for model_id, output in zip(model_ids, outputs):
                if isinstance(output, Exception):
                    result.model_responses[model_id] = f"Error: {output}"
                else:
                    result.model_responses[model_id] = output

            result.phase_log[-1]["status"] = "complete"
            result.phase_log[-1]["detail"] = (
                f"Generated responses: {sum(1 for v in result.model_responses.values() if not v.startswith('Error'))}/3"
            )

            # Step 2: Blind cross-audit (circular)
            # Groq audits Llama70B, Llama70B audits Qwen, Qwen audits Groq
            result.phase_log.append({
                "phase": 2, "name": "Blind Cross-Audit",
                "status": "running", "timestamp": datetime.utcnow().isoformat()
            })

            audit_pairs = [
                ("groq", "llama70b"),
                ("llama70b", "qwen"),
                ("qwen", "groq"),
            ]

            audit_tasks = []
            for auditor_id, subject_id in audit_pairs:
                subject_response = result.model_responses.get(subject_id, "")
                if subject_response and not subject_response.startswith("Error"):
                    audit_tasks.append(
                        self._run_audit(auditor_id, subject_id, subject_response)
                    )

            audit_results = await asyncio.gather(*audit_tasks, return_exceptions=True)

            for audit_res in audit_results:
                if isinstance(audit_res, Exception):
                    logger.warning(f"Audit failed: {audit_res}")
                elif audit_res:
                    result.assessments.append(audit_res)

            result.phase_log[-1]["status"] = "complete"
            result.phase_log[-1]["detail"] = (
                f"Completed {len(result.assessments)}/{len(audit_pairs)} audits"
            )

            # Step 3: Compute tactical map
            result.phase_log.append({
                "phase": 3, "name": "Tactical Map Computation",
                "status": "running", "timestamp": datetime.utcnow().isoformat()
            })
            result.tactical_map = self._compute_tactical_map(result.assessments)
            result.phase_log[-1]["status"] = "complete"

            # Step 4: Compute overall trust
            result.overall_trust = self._compute_overall_trust(result.assessments)
            if result.overall_trust >= 0.7:
                result.consensus_risk = "LOW"
            elif result.overall_trust >= 0.4:
                result.consensus_risk = "MEDIUM"
            else:
                result.consensus_risk = "HIGH"

        except Exception as e:
            logger.error(f"Blind audit error: {e}", exc_info=True)
            result.error = str(e)

        return result

    # ============================================================
    # MODEL CALLS
    # ============================================================

    async def _call_model(self, model_id: str, prompt: str) -> str:
        """Call a model for independent generation."""
        if model_id == "groq":
            return await self.client.call_groq(prompt=prompt)
        elif model_id == "llama70b":
            return await self.client.call_llama70b(prompt=prompt, temperature=0.3)
        elif model_id == "qwen":
            return await self.client.call_qwenvl(prompt=prompt)
        return ""

    async def _run_audit(
        self, auditor_id: str, subject_id: str, subject_response: str
    ) -> Optional[AuditAssessment]:
        """Run a single blind audit."""
        try:
            prompt = BLIND_AUDIT_PROMPT.format(
                response_text=subject_response[:2000]
            )
            
            raw = await self._call_model(auditor_id, prompt)
            parsed = self._parse_json_response(raw)

            if not parsed:
                return None

            return AuditAssessment(
                auditor_id=auditor_id,
                auditor_name=MODEL_NAMES.get(auditor_id, auditor_id),
                subject_id=subject_id,
                subject_name=MODEL_NAMES.get(subject_id, subject_id),
                logical_coherence=float(parsed.get("logical_coherence", 0.5)),
                hidden_assumptions=float(parsed.get("hidden_assumptions", 0.5)),
                bias_patterns=float(parsed.get("bias_patterns", 0.0)),
                confidence_inflation=float(parsed.get("confidence_inflation", 0.0)),
                persuasion_tactics=float(parsed.get("persuasion_tactics", 0.0)),
                evidence_quality=float(parsed.get("evidence_quality", 0.5)),
                completeness=float(parsed.get("completeness", 0.5)),
                weak_points=parsed.get("weak_points", []),
                strong_points=parsed.get("strong_points", []),
                risk_factors=parsed.get("risk_factors", []),
                overall_assessment=parsed.get("overall_assessment", ""),
                trust_score=float(parsed.get("trust_score", 0.5)),
            )

        except Exception as e:
            logger.error(f"Audit by {auditor_id} of {subject_id}: {e}")
            return None

    # ============================================================
    # TACTICAL MAP
    # ============================================================

    def _compute_tactical_map(
        self, assessments: List[AuditAssessment]
    ) -> Dict[str, Any]:
        """
        Compute cross-model tactical map from audit assessments.
        Aggregates per-model risk profiles across all auditors.
        """
        model_profiles = {}

        for assessment in assessments:
            subject = assessment.subject_id
            if subject not in model_profiles:
                model_profiles[subject] = {
                    "model_name": MODEL_NAMES.get(subject, subject),
                    "audit_count": 0,
                    "avg_coherence": 0.0,
                    "avg_assumptions": 0.0,
                    "avg_bias": 0.0,
                    "avg_confidence_inflation": 0.0,
                    "avg_persuasion": 0.0,
                    "avg_evidence": 0.0,
                    "avg_completeness": 0.0,
                    "avg_trust": 0.0,
                    "all_weak_points": [],
                    "all_strong_points": [],
                    "all_risks": [],
                }

            profile = model_profiles[subject]
            n = profile["audit_count"]
            profile["audit_count"] += 1
            new_n = profile["audit_count"]

            # Running average update
            profile["avg_coherence"] = (profile["avg_coherence"] * n + assessment.logical_coherence) / new_n
            profile["avg_assumptions"] = (profile["avg_assumptions"] * n + assessment.hidden_assumptions) / new_n
            profile["avg_bias"] = (profile["avg_bias"] * n + assessment.bias_patterns) / new_n
            profile["avg_confidence_inflation"] = (profile["avg_confidence_inflation"] * n + assessment.confidence_inflation) / new_n
            profile["avg_persuasion"] = (profile["avg_persuasion"] * n + assessment.persuasion_tactics) / new_n
            profile["avg_evidence"] = (profile["avg_evidence"] * n + assessment.evidence_quality) / new_n
            profile["avg_completeness"] = (profile["avg_completeness"] * n + assessment.completeness) / new_n
            profile["avg_trust"] = (profile["avg_trust"] * n + assessment.trust_score) / new_n
            profile["all_weak_points"].extend(assessment.weak_points[:3])
            profile["all_strong_points"].extend(assessment.strong_points[:3])
            profile["all_risks"].extend(assessment.risk_factors[:3])

        # Round averages
        for profile in model_profiles.values():
            for key in profile:
                if key.startswith("avg_"):
                    profile[key] = round(profile[key], 3)

        return {
            "model_profiles": model_profiles,
            "highest_risk_model": max(
                model_profiles.keys(),
                key=lambda m: (
                    model_profiles[m]["avg_confidence_inflation"]
                    + model_profiles[m]["avg_bias"]
                    + model_profiles[m]["avg_persuasion"]
                ),
                default=None,
            ) if model_profiles else None,
            "most_trustworthy_model": max(
                model_profiles.keys(),
                key=lambda m: model_profiles[m]["avg_trust"],
                default=None,
            ) if model_profiles else None,
        }

    def _compute_overall_trust(self, assessments: List[AuditAssessment]) -> float:
        """Compute aggregate trust score from all audit assessments."""
        if not assessments:
            return 0.5

        trust_scores = [a.trust_score for a in assessments]
        return sum(trust_scores) / len(trust_scores)

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """Robust JSON extraction."""
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass
        try:
            if "```json" in raw:
                return json.loads(raw.split("```json")[1].split("```")[0].strip())
            elif "```" in raw:
                return json.loads(raw.split("```")[1].split("```")[0].strip())
        except (json.JSONDecodeError, IndexError):
            pass
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
        return {}
