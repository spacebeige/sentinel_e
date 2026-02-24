"""
Forensic Evidence Engine — Sentinel-E Evidence Mode

5-Phase Triangular Forensic Pipeline:
  Phase 1: Independent model retrieval (each model extracts claims independently)
  Phase 2: Grouped cross-verification (triangular checking)
  Phase 3: Algorithmic contradiction detection (deterministic comparison)
  Phase 4: Bayesian confidence update (computed, not fixed)
  Phase 5: Verbatim citation mode (exact quotes, URLs, reliability)

Each model receives ONLY the user question. No mention of cross-verification.
Triangular verification: each pair verifies the third.
All confidence values are COMPUTED, never fixed.
"""

import asyncio
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("ForensicEvidenceEngine")

# Trigger words that activate verbatim citation mode
EVIDENCE_TRIGGER_WORDS = [
    "cite", "source", "did that happen", "prove", "evidence",
    "verify", "proof", "citation", "reference", "fact check",
    "is that true", "really", "actually happened", "show me",
]

CLAIM_EXTRACTION_PROMPT = """You are a forensic fact analyst. Extract ALL factual claims from your analysis.

USER QUESTION: {query}

Analyze the question and provide your best factual response. Then extract every distinct factual claim.

CRITICAL: For EACH claim, provide:
1. The exact statement
2. A date/timeframe if applicable (or "unknown")
3. The likely source type (academic, news, government, industry, unknown)
4. A relevant quote if you can recall one (or "none")  
5. Your confidence in this specific claim (0.0-1.0)

OUTPUT FORMAT (JSON only, no markdown):
{{
  "analysis": "Your full analysis text here",
  "claims": [
    {{
      "statement": "Exact factual claim",
      "date": "Date or timeframe or unknown",
      "source_type": "academic|news|government|industry|unknown",
      "quote": "Relevant quote or none",
      "confidence": 0.0
    }}
  ]
}}"""

VERIFICATION_PROMPT = """You are a forensic fact verifier. You must evaluate the following factual claims for accuracy and evidence strength.

DO NOT reveal that you are checking another system's work. DO NOT mention AI, models, or verification pipelines.

Evaluate each claim as if you are an independent researcher verifying factual statements:

CLAIMS TO VERIFY:
{claims_json}

For EACH claim, evaluate:
1. Is the statement factually accurate? (true/false/uncertain)
2. Are there any contradictions with known facts?
3. How credible is the sourcing?
4. What evidence supports or contradicts this claim?
5. Your confidence in the claim's accuracy (0.0-1.0)

OUTPUT FORMAT (JSON only, no markdown):
{{
  "verifications": [
    {{
      "claim_index": 0,
      "statement": "The original claim",
      "verdict": "confirmed|contradicted|uncertain",
      "contradictions": ["Any contradicting facts"],
      "supporting_evidence": ["Supporting evidence"],
      "source_credibility": 0.0,
      "confidence": 0.0,
      "reasoning": "Brief explanation"
    }}
  ]
}}"""

VERBATIM_CITATION_PROMPT = """You are a citation specialist. For the following query, provide ONLY verifiable citations and exact quotes. No summarization. No paraphrasing.

QUERY: {query}

For each piece of evidence:
1. Extract the EXACT paragraph or statement
2. Show the verbatim quote  
3. Provide the URL or reference
4. Rate the source reliability (0.0-1.0)

OUTPUT FORMAT (JSON only, no markdown):
{{
  "citations": [
    {{
      "quote": "Exact verbatim text",
      "source": "URL or reference",
      "reliability": 0.0,
      "context": "Brief context for this citation"
    }}
  ]
}}"""


@dataclass
class ForensicClaim:
    """A single claim extracted during forensic analysis."""
    statement: str
    date: str = "unknown"
    source_type: str = "unknown"
    quote: str = "none"
    confidence: float = 0.5
    model_origin: str = ""
    verification_verdicts: List[Dict[str, Any]] = field(default_factory=list)
    final_confidence: float = 0.5
    agreement_count: int = 0
    contradiction_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "date": self.date,
            "source_type": self.source_type,
            "quote": self.quote,
            "initial_confidence": round(self.confidence, 4),
            "model_origin": self.model_origin,
            "verifications": self.verification_verdicts,
            "final_confidence": round(self.final_confidence, 4),
            "agreement_count": self.agreement_count,
            "contradiction_count": self.contradiction_count,
        }


@dataclass
class ForensicResult:
    """Complete result from 5-phase forensic pipeline."""
    query: str
    phase_log: List[Dict[str, Any]] = field(default_factory=list)
    model_claims: Dict[str, List[ForensicClaim]] = field(default_factory=dict)
    all_claims: List[ForensicClaim] = field(default_factory=list)
    cross_verifications: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    bayesian_confidence: float = 0.5
    agreement_score: float = 0.0
    source_reliability_avg: float = 0.0
    verbatim_citations: List[Dict[str, Any]] = field(default_factory=list)
    is_citation_mode: bool = False
    pipeline_succeeded: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "phase_log": self.phase_log,
            "model_claims": {
                k: [c.to_dict() for c in v] for k, v in self.model_claims.items()
            },
            "total_claims": len(self.all_claims),
            "all_claims": [c.to_dict() for c in self.all_claims],
            "cross_verifications": self.cross_verifications,
            "contradictions": self.contradictions,
            "bayesian_confidence": round(self.bayesian_confidence, 4),
            "agreement_score": round(self.agreement_score, 4),
            "source_reliability_avg": round(self.source_reliability_avg, 4),
            "verbatim_citations": self.verbatim_citations,
            "is_citation_mode": self.is_citation_mode,
            "pipeline_succeeded": self.pipeline_succeeded,
        }


class ForensicEvidenceEngine:
    """
    5-Phase Triangular Forensic Evidence Engine.
    
    Each model operates independently. Cross-verification is blind.
    All confidence values are computed, never fixed.
    """

    def __init__(self, cloud_client):
        self.client = cloud_client

    @staticmethod
    def detect_evidence_trigger(text: str) -> bool:
        """Detect if user input contains evidence trigger words."""
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in EVIDENCE_TRIGGER_WORDS)

    @staticmethod
    def detect_citation_trigger(text: str) -> bool:
        """Detect if user input specifically requests citations/verbatim quotes."""
        text_lower = text.lower()
        citation_triggers = ["cite", "source", "proof", "quote", "evidence", "verify"]
        return any(trigger in text_lower for trigger in citation_triggers)

    async def run_forensic_pipeline(
        self, query: str, history: List[Dict[str, str]] = None
    ) -> ForensicResult:
        """
        Execute the full 5-phase forensic pipeline.
        """
        result = ForensicResult(query=query)
        history = history or []

        try:
            # Phase 1: Independent Retrieval
            result.phase_log.append({
                "phase": 1, "name": "Independent Retrieval",
                "status": "running", "timestamp": datetime.utcnow().isoformat()
            })
            await self._phase1_independent_retrieval(query, result)
            result.phase_log[-1]["status"] = "complete"

            # Check: if no claims extracted, pipeline fails
            total_claims = sum(len(v) for v in result.model_claims.values())
            if total_claims == 0:
                result.error = "No claims extracted from any model. Pipeline failed."
                result.phase_log.append({
                    "phase": "ABORT", "name": "Pipeline Failed",
                    "status": "failed", "detail": "Zero claims extracted",
                    "timestamp": datetime.utcnow().isoformat()
                })
                return result

            # Phase 2: Grouped Cross-Verification
            result.phase_log.append({
                "phase": 2, "name": "Triangular Cross-Verification",
                "status": "running", "timestamp": datetime.utcnow().isoformat()
            })
            await self._phase2_cross_verification(result)
            result.phase_log[-1]["status"] = "complete"

            # Phase 3: Algorithmic Contradiction Detection
            result.phase_log.append({
                "phase": 3, "name": "Contradiction Detection",
                "status": "running", "timestamp": datetime.utcnow().isoformat()
            })
            self._phase3_contradiction_detection(result)
            result.phase_log[-1]["status"] = "complete"

            # Phase 4: Bayesian Confidence Update
            result.phase_log.append({
                "phase": 4, "name": "Bayesian Confidence Update",
                "status": "running", "timestamp": datetime.utcnow().isoformat()
            })
            self._phase4_bayesian_confidence(result)
            result.phase_log[-1]["status"] = "complete"

            # Phase 5: Verbatim Citation (if triggered)
            is_citation = self.detect_citation_trigger(query)
            result.is_citation_mode = is_citation
            if is_citation:
                result.phase_log.append({
                    "phase": 5, "name": "Verbatim Citation",
                    "status": "running", "timestamp": datetime.utcnow().isoformat()
                })
                await self._phase5_verbatim_citations(query, result)
                result.phase_log[-1]["status"] = "complete"

            result.pipeline_succeeded = True

        except Exception as e:
            logger.error(f"Forensic pipeline error: {e}", exc_info=True)
            result.error = str(e)

        return result

    # ============================================================
    # PHASE 1 — Independent Retrieval
    # ============================================================

    async def _phase1_independent_retrieval(
        self, query: str, result: ForensicResult
    ):
        """
        Each model independently analyzes the query and extracts claims.
        No mention of cross-verification. Pure independent analysis.
        """
        prompt = CLAIM_EXTRACTION_PROMPT.format(query=query)

        # Parallel execution
        tasks = [
            self._extract_model_claims("groq", prompt),
            self._extract_model_claims("llama70b", prompt),
            self._extract_model_claims("qwen", prompt),
        ]
        
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        model_ids = ["groq", "llama70b", "qwen"]
        for model_id, output in zip(model_ids, outputs):
            if isinstance(output, Exception):
                logger.error(f"Phase 1: {model_id} failed: {output}")
                result.model_claims[model_id] = []
                continue
            
            result.model_claims[model_id] = output
            result.all_claims.extend(output)

        result.phase_log[-1]["detail"] = (
            f"Claims extracted — Groq: {len(result.model_claims.get('groq', []))}, "
            f"Llama70B: {len(result.model_claims.get('llama70b', []))}, "
            f"Qwen: {len(result.model_claims.get('qwen', []))}"
        )

    async def _extract_model_claims(
        self, model_id: str, prompt: str
    ) -> List[ForensicClaim]:
        """Call a model and parse its claim extraction output."""
        try:
            if model_id == "groq":
                raw = await self.client.call_groq(prompt=prompt)
            elif model_id == "llama70b":
                raw = await self.client.call_llama70b(prompt=prompt, temperature=0.3)
            elif model_id == "qwen":
                raw = await self.client.call_qwenvl(prompt=prompt)
            else:
                return []

            # Parse JSON from response
            parsed = self._parse_json_response(raw)
            claims_data = parsed.get("claims", [])

            claims = []
            for cd in claims_data:
                claims.append(ForensicClaim(
                    statement=cd.get("statement", ""),
                    date=cd.get("date", "unknown"),
                    source_type=cd.get("source_type", "unknown"),
                    quote=cd.get("quote", "none"),
                    confidence=float(cd.get("confidence", 0.5)),
                    model_origin=model_id,
                ))

            return claims

        except Exception as e:
            logger.error(f"Claim extraction from {model_id}: {e}")
            return []

    # ============================================================
    # PHASE 2 — Grouped Cross-Verification (Triangular)
    # ============================================================

    async def _phase2_cross_verification(self, result: ForensicResult):
        """
        Triangular cross-verification:
          Groq + Llama70B verify Qwen's claims
          Llama70B + Qwen verify Groq's claims
          Groq + Qwen verify Llama70B's claims
        
        Verification prompt does NOT reveal model identity or that it's 
        checking another AI — uses forensic framing only.
        """
        verification_groups = [
            {
                "verifiers": ["groq", "llama70b"],
                "subject": "qwen",
                "label": "Groq + Llama70B verify Qwen",
            },
            {
                "verifiers": ["llama70b", "qwen"],
                "subject": "groq",
                "label": "Llama70B + Qwen verify Groq",
            },
            {
                "verifiers": ["groq", "qwen"],
                "subject": "llama70b",
                "label": "Groq + Qwen verify Llama70B",
            },
        ]

        for group in verification_groups:
            subject_claims = result.model_claims.get(group["subject"], [])
            if not subject_claims:
                continue

            # Format claims for verification prompt
            claims_for_prompt = [
                {"index": i, "statement": c.statement, "date": c.date}
                for i, c in enumerate(subject_claims)
            ]
            claims_json = json.dumps(claims_for_prompt, indent=2)
            prompt = VERIFICATION_PROMPT.format(claims_json=claims_json)

            # Run both verifiers in parallel
            tasks = []
            for verifier_id in group["verifiers"]:
                tasks.append(self._run_verification(verifier_id, prompt))

            verifier_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process verification results
            for verifier_id, ver_result in zip(group["verifiers"], verifier_results):
                if isinstance(ver_result, Exception):
                    logger.warning(f"Verification by {verifier_id} failed: {ver_result}")
                    continue

                result.cross_verifications.append({
                    "verifier": verifier_id,
                    "subject": group["subject"],
                    "group": group["label"],
                    "verifications": ver_result,
                })

                # Attach verdicts to subject claims
                for ver in ver_result:
                    idx = ver.get("claim_index", -1)
                    if 0 <= idx < len(subject_claims):
                        subject_claims[idx].verification_verdicts.append({
                            "verifier": verifier_id,
                            "verdict": ver.get("verdict", "uncertain"),
                            "confidence": ver.get("confidence", 0.5),
                            "source_credibility": ver.get("source_credibility", 0.5),
                            "contradictions": ver.get("contradictions", []),
                            "supporting_evidence": ver.get("supporting_evidence", []),
                        })

    async def _run_verification(
        self, model_id: str, prompt: str
    ) -> List[Dict[str, Any]]:
        """Run verification on a single model."""
        try:
            if model_id == "groq":
                raw = await self.client.call_groq(prompt=prompt)
            elif model_id == "llama70b":
                raw = await self.client.call_llama70b(prompt=prompt, temperature=0.3)
            elif model_id == "qwen":
                raw = await self.client.call_qwenvl(prompt=prompt)
            else:
                return []

            parsed = self._parse_json_response(raw)
            return parsed.get("verifications", [])

        except Exception as e:
            logger.error(f"Verification by {model_id}: {e}")
            return []

    # ============================================================
    # PHASE 3 — Algorithmic Contradiction Detection
    # ============================================================

    def _phase3_contradiction_detection(self, result: ForensicResult):
        """
        Deterministic contradiction detection across verified claims.
        Uses structured data comparison, NOT model opinion.
        """
        contradictions = []

        # Cross-model comparison
        model_ids = list(result.model_claims.keys())
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                claims_a = result.model_claims[model_ids[i]]
                claims_b = result.model_claims[model_ids[j]]

                for ca in claims_a:
                    for cb in claims_b:
                        contradiction = self._check_claim_contradiction(ca, cb)
                        if contradiction:
                            contradictions.append({
                                "model_a": model_ids[i],
                                "model_b": model_ids[j],
                                "claim_a": ca.statement[:200],
                                "claim_b": cb.statement[:200],
                                "type": contradiction["type"],
                                "severity": contradiction["severity"],
                            })

        # Update claim-level contradiction counts
        for claim in result.all_claims:
            for verdict in claim.verification_verdicts:
                if verdict.get("verdict") == "confirmed":
                    claim.agreement_count += 1
                elif verdict.get("verdict") == "contradicted":
                    claim.contradiction_count += 1

        result.contradictions = contradictions
        result.phase_log[-1]["detail"] = (
            f"Detected {len(contradictions)} contradiction(s) across model pairs"
        )

    def _check_claim_contradiction(
        self, claim_a: ForensicClaim, claim_b: ForensicClaim
    ) -> Optional[Dict[str, Any]]:
        """
        Deterministic claim comparison.
        Check dates, sentiment, and factual opposition.
        """
        # Date contradiction
        if (claim_a.date != "unknown" and claim_b.date != "unknown"
                and claim_a.date != claim_b.date):
            # Check if claims are about the same topic
            overlap = self._word_overlap(claim_a.statement, claim_b.statement)
            if overlap > 0.3:
                return {
                    "type": "date_contradiction",
                    "severity": 0.7,
                }

        # Sentiment opposition on overlapping topic
        overlap = self._word_overlap(claim_a.statement, claim_b.statement)
        if overlap > 0.25:
            sent_a = self._detect_sentiment(claim_a.statement)
            sent_b = self._detect_sentiment(claim_b.statement)
            if sent_a != "neutral" and sent_b != "neutral" and sent_a != sent_b:
                return {
                    "type": "sentiment_opposition",
                    "severity": 0.6,
                }

        # Numerical contradiction
        nums_a = re.findall(r'\b\d+\.?\d*%?\b', claim_a.statement)
        nums_b = re.findall(r'\b\d+\.?\d*%?\b', claim_b.statement)
        if nums_a and nums_b and overlap > 0.25:
            # Same topic, different numbers
            if set(nums_a) != set(nums_b):
                return {
                    "type": "numerical_disagreement",
                    "severity": 0.5,
                }

        return None

    # ============================================================
    # PHASE 4 — Bayesian Confidence Update
    # ============================================================

    def _phase4_bayesian_confidence(self, result: ForensicResult):
        """
        Compute Bayesian confidence update for each claim and overall.
        
        Formula per claim:
          final_confidence = 
            (agreement_weight * agreement_score)
            + (source_weight * avg_source_reliability)
            - (contradiction_penalty)
        
        NOT a fixed 66%. Computed from actual verification data.
        """
        agreement_weight = 0.45
        source_weight = 0.35
        contradiction_weight = 0.20

        total_claims = len(result.all_claims)
        if total_claims == 0:
            result.bayesian_confidence = 0.1
            return

        claim_confidences = []
        source_reliabilities = []

        for claim in result.all_claims:
            # Agreement score: fraction of verifiers that confirmed
            total_verdicts = len(claim.verification_verdicts)
            if total_verdicts > 0:
                agreement_score = claim.agreement_count / total_verdicts
                avg_source_cred = sum(
                    v.get("source_credibility", 0.5) for v in claim.verification_verdicts
                ) / total_verdicts
            else:
                agreement_score = 0.5  # No verification data
                avg_source_cred = 0.5

            # Contradiction penalty: scaled by number of contradictions
            contradiction_penalty = min(claim.contradiction_count * 0.15, 0.5)

            # Bayesian update
            updated = (
                agreement_weight * agreement_score
                + source_weight * avg_source_cred
                - contradiction_weight * contradiction_penalty
            )

            # Clamp and apply to claim
            claim.final_confidence = max(0.05, min(0.95, updated))
            claim_confidences.append(claim.final_confidence)
            source_reliabilities.append(avg_source_cred)

        # Overall Bayesian confidence
        result.bayesian_confidence = sum(claim_confidences) / len(claim_confidences)
        result.agreement_score = sum(
            1 for c in result.all_claims if c.agreement_count > c.contradiction_count
        ) / max(total_claims, 1)
        result.source_reliability_avg = (
            sum(source_reliabilities) / len(source_reliabilities) if source_reliabilities else 0.5
        )

        result.phase_log[-1]["detail"] = (
            f"Bayesian confidence: {result.bayesian_confidence:.4f}, "
            f"Agreement: {result.agreement_score:.4f}, "
            f"Source reliability: {result.source_reliability_avg:.4f}"
        )

    # ============================================================
    # PHASE 5 — Verbatim Citation Mode
    # ============================================================

    async def _phase5_verbatim_citations(
        self, query: str, result: ForensicResult
    ):
        """
        Extract verbatim citations — exact quotes, URLs, reliability.
        No summarization. No paraphrasing.
        """
        prompt = VERBATIM_CITATION_PROMPT.format(query=query)

        # Get citations from all 3 models in parallel
        tasks = [
            self._get_citations("groq", prompt),
            self._get_citations("llama70b", prompt),
            self._get_citations("qwen", prompt),
        ]

        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        all_citations = []
        for model_id, output in zip(["groq", "llama70b", "qwen"], outputs):
            if isinstance(output, Exception):
                continue
            for citation in output:
                citation["model_source"] = model_id
                all_citations.append(citation)

        # Deduplicate by quote similarity
        result.verbatim_citations = self._deduplicate_citations(all_citations)

    async def _get_citations(
        self, model_id: str, prompt: str
    ) -> List[Dict[str, Any]]:
        """Get verbatim citations from a model."""
        try:
            if model_id == "groq":
                raw = await self.client.call_groq(prompt=prompt)
            elif model_id == "llama70b":
                raw = await self.client.call_llama70b(prompt=prompt, temperature=0.3)
            elif model_id == "qwen":
                raw = await self.client.call_qwenvl(prompt=prompt)
            else:
                return []

            parsed = self._parse_json_response(raw)
            return parsed.get("citations", [])
        except Exception as e:
            logger.error(f"Citation extraction from {model_id}: {e}")
            return []

    def _deduplicate_citations(
        self, citations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate citations by quote similarity."""
        if not citations:
            return []

        unique = []
        seen_quotes = set()

        for citation in citations:
            quote = citation.get("quote", "")[:100].lower().strip()
            if quote and quote not in seen_quotes:
                seen_quotes.add(quote)
                unique.append(citation)

        return unique

    # ============================================================
    # HELPERS
    # ============================================================

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """Robust JSON extraction from model output."""
        try:
            # Try direct parse
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        try:
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            pass

        # Try finding JSON object
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

        return {}

    def _word_overlap(self, text_a: str, text_b: str) -> float:
        """Compute word overlap between two texts."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
            "for", "and", "or", "on", "at", "by", "it", "this", "that", "with",
        }
        words_a = set(text_a.lower().split()) - stop_words
        words_b = set(text_b.lower().split()) - stop_words
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / max(len(words_a | words_b), 1)

    def _detect_sentiment(self, text: str) -> str:
        """Simple sentiment detection."""
        positive = {"increase", "improve", "benefit", "effective", "positive", "growth", "better", "good", "safe", "success"}
        negative = {"decrease", "worsen", "harm", "ineffective", "negative", "decline", "worse", "bad", "unsafe", "failure"}
        words = set(text.lower().split())
        pos_count = len(words & positive)
        neg_count = len(words & negative)
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"
