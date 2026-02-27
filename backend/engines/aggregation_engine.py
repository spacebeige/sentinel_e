"""
Aggregation Engine — Sentinel-E Standard Mode (v5.1)

TRUE parallel execution of ALL enabled models.
No debate. No sequential calls. No stale state.

Pipeline:
1. Query model registry for all enabled models
2. Run all models via asyncio.gather (true parallelism)
3. Normalize outputs
4. Extract core claims from each
5. Compute divergence score between models
6. Generate synthesis
7. Compute confidence aggregation
8. Return structured AggregationResult

v5.1: Dynamically iterates model registry — no hardcoded model list.
      New models (Nemotron, Qwen3 Coder, Kimi 2.5, etc.) are
      automatically included when their API key is present.

This engine is the ONLY path for Standard mode.
Debate mode is separate and must be explicitly selected.
"""

import asyncio
import logging
import re
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("AggregationEngine")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ModelOutput:
    """Output from a single model in parallel aggregation."""
    model_id: str
    model_name: str
    raw_output: str
    claims: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    word_count: int = 0
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "output": self.raw_output[:2000],
            "claims": self.claims,
            "confidence": round(self.confidence, 4),
            "word_count": self.word_count,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class AggregationResult:
    """Complete result from parallel aggregation."""
    query: str
    model_outputs: List[ModelOutput] = field(default_factory=list)
    synthesis: str = ""
    divergence_score: float = 0.0
    disagreement_details: List[Dict[str, Any]] = field(default_factory=list)
    agreement_points: List[str] = field(default_factory=list)
    confidence_aggregation: float = 0.5
    confidence_per_model: Dict[str, float] = field(default_factory=dict)
    boundary_severity: float = 0.0
    models_succeeded: int = 0
    models_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "model_outputs": [m.to_dict() for m in self.model_outputs],
            "synthesis": self.synthesis,
            "divergence_score": round(self.divergence_score, 4),
            "disagreement_details": self.disagreement_details,
            "agreement_points": self.agreement_points,
            "confidence_aggregation": round(self.confidence_aggregation, 4),
            "confidence_per_model": {k: round(v, 4) for k, v in self.confidence_per_model.items()},
            "boundary_severity": round(self.boundary_severity, 4),
            "models_succeeded": self.models_succeeded,
            "models_failed": self.models_failed,
        }


# ============================================================
# AGGREGATION ENGINE
# ============================================================

class AggregationEngine:
    """
    Dynamic parallel multi-model aggregation engine for Standard mode.
    
    Queries the model registry and runs ALL enabled models in parallel.
    No hardcoded model list — new providers are automatically included.
    """

    STANDARD_SYSTEM_PROMPT = (
        "You are a rigorous analytical assistant. Provide a clear, comprehensive, "
        "and well-structured response. Include specific details, evidence where possible, "
        "and acknowledge uncertainties. Do not hedge excessively — commit to your analysis."
    )

    def __init__(self, cloud_client):
        """
        Args:
            cloud_client: MCOModelBridge instance with call_model() and get_enabled_model_ids()
        """
        self.client = cloud_client

    async def run_parallel_aggregation(
        self, query: str, history: List[Dict[str, str]] = None
    ) -> AggregationResult:
        """
        Execute parallel aggregation across all 3 models.
        
        Steps:
        1. Fire all 3 models simultaneously via asyncio.gather
        2. Normalize outputs
        3. Extract claims from each
        4. Compute divergence
        5. Generate synthesis
        6. Compute aggregated confidence
        """
        result = AggregationResult(query=query)
        history = history or []

        # Step 1: True parallel execution — dynamically from registry
        logger.info(f"Starting parallel aggregation for: {query[:80]}...")
        
        start_time = datetime.utcnow()
        
        # Build model list dynamically from bridge — all enabled models participate
        models_info = self.client.get_enabled_models_info()
        
        tasks = [
            self._run_model(m["legacy_id"], m["name"], query, history)
            for m in models_info
        ]
        
        model_outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Step 2: Process outputs
        for output in model_outputs:
            if isinstance(output, Exception):
                result.model_outputs.append(ModelOutput(
                    model_id="error", model_name="Error",
                    raw_output="", error=str(output)
                ))
                result.models_failed += 1
            elif output.error:
                result.model_outputs.append(output)
                result.models_failed += 1
            else:
                result.model_outputs.append(output)
                result.models_succeeded += 1

        successful_outputs = [m for m in result.model_outputs if not m.error]
        
        if not successful_outputs:
            result.synthesis = "All models failed to generate a response."
            result.confidence_aggregation = 0.01
            return result

        # Step 3: Extract claims from each
        for output in successful_outputs:
            output.claims = self._extract_claims(output.raw_output)

        # Step 4: Compute divergence
        divergence = self._compute_divergence(successful_outputs)
        result.divergence_score = divergence["score"]
        result.disagreement_details = divergence["disagreements"]
        result.agreement_points = divergence["agreements"]

        # Step 5: Generate synthesis
        result.synthesis = self._generate_synthesis(successful_outputs, divergence)

        # Step 6: Compute confidence
        confidence_data = self._compute_confidence_aggregation(
            successful_outputs, divergence
        )
        result.confidence_aggregation = confidence_data["aggregated"]
        result.confidence_per_model = confidence_data["per_model"]

        # Step 7: Compute boundary severity dynamically
        result.boundary_severity = self._compute_boundary_severity(
            divergence["score"], confidence_data["aggregated"], result.models_failed
        )

        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            f"Aggregation complete: {result.models_succeeded} succeeded, "
            f"{result.models_failed} failed, divergence={result.divergence_score:.3f}, "
            f"confidence={result.confidence_aggregation:.3f}, {elapsed:.0f}ms"
        )

        return result

    # ============================================================
    # MODEL EXECUTION
    # ============================================================

    async def _run_model(
        self, model_id: str, model_name: str, query: str,
        history: List[Dict[str, str]]
    ) -> ModelOutput:
        """Run a single model and return structured output."""
        start = datetime.utcnow()
        try:
            # Unified model dispatch through MCOModelBridge
            raw = await self.client.call_model(
                model_id=model_id,
                prompt=query,
                system_role=self.STANDARD_SYSTEM_PROMPT,
            )

            elapsed = (datetime.utcnow() - start).total_seconds() * 1000
            
            # Check for API error strings (provider-agnostic pattern)
            error = None
            if any(marker in raw for marker in [
                "Error:", "Exception:", "API Key missing", "not available",
                "not found in registry",
            ]) and len(raw) < 500:
                error = raw

            return ModelOutput(
                model_id=model_id,
                model_name=model_name,
                raw_output=raw if not error else "",
                word_count=len(raw.split()) if not error else 0,
                error=error,
                latency_ms=elapsed,
            )
        except Exception as e:
            elapsed = (datetime.utcnow() - start).total_seconds() * 1000
            logger.error(f"Model {model_id} failed: {e}")
            return ModelOutput(
                model_id=model_id, model_name=model_name,
                raw_output="", error=str(e), latency_ms=elapsed,
            )

    # ============================================================
    # CLAIM EXTRACTION
    # ============================================================

    def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract atomic claims from model output."""
        if not text:
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims = []

        claim_patterns = [
            r'\b(is|are|was|were|has|have|will|can|does|do)\b',
            r'\b(causes?|leads?\s+to|results?\s+in|increases?|decreases?)\b',
            r'\b(according\s+to|research\s+shows?|studies?\s+(show|indicate|suggest))\b',
            r'\b(always|never|typically|generally|often|usually|rarely)\b',
            r'\b(\d+%|\d+\s*percent)\b',
        ]

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue

            strength = sum(
                1 for pattern in claim_patterns
                if re.search(pattern, sentence, re.IGNORECASE)
            )

            if strength >= 1:
                claims.append({
                    "id": f"claim_{i}",
                    "text": sentence[:300],
                    "strength": min(strength / 3.0, 1.0),
                })

        return claims

    # ============================================================
    # DIVERGENCE COMPUTATION
    # ============================================================

    def _compute_divergence(self, outputs: List[ModelOutput]) -> Dict[str, Any]:
        """
        Compute divergence between model outputs using content analysis.
        
        Returns divergence score (0-1), specific disagreements, and agreements.
        """
        if len(outputs) < 2:
            return {"score": 0.0, "disagreements": [], "agreements": []}

        # Extract keyword sets per model (excluding stop words)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
            "for", "and", "or", "on", "at", "by", "it", "this", "that", "with",
            "as", "be", "but", "not", "from", "they", "we", "you", "can", "will",
            "if", "so", "than", "its", "has", "have", "had", "may", "also",
            "which", "their", "would", "been", "more", "these", "those",
        }

        model_keywords = {}
        for output in outputs:
            words = set(output.raw_output.lower().split())
            words -= stop_words
            # Filter short words and non-alpha
            words = {w for w in words if len(w) > 3 and w.isalpha()}
            model_keywords[output.model_id] = words

        # Pairwise Jaccard distances
        models = list(model_keywords.keys())
        pairwise_distances = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                set_a = model_keywords[models[i]]
                set_b = model_keywords[models[j]]
                union = set_a | set_b
                intersection = set_a & set_b
                if union:
                    jaccard = len(intersection) / len(union)
                    distance = 1.0 - jaccard
                else:
                    distance = 1.0
                pairwise_distances.append(distance)

        divergence_score = sum(pairwise_distances) / max(len(pairwise_distances), 1)

        # Find specific disagreements: words unique to only one model
        all_words = set()
        for words in model_keywords.values():
            all_words |= words

        disagreements = []
        agreements = []

        for word in all_words:
            present_in = [m for m in models if word in model_keywords[m]]
            if len(present_in) == 1:
                # Unique to one model — potential divergence
                pass  # Individual divergences, too noisy
            elif len(present_in) == len(models):
                # Agreement across all models
                if len(word) > 5:  # Only substantial words
                    agreements.append(word)

        # Sentiment-based disagreements
        positive = {"increase", "improve", "benefit", "effective", "positive", "growth", "better", "good", "safe"}
        negative = {"decrease", "worsen", "harm", "ineffective", "negative", "decline", "worse", "bad", "unsafe"}

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a_pos = model_keywords[models[i]] & positive
                a_neg = model_keywords[models[i]] & negative
                b_pos = model_keywords[models[j]] & positive
                b_neg = model_keywords[models[j]] & negative

                if (a_pos and b_neg) or (a_neg and b_pos):
                    disagreements.append({
                        "model_a": models[i],
                        "model_b": models[j],
                        "type": "sentiment_opposition",
                        "detail": f"{models[i]} sentiment differs from {models[j]}",
                    })

        return {
            "score": round(min(divergence_score, 1.0), 4),
            "disagreements": disagreements,
            "agreements": sorted(agreements)[:20],
        }

    # ============================================================
    # SYNTHESIS GENERATION
    # ============================================================

    def _generate_synthesis(
        self, outputs: List[ModelOutput], divergence: Dict[str, Any]
    ) -> str:
        """
        Generate a synthesis from multiple model outputs.
        Picks the most comprehensive response and annotates with divergence.
        """
        if not outputs:
            return "No model outputs available for synthesis."

        # Select best output by word count and claim count
        scored = []
        for o in outputs:
            score = o.word_count * 0.3 + len(o.claims) * 50
            scored.append((score, o))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]

        synthesis = best.raw_output

        # Append divergence notice if significant
        if divergence["score"] > 0.3 and divergence["disagreements"]:
            synthesis += "\n\n---\n**⚠️ Cross-Model Divergence Detected**\n"
            for d in divergence["disagreements"][:3]:
                synthesis += f"- {d.get('detail', 'Models showed opposing analysis')}\n"

        # Append agreement highlights
        if divergence["agreements"]:
            top_agreements = divergence["agreements"][:10]
            synthesis += f"\n\n**Cross-Model Agreement**: All models converged on: {', '.join(top_agreements)}"

        return synthesis

    # ============================================================
    # CONFIDENCE AGGREGATION
    # ============================================================

    def _compute_confidence_aggregation(
        self, outputs: List[ModelOutput], divergence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute aggregated confidence from model outputs and divergence.
        
        Formula:
          base = average of model confidences (from claim strength)
          penalty = divergence_score * 0.3
          bonus = agreement_ratio * 0.1
          aggregated = base - penalty + bonus
        """
        per_model = {}
        for output in outputs:
            if output.claims:
                avg_strength = sum(c["strength"] for c in output.claims) / len(output.claims)
                model_conf = 0.4 + avg_strength * 0.5  # Scale 0.4-0.9
            else:
                model_conf = 0.5  # No claims extracted
            per_model[output.model_id] = model_conf

        if not per_model:
            return {"aggregated": 0.3, "per_model": {}}

        base_confidence = sum(per_model.values()) / len(per_model)
        divergence_penalty = divergence["score"] * 0.3
        agreement_bonus = min(len(divergence["agreements"]) / 20.0, 0.1)

        aggregated = base_confidence - divergence_penalty + agreement_bonus
        aggregated = max(0.05, min(0.95, aggregated))

        return {
            "aggregated": round(aggregated, 4),
            "per_model": per_model,
        }

    # ============================================================
    # DYNAMIC BOUNDARY SEVERITY
    # ============================================================

    def _compute_boundary_severity(
        self, divergence_score: float, confidence: float,
        failed_models: int
    ) -> float:
        """
        Compute boundary severity dynamically.
        No hardcoded values.
        
        Formula:
          epistemic_risk = 1 - confidence
          instability = divergence_entropy(divergence_score)
          failure_penalty = failed_models * 15
          severity = (epistemic_risk * 40) + (divergence_score * 30) + (instability * 20) + failure_penalty
        """
        epistemic_risk = 1.0 - confidence
        
        # Divergence entropy: higher divergence = more instability
        if divergence_score > 0:
            instability = -divergence_score * math.log2(max(divergence_score, 0.001))
        else:
            instability = 0.0
        instability = min(instability, 1.0)

        failure_penalty = failed_models * 15

        severity = (
            epistemic_risk * 40
            + divergence_score * 30
            + instability * 20
            + failure_penalty
        )

        return round(min(100.0, max(0.0, severity)), 2)
