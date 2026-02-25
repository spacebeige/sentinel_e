"""
============================================================
Arbitration Engine — Multi-Model Scoring & Selection
============================================================
Implements the mandatory scoring formula:

  FinalScore = 0.30*T + 0.25*K + 0.15*S + 0.15*C - 0.15*D

Where:
  T = Topic Alignment     — cosine(output_embedding, topic_centroid)
  K = Knowledge Grounding — supported_claims / total_claims
  S = Specificity         — (named_entities + technical_terms) / tokens
  C = Confidence Calibration — factual_support_alignment
  D = Drift Penalty       — 1 - cosine(output_embedding, session_context_embedding)

No raw output modification.
No score tampering.
Transparent scoring for both modes.
============================================================
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple

from metacognitive.schemas import (
    ArbitrationScore,
    CognitiveGatewayOutput,
    KnowledgeBlock,
    ModelResult,
    OperatingMode,
)
from metacognitive.embedding import (
    embed_text,
    cosine_similarity,
    drift_score as compute_drift,
    count_named_entities,
    count_technical_terms,
    token_count_approx,
)

logger = logging.getLogger("MCO-Arbitration")


class ArbitrationEngine:
    """
    Scores model outputs according to the 5-metric formula.
    Selects winner (Standard mode) or exposes all (Experimental).

    Rules:
      ✗ No raw output modification before scoring
      ✗ No score tampering
      ✓ Full metric exposure in experimental mode
      ✓ Transparent scoring
    """

    # ── Public Interface ─────────────────────────────────────

    def score_outputs(
        self,
        outputs: List[CognitiveGatewayOutput],
        topic_centroid: List[float],
        session_context_embedding: List[float],
        knowledge_bundle: List[KnowledgeBlock],
    ) -> List[ArbitrationScore]:
        """
        Score all model outputs using the 5-metric formula.
        Returns list of ArbitrationScore, one per output.
        """
        scores = []
        for output in outputs:
            if not output.success or not output.raw_output:
                scores.append(ArbitrationScore(
                    model_name=output.model_name,
                    topic_alignment=0.0,
                    knowledge_grounding=0.0,
                    specificity=0.0,
                    confidence_calibration=0.0,
                    drift_penalty=1.0,
                    final_score=0.0,
                ))
                continue

            score = self._compute_score(
                output=output,
                topic_centroid=topic_centroid,
                session_context_embedding=session_context_embedding,
                knowledge_bundle=knowledge_bundle,
            )
            scores.append(score)

        return scores

    def select_winner(
        self,
        outputs: List[CognitiveGatewayOutput],
        scores: List[ArbitrationScore],
    ) -> Tuple[CognitiveGatewayOutput, ArbitrationScore]:
        """
        Standard Mode: Select the output with highest FinalScore.
        Returns (winning_output, winning_score).
        """
        if not outputs or not scores:
            return (
                CognitiveGatewayOutput(
                    model_name="none", raw_output="No outputs available.",
                    success=False,
                ),
                ArbitrationScore(model_name="none"),
            )

        best_idx = 0
        best_score = -float("inf")
        for i, s in enumerate(scores):
            if s.final_score > best_score:
                best_score = s.final_score
                best_idx = i

        return outputs[best_idx], scores[best_idx]

    def build_results(
        self,
        outputs: List[CognitiveGatewayOutput],
        scores: List[ArbitrationScore],
    ) -> List[ModelResult]:
        """Build ModelResult list pairing outputs with scores."""
        results = []
        for output, score in zip(outputs, scores):
            results.append(ModelResult(output=output, score=score))
        return results

    def compute_divergence_metrics(
        self,
        outputs: List[CognitiveGatewayOutput],
        scores: List[ArbitrationScore],
    ) -> Dict[str, Any]:
        """
        Experimental Mode: Compute divergence metrics between models.
        Returns divergence heatmap, stability delta, etc.
        """
        if len(outputs) < 2:
            return {"divergence": "insufficient_models"}

        # Compute pairwise output similarity
        embeddings = {}
        for output in outputs:
            if output.success and output.raw_output:
                embeddings[output.model_name] = embed_text(output.raw_output)

        pairwise = {}
        names = list(embeddings.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = cosine_similarity(embeddings[names[i]], embeddings[names[j]])
                pairwise[f"{names[i]} ↔ {names[j]}"] = round(sim, 4)

        # Score spread
        final_scores = [s.final_score for s in scores if s.final_score > 0]
        score_spread = max(final_scores) - min(final_scores) if final_scores else 0.0

        # Stability delta (std dev of scores)
        import statistics
        score_std = statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0

        return {
            "pairwise_similarity": pairwise,
            "score_spread": round(score_spread, 4),
            "score_std_dev": round(score_std, 4),
            "model_count": len(outputs),
            "successful_count": sum(1 for o in outputs if o.success),
            "per_model_scores": {
                s.model_name: round(s.final_score, 4)
                for s in scores
            },
        }

    # ── Internal Scoring ─────────────────────────────────────

    def _compute_score(
        self,
        output: CognitiveGatewayOutput,
        topic_centroid: List[float],
        session_context_embedding: List[float],
        knowledge_bundle: List[KnowledgeBlock],
    ) -> ArbitrationScore:
        """Compute all 5 metrics and the final score."""

        text = output.raw_output
        output_embedding = embed_text(text)

        # ── T: Topic Alignment ───────────────────────────────
        if topic_centroid:
            T = cosine_similarity(output_embedding, topic_centroid)
        else:
            T = 0.5  # Neutral if no centroid

        # ── K: Knowledge Grounding ───────────────────────────
        K = self._compute_grounding(text, knowledge_bundle)

        # ── S: Specificity ───────────────────────────────────
        S = self._compute_specificity(text)

        # ── C: Confidence Calibration ────────────────────────
        C = self._compute_confidence_calibration(output, knowledge_bundle)

        # ── D: Drift Penalty ─────────────────────────────────
        if session_context_embedding:
            D = compute_drift(output_embedding, session_context_embedding)
        else:
            D = 0.0  # No drift if no context yet

        score = ArbitrationScore(
            model_name=output.model_name,
            topic_alignment=round(T, 4),
            knowledge_grounding=round(K, 4),
            specificity=round(S, 4),
            confidence_calibration=round(C, 4),
            drift_penalty=round(D, 4),
        )
        score.compute_final()
        return score

    def _compute_grounding(
        self,
        text: str,
        knowledge_bundle: List[KnowledgeBlock],
    ) -> float:
        """
        K = supported_claims / total_claims
        Uses embedding similarity to check if output claims
        are supported by knowledge bundle.
        """
        if not knowledge_bundle:
            return 0.5  # Neutral when no knowledge available

        # Extract claim-like sentences from output
        claims = self._extract_claims(text)
        if not claims:
            return 0.5

        # Check each claim against knowledge bundle
        supported = 0
        for claim in claims:
            claim_emb = embed_text(claim)
            max_sim = 0.0
            for block in knowledge_bundle:
                if block.embedding:
                    sim = cosine_similarity(claim_emb, block.embedding)
                    max_sim = max(max_sim, sim)
            if max_sim > 0.5:  # Threshold for "supported"
                supported += 1

        return supported / len(claims) if claims else 0.5

    def _compute_specificity(self, text: str) -> float:
        """
        S = (named_entities + technical_terms) / tokens
        Normalized to [0, 1].
        """
        entities = count_named_entities(text)
        tech_terms = count_technical_terms(text)
        tokens = token_count_approx(text)

        raw = (entities + tech_terms) / max(tokens, 1)
        # Normalize: cap at 0.3 density → 1.0 score
        return min(1.0, raw / 0.3)

    def _compute_confidence_calibration(
        self,
        output: CognitiveGatewayOutput,
        knowledge_bundle: List[KnowledgeBlock],
    ) -> float:
        """
        C = factual_support_alignment
        Measures how well the model's confidence estimate
        aligns with actual factual support.
        """
        if output.confidence_estimate is not None:
            model_confidence = output.confidence_estimate
        else:
            # Estimate confidence from output language
            model_confidence = self._estimate_confidence_from_text(output.raw_output)

        # Compute factual support
        if knowledge_bundle:
            output_emb = embed_text(output.raw_output)
            support_scores = [
                cosine_similarity(output_emb, b.embedding)
                for b in knowledge_bundle
                if b.embedding
            ]
            factual_support = max(support_scores) if support_scores else 0.3
        else:
            factual_support = 0.5

        # Calibration = how close confidence is to actual support
        # Perfect calibration = 1.0
        gap = abs(model_confidence - factual_support)
        return max(0.0, 1.0 - gap)

    def _estimate_confidence_from_text(self, text: str) -> float:
        """
        Heuristic confidence estimation from output language.
        Hedging phrases reduce confidence.
        """
        text_lower = text.lower()

        hedging = [
            "i think", "i believe", "might", "could be", "possibly",
            "it seems", "perhaps", "arguably", "it appears", "uncertain",
            "not sure", "may or may not", "speculative",
        ]
        assertive = [
            "clearly", "definitely", "certainly", "it is", "this is",
            "the answer is", "in fact", "without doubt", "evidence shows",
            "data confirms",
        ]

        hedge_count = sum(1 for h in hedging if h in text_lower)
        assert_count = sum(1 for a in assertive if a in text_lower)

        # Base confidence
        base = 0.5
        base += assert_count * 0.1
        base -= hedge_count * 0.1
        return max(0.1, min(0.95, base))

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claim-like sentences from text.
        Uses heuristic: sentences with assertive structure.
        """
        sentences = re.split(r'[.!?]\s+', text)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            # Skip questions
            if sent.endswith("?"):
                continue
            # Skip meta-commentary
            if any(p in sent.lower() for p in [
                "let me", "i'll", "here is", "below is", "as follows",
                "in summary", "to summarize",
            ]):
                continue
            claims.append(sent)
        return claims[:20]  # Cap at 20 claims for performance
