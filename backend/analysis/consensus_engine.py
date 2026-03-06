"""
============================================================
Consensus Engine — Sentinel-E Battle Platform v2
============================================================
Derives agreement-based ranking directly from model outputs.

Scoring formula (per model):
    score = 0.35 * reasoning_coherence
          + 0.25 * evidence_support
          + 0.20 * consensus_alignment
          - 0.10 * contradiction_rate
          + 0.10 * confidence_calibration

Design rationale:
    • reasoning_coherence (35%) is the dominant signal. A model that
      produces incoherent chains even with correct conclusions is
      unreliable at inference time. High weighting prevents hallucination-
      dominant responses from winning by agreement alone.
    
    • evidence_support (25%) rewards claim-grounding. Ungrounded
      assertions that happen to match the consensus are systematically
      down-ranked — this is the primary anti-hallucination mechanism.
    
    • consensus_alignment (20%) captures agreement clusters. When two
      or more models converge on a position from independent reasoning
      paths, the result has higher epistemic value than isolated claims.
    
    • contradiction_rate (−10%) is a direct hallucination penalty.
      Models that internally contradict themselves within a single debate
      round are penalised even if their final position is well-supported.
    
    • confidence_calibration (10%) measures whether stated confidence
      tracks evidence quality. Over-confident responses on poorly
      supported claims are down-scored; calibrated uncertainty is rewarded.

Agreement clusters:
    Detected via TF-IDF + cosine similarity on final-round positions.
    Models within semantic distance < 0.4 are co-clustered.
    Cluster size influences consensus_alignment score.

Contradiction detection:
    Sentence-level negation + key-claim reversal within a single output.
    Uses lightweight regex patterns; no external NLP dependency.

No external API calls. No I/O. Pure in-process computation.
============================================================
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from core.ensemble_schemas import (
    ConsensusScore,
    ModelReasoningMetrics,
    StructuredModelOutput,
)

logger = logging.getLogger("ConsensusEngine")


# ── Composite Weight Constants ───────────────────────────────
W_REASONING_COHERENCE   = 0.35
W_EVIDENCE_SUPPORT      = 0.25
W_CONSENSUS_ALIGNMENT   = 0.20
W_CONFIDENCE_CALIB      = 0.10
W_CONTRADICTION_PENALTY = 0.10   # subtracted

# Semantic distance threshold for clustering (0 = identical, 1 = orthogonal)
CLUSTER_DISTANCE_THRESHOLD = 0.40

# Evidence marker patterns (signals that a claim is grounded)
_EVIDENCE_PATTERNS = [
    r"\bfor example\b", r"\bfor instance\b", r"\bsuch as\b",
    r"\baccording to\b", r"\bstudies show\b", r"\bresearch indicates\b",
    r"\bdata suggests\b", r"\bevidence shows\b", r"\bspeci[fic]+ally\b",
    r"\bmeaning that\b", r"\bwhich means\b", r"\bbecause\b",
    r"\btherefore\b", r"\bconsequently\b", r"\bthus\b",
    r"\bdemonstrat", r"\bproves?\b", r"\bverif", r"\bcit[ei]",
    r"\d+%", r"\$\d+", r"\d{4}",                 # Numbers / years
]
_EVIDENCE_RE = re.compile("|".join(_EVIDENCE_PATTERNS), re.IGNORECASE)

# Contradiction signal patterns (negation of a close prior claim)
_NEGATION_PATTERNS = [
    r"\bnot\b", r"\bnever\b", r"\bno\b(?!t)", r"\bfalse\b",
    r"\bwrong\b", r"\bincorrect\b", r"\bcontrary\b",
    r"\bhowever\b.*\bbut\b", r"\bdespite\b", r"\balthough\b",
    r"\byet\b", r"\bnevertheless\b", r"\bnonetheless\b",
]
_NEGATION_RE = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE)

# Coherence signals (structured logical flow)
_COHERENCE_PATTERNS = [
    r"\bfirst(ly)?\b", r"\bsecond(ly)?\b", r"\bthird(ly)?\b",
    r"\bfinal(ly)?\b", r"\bin conclusion\b", r"\bto summarize\b",
    r"\bmoreover\b", r"\bfurthermore\b", r"\bin addition\b",
    r"\bas a result\b", r"\bthis means\b", r"\btherefore\b",
    r"\bif .+, then\b", r"\bgiven that\b",
]
_COHERENCE_RE = re.compile("|".join(_COHERENCE_PATTERNS), re.IGNORECASE)


@dataclass
class AgreementCluster:
    """A group of models that have converged on semantically similar positions."""
    cluster_id: int
    model_ids: List[str] = field(default_factory=list)
    centroid_similarity: float = 0.0   # Mean pairwise similarity within cluster
    representative_model: str = ""


@dataclass
class ConsensusEngineOutput:
    """Full output of the consensus computation."""
    scores: List[ConsensusScore]
    clusters: List[AgreementCluster]
    contradicting_pairs: List[Tuple[str, str]]  # model pairs with low similarity
    stability_score: float                        # 0=fragmented, 1=fully converged
    dominant_cluster_id: Optional[int]
    dissenting_models: List[str]


class ConsensusEngine:
    """
    Derives consensus quality scores from structured debate outputs.

    Usage:
        engine = ConsensusEngine()
        result = engine.evaluate(outputs)
        # result.scores[i].composite_score → ranking signal
    """

    # ── Public interface ──────────────────────────────────────

    def evaluate(
        self,
        outputs: List[StructuredModelOutput],
        final_round_only: bool = True,
    ) -> ConsensusEngineOutput:
        """
        Run the full consensus evaluation pipeline.

        Args:
            outputs:          List of StructuredModelOutput from the debate engine.
            final_round_only: If True, clustering uses final_position text only.
                              If False, full reasoning is used (richer but noisier).

        Returns:
            ConsensusEngineOutput with ranked ConsensusScore[] and cluster data.
        """
        succeeded = [o for o in outputs if o.succeeded]
        if not succeeded:
            logger.warning("ConsensusEngine: no successful outputs to evaluate")
            return ConsensusEngineOutput(
                scores=[], clusters=[], contradicting_pairs=[],
                stability_score=0.0, dominant_cluster_id=None,
                dissenting_models=[],
            )

        # Step 1 — Per-model signal extraction
        evidence_scores     = self._score_evidence_density(succeeded)
        coherence_scores    = self._score_reasoning_coherence(succeeded)
        contradiction_rates = self._score_contradiction_rate(succeeded)
        confidence_calib    = self._score_confidence_calibration(succeeded)

        # Step 2 — Build similarity matrix for consensus alignment
        texts = [
            (o.position if final_round_only else f"{o.position} {o.reasoning}")
            for o in succeeded
        ]
        sim_matrix = self._compute_similarity_matrix(texts)

        # Step 3 — Cluster detection
        clusters = self._detect_clusters(succeeded, sim_matrix)
        dominant_cluster = self._dominant_cluster(clusters)

        # Step 4 — Consensus alignment per model
        alignment_scores = self._score_consensus_alignment(
            succeeded, sim_matrix, clusters
        )

        # Step 5 — Composite score per model
        scores: List[ConsensusScore] = []
        for i, o in enumerate(succeeded):
            rc   = coherence_scores.get(o.model_id, 0.0)
            ev   = evidence_scores.get(o.model_id, 0.0)
            ca   = alignment_scores.get(o.model_id, 0.0)
            cc   = confidence_calib.get(o.model_id, 0.0)
            cr   = contradiction_rates.get(o.model_id, 0.0)

            composite = (
                W_REASONING_COHERENCE   * rc
                + W_EVIDENCE_SUPPORT    * ev
                + W_CONSENSUS_ALIGNMENT * ca
                + W_CONFIDENCE_CALIB    * cc
                - W_CONTRADICTION_PENALTY * cr
            )
            composite = max(0.0, min(1.0, composite))

            scores.append(ConsensusScore(
                model=o.model_id,
                model_name=o.model_name,
                reasoning_coherence=round(rc, 4),
                evidence_support=round(ev, 4),
                consensus_alignment=round(ca, 4),
                confidence_calibration=round(cc, 4),
                contradiction_rate=round(cr, 4),
                composite_score=round(composite, 4),
            ))

        # Step 6 — Rank by composite score
        scores.sort(key=lambda s: s.composite_score, reverse=True)
        for rank, s in enumerate(scores, start=1):
            s.rank = rank

        # Step 7 — Identify contradiction pairs and dissenters
        contra_pairs = self._find_contradiction_pairs(succeeded, sim_matrix)
        dissenters   = self._find_dissenters(succeeded, sim_matrix)

        stability = self._compute_stability(sim_matrix, len(succeeded))

        logger.info(
            "ConsensusEngine: evaluated %d models, "
            "%d clusters, stability=%.3f",
            len(succeeded), len(clusters), stability,
        )

        return ConsensusEngineOutput(
            scores=scores,
            clusters=clusters,
            contradicting_pairs=contra_pairs,
            stability_score=round(stability, 4),
            dominant_cluster_id=(
                dominant_cluster.cluster_id if dominant_cluster else None
            ),
            dissenting_models=dissenters,
        )

    # ── Private — Signal Extraction ──────────────────────────

    def _score_evidence_density(
        self, outputs: List[StructuredModelOutput]
    ) -> Dict[str, float]:
        """
        Compute evidence density: ratio of evidence-marker sentences
        to total sentences in the reasoning field.

        Rationale: Ungrounded assertions are the dominant hallucination
        vector. Models that back claims with specific examples, data,
        or causal chains are more reliable than those that assert boldly.
        """
        result: Dict[str, float] = {}
        for o in outputs:
            text  = f"{o.position} {o.reasoning}".strip()
            sentences = self._split_sentences(text)
            if not sentences:
                result[o.model_id] = 0.0
                continue
            evidence_count = sum(
                1 for s in sentences if _EVIDENCE_RE.search(s)
            )
            result[o.model_id] = evidence_count / len(sentences)
        return result

    def _score_reasoning_coherence(
        self, outputs: List[StructuredModelOutput]
    ) -> Dict[str, float]:
        """
        Coherence score based on:
          • Structural connector density (logical flow markers)
          • Presence of at least one explicit assumption
          • Self-identified vulnerabilities (meta-cognitive awareness)
          • Sentence count (minimum threshold for depth)

        Rationale: Coherence is the structural skeleton of a sound argument.
        The presence of explicit flow markers indicates the model has
        organised its reasoning rather than stream-of-consciousness output.
        """
        result: Dict[str, float] = {}
        for o in outputs:
            text = f"{o.position} {o.reasoning}".strip()
            sentences = self._split_sentences(text)
            n = len(sentences)
            if n == 0:
                result[o.model_id] = 0.0
                continue

            # Flow connector density
            connector_count = sum(
                1 for s in sentences if _COHERENCE_RE.search(s)
            )
            connector_density = min(1.0, connector_count / max(1, n) * 3.0)

            # Bonus for explicit assumptions
            assumption_bonus = 0.1 if o.assumptions else 0.0

            # Bonus for self-identified vulnerabilities (meta-reasoning)
            vuln_bonus = 0.1 if o.vulnerabilities else 0.0

            # Depth signal: more sentences (up to ~10) → higher depth score
            depth_score = min(1.0, n / 10.0)

            score = (
                0.50 * connector_density
                + 0.20 * depth_score
                + assumption_bonus
                + vuln_bonus
            )
            result[o.model_id] = min(1.0, score)
        return result

    def _score_contradiction_rate(
        self, outputs: List[StructuredModelOutput]
    ) -> Dict[str, float]:
        """
        Estimate intra-response contradiction rate.

        Method:
          1. Extract key noun-phrases from position statement.
          2. Check if later sentences negate these phrases + contain
             a negation marker.
          3. Rate = contradicting_sentences / total_sentences.

        This is a lightweight heuristic — not an NLI classifier.
        It catches the common pattern: "X is the best approach …
        however X is problematic because …" within the same response.
        """
        result: Dict[str, float] = {}
        for o in outputs:
            text = f"{o.position} {o.reasoning}".strip()
            sentences = self._split_sentences(text)
            if len(sentences) < 3:
                result[o.model_id] = 0.0
                continue

            # Extract key terms from first 2 sentences (position claim)
            anchor_text = " ".join(sentences[:2]).lower()
            anchor_words = set(re.findall(r"\b\w{5,}\b", anchor_text)) - _STOP_WORDS

            contradictions = 0
            for sent in sentences[2:]:
                sent_lower = sent.lower()
                has_negation = bool(_NEGATION_RE.search(sent_lower))
                has_anchor   = any(w in sent_lower for w in anchor_words)
                if has_negation and has_anchor:
                    contradictions += 1

            total = max(1, len(sentences) - 2)
            result[o.model_id] = min(1.0, contradictions / total)
        return result

    def _score_confidence_calibration(
        self, outputs: List[StructuredModelOutput]
    ) -> Dict[str, float]:
        """
        Confidence calibration score.

        Penalises models that state high confidence (>0.8) but have
        low evidence density. Rewards models whose stated confidence
        tracks their evidence quality.

        Rationale: Hallucinating models tend to assert high confidence
        regardless of evidence. A calibrated model expresses uncertainty
        when its evidence is thin.
        """
        result: Dict[str, float] = {}
        evidence_scores = self._score_evidence_density(outputs)
        for o in outputs:
            ev  = evidence_scores.get(o.model_id, 0.0)
            c   = o.confidence
            # Gap between stated confidence and evidence quality
            gap = abs(c - ev)
            # Perfect calibration = 0 gap → score 1.0
            # gap of 1.0 → score 0.0
            result[o.model_id] = 1.0 - gap
        return result

    # ── Private — Similarity & Clustering ─────────────────────

    def _compute_similarity_matrix(
        self, texts: List[str]
    ) -> List[List[float]]:
        """Compute pairwise TF-IDF cosine similarity matrix."""
        n = len(texts)
        if n == 0:
            return []
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=500)
            vectors = tfidf.fit_transform(texts)
            sims = sk_cosine(vectors, vectors).tolist()
        except Exception as exc:
            logger.warning("ConsensusEngine: similarity fallback due to %s", exc)
            sims = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return sims

    def _detect_clusters(
        self,
        outputs: List[StructuredModelOutput],
        sim_matrix: List[List[float]],
    ) -> List[AgreementCluster]:
        """
        Single-linkage clustering: models within CLUSTER_DISTANCE_THRESHOLD
        of each other are co-clustered.

        Agreement clusters provide epistemic value: when independent
        reasoning paths converge on the same position, that position is
        more likely to be correct than a single isolated assertion.
        """
        n = len(outputs)
        if n == 0:
            return []

        assigned: Dict[int, int] = {}     # model_idx → cluster_id
        cluster_counter = 0

        for i in range(n):
            if i in assigned:
                continue
            cid = cluster_counter
            cluster_counter += 1
            assigned[i] = cid
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                similarity = sim_matrix[i][j] if sim_matrix else 0.0
                if similarity >= (1.0 - CLUSTER_DISTANCE_THRESHOLD):
                    assigned[j] = cid

        # Build cluster objects
        cluster_map: Dict[int, List[int]] = {}
        for idx, cid in assigned.items():
            cluster_map.setdefault(cid, []).append(idx)

        clusters: List[AgreementCluster] = []
        for cid, members in cluster_map.items():
            if len(members) == 0:
                continue
            # Centroid similarity = mean pairwise sim within cluster
            if len(members) == 1:
                centroid_sim = 1.0
            else:
                sims_within = [
                    sim_matrix[a][b]
                    for a in members for b in members if a < b
                ]
                centroid_sim = sum(sims_within) / max(1, len(sims_within))

            model_ids = [outputs[idx].model_id for idx in members]
            representative = outputs[members[0]].model_id  # highest-confidence

            clusters.append(AgreementCluster(
                cluster_id=cid,
                model_ids=model_ids,
                centroid_similarity=round(centroid_sim, 4),
                representative_model=representative,
            ))

        clusters.sort(key=lambda c: len(c.model_ids), reverse=True)
        return clusters

    def _score_consensus_alignment(
        self,
        outputs: List[StructuredModelOutput],
        sim_matrix: List[List[float]],
        clusters: List[AgreementCluster],
    ) -> Dict[str, float]:
        """
        Consensus alignment per model: mean similarity to all other models,
        boosted by cluster membership size.

        A model in a large consensus cluster gets a higher alignment score
        than a lone dissenter — even if both have coherent arguments.
        This reflects the epistemic principle that independent convergence
        is stronger evidence than isolated claims.
        """
        result: Dict[str, float] = {}
        n = len(outputs)
        if n <= 1:
            for o in outputs:
                result[o.model_id] = 1.0
            return result

        # Build cluster size lookup
        cluster_membership: Dict[str, int] = {}
        for c in clusters:
            for mid in c.model_ids:
                cluster_membership[mid] = len(c.model_ids)

        for i, o in enumerate(outputs):
            row = sim_matrix[i] if sim_matrix else []
            mean_sim = (sum(row) - 1.0) / (n - 1) if n > 1 else 0.0
            cluster_bonus = (cluster_membership.get(o.model_id, 1) - 1) / max(1, n - 1)
            score = 0.70 * mean_sim + 0.30 * cluster_bonus
            result[o.model_id] = min(1.0, max(0.0, score))
        return result

    def _find_contradiction_pairs(
        self,
        outputs: List[StructuredModelOutput],
        sim_matrix: List[List[float]],
        low_sim_threshold: float = 0.20,
    ) -> List[Tuple[str, str]]:
        """
        Identify pairs of models with very low semantic similarity
        (positional contradictions at the debate level).
        """
        pairs: List[Tuple[str, str]] = []
        n = len(outputs)
        for i in range(n):
            for j in range(i + 1, n):
                val = sim_matrix[i][j] if sim_matrix else 1.0
                if val < low_sim_threshold:
                    pairs.append((outputs[i].model_id, outputs[j].model_id))
        return pairs

    def _find_dissenters(
        self,
        outputs: List[StructuredModelOutput],
        sim_matrix: List[List[float]],
        threshold: float = 0.35,
    ) -> List[str]:
        """Models whose mean similarity to others is below `threshold`."""
        n = len(outputs)
        if n <= 1:
            return []
        dissenters: List[str] = []
        for i, o in enumerate(outputs):
            row = sim_matrix[i] if sim_matrix else []
            mean_sim = (sum(row) - 1.0) / (n - 1)
            if mean_sim < threshold:
                dissenters.append(o.model_id)
        return dissenters

    def _compute_stability(
        self, sim_matrix: List[List[float]], n: int
    ) -> float:
        """
        Global consensus stability: mean of all pairwise similarities.
        0 = total fragmentation, 1 = full convergence.
        """
        if n <= 1 or not sim_matrix:
            return 1.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += sim_matrix[i][j]
                count += 1
        return total / max(1, count)

    def _dominant_cluster(
        self, clusters: List[AgreementCluster]
    ) -> Optional[AgreementCluster]:
        """Return the largest cluster (already sorted by size desc)."""
        return clusters[0] if clusters else None

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Lightweight sentence splitter — no NLTK dependency."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 8]


# Common English stop words used in contradiction detection
_STOP_WORDS: Set[str] = {
    "about", "above", "after", "again", "against", "also", "another",
    "because", "before", "between", "could", "during", "either",
    "every", "first", "from", "have", "having", "hence", "their",
    "these", "those", "though", "through", "under", "using", "where",
    "which", "while", "within", "would", "there", "other", "since",
    "still", "being",
}
