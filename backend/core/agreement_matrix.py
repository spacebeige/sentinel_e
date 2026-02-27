"""
============================================================
Agreement Matrix — Pairwise Model Similarity Engine
============================================================
Computes pairwise agreement between all model outputs.
Produces disagreement entropy, contradiction density,
and cluster detection.

No mode awareness. No provider awareness. Pure computation.
"""

from __future__ import annotations

import math
import logging
import re
from typing import Dict, List, Set, Tuple

from core.ensemble_schemas import (
    AgreementMatrix,
    PairwiseScore,
    StructuredModelOutput,
    StanceVector,
)

logger = logging.getLogger("AgreementMatrixEngine")


class AgreementMatrixEngine:
    """Computes full pairwise agreement matrix from structured model outputs."""

    # ── Public API ───────────────────────────────────────────

    def compute(
        self, outputs: List[StructuredModelOutput]
    ) -> AgreementMatrix:
        """
        Compute full pairwise agreement matrix.

        Args:
            outputs: List of structured model outputs (only successful ones).

        Returns:
            AgreementMatrix with all pair scores and aggregate metrics.
        """
        if len(outputs) < 2:
            return AgreementMatrix(
                mean_agreement=1.0 if len(outputs) == 1 else 0.0,
                min_agreement=1.0 if len(outputs) == 1 else 0.0,
                max_agreement=1.0 if len(outputs) == 1 else 0.0,
            )

        pairs: List[PairwiseScore] = []
        agreements: List[float] = []

        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                score = self._compute_pair(outputs[i], outputs[j])
                pairs.append(score)
                agreements.append(score.overall_agreement)

        mean_agr = sum(agreements) / len(agreements) if agreements else 0.0
        min_agr = min(agreements) if agreements else 0.0
        max_agr = max(agreements) if agreements else 0.0

        clusters = self._detect_clusters(outputs, pairs)
        dissenters = self._find_dissenters(outputs, pairs, mean_agr)

        return AgreementMatrix(
            pairs=pairs,
            mean_agreement=mean_agr,
            min_agreement=min_agr,
            max_agreement=max_agr,
            agreement_clusters=clusters,
            dissenting_models=dissenters,
        )

    def compute_disagreement_entropy(
        self, outputs: List[StructuredModelOutput]
    ) -> float:
        """
        Compute Shannon entropy of position distribution.

        Groups models by position similarity, then computes entropy
        over the group distribution. Higher entropy = more disagreement.
        """
        if len(outputs) < 2:
            return 0.0

        # Cluster positions by keyword overlap
        clusters: List[Set[str]] = []
        assigned: Set[int] = set()

        for i in range(len(outputs)):
            if i in assigned:
                continue
            cluster = {outputs[i].model_id}
            assigned.add(i)
            words_i = self._extract_keywords(outputs[i].position)

            for j in range(i + 1, len(outputs)):
                if j in assigned:
                    continue
                words_j = self._extract_keywords(outputs[j].position)
                sim = self._jaccard(words_i, words_j)
                if sim > 0.4:  # Same cluster threshold
                    cluster.add(outputs[j].model_id)
                    assigned.add(j)

            clusters.append(cluster)

        # Add any unassigned as singletons
        for i, o in enumerate(outputs):
            if i not in assigned:
                clusters.append({o.model_id})

        # Compute Shannon entropy
        total = sum(len(c) for c in clusters)
        if total == 0:
            return 0.0

        entropy = 0.0
        for cluster in clusters:
            p = len(cluster) / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by max possible entropy (all unique positions)
        max_entropy = math.log2(len(outputs)) if len(outputs) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def compute_contradiction_density(
        self, outputs: List[StructuredModelOutput]
    ) -> float:
        """
        Fraction of model pairs with contradicting positions.

        A contradiction is detected when:
        - Positions use negation patterns (one says X, another says NOT X)
        - Stance vectors are in opposing quadrants
        - Confidence is high on both sides (both believe they're right)
        """
        if len(outputs) < 2:
            return 0.0

        contradictions = 0
        total_pairs = 0

        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                total_pairs += 1
                if self._detect_contradiction(outputs[i], outputs[j]):
                    contradictions += 1

        return contradictions / total_pairs if total_pairs > 0 else 0.0

    # ── Pairwise Computation ─────────────────────────────────

    def _compute_pair(
        self, a: StructuredModelOutput, b: StructuredModelOutput
    ) -> PairwiseScore:
        """Compute pairwise similarity between two model outputs."""
        pos_sim = self._text_similarity(a.position, b.position)
        reason_sim = self._text_similarity(a.reasoning, b.reasoning)
        stance_dist = self._stance_distance(a.stance_vector, b.stance_vector)
        assumption_overlap = self._list_overlap(a.assumptions, b.assumptions)

        # Overall agreement: weighted combination
        # Position and reasoning matter most; stance distance is a penalty
        overall = (
            0.35 * pos_sim
            + 0.30 * reason_sim
            + 0.20 * assumption_overlap
            + 0.15 * max(0, 1.0 - stance_dist)
        )

        return PairwiseScore(
            model_a=a.model_id,
            model_b=b.model_id,
            position_similarity=pos_sim,
            reasoning_similarity=reason_sim,
            stance_distance=stance_dist,
            assumption_overlap=assumption_overlap,
            overall_agreement=max(0.0, min(1.0, overall)),
        )

    # ── Text Similarity ──────────────────────────────────────

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Trigram Jaccard similarity between two texts."""
        if not text_a or not text_b:
            return 0.0

        trigrams_a = self._get_trigrams(text_a.lower())
        trigrams_b = self._get_trigrams(text_b.lower())

        if not trigrams_a or not trigrams_b:
            # Fall back to word-level Jaccard
            words_a = self._extract_keywords(text_a)
            words_b = self._extract_keywords(text_b)
            return self._jaccard(words_a, words_b)

        return self._jaccard(trigrams_a, trigrams_b)

    def _get_trigrams(self, text: str) -> Set[str]:
        """Extract character trigrams."""
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        trigrams: Set[str] = set()
        for word in words:
            for i in range(max(0, len(word) - 2)):
                trigrams.add(word[i:i+3])
        return trigrams

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords."""
        if not text:
            return set()
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'and',
            'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
            'neither', 'each', 'every', 'all', 'any', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'only', 'own',
            'same', 'than', 'too', 'very', 'just', 'because', 'this',
            'that', 'these', 'those', 'it', 'its', 'i', 'we', 'they',
        }
        words = re.findall(r'[a-z]+', text.lower())
        return {w for w in words if len(w) > 2 and w not in stop_words}

    def _jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Jaccard similarity."""
        if not set_a and not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    # ── Stance Distance ──────────────────────────────────────

    def _stance_distance(self, a: StanceVector, b: StanceVector) -> float:
        """Euclidean distance in 5D stance space, normalized to [0, 1]."""
        va = a.to_vector()
        vb = b.to_vector()
        dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))
        # Max possible distance in 5D unit cube = sqrt(5) ≈ 2.236
        return min(dist / math.sqrt(5), 1.0)

    # ── List Overlap ─────────────────────────────────────────

    def _list_overlap(self, list_a: List[str], list_b: List[str]) -> float:
        """Semantic overlap between two lists of strings."""
        if not list_a or not list_b:
            return 0.0

        keywords_a = set()
        for item in list_a:
            keywords_a.update(self._extract_keywords(item))

        keywords_b = set()
        for item in list_b:
            keywords_b.update(self._extract_keywords(item))

        return self._jaccard(keywords_a, keywords_b)

    # ── Contradiction Detection ──────────────────────────────

    def _detect_contradiction(
        self, a: StructuredModelOutput, b: StructuredModelOutput
    ) -> bool:
        """Detect if two outputs contradict each other."""
        # Check 1: Negation patterns
        neg_patterns = [
            (r'\bnot\b', r'\bshould\b'),
            (r'\bno\b', r'\byes\b'),
            (r'\bimpossible\b', r'\bpossible\b'),
            (r'\bincorrect\b', r'\bcorrect\b'),
            (r'\bfalse\b', r'\btrue\b'),
            (r'\bdisagree\b', r'\bagree\b'),
            (r'\bunsafe\b', r'\bsafe\b'),
            (r'\binvalid\b', r'\bvalid\b'),
        ]

        pos_a = a.position.lower()
        pos_b = b.position.lower()

        for neg, pos in neg_patterns:
            a_has_neg = bool(re.search(neg, pos_a))
            b_has_neg = bool(re.search(neg, pos_b))
            a_has_pos = bool(re.search(pos, pos_a))
            b_has_pos = bool(re.search(pos, pos_b))

            if (a_has_neg and b_has_pos) or (a_has_pos and b_has_neg):
                return True

        # Check 2: High confidence + low agreement
        if a.confidence > 0.7 and b.confidence > 0.7:
            keywords_a = self._extract_keywords(a.position)
            keywords_b = self._extract_keywords(b.position)
            if self._jaccard(keywords_a, keywords_b) < 0.15:
                return True

        # Check 3: Opposing stance vectors
        stance_dist = self._stance_distance(a.stance_vector, b.stance_vector)
        if stance_dist > 0.7 and a.confidence > 0.6 and b.confidence > 0.6:
            return True

        return False

    # ── Cluster Detection ────────────────────────────────────

    def _detect_clusters(
        self,
        outputs: List[StructuredModelOutput],
        pairs: List[PairwiseScore],
    ) -> List[List[str]]:
        """Simple greedy clustering based on agreement threshold."""
        if len(outputs) < 2:
            return [[o.model_id] for o in outputs]

        # Build adjacency from pairs with high agreement
        adj: Dict[str, Set[str]] = {o.model_id: set() for o in outputs}
        for pair in pairs:
            if pair.overall_agreement >= 0.5:
                adj[pair.model_a].add(pair.model_b)
                adj[pair.model_b].add(pair.model_a)

        # Greedy connected components
        visited: Set[str] = set()
        clusters: List[List[str]] = []

        for output in outputs:
            mid = output.model_id
            if mid in visited:
                continue
            cluster: List[str] = []
            stack = [mid]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                stack.extend(adj.get(node, set()) - visited)
            clusters.append(sorted(cluster))

        return clusters

    def _find_dissenters(
        self,
        outputs: List[StructuredModelOutput],
        pairs: List[PairwiseScore],
        mean_agreement: float,
    ) -> List[str]:
        """Find models whose average agreement is below the global mean."""
        if len(outputs) < 2:
            return []

        model_agreements: Dict[str, List[float]] = {o.model_id: [] for o in outputs}
        for pair in pairs:
            model_agreements[pair.model_a].append(pair.overall_agreement)
            model_agreements[pair.model_b].append(pair.overall_agreement)

        dissenters: List[str] = []
        threshold = mean_agreement - 0.15  # Models notably below mean
        for mid, agrs in model_agreements.items():
            avg = sum(agrs) / len(agrs) if agrs else 0.0
            if avg < threshold:
                dissenters.append(mid)

        return dissenters
