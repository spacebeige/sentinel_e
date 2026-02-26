"""
============================================================
Ensemble Session Engine — Sentinel-E v6.0
============================================================
Tracks session evolution across ensemble requests.
Every response updates session state. No mode awareness.

Wraps and extends the existing SessionIntelligence with
ensemble-specific tracking: entropy history, model reliability,
topic clusters, and fragility evolution.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

from backend.core.ensemble_schemas import (
    SessionIntelligenceSnapshot,
    EnsembleMetrics,
    AgreementMatrix,
    CalibratedConfidence,
    StructuredModelOutput,
)

logger = logging.getLogger("EnsembleSessionEngine")


class EnsembleSessionEngine:
    """
    Persistent session state that evolves with every ensemble request.

    Not mode-aware. Not provider-aware. Pure session tracking.
    """

    def __init__(self, session_id: str = ""):
        self._session_id = session_id
        self._message_count: int = 0
        self._boundary_hits: int = 0
        self._confidence_history: List[float] = []
        self._entropy_history: List[float] = []
        self._fragility_history: List[float] = []
        self._topic_keywords: List[Set[str]] = []
        self._model_success_counts: Dict[str, int] = defaultdict(int)
        self._model_total_counts: Dict[str, int] = defaultdict(int)
        self._queries: List[str] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str):
        self._session_id = value

    def update(
        self,
        query: str,
        outputs: List[StructuredModelOutput],
        metrics: EnsembleMetrics,
        matrix: AgreementMatrix,
        confidence: CalibratedConfidence,
        boundary_hit: bool = False,
    ) -> SessionIntelligenceSnapshot:
        """
        Update session state with new request results.
        Returns current snapshot.
        """
        self._message_count += 1
        self._queries.append(query)

        if boundary_hit:
            self._boundary_hits += 1

        self._confidence_history.append(confidence.final_confidence)
        self._entropy_history.append(metrics.disagreement_entropy)
        self._fragility_history.append(metrics.fragility_score)

        for output in outputs:
            self._model_total_counts[output.model_id] += 1
            if output.succeeded:
                self._model_success_counts[output.model_id] += 1

        query_keywords = self._extract_keywords(query)
        self._topic_keywords.append(query_keywords)

        return self.snapshot()

    def snapshot(self) -> SessionIntelligenceSnapshot:
        """Get current session state snapshot."""
        return SessionIntelligenceSnapshot(
            session_id=self._session_id,
            message_count=self._message_count,
            boundary_hits=self._boundary_hits,
            depth=self._compute_depth(),
            volatility=self._compute_volatility(),
            topic_clusters=self._compute_topic_clusters(),
            confidence_history=list(self._confidence_history),
            entropy_history=list(self._entropy_history),
            fragility_history=list(self._fragility_history),
            model_reliability=self._compute_reliability(),
            cumulative_agreement=self._compute_cumulative_agreement(),
            inferred_domain=self._infer_domain(),
            user_expertise_estimate=self._estimate_expertise(),
        )

    def _compute_depth(self) -> float:
        if self._message_count == 0:
            return 0.0
        base = min(math.log2(self._message_count + 1) / 4.0, 1.0)
        evolution_bonus = 0.0
        if len(self._topic_keywords) >= 2:
            follow_up_count = 0
            for i in range(1, len(self._topic_keywords)):
                overlap = self._topic_keywords[i] & self._topic_keywords[i - 1]
                if len(overlap) > 0:
                    follow_up_count += 1
            evolution_bonus = follow_up_count / (len(self._topic_keywords) - 1)
        return min(1.0, base * 0.6 + evolution_bonus * 0.4)

    def _compute_volatility(self) -> float:
        if len(self._confidence_history) < 2:
            return 0.0
        mean_conf = sum(self._confidence_history) / len(self._confidence_history)
        variance = sum(
            (c - mean_conf) ** 2 for c in self._confidence_history
        ) / len(self._confidence_history)
        conf_vol = math.sqrt(variance)
        topic_vol = 0.0
        if len(self._topic_keywords) >= 2:
            jumps = []
            for i in range(1, len(self._topic_keywords)):
                a = self._topic_keywords[i - 1]
                b = self._topic_keywords[i]
                union = a | b
                inter = a & b
                jump = 1.0 - (len(inter) / len(union) if union else 0)
                jumps.append(jump)
            topic_vol = sum(jumps) / len(jumps)
        return min(1.0, conf_vol * 0.5 + topic_vol * 0.5)

    def _compute_topic_clusters(self) -> List[str]:
        if not self._queries:
            return []
        keyword_freq: Dict[str, int] = defaultdict(int)
        for kw_set in self._topic_keywords:
            for kw in kw_set:
                keyword_freq[kw] += 1
        sorted_kw = sorted(keyword_freq.items(), key=lambda x: -x[1])
        return [kw for kw, _ in sorted_kw[:8]]

    def _compute_reliability(self) -> Dict[str, float]:
        reliability: Dict[str, float] = {}
        for model_id in self._model_total_counts:
            total = self._model_total_counts[model_id]
            success = self._model_success_counts.get(model_id, 0)
            reliability[model_id] = success / total if total > 0 else 0.0
        return reliability

    def _compute_cumulative_agreement(self) -> float:
        if not self._entropy_history:
            return 0.0
        mean_entropy = sum(self._entropy_history) / len(self._entropy_history)
        return max(0.0, 1.0 - mean_entropy)

    def _infer_domain(self) -> str:
        clusters = self._compute_topic_clusters()
        if not clusters:
            return "general"
        domain_signals = {
            "code": {"code", "function", "class", "api", "bug", "error",
                     "python", "javascript", "typescript", "react", "database"},
            "security": {"security", "vulnerability", "attack", "threat",
                         "encryption", "authentication", "authorization"},
            "science": {"hypothesis", "experiment", "theory", "research",
                        "data", "analysis", "study", "evidence"},
            "business": {"strategy", "market", "revenue", "growth",
                         "customer", "product", "roi", "investment"},
            "philosophy": {"ethics", "moral", "consciousness", "existence",
                           "truth", "knowledge", "belief", "value"},
            "medical": {"health", "disease", "treatment", "diagnosis",
                        "patient", "clinical", "medical", "symptoms"},
        }
        cluster_set = set(clusters)
        best_domain = "general"
        best_score = 0
        for domain, signals in domain_signals.items():
            score = len(cluster_set & signals)
            if score > best_score:
                best_score = score
                best_domain = domain
        return best_domain

    def _estimate_expertise(self) -> float:
        if not self._queries:
            return 0.5
        total_length = sum(len(q) for q in self._queries)
        avg_length = total_length / len(self._queries)
        length_score = min(avg_length / 200.0, 1.0)
        depth = self._compute_depth()
        return min(1.0, length_score * 0.4 + depth * 0.6)

    def _extract_keywords(self, text: str) -> Set[str]:
        if not text:
            return set()
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'and', 'but', 'or',
            'not', 'this', 'that', 'what', 'how', 'why', 'when', 'where',
            'who', 'which', 'it', 'its', 'i', 'me', 'my', 'we', 'you',
            'your', 'they', 'them', 'their', 'about', 'there', 'here',
        }
        words = re.findall(r'[a-z]+', text.lower())
        return {w for w in words if len(w) > 2 and w not in stop_words}

    def to_dict(self) -> Dict:
        return {
            "session_id": self._session_id,
            "message_count": self._message_count,
            "boundary_hits": self._boundary_hits,
            "confidence_history": self._confidence_history,
            "entropy_history": self._entropy_history,
            "fragility_history": self._fragility_history,
            "queries": self._queries,
            "model_success_counts": dict(self._model_success_counts),
            "model_total_counts": dict(self._model_total_counts),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EnsembleSessionEngine":
        engine = cls(session_id=data.get("session_id", ""))
        engine._message_count = data.get("message_count", 0)
        engine._boundary_hits = data.get("boundary_hits", 0)
        engine._confidence_history = data.get("confidence_history", [])
        engine._entropy_history = data.get("entropy_history", [])
        engine._fragility_history = data.get("fragility_history", [])
        engine._queries = data.get("queries", [])
        for q in engine._queries:
            engine._topic_keywords.append(engine._extract_keywords(q))
        engine._model_success_counts = defaultdict(
            int, data.get("model_success_counts", {})
        )
        engine._model_total_counts = defaultdict(
            int, data.get("model_total_counts", {})
        )
        return engine
