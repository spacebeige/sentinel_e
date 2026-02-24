"""
Hallucination Control Gate — Sentinel-E Autonomous Reasoning Engine

Post-generation verification pipeline:
  1. Decompose response into atomic sentences
  2. Classify each sentence (factual, opinion, connective)
  3. Verify factual claims against evidence & knowledge graph
  4. Compute coverage score
  5. Strip/soften unsupported claims or trigger regeneration

Integrates with:
  - backend/core/evidence_engine.py (EvidenceEngine)
  - backend/core/confidence_engine.py (ConfidenceComponents)
  - backend/core/knowledge_memory.py (KnowledgeGraph)
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("HallucinationGate")

# ============================================================
# CONFIGURATION
# ============================================================

COVERAGE_THRESHOLDS = {
    "standard":  0.50,
    "debate":    0.55,
    "evidence":  0.70,
    "glass":     0.60,
    "stress":    0.50,
}

SUPPORT_THRESHOLD_VERIFIED = 0.75
SUPPORT_THRESHOLD_PARTIAL = 0.55

MAX_REGEN_ATTEMPTS = {
    "standard": 1,
    "debate": 2,
    "evidence": 3,
    "glass": 2,
    "stress": 1,
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class VerifiedSentence:
    text: str
    support_score: float
    source_refs: List[str] = field(default_factory=list)
    status: str = "verified"  # verified | partially_supported

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "support_score": round(self.support_score, 4),
            "source_refs": self.source_refs[:5],
            "status": self.status,
        }


@dataclass
class UnsupportedSentence:
    text: str
    best_match_score: float
    status: str = "unsupported"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "best_match_score": round(self.best_match_score, 4),
            "status": self.status,
        }


@dataclass
class VerificationResult:
    status: str = "verified"  # verified | regenerate
    coverage: float = 1.0
    clean_response: str = ""
    verified: List[VerifiedSentence] = field(default_factory=list)
    unsupported: List[UnsupportedSentence] = field(default_factory=list)
    opinion_count: int = 0
    traceability_map: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "coverage": round(self.coverage, 4),
            "verified_count": len(self.verified),
            "unsupported_count": len(self.unsupported),
            "opinion_count": self.opinion_count,
            "confidence_score": round(self.confidence_score, 4),
            "traceability_map": self.traceability_map[:10],
        }


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences. Handles common abbreviations."""
    # Protect common abbreviations
    protected = text
    abbrevs = ["Mr.", "Mrs.", "Dr.", "Prof.", "Jr.", "Sr.", "vs.", "etc.", "i.e.", "e.g."]
    for abbr in abbrevs:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', protected)

    # Restore abbreviations
    sentences = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip()
        if p and len(p) > 5:  # Skip very short fragments
            sentences.append(p)

    return sentences


# ============================================================
# SENTENCE CLASSIFICATION
# ============================================================

FACTUAL_PATTERNS = [
    r"\b(is|are|was|were|has|have|had)\b.*\b(approximately|about|roughly|exactly|precisely)\b",
    r"\b\d+(\.\d+)?%",  # Percentages
    r"\b\d{4}\b",  # Years
    r"\b(according\s+to|based\s+on|research\s+(shows?|indicates?|suggests?))\b",
    r"\b(studies?\s+(show|indicate|suggest|found|reveal))\b",
    r"\b(proven|confirmed|established|documented|recorded)\b",
    r"\b(million|billion|trillion|thousand)\b",
    r"\b(increase|decrease|rise|fall|grow|decline|drop).*by\b",
]

CAUSAL_PATTERNS = [
    r"\b(cause[sd]?|leads?\s+to|results?\s+in|due\s+to|because)\b",
    r"\b(therefore|consequently|thus|hence|as\s+a\s+result)\b",
    r"\b(if.*then|when.*will|whenever)\b",
]

OPINION_PATTERNS = [
    r"\b(I\s+think|I\s+believe|in\s+my\s+opinion|arguably|perhaps|probably)\b",
    r"\b(could\s+be|might\s+be|may\s+be|possibly)\b",
    r"\b(seems?\s+like|appears?\s+to|looks?\s+like)\b",
]


def classify_sentence_type(sentence: str) -> str:
    """Classify a sentence as factual, causal, statistical, opinion, or connective."""
    s_lower = sentence.lower()

    # Check opinion first (hedged language takes priority)
    for pattern in OPINION_PATTERNS:
        if re.search(pattern, s_lower):
            return "opinion"

    # Check causal
    for pattern in CAUSAL_PATTERNS:
        if re.search(pattern, s_lower):
            return "causal_claim"

    # Check factual/statistical
    for pattern in FACTUAL_PATTERNS:
        if re.search(pattern, s_lower):
            return "factual_claim"

    # If it contains numbers + specific nouns, likely factual
    if re.search(r'\b\d+', s_lower) and len(sentence.split()) > 5:
        return "statistical_claim"

    # Default: connective/procedural (pass-through)
    return "connective"


# ============================================================
# COSINE SIMILARITY
# ============================================================

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# HALLUCINATION GATE
# ============================================================

class HallucinationGate:
    """
    Post-generation verification pipeline.
    Decomposes response into sentences, verifies factual claims
    against evidence, and strips/softens unsupported content.
    """

    def __init__(self, embed_fn=None, mode: str = "standard"):
        """
        Args:
            embed_fn: function(text) -> np.ndarray. Required for
                      semantic matching against evidence.
            mode: Sentinel mode (governs thresholds).
        """
        self._embed_fn = embed_fn
        self.mode = mode

    def verify(
        self,
        response_text: str,
        evidence_chunks: List[Dict[str, Any]] = None,
        knowledge_claims: List[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Verify a model response against available evidence.
        
        Args:
            response_text: The generated response text.
            evidence_chunks: List of dicts with 'content', 'embedding', 'source_url'.
            knowledge_claims: List of dicts with 'claim_text', 'source_urls', 'embedding'.
        
        Returns:
            VerificationResult with coverage score and clean response.
        """
        evidence_chunks = evidence_chunks or []
        knowledge_claims = knowledge_claims or []

        sentences = split_into_sentences(response_text)
        if not sentences:
            return VerificationResult(
                status="verified",
                coverage=1.0,
                clean_response=response_text,
            )

        verified = []
        unsupported = []
        opinion_count = 0
        traceability = []

        for sentence in sentences:
            stype = classify_sentence_type(sentence)

            if stype in ("factual_claim", "causal_claim", "statistical_claim"):
                # Must be verifiable
                support = self._find_support(sentence, evidence_chunks, knowledge_claims)

                if support["max_similarity"] >= SUPPORT_THRESHOLD_VERIFIED:
                    vs = VerifiedSentence(
                        text=sentence,
                        support_score=support["max_similarity"],
                        source_refs=support["source_refs"],
                        status="verified",
                    )
                    verified.append(vs)
                    traceability.append({
                        "sentence": sentence[:100],
                        "sources": support["source_refs"][:3],
                        "score": round(support["max_similarity"], 3),
                        "status": "verified",
                    })
                elif support["max_similarity"] >= SUPPORT_THRESHOLD_PARTIAL:
                    vs = VerifiedSentence(
                        text=sentence,
                        support_score=support["max_similarity"],
                        source_refs=support["source_refs"],
                        status="partially_supported",
                    )
                    verified.append(vs)
                    traceability.append({
                        "sentence": sentence[:100],
                        "sources": support["source_refs"][:3],
                        "score": round(support["max_similarity"], 3),
                        "status": "partial",
                    })
                else:
                    us = UnsupportedSentence(
                        text=sentence,
                        best_match_score=support["max_similarity"],
                    )
                    unsupported.append(us)
                    traceability.append({
                        "sentence": sentence[:100],
                        "sources": [],
                        "score": round(support["max_similarity"], 3),
                        "status": "unsupported",
                    })

            elif stype == "opinion":
                opinion_count += 1

            # connective: pass-through, no verification needed

        # Compute coverage
        total_factual = len(verified) + len(unsupported)
        coverage = len(verified) / total_factual if total_factual > 0 else 1.0

        # Check against mode threshold
        threshold = COVERAGE_THRESHOLDS.get(self.mode, 0.50)

        if coverage < threshold:
            return VerificationResult(
                status="regenerate",
                coverage=coverage,
                clean_response=response_text,
                verified=verified,
                unsupported=unsupported,
                opinion_count=opinion_count,
                traceability_map=traceability,
                confidence_score=0.0,
            )

        # Build clean response
        clean_response = self._clean_response(response_text, unsupported)

        # Compute confidence
        confidence = self._compute_confidence(verified, unsupported, coverage)

        return VerificationResult(
            status="verified",
            coverage=coverage,
            clean_response=clean_response,
            verified=verified,
            unsupported=unsupported,
            opinion_count=opinion_count,
            traceability_map=traceability,
            confidence_score=confidence,
        )

    def _find_support(
        self,
        sentence: str,
        evidence_chunks: List[Dict[str, Any]],
        knowledge_claims: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Find supporting evidence for a sentence."""
        best_sim = 0.0
        source_refs = []

        if self._embed_fn is None:
            # No embedding function — fall back to keyword overlap
            return self._keyword_support(sentence, evidence_chunks, knowledge_claims)

        try:
            sent_embedding = self._embed_fn(sentence)
            if isinstance(sent_embedding, list):
                sent_embedding = np.array(sent_embedding, dtype=np.float32)
        except Exception:
            return self._keyword_support(sentence, evidence_chunks, knowledge_claims)

        # Check evidence chunks
        for chunk in evidence_chunks:
            chunk_emb = chunk.get("embedding")
            if chunk_emb is None:
                continue
            if isinstance(chunk_emb, list):
                chunk_emb = np.array(chunk_emb, dtype=np.float32)
            sim = _cosine_sim(sent_embedding, chunk_emb)
            if sim > best_sim:
                best_sim = sim
                source_refs = [chunk.get("source_url", "")]
            elif sim > 0.60:
                source_refs.append(chunk.get("source_url", ""))

        # Check knowledge claims
        for claim in knowledge_claims:
            claim_emb = claim.get("embedding")
            if claim_emb is None:
                # Try to embed
                try:
                    claim_emb = self._embed_fn(claim.get("claim_text", ""))
                    if isinstance(claim_emb, list):
                        claim_emb = np.array(claim_emb, dtype=np.float32)
                except Exception:
                    continue
            else:
                if isinstance(claim_emb, list):
                    claim_emb = np.array(claim_emb, dtype=np.float32)

            sim = _cosine_sim(sent_embedding, claim_emb)
            if sim > best_sim:
                best_sim = sim
                source_refs = claim.get("source_urls", [])
            elif sim > 0.60:
                source_refs.extend(claim.get("source_urls", []))

        return {
            "max_similarity": best_sim,
            "source_refs": list(set(r for r in source_refs if r)),
        }

    def _keyword_support(
        self,
        sentence: str,
        evidence_chunks: List[Dict[str, Any]],
        knowledge_claims: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fallback: keyword-based support detection."""
        sent_words = set(sentence.lower().split())
        best_overlap = 0.0
        source_refs = []

        for chunk in evidence_chunks:
            content = chunk.get("content", "")
            content_words = set(content.lower().split())
            if not content_words:
                continue
            overlap = len(sent_words & content_words) / max(len(sent_words | content_words), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                source_refs = [chunk.get("source_url", "")]

        for claim in knowledge_claims:
            text = claim.get("claim_text", "")
            claim_words = set(text.lower().split())
            if not claim_words:
                continue
            overlap = len(sent_words & claim_words) / max(len(sent_words | claim_words), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                source_refs = claim.get("source_urls", [])

        return {
            "max_similarity": best_overlap,
            "source_refs": list(set(r for r in source_refs if r)),
        }

    def _clean_response(self, response_text: str, unsupported: List[UnsupportedSentence]) -> str:
        """Strip or soften unsupported claims in the response."""
        clean = response_text
        for us in unsupported:
            if self.mode == "evidence":
                # Evidence mode: explicit tag
                clean = clean.replace(us.text, f"[Unverified: {us.text}]")
            else:
                # Standard/other: soften language
                softened = self._soften(us.text)
                clean = clean.replace(us.text, softened)
        return clean

    def _soften(self, sentence: str) -> str:
        """Soften a factual claim to indicate uncertainty."""
        # Simple hedging: prepend qualifier
        # "X is Y" → "X may be Y (unverified)"
        hedged = sentence.rstrip(".")
        return f"{hedged} (unverified)."

    def _compute_confidence(
        self,
        verified: List[VerifiedSentence],
        unsupported: List[UnsupportedSentence],
        coverage: float,
    ) -> float:
        """Compute response confidence from verification results."""
        base = coverage

        # Penalty for partially supported
        partial_count = sum(1 for v in verified if v.status == "partially_supported")
        partial_penalty = partial_count * 0.05

        # Bonus for high-confidence verified sentences
        high_conf = sum(1 for v in verified if v.support_score > 0.85)
        bonus = min(high_conf * 0.03, 0.15)

        return max(0.0, min(1.0, base - partial_penalty + bonus))
