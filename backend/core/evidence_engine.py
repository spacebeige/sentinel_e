"""
Evidence Engine — Sentinel-E Omega Cognitive Kernel 3.0

Provides evidence-backed reasoning via external search:
- Tavily web search integration (from sentinel_x_omega_cognitive.py)
- Source reliability scoring
- Contradiction detection (cross-source comparison)
- Data lineage tracking
- Evidence confidence computation

Output: EvidenceResult dict with sources, contradictions, lineage, confidence
"""

import os
import re
import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger("Omega-EvidenceEngine")


# ============================================================
# SOURCE RELIABILITY HEURISTICS
# ============================================================

TRUSTED_DOMAINS = {
    # Tier 1 — High reliability
    "arxiv.org": 0.90,
    "nature.com": 0.92,
    "science.org": 0.92,
    "ieee.org": 0.88,
    "acm.org": 0.88,
    "gov": 0.85,
    "edu": 0.83,
    "who.int": 0.90,
    "nih.gov": 0.90,
    "cdc.gov": 0.88,
    # Tier 2 — Moderate reliability
    "wikipedia.org": 0.70,
    "stackoverflow.com": 0.72,
    "github.com": 0.70,
    "docs.python.org": 0.85,
    "docs.microsoft.com": 0.82,
    "developer.mozilla.org": 0.85,
    "cloud.google.com": 0.80,
    # Tier 3 — News/media (variable)
    "reuters.com": 0.78,
    "bbc.com": 0.75,
    "nytimes.com": 0.75,
    "washingtonpost.com": 0.73,
    # Tier 4 — Community/blog (lower reliability)
    "medium.com": 0.55,
    "reddit.com": 0.45,
    "quora.com": 0.50,
    "twitter.com": 0.40,
    "x.com": 0.40,
}

UNRELIABLE_PATTERNS = [
    r"(sponsored|advertisement|affiliate|paid\s+promotion)",
    r"(click\s+here|subscribe|sign\s+up\s+now)",
    r"(miracle|guaranteed|revolutionary|breakthrough)",
]


@dataclass
class EvidenceSource:
    """A single evidence source from external search."""
    url: str
    title: str
    content_snippet: str
    reliability_score: float = 0.5      # 0.0–1.0
    domain: str = ""
    retrieved_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    content_hash: str = ""               # For dedup

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content_snippet": self.content_snippet[:300],
            "reliability_score": round(self.reliability_score, 3),
            "domain": self.domain,
            "retrieved_at": self.retrieved_at,
        }


@dataclass
class Contradiction:
    """Detected contradiction between sources."""
    source_a_url: str
    source_b_url: str
    claim_a: str
    claim_b: str
    severity: float = 0.5               # 0.0–1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_a": self.source_a_url,
            "source_b": self.source_b_url,
            "claim_a": self.claim_a[:200],
            "claim_b": self.claim_b[:200],
            "severity": round(self.severity, 3),
        }


@dataclass
class EvidenceResult:
    """Complete evidence analysis result."""
    query: str
    sources: List[EvidenceSource] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    evidence_confidence: float = 0.5     # 0.0–1.0
    source_agreement: float = 0.0        # 0.0–1.0
    lineage: List[Dict[str, str]] = field(default_factory=list)  # Processing steps log
    search_executed: bool = False
    error: Optional[str] = None
    # 3.X additions
    claims: List[Dict[str, Any]] = field(default_factory=list)  # Extracted atomic claims
    claim_overlap_matrix: Dict[str, Any] = field(default_factory=dict)  # Source-to-claim mapping
    evidence_strength: Dict[str, Any] = field(default_factory=dict)  # Per-source strength scores
    traceability_map: List[Dict[str, Any]] = field(default_factory=list)  # Answer sentence → sources

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "sources": [s.to_dict() for s in self.sources],
            "source_count": len(self.sources),
            "contradictions": [c.to_dict() for c in self.contradictions],
            "contradiction_count": len(self.contradictions),
            "evidence_confidence": round(self.evidence_confidence, 4),
            "source_agreement": round(self.source_agreement, 4),
            "lineage": self.lineage,
            "search_executed": self.search_executed,
            "error": self.error,
            "claims": self.claims,
            "claim_overlap_matrix": self.claim_overlap_matrix,
            "evidence_strength": self.evidence_strength,
            "traceability_map": self.traceability_map,
        }


class EvidenceEngine:
    """
    Evidence-backed reasoning engine.
    
    Pipeline:
    1. External search via Tavily API
    2. Source reliability scoring (domain heuristics + content analysis)
    3. Contradiction detection (cross-source keyword comparison)
    4. Evidence confidence computation
    5. Data lineage tracking
    """

    def __init__(self):
        self._tavily_client = None
        self._tavily_available = False
        self._init_tavily()

    def _init_tavily(self):
        """Initialize Tavily client if API key available."""
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if api_key:
                from tavily import TavilyClient
                self._tavily_client = TavilyClient(api_key=api_key)
                self._tavily_available = True
                logger.info("Evidence Engine: Tavily client initialized.")
            else:
                logger.warning("Evidence Engine: TAVILY_API_KEY not set. Running in offline mode.")
        except ImportError:
            logger.warning("Evidence Engine: tavily package not installed. Running in offline mode.")
        except Exception as e:
            logger.error(f"Evidence Engine: Tavily init failed: {e}")

    async def search_evidence(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
    ) -> EvidenceResult:
        """
        Execute evidence search and analysis pipeline.
        
        Args:
            query: The claim/question to gather evidence for
            max_results: Maximum search results to retrieve
            search_depth: Tavily search depth ("basic" or "advanced")
            
        Returns:
            EvidenceResult with sources, contradictions, and confidence
        """
        result = EvidenceResult(query=query)
        result.lineage.append({
            "step": "init",
            "timestamp": datetime.utcnow().isoformat(),
            "detail": f"Evidence search initiated for: {query[:100]}",
        })

        # Step 1: External Search
        raw_results = await self._execute_search(query, max_results, search_depth)
        result.lineage.append({
            "step": "search",
            "timestamp": datetime.utcnow().isoformat(),
            "detail": f"Retrieved {len(raw_results)} results from {'Tavily' if self._tavily_available else 'offline'}",
        })

        if not raw_results:
            result.error = "No search results retrieved"
            result.evidence_confidence = 0.1
            return result

        result.search_executed = True

        # Step 2: Source Processing & Reliability Scoring
        for raw in raw_results:
            source = self._process_source(raw)
            result.sources.append(source)

        result.lineage.append({
            "step": "reliability_scoring",
            "timestamp": datetime.utcnow().isoformat(),
            "detail": f"Scored {len(result.sources)} sources. "
                      f"Avg reliability: {sum(s.reliability_score for s in result.sources) / len(result.sources):.2f}",
        })

        # Step 3: Contradiction Detection
        result.contradictions = self._detect_contradictions(result.sources)
        result.lineage.append({
            "step": "contradiction_detection",
            "timestamp": datetime.utcnow().isoformat(),
            "detail": f"Found {len(result.contradictions)} potential contradictions",
        })

        # Step 4: Source Agreement Computation
        result.source_agreement = self._compute_agreement(result.sources, result.contradictions)

        # Step 5: Evidence Confidence
        result.evidence_confidence = self._compute_confidence(result)
        result.lineage.append({
            "step": "confidence",
            "timestamp": datetime.utcnow().isoformat(),
            "detail": f"Final evidence confidence: {result.evidence_confidence:.4f}",
        })

        logger.info(
            f"Evidence search complete: {len(result.sources)} sources, "
            f"{len(result.contradictions)} contradictions, "
            f"confidence={result.evidence_confidence:.4f}"
        )

        return result

    async def _execute_search(
        self, query: str, max_results: int, search_depth: str
    ) -> List[Dict[str, Any]]:
        """Execute search via Tavily or return empty for offline mode."""
        if not self._tavily_available or not self._tavily_client:
            logger.info("Evidence Engine: offline mode — no search executed")
            return []

        try:
            import asyncio
            # Tavily client is sync, run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._tavily_client.search(
                    query=query,
                    search_depth=search_depth,
                    max_results=max_results,
                ),
            )
            return response.get("results", [])
        except Exception as e:
            logger.error(f"Evidence search failed: {e}")
            return []

    def _process_source(self, raw: Dict[str, Any]) -> EvidenceSource:
        """Process a raw search result into an EvidenceSource."""
        url = raw.get("url", "")
        title = raw.get("title", "Unknown")
        content = raw.get("content", "")[:500]

        # Extract domain
        domain = self._extract_domain(url)

        # Reliability scoring
        reliability = self._score_reliability(url, domain, content)

        # Content hash for dedup
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]

        return EvidenceSource(
            url=url,
            title=title,
            content_snippet=content,
            reliability_score=reliability,
            domain=domain,
            content_hash=content_hash,
        )

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower().replace("www.", "")
        except Exception:
            return ""

    def _score_reliability(self, url: str, domain: str, content: str) -> float:
        """Score source reliability based on domain + content heuristics."""
        base_score = 0.50

        # Domain-based scoring
        for trusted_domain, score in TRUSTED_DOMAINS.items():
            if domain.endswith(trusted_domain):
                base_score = score
                break

        # TLD-based fallback
        if base_score == 0.50:
            if domain.endswith(".gov"):
                base_score = 0.85
            elif domain.endswith(".edu"):
                base_score = 0.80
            elif domain.endswith(".org"):
                base_score = 0.65

        # Content-based penalties
        content_lower = content.lower()
        for pattern in UNRELIABLE_PATTERNS:
            if re.search(pattern, content_lower):
                base_score -= 0.10

        # Penalize very short content
        if len(content.split()) < 20:
            base_score -= 0.10

        return max(0.1, min(base_score, 1.0))

    def _detect_contradictions(self, sources: List[EvidenceSource]) -> List[Contradiction]:
        """
        Detect contradictions between sources using keyword-based heuristics.
        
        Strategy: Compare key claims in each source pair.
        If sources use opposing sentiment for similar topics, flag as contradiction.
        """
        contradictions = []

        # Opposing signal words
        positive_signals = {"increase", "improve", "benefit", "effective", "safe", "positive", "growth", "success"}
        negative_signals = {"decrease", "worsen", "harm", "ineffective", "unsafe", "negative", "decline", "failure"}

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                src_a = sources[i]
                src_b = sources[j]

                words_a = set(src_a.content_snippet.lower().split())
                words_b = set(src_b.content_snippet.lower().split())

                # Check for opposing sentiment signals on overlapping topics
                a_positive = words_a & positive_signals
                a_negative = words_a & negative_signals
                b_positive = words_b & positive_signals
                b_negative = words_b & negative_signals

                # Contradiction: A is positive while B is negative (or vice versa)
                has_opposition = (
                    (len(a_positive) > 0 and len(b_negative) > 0)
                    or (len(a_negative) > 0 and len(b_positive) > 0)
                )

                if has_opposition:
                    # Check topic overlap (shared non-trivial words)
                    shared = words_a & words_b
                    # Filter out common stop words
                    stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or", "on", "at", "by", "it", "this", "that", "with"}
                    shared -= stop_words
                    
                    if len(shared) >= 3:  # Sufficient overlap to call it related
                        severity = min(0.5 + len(shared) * 0.05, 1.0)
                        contradictions.append(Contradiction(
                            source_a_url=src_a.url,
                            source_b_url=src_b.url,
                            claim_a=src_a.content_snippet[:150],
                            claim_b=src_b.content_snippet[:150],
                            severity=severity,
                        ))

        return contradictions

    def _compute_agreement(
        self, sources: List[EvidenceSource], contradictions: List[Contradiction]
    ) -> float:
        """Compute inter-source agreement score."""
        if not sources:
            return 0.0

        n = len(sources)
        total_pairs = n * (n - 1) / 2 if n > 1 else 1
        contradiction_ratio = len(contradictions) / total_pairs

        # Agreement = 1 - contradiction_ratio, weighted by source reliability
        avg_reliability = sum(s.reliability_score for s in sources) / n
        agreement = (1.0 - contradiction_ratio) * avg_reliability

        return round(max(0.0, min(agreement, 1.0)), 4)

    def _compute_confidence(self, result: EvidenceResult) -> float:
        """
        Compute overall evidence confidence.
        
        Factors:
        - Number of sources (more = better, diminishing returns)
        - Source reliability (average)
        - Contradiction penalty
        - Agreement bonus
        """
        if not result.sources:
            return 0.1

        n = len(result.sources)
        avg_reliability = sum(s.reliability_score for s in result.sources) / n

        # Source count factor (diminishing returns)
        count_factor = min(n / 5.0, 1.0)

        # Contradiction penalty
        contradiction_penalty = len(result.contradictions) * 0.08

        # Base confidence
        confidence = (
            0.4 * count_factor
            + 0.35 * avg_reliability
            + 0.25 * result.source_agreement
            - contradiction_penalty
        )

        return round(max(0.05, min(confidence, 0.99)), 4)

    # ============================================================
    # 3.X — CLAIM EXTRACTION & RESEARCH-GRADE ANALYSIS
    # ============================================================

    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract atomic claims from an answer or source text.
        
        Splits text into sentences, then identifies claim-like statements
        (declarative, factual, or causal assertions).
        """
        if not text:
            return []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims = []
        
        # Claim indicators: declarative/factual patterns
        claim_patterns = [
            r'\b(is|are|was|were|has|have|will|can|should|must|does|do)\b',
            r'\b(causes?|leads?\s+to|results?\s+in|increases?|decreases?)\b',
            r'\b(according\s+to|research\s+shows?|studies?\s+(show|indicate|suggest))\b',
            r'\b(always|never|typically|generally|often|usually|rarely)\b',
            r'\b(because|therefore|thus|hence|consequently)\b',
            r'\b(\d+%|\d+\s*percent)\b',
        ]
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            
            # Count claim signals
            claim_strength = 0
            for pattern in claim_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claim_strength += 1
            
            if claim_strength >= 1:
                claims.append({
                    "id": f"claim_{i}",
                    "text": sentence[:300],
                    "strength": min(claim_strength / 3.0, 1.0),
                    "type": self._classify_claim(sentence),
                })
        
        return claims

    def _classify_claim(self, sentence: str) -> str:
        """Classify a claim as factual, causal, evaluative, or prescriptive."""
        s = sentence.lower()
        if re.search(r'\b(causes?|leads?\s+to|results?\s+in|because|therefore)\b', s):
            return "causal"
        if re.search(r'\b(should|must|ought|need\s+to|recommend)\b', s):
            return "prescriptive"
        if re.search(r'\b(good|bad|better|worse|best|worst|effective|ineffective)\b', s):
            return "evaluative"
        return "factual"

    def build_claim_overlap_matrix(
        self, answer_claims: List[Dict], source_claims: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Build a matrix mapping answer claims to source claims.
        
        Args:
            answer_claims: Claims extracted from the answer
            source_claims: {source_url: [claims]} for each source
            
        Returns:
            Matrix with support/contradiction relationships
        """
        matrix = {
            "answer_claims_count": len(answer_claims),
            "sources_analyzed": len(source_claims),
            "claim_mappings": [],
            "unsupported_claims": [],
            "conflicting_claims": [],
        }
        
        for ac in answer_claims:
            ac_words = set(ac["text"].lower().split())
            ac_words -= {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or", "on", "at", "by", "it", "this", "that", "with"}
            
            supporting_sources = []
            conflicting_sources = []
            
            for source_url, s_claims in source_claims.items():
                best_overlap = 0
                is_conflicting = False
                
                for sc in s_claims:
                    sc_words = set(sc["text"].lower().split())
                    sc_words -= {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or", "on", "at", "by", "it", "this", "that", "with"}
                    
                    if not ac_words or not sc_words:
                        continue
                    
                    overlap = len(ac_words & sc_words) / max(len(ac_words | sc_words), 1)
                    best_overlap = max(best_overlap, overlap)
                    
                    # Check for sentiment opposition
                    if overlap > 0.2 and self._has_opposing_sentiment(ac["text"], sc["text"]):
                        is_conflicting = True
                
                if best_overlap > 0.25:
                    if is_conflicting:
                        conflicting_sources.append(source_url)
                    else:
                        supporting_sources.append(source_url)
            
            mapping = {
                "claim_id": ac["id"],
                "claim_text": ac["text"][:200],
                "claim_type": ac["type"],
                "supporting_sources": supporting_sources,
                "conflicting_sources": conflicting_sources,
                "support_level": len(supporting_sources) / max(len(source_claims), 1),
            }
            
            matrix["claim_mappings"].append(mapping)
            
            if not supporting_sources:
                matrix["unsupported_claims"].append(ac["id"])
            if conflicting_sources:
                matrix["conflicting_claims"].append(ac["id"])
        
        return matrix

    def compute_evidence_strength(self, sources: List[EvidenceSource]) -> Dict[str, Any]:
        """
        Compute per-source evidence strength using composite scoring.
        
        Factors:
        - domain_authority: From TRUSTED_DOMAINS heuristic
        - recency: How recent the source appears to be
        - internal_coherence: Length and structure of content
        - citation_depth: Presence of references/citations in content
        - agreement: How much this source agrees with others
        """
        if not sources:
            return {"scores": {}, "overall_strength": 0.0}
        
        scores = {}
        for source in sources:
            # Domain authority (already computed as reliability_score)
            domain_authority = source.reliability_score
            
            # Recency (heuristic: look for year patterns in content)
            recency = self._estimate_recency(source.content_snippet)
            
            # Internal coherence (content length and structure)
            words = source.content_snippet.split()
            word_count = len(words)
            has_numbers = bool(re.search(r'\d+', source.content_snippet))
            has_structured = bool(re.search(r'(\d\.|•|-)\s', source.content_snippet))
            internal_coherence = min(
                (word_count / 100) * 0.4 + (0.3 if has_numbers else 0) + (0.3 if has_structured else 0),
                1.0
            )
            
            # Citation depth (references in content)
            citation_patterns = re.findall(
                r'(et\s+al|doi:|http|reference|source|according\s+to|study\s+by)',
                source.content_snippet, re.IGNORECASE
            )
            citation_depth = min(len(citation_patterns) / 3.0, 1.0)
            
            # Agreement with others (percentage of other sources with similar content)
            other_sources = [s for s in sources if s.url != source.url]
            agreement = self._compute_pairwise_agreement(source, other_sources)
            
            # Composite strength
            strength = (
                0.30 * domain_authority
                + 0.15 * recency
                + 0.20 * internal_coherence
                + 0.15 * citation_depth
                + 0.20 * agreement
            )
            
            scores[source.url] = {
                "domain_authority": round(domain_authority, 3),
                "recency": round(recency, 3),
                "internal_coherence": round(internal_coherence, 3),
                "citation_depth": round(citation_depth, 3),
                "agreement_with_others": round(agreement, 3),
                "composite_strength": round(strength, 3),
            }
        
        overall = sum(s["composite_strength"] for s in scores.values()) / len(scores) if scores else 0.0
        
        return {
            "scores": scores,
            "overall_strength": round(overall, 3),
        }

    def build_traceability_map(
        self, answer: str, sources: List[EvidenceSource]
    ) -> List[Dict[str, Any]]:
        """
        Map answer sentences to supporting sources.
        
        For each sentence in the answer, find which sources contain
        matching content. If a sentence has no source support, mark it
        as unsupported.
        """
        if not answer or not sources:
            return []
        
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        trace_map = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            sentence_words = set(sentence.lower().split())
            sentence_words -= {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or", "on", "at", "by", "it", "this", "that", "with"}
            
            supporting = []
            for source in sources:
                source_words = set(source.content_snippet.lower().split())
                source_words -= {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or", "on", "at", "by", "it", "this", "that", "with"}
                
                if not sentence_words or not source_words:
                    continue
                
                overlap = len(sentence_words & source_words) / max(len(sentence_words), 1)
                if overlap > 0.3:
                    supporting.append({
                        "url": source.url,
                        "title": source.title,
                        "overlap_score": round(overlap, 3),
                    })
            
            trace_map.append({
                "sentence_index": i,
                "sentence": sentence[:200],
                "supporting_sources": supporting,
                "is_supported": len(supporting) > 0,
            })
        
        return trace_map

    async def run_full_evidence_analysis(
        self,
        query: str,
        answer: str,
        max_results: int = 5,
    ) -> EvidenceResult:
        """
        Run the complete 3.X evidence analysis pipeline.
        
        Steps:
        1. Search for external evidence
        2. Extract claims from answer
        3. Extract claims from each source
        4. Build claim overlap matrix
        5. Compute evidence strength
        6. Build traceability map
        7. Compute final confidence
        """
        # Step 1: Base evidence search
        result = await self.search_evidence(query, max_results)
        
        # Step 2: Extract claims from answer
        result.claims = self.extract_claims(answer)
        result.lineage.append({
            "step": "claim_extraction",
            "timestamp": datetime.utcnow().isoformat(),
            "detail": f"Extracted {len(result.claims)} claims from answer",
        })
        
        # Step 3: Extract claims from each source
        source_claims = {}
        for source in result.sources:
            s_claims = self.extract_claims(source.content_snippet)
            if s_claims:
                source_claims[source.url] = s_claims
        
        # Step 4: Build claim overlap matrix
        if result.claims and source_claims:
            result.claim_overlap_matrix = self.build_claim_overlap_matrix(
                result.claims, source_claims
            )
            result.lineage.append({
                "step": "claim_overlap",
                "timestamp": datetime.utcnow().isoformat(),
                "detail": (
                    f"Mapped {len(result.claims)} answer claims to {len(source_claims)} sources. "
                    f"Unsupported: {len(result.claim_overlap_matrix.get('unsupported_claims', []))}. "
                    f"Conflicting: {len(result.claim_overlap_matrix.get('conflicting_claims', []))}."
                ),
            })
        
        # Step 5: Compute evidence strength
        result.evidence_strength = self.compute_evidence_strength(result.sources)
        result.lineage.append({
            "step": "evidence_strength",
            "timestamp": datetime.utcnow().isoformat(),
            "detail": f"Overall evidence strength: {result.evidence_strength.get('overall_strength', 0):.3f}",
        })
        
        # Step 6: Build traceability map
        result.traceability_map = self.build_traceability_map(answer, result.sources)
        unsupported_count = sum(1 for t in result.traceability_map if not t["is_supported"])
        result.lineage.append({
            "step": "traceability",
            "timestamp": datetime.utcnow().isoformat(),
            "detail": f"Traced {len(result.traceability_map)} sentences. {unsupported_count} unsupported.",
        })
        
        # Step 7: Adjust confidence based on claim analysis
        if result.claim_overlap_matrix:
            unsupported = len(result.claim_overlap_matrix.get("unsupported_claims", []))
            total = result.claim_overlap_matrix.get("answer_claims_count", 1)
            unsupported_ratio = unsupported / max(total, 1)
            # Penalize confidence for unsupported claims
            result.evidence_confidence = max(
                0.05,
                result.evidence_confidence - unsupported_ratio * 0.15
            )
        
        return result

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _estimate_recency(self, text: str) -> float:
        """Estimate source recency from year mentions in text."""
        current_year = datetime.utcnow().year
        years = re.findall(r'\b(20\d{2})\b', text)
        if years:
            most_recent = max(int(y) for y in years)
            age = current_year - most_recent
            if age <= 1:
                return 0.95
            elif age <= 3:
                return 0.75
            elif age <= 5:
                return 0.55
            else:
                return 0.30
        return 0.50  # Unknown recency

    def _compute_pairwise_agreement(
        self, source: EvidenceSource, others: List[EvidenceSource]
    ) -> float:
        """Compute agreement between a source and all other sources."""
        if not others:
            return 0.5
        
        src_words = set(source.content_snippet.lower().split())
        src_words -= {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or"}
        
        overlaps = []
        for other in others:
            other_words = set(other.content_snippet.lower().split())
            other_words -= {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or"}
            
            if src_words and other_words:
                jaccard = len(src_words & other_words) / len(src_words | other_words)
                overlaps.append(jaccard)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.5

    def _has_opposing_sentiment(self, text_a: str, text_b: str) -> bool:
        """Check if two texts have opposing sentiment on the same topic."""
        positive = {"increase", "improve", "benefit", "effective", "safe", "positive", "growth", "success", "better", "good"}
        negative = {"decrease", "worsen", "harm", "ineffective", "unsafe", "negative", "decline", "failure", "worse", "bad"}
        
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        a_pos = words_a & positive
        a_neg = words_a & negative
        b_pos = words_b & positive
        b_neg = words_b & negative
        
        return (len(a_pos) > 0 and len(b_neg) > 0) or (len(a_neg) > 0 and len(b_pos) > 0)
