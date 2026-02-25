"""
Reasoning Engine Orchestrator — Sentinel-E Autonomous Reasoning Engine

Master orchestrator that wires together all components:
  - IntentHasher
  - EvidenceCache
  - SessionMemoryTier / EvidenceMemory / KnowledgeGraph
  - ContextCompiler
  - TopicBoundaryDetector / FollowUpAnchor / ContextDecay
  - HallucinationGate
  - CognitiveRAG (existing)
  - EvidenceEngine (existing)
  - ConfidenceEngine (existing)

This is the single entry point for the autonomous reasoning pipeline.
Replaces ad-hoc wiring in orchestration layer.
"""

import logging
import time
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.intent_hasher import IntentHasher, IntentHash
from core.evidence_cache import EvidenceCache, EvidenceCacheEntry, CachedChunk
from core.knowledge_memory import (
    SessionMemoryTier, EvidenceMemory, KnowledgeGraph, EvidenceObject, cosine_similarity,
)
from core.context_compiler import ContextCompiler, CompiledContext
from core.topic_boundary import (
    TopicBoundaryDetector, FollowUpAnchorEngine, ContextDecayEngine,
)
from core.hallucination_gate import HallucinationGate

logger = logging.getLogger("ReasoningEngine")


# ============================================================
# MODE-SPECIFIC CONFIGURATION
# ============================================================

MODE_CONFIG = {
    "standard":   {"cache_threshold": 0.87, "coverage_threshold": 0.50, "graph_depth": 1, "decay_lambda": 0.15, "regen_attempts": 1},
    "debate":     {"cache_threshold": 0.87, "coverage_threshold": 0.55, "graph_depth": 2, "decay_lambda": 0.10, "regen_attempts": 2},
    "evidence":   {"cache_threshold": 0.80, "coverage_threshold": 0.70, "graph_depth": 3, "decay_lambda": 0.08, "regen_attempts": 3},
    "glass":      {"cache_threshold": 0.87, "coverage_threshold": 0.60, "graph_depth": 2, "decay_lambda": 0.12, "regen_attempts": 2},
    "stress":     {"cache_threshold": 0.87, "coverage_threshold": 0.50, "graph_depth": 1, "decay_lambda": 0.20, "regen_attempts": 1},
}


@dataclass
class ReasoningResult:
    """Result of the full reasoning pipeline."""
    # Context
    compiled_context: Optional[CompiledContext] = None
    # Retrieval
    retrieval_executed: bool = False
    cache_hit: bool = False
    cache_level: str = ""  # l1 | l2 | l3 | miss
    evidence_count: int = 0
    # Topic
    topic_shift: bool = False
    drift_score: float = 0.0
    followup_type: str = ""
    clarification_needed: bool = False
    clarification_prompt: str = ""
    # Confidence
    evidence_confidence: float = 0.0
    coverage_score: float = 0.0
    # Hallucination
    verification_status: str = ""
    unsupported_count: int = 0
    traceability_map: List[Dict[str, Any]] = field(default_factory=list)
    # Intent
    intent_hash: str = ""
    intent_type: str = ""
    # Timing
    pipeline_latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_executed": self.retrieval_executed,
            "cache_hit": self.cache_hit,
            "cache_level": self.cache_level,
            "evidence_count": self.evidence_count,
            "topic_shift": self.topic_shift,
            "drift_score": round(self.drift_score, 4),
            "followup_type": self.followup_type,
            "clarification_needed": self.clarification_needed,
            "evidence_confidence": round(self.evidence_confidence, 4),
            "coverage_score": round(self.coverage_score, 4),
            "verification_status": self.verification_status,
            "unsupported_count": self.unsupported_count,
            "intent_hash": self.intent_hash[:16] + "..." if self.intent_hash else "",
            "intent_type": self.intent_type,
            "pipeline_latency_ms": self.pipeline_latency_ms,
        }


class ReasoningEngine:
    """
    Master orchestrator for the autonomous reasoning pipeline.
    
    Lifecycle per query:
      1. Hash intent
      2. Check topic boundary
      3. Anchor follow-up
      4. Check evidence cache
      5. Retrieve (if cache miss)
      6. Compile context
      7. (Model call happens externally)
      8. Verify response (hallucination gate)
      9. Feed back into memory
    """

    def __init__(
        self,
        mode: str = "standard",
        embed_fn=None,
        compress_fn=None,
        redis_client=None,
    ):
        self.mode = mode
        self._embed_fn = embed_fn
        self._config = MODE_CONFIG.get(mode, MODE_CONFIG["standard"])

        # Core components
        self.intent_hasher = IntentHasher(embed_fn=embed_fn)
        self.evidence_cache = EvidenceCache(redis_client=redis_client)
        self.session_memory = SessionMemoryTier(embed_fn=embed_fn, compress_fn=compress_fn)
        self.evidence_memory = EvidenceMemory()
        self.knowledge_graph = KnowledgeGraph()
        self.context_compiler = ContextCompiler()
        self.topic_detector = TopicBoundaryDetector()
        self.followup_engine = FollowUpAnchorEngine()
        self.decay_engine = ContextDecayEngine(decay_lambda=self._config["decay_lambda"])
        self.hallucination_gate = HallucinationGate(embed_fn=embed_fn, mode=mode)

        # State
        self._intent_history: List[Dict[str, Any]] = []
        self._session_id = ""

    def set_session(self, session_id: str):
        self._session_id = session_id

    # ============================================================
    # PHASE 1: PRE-GENERATION (before model call)
    # ============================================================

    async def prepare_context(
        self,
        query: str,
        system_prompt: str = "",
        rag_pipeline=None,
    ) -> ReasoningResult:
        """
        Run the full pre-generation pipeline:
          1. Hash intent
          2. Detect topic boundary
          3. Anchor follow-up
          4. Check evidence cache / retrieve
          5. Compile structured context
        
        Args:
            query: User query text
            system_prompt: Base system prompt for the model
            rag_pipeline: CognitiveRAG instance (for live retrieval)
            
        Returns:
            ReasoningResult with compiled_context ready for model injection
        """
        start_time = time.time()
        result = ReasoningResult()

        # --- 1. Hash intent ---
        intent = self.intent_hasher.hash_intent(query, self._session_id)
        result.intent_hash = intent.exact_hash
        result.intent_type = intent.intent_type

        # --- 2. Detect topic boundary ---
        recent_user_embeddings = [
            m.embedding for m in self.session_memory.messages
            if m.role == "user" and m.embedding is not None
        ][-5:]

        boundary = self.topic_detector.detect(
            query_embedding=intent.embedding,
            topic_embedding=self.session_memory.topic_embedding,
            recent_user_embeddings=recent_user_embeddings if recent_user_embeddings else None,
            current_cluster_id=self.session_memory.topic_cluster_id,
        )
        result.topic_shift = boundary.is_shift
        result.drift_score = boundary.drift_score

        if boundary.action == "archive_and_reset":
            self.session_memory.archive_and_reset()
            logger.info("Topic shift → archived and reset session memory")

        # --- 3. Anchor follow-up ---
        anchor = self.followup_engine.anchor(
            current_embedding=intent.embedding,
            current_hash=intent.exact_hash,
            intent_history=self._intent_history,
            current_cluster_id=self.session_memory.topic_cluster_id,
        )
        result.followup_type = anchor.anchor_type
        result.clarification_needed = anchor.clarification_needed
        result.clarification_prompt = anchor.clarification_prompt

        # --- 4. Check evidence cache ---
        evidence_chunks = []
        knowledge_claims = []
        cache_entry = None
        retrieval_result = None

        if intent.retrieval_p >= 0.3:  # Only check cache if retrieval might be needed
            cache_entry = await self.evidence_cache.check(
                cache_key=intent.exact_hash,
                query_embedding=intent.embedding,
                temporal_flag=intent.temporal_flag,
                intent_type=intent.intent_type,
            )

        if cache_entry is not None:
            # Cache hit
            result.cache_hit = True
            result.cache_level = "l1" if self.evidence_cache._metrics.get("l1_hit", 0) > 0 else "l2"
            evidence_chunks = [
                {"content": c.content, "source_url": c.source_url, "reliability_score": c.reliability_score}
                for c in cache_entry.chunks
            ]
            result.evidence_count = len(evidence_chunks)
            result.evidence_confidence = cache_entry.confidence_score
        elif intent.retrieval_p >= 0.6 and rag_pipeline:
            # Cache miss → live retrieval
            try:
                retrieval_result = await rag_pipeline.process(query)
                result.retrieval_executed = retrieval_result.retrieval_executed

                if retrieval_result.sources:
                    evidence_chunks = [
                        {
                            "content": s.content,
                            "source_url": s.url,
                            "reliability_score": s.reliability_score,
                            "domain": s.domain,
                        }
                        for s in retrieval_result.sources
                    ]
                    result.evidence_count = len(evidence_chunks)
                    result.evidence_confidence = retrieval_result.average_reliability

                    # Store in cache
                    entry = EvidenceCacheEntry(
                        cache_key=intent.exact_hash,
                        query_canonical=intent.canonical,
                        query_embedding=intent.embedding.tolist() if intent.embedding is not None else None,
                        chunks=[
                            CachedChunk(
                                content=s.content,
                                source_url=s.url,
                                source_domain=s.domain,
                                reliability_score=s.reliability_score,
                                content_hash=s.content_hash,
                            )
                            for s in retrieval_result.sources
                        ],
                        source_urls=[s.url for s in retrieval_result.sources],
                        confidence_score=retrieval_result.average_reliability,
                        intent_type=intent.intent_type,
                        citations_text=retrieval_result.citations_text,
                        contradictions=retrieval_result.contradictions,
                    )
                    await self.evidence_cache.store(entry)

                    # Store in evidence memory (Tier 2)
                    ev_obj = EvidenceObject(
                        query_origin=query,
                        chunks=[{"content": s.content, "source_url": s.url} for s in retrieval_result.sources],
                        source_metadata=[{"url": s.url, "domain": s.domain, "reliability": s.reliability_score} for s in retrieval_result.sources],
                        confidence_score=retrieval_result.average_reliability,
                    )
                    if intent.embedding is not None:
                        ev_obj.topic_embedding = intent.embedding
                    self.evidence_memory.store(ev_obj)

            except Exception as e:
                logger.error(f"Live retrieval failed: {e}")
        else:
            # Check knowledge graph coverage (L3)
            if intent.embedding is not None:
                kg_coverage = self.knowledge_graph.compute_coverage(query, intent.embedding)
                if kg_coverage["score"] >= 0.80 and not intent.temporal_flag:
                    result.cache_hit = True
                    result.cache_level = "l3"
                    knowledge_claims = [
                        {"claim_text": c.claim_text, "source_urls": c.source_urls, "confidence": c.confidence}
                        for c in kg_coverage["claims"]
                    ]

        # --- 5. Get knowledge graph claims ---
        if intent.embedding is not None and not knowledge_claims:
            related_entities = self.knowledge_graph.get_related_entities(intent.embedding, k=15)
            if related_entities:
                kg_claims = self.knowledge_graph.get_claims_for_entities(
                    [e.entity_id for e in related_entities]
                )
                knowledge_claims = [
                    {"claim_text": c.claim_text, "source_urls": c.source_urls, "confidence": c.confidence,
                     "status": c.status, "conflict_marker": c.conflict_marker}
                    for c in kg_claims
                ]

        # --- 6. Compile context ---
        conflict_flags = [c for c in knowledge_claims if c.get("status") == "disputed"]
        conversation_msgs = self.session_memory.get_context_messages(min_weight=0.3)

        # Freshness ratio
        all_evidence = evidence_chunks + [{"confidence": c.get("confidence", 0)} for c in knowledge_claims]
        freshness_ratio = 0.0
        if cache_entry:
            cache_entry.update_freshness()
            freshness_ratio = 1.0 if cache_entry.freshness_class in ("live", "recent") else 0.5

        compiled = self.context_compiler.compile(
            mode=self.mode,
            verified_evidence=[
                {
                    "content": c.get("content", c.get("claim_text", "")),
                    "confidence": c.get("confidence", c.get("reliability_score", 0)),
                    "sources": 1,
                    "domain": c.get("domain", ""),
                }
                for c in (evidence_chunks + knowledge_claims)[:20]
            ] if evidence_chunks or knowledge_claims else None,
            conflict_flags=conflict_flags if conflict_flags else None,
            evidence_confidence=result.evidence_confidence,
            source_agreement=0.0,
            entity_count=len(self.knowledge_graph.entities),
            claim_count=len(knowledge_claims),
            freshness_ratio=freshness_ratio,
            conversation_messages=conversation_msgs,
        )
        result.compiled_context = compiled

        # --- Update state ---
        self.session_memory.add_message("user", query, embedding=intent.embedding)
        self._intent_history.append(intent.to_dict())
        if len(self._intent_history) > 50:
            self._intent_history = self._intent_history[-50:]

        result.pipeline_latency_ms = int((time.time() - start_time) * 1000)
        return result

    # ============================================================
    # PHASE 2: POST-GENERATION (after model call)
    # ============================================================

    async def verify_and_store(
        self,
        response_text: str,
        evidence_chunks: List[Dict[str, Any]] = None,
        knowledge_claims: List[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Post-generation pipeline:
          1. Verify response against evidence (hallucination gate)
          2. Store assistant message in session memory
          3. Return verification results
        """
        result = ReasoningResult()

        verification = self.hallucination_gate.verify(
            response_text=response_text,
            evidence_chunks=evidence_chunks,
            knowledge_claims=knowledge_claims,
        )

        result.verification_status = verification.status
        result.coverage_score = verification.coverage
        result.unsupported_count = len(verification.unsupported)
        result.traceability_map = verification.traceability_map
        result.evidence_confidence = verification.confidence_score

        # Store assistant response in session memory
        self.session_memory.add_message("assistant", response_text)

        return result

    # ============================================================
    # UTILITY
    # ============================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get full diagnostic state of the reasoning engine."""
        return {
            "mode": self.mode,
            "config": self._config,
            "session_id": self._session_id,
            "topic_cluster_id": self.session_memory.topic_cluster_id,
            "message_count": len(self.session_memory.messages),
            "archive_count": len(self.session_memory.archives),
            "evidence_memory_count": len(self.evidence_memory._store),
            "kg_entity_count": len(self.knowledge_graph.entities),
            "kg_claim_count": len(self.knowledge_graph.claims),
            "cache_metrics": self.evidence_cache.metrics,
            "intent_history_length": len(self._intent_history),
        }
