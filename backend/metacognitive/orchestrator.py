"""
============================================================
Meta-Cognitive Orchestrator — Executive Controller
============================================================
Coordinates three physically and logically separated APIs:

  API 1 — Cognitive Model Gateway   (pure reasoning)
  API 2 — Knowledge & Retrieval     (live data acquisition)
  API 3 — Session & Persistence     (structured state)

Enforces the mandatory 10-step execution protocol
for every user query. No steps may be skipped.

Does NOT:
  ✗ Modify model weights
  ✗ Modify sampling
  ✗ Rewrite raw model outputs before scoring
  ✗ Merge API responsibilities

DOES:
  ✓ Orchestration
  ✓ Stability enforcement
  ✓ Grounding enforcement
  ✓ Arbitration
============================================================
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Any

from metacognitive.schemas import (
    OperatingMode,
    QueryMode,
    CognitiveGatewayInput,
    CognitiveGatewayOutput,
    KnowledgeBlock,
    KnowledgeRetrievalInput,
    ArbitrationScore,
    ModelResult,
    SessionState,
    BehavioralRecord,
    OrchestratorRequest,
    OrchestratorResponse,
)
from metacognitive.cognitive_gateway import (
    CognitiveModelGateway,
    get_models_for_task,
    COGNITIVE_MODEL_REGISTRY,
)
from metacognitive.knowledge_engine import (
    KnowledgeRetrievalEngine,
    VOLATILITY_THRESHOLD,
)
from metacognitive.session_engine import SessionPersistenceEngine
from metacognitive.arbitration import ArbitrationEngine
from metacognitive.knowledge_graph import KnowledgeGraph, NodeType, EdgeType
from metacognitive.embedding import (
    embed_text,
    cosine_similarity,
    compute_centroid,
    centroid_variance,
    volatility_score as compute_volatility,
)

logger = logging.getLogger("MCO-Orchestrator")


# ============================================================
# Configuration
# ============================================================

# Recursive context stabilization parameters
MAX_STABILIZATION_CYCLES = 5
CENTROID_EPSILON = 0.05  # Stop when variance < epsilon

# Task detection heuristics
CODE_PATTERNS = [
    r'\b(code|program|function|class|implement|debug|compile|syntax|algorithm|api)\b',
    r'\b(python|javascript|typescript|rust|go|java|c\+\+|sql|html|css)\b',
    r'```',
]
IMAGE_PATTERNS = [
    r'\b(image|photo|picture|visual|diagram|chart|graph|screenshot|vision)\b',
    r'\b(look at|see|show|display|render|draw)\b',
]
LONGCTX_PATTERNS = [
    r'\b(document|paper|article|book|chapter|report|analysis|lengthy|comprehensive)\b',
    r'\b(entire|full|complete|all of|everything)\b',
]


class MetaCognitiveOrchestrator:
    """
    Executive controller of the distributed cognitive research architecture.

    For every user query, executes the mandatory 10-step protocol:
      1. Load Session State
      2. Compute Embeddings
      3. Volatility Enforcement
      4. Recursive Context Stabilization
      5. Parallel Model Invocation
      6. Store Raw Outputs
      7. Arbitration Scoring
      8. Mode Handling
      9. Update Persistence Layer
      10. Background Scheduling
    """

    def __init__(self):
        # Three isolated APIs
        self.cognitive_gateway = CognitiveModelGateway()
        self.knowledge_engine = KnowledgeRetrievalEngine()
        self.session_engine = SessionPersistenceEngine()

        # Supporting systems
        self.arbitration = ArbitrationEngine()
        self.knowledge_graph = KnowledgeGraph()

        logger.info("Meta-Cognitive Orchestrator initialized")

    def set_redis(self, redis_client):
        """Inject Redis for session persistence."""
        self.session_engine.set_redis(redis_client)

    async def close(self):
        """Cleanup resources."""
        await self.cognitive_gateway.close()
        await self.knowledge_engine.close()

    # ============================================================
    # MAIN EXECUTION — 10-STEP PROTOCOL
    # ============================================================

    async def process(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResponse:
        """
        Execute the mandatory 10-step protocol.
        No steps may be skipped.
        """
        start_time = time.monotonic()

        logger.info(
            f"Processing query [{request.mode.value}]: {request.query[:80]}..."
        )

        # ─────────────────────────────────────────────────────
        # STEP 1: Load Session State
        # ─────────────────────────────────────────────────────
        session = await self._step1_load_session(request)
        logger.debug(f"Step 1 complete: session={session.session_id}")

        # ─────────────────────────────────────────────────────
        # STEP 2: Compute Embeddings
        # ─────────────────────────────────────────────────────
        query_embedding, vol_score, d_score = self._step2_compute_embeddings(
            request, session
        )
        logger.debug(
            f"Step 2 complete: volatility={vol_score:.3f}, drift={d_score:.3f}"
        )

        # ─────────────────────────────────────────────────────
        # STEP 3: Volatility Enforcement
        # ─────────────────────────────────────────────────────
        knowledge_bundle = await self._step3_volatility_enforcement(
            request, query_embedding, vol_score, session
        )
        logger.debug(
            f"Step 3 complete: {len(knowledge_bundle)} knowledge blocks"
        )

        # ─────────────────────────────────────────────────────
        # STEP 4: Recursive Context Stabilization
        # ─────────────────────────────────────────────────────
        stabilized_context = self._step4_recursive_stabilization(
            request, session, query_embedding, knowledge_bundle
        )
        logger.debug("Step 4 complete: context stabilized")

        # ─────────────────────────────────────────────────────
        # STEP 5: Parallel Model Invocation
        # ─────────────────────────────────────────────────────
        model_outputs = await self._step5_parallel_invocation(
            request, stabilized_context, knowledge_bundle, session
        )
        logger.debug(
            f"Step 5 complete: {len(model_outputs)} model outputs"
        )

        # ─────────────────────────────────────────────────────
        # STEP 6: Store Raw Outputs
        # ─────────────────────────────────────────────────────
        self._step6_store_raw_outputs(session, model_outputs)
        logger.debug("Step 6 complete: raw outputs stored")

        # ─────────────────────────────────────────────────────
        # STEP 7: Arbitration Scoring
        # ─────────────────────────────────────────────────────
        scores = self._step7_arbitration_scoring(
            model_outputs, session, knowledge_bundle
        )
        logger.debug("Step 7 complete: scoring done")

        # ─────────────────────────────────────────────────────
        # STEP 8: Mode Handling
        # ─────────────────────────────────────────────────────
        response = self._step8_mode_handling(
            request, session, model_outputs, scores, knowledge_bundle
        )
        logger.debug(f"Step 8 complete: mode={request.mode.value}")

        # ─────────────────────────────────────────────────────
        # STEP 9: Update Persistence Layer
        # ─────────────────────────────────────────────────────
        await self._step9_update_persistence(
            session, model_outputs, scores, query_embedding
        )
        logger.debug("Step 9 complete: persistence updated")

        # ─────────────────────────────────────────────────────
        # STEP 10: Background Scheduling
        # ─────────────────────────────────────────────────────
        self._step10_background_scheduling(session, request)
        logger.debug("Step 10 complete: scheduling done")

        # Final timing
        elapsed = (time.monotonic() - start_time) * 1000
        response.latency_ms = elapsed
        response.session_state = session

        logger.info(
            f"Query processed in {elapsed:.0f}ms. "
            f"Winner: {response.winning_model} "
            f"(score={response.winning_score:.3f})"
        )

        return response

    # ============================================================
    # STEP IMPLEMENTATIONS
    # ============================================================

    async def _step1_load_session(
        self,
        request: OrchestratorRequest,
    ) -> SessionState:
        """Step 1: Load session state from Session API."""
        session_id = request.session_id or request.chat_id

        if session_id:
            # Try in-memory first, then Redis
            session = self.session_engine.get_session(session_id)
            if not session:
                session = await self.session_engine.restore_from_redis(session_id)
            if session:
                return session

        # Create new session
        session = self.session_engine.create_session(
            session_id=session_id,
            mode=request.mode,
        )

        # Register in knowledge graph
        self.knowledge_graph.register_session(
            session.session_id,
            label=f"Session {request.query[:30]}",
        )

        return session

    def _step2_compute_embeddings(
        self,
        request: OrchestratorRequest,
        session: SessionState,
    ) -> tuple:
        """
        Step 2: Compute query embedding, volatility, and drift.
        """
        query_embedding = embed_text(request.query)

        # Volatility: how far is query from topic centroid?
        vol_score = self.session_engine.compute_volatility(
            session.session_id, query_embedding
        )

        # Drift: how much has context shifted?
        if session.topic_centroid_embedding:
            d_score = 1.0 - cosine_similarity(
                query_embedding, session.topic_centroid_embedding
            )
        else:
            d_score = 0.0

        return query_embedding, vol_score, d_score

    async def _step3_volatility_enforcement(
        self,
        request: OrchestratorRequest,
        query_embedding: List[float],
        vol_score: float,
        session: SessionState,
    ) -> List[KnowledgeBlock]:
        """
        Step 3: If volatility > threshold, retrieval is MANDATORY.
        No fallback to internal knowledge if retrieval is possible.
        """
        knowledge_bundle: List[KnowledgeBlock] = []

        should_retrieve = (
            vol_score > VOLATILITY_THRESHOLD
            or request.force_retrieval
        )

        if should_retrieve:
            logger.info(
                f"Retrieval mandatory: volatility={vol_score:.3f} "
                f"(threshold={VOLATILITY_THRESHOLD})"
            )

            retrieval_input = KnowledgeRetrievalInput(
                query_embedding=query_embedding,
                query_text=request.query,
                volatility_score=vol_score,
                domain=self._detect_domain(request.query),
                concept_expansion_depth=1 if vol_score > 0.6 else 0,
            )

            result = await self.knowledge_engine.retrieve(retrieval_input)
            knowledge_bundle = result.knowledge_bundle

            logger.info(
                f"Retrieval complete: {len(knowledge_bundle)} blocks, "
                f"confidence={result.retrieval_confidence:.3f}"
            )
        else:
            logger.debug(
                f"Volatility {vol_score:.3f} below threshold. "
                "Retrieval not enforced."
            )

        return knowledge_bundle

    def _step4_recursive_stabilization(
        self,
        request: OrchestratorRequest,
        session: SessionState,
        query_embedding: List[float],
        knowledge_bundle: List[KnowledgeBlock],
    ) -> Dict[str, Any]:
        """
        Step 4: Recursive Context Stabilization.
        
        For K cycles:
          - Recompute topic centroid
          - Rank memory blocks by cosine similarity
          - Suppress low-signal blocks
          - Elevate structured goals
          - Detect instability
        Stop when centroid variance < epsilon.
        """
        # Add query to memory
        self.session_engine.add_memory_block(
            session.session_id,
            content=request.query,
            source="user",
            embedding=query_embedding,
        )

        # Add knowledge to memory
        for block in knowledge_bundle:
            self.session_engine.add_memory_block(
                session.session_id,
                content=block.content[:500],
                source="retrieval",
                embedding=block.embedding,
            )

        # Recursive stabilization loop
        for cycle in range(MAX_STABILIZATION_CYCLES):
            # Recompute centroid
            self.session_engine.update_topic_centroid(
                session.session_id, query_embedding
            )

            # Rank by similarity to centroid
            self.session_engine.rank_memory_blocks(
                session.session_id, query_embedding
            )

            # Suppress low-signal
            self.session_engine.suppress_low_signal_blocks(
                session.session_id, threshold=0.1
            )

            # Check stability
            if self.session_engine.is_centroid_stable(
                session.session_id, epsilon=CENTROID_EPSILON
            ):
                logger.debug(f"Context stabilized after {cycle + 1} cycles")
                break

        # Build stabilized context
        session_summary = self.session_engine.get_session_summary(
            session.session_id
        )
        active_goals = self.session_engine.get_active_goals(
            session.session_id
        )

        stabilized = {
            "topic_centroid_active": bool(session.topic_centroid_embedding),
            "drift_score": session.drift_score,
            "volatility_score": session.volatility_score,
            "active_goals": [g.description for g in active_goals],
            "turn_count": session.turn_count,
            "recent_context": session_summary.get("recent_context", []),
        }

        return stabilized

    async def _step5_parallel_invocation(
        self,
        request: OrchestratorRequest,
        stabilized_context: Dict[str, Any],
        knowledge_bundle: List[KnowledgeBlock],
        session: SessionState,
    ) -> List[CognitiveGatewayOutput]:
        """
        Step 5: Parallel Model Invocation.
        
        Model selection based on task characteristics:
          Code-heavy → Qwen3 Coder prioritized
          Image-heavy → Qwen3 VL prioritized
          Conceptual → Llama prioritized
          Baseline → Nemotron always
          Long-context → Kimi 2.5

        All receive IDENTICAL stabilized context.
        """
        # Detect task type
        code_heavy = self._detect_task_type(request.query, CODE_PATTERNS)
        image_heavy = self._detect_task_type(request.query, IMAGE_PATTERNS)
        long_context = self._detect_task_type(request.query, LONGCTX_PATTERNS)

        # Select models — Single Model Focus or task-based selection
        if request.selected_model:
            model_keys = [request.selected_model]
            logger.info(f"Single Model Focus: {request.selected_model}")
        else:
            model_keys = get_models_for_task(
                code_heavy=code_heavy,
                image_heavy=image_heavy,
                conceptual=not code_heavy and not image_heavy,
                long_context=long_context,
            )

        logger.info(f"Invoking {len(model_keys)} models: {model_keys}")

        # Build gateway input (identical for all)
        session_summary = self.session_engine.get_session_summary(
            session.session_id
        )

        gateway_input = CognitiveGatewayInput(
            stabilized_context=stabilized_context,
            knowledge_bundle=knowledge_bundle,
            session_summary=session_summary,
            user_query=request.query,
            mode=QueryMode.RESEARCH if request.mode == OperatingMode.EXPERIMENTAL else QueryMode.RAW,
        )

        # Parallel invocation — all models get identical context
        outputs = await self.cognitive_gateway.invoke_parallel(
            model_keys, gateway_input
        )

        return outputs

    def _step6_store_raw_outputs(
        self,
        session: SessionState,
        outputs: List[CognitiveGatewayOutput],
    ):
        """
        Step 6: Store raw outputs.
        Logged in session. No modification.
        """
        for output in outputs:
            if output.success and output.raw_output:
                # Store in session memory
                self.session_engine.add_memory_block(
                    session.session_id,
                    content=f"[{output.model_name}] {output.raw_output[:500]}",
                    source="model",
                )

                # Register in knowledge graph
                self.knowledge_graph.register_output(
                    session.session_id,
                    model_name=output.model_name,
                    output_text=output.raw_output,
                )

    def _step7_arbitration_scoring(
        self,
        outputs: List[CognitiveGatewayOutput],
        session: SessionState,
        knowledge_bundle: List[KnowledgeBlock],
    ) -> List[ArbitrationScore]:
        """
        Step 7: Score all outputs using the 5-metric formula.
        
        FinalScore = 0.30*T + 0.25*K + 0.15*S + 0.15*C - 0.15*D
        """
        scores = self.arbitration.score_outputs(
            outputs=outputs,
            topic_centroid=session.topic_centroid_embedding,
            session_context_embedding=session.topic_centroid_embedding,
            knowledge_bundle=knowledge_bundle,
        )

        # Log scores
        for score in scores:
            logger.info(
                f"  {score.model_name}: "
                f"T={score.topic_alignment:.3f} "
                f"K={score.knowledge_grounding:.3f} "
                f"S={score.specificity:.3f} "
                f"C={score.confidence_calibration:.3f} "
                f"D={score.drift_penalty:.3f} "
                f"→ Final={score.final_score:.3f}"
            )

        return scores

    def _step8_mode_handling(
        self,
        request: OrchestratorRequest,
        session: SessionState,
        outputs: List[CognitiveGatewayOutput],
        scores: List[ArbitrationScore],
        knowledge_bundle: List[KnowledgeBlock],
    ) -> OrchestratorResponse:
        """
        Step 8: Mode-specific output handling.

        Standard Mode:
          - Select highest FinalScore
          - Return aggregated answer

        Experimental Mode:
          - Return ALL outputs
          - Display scoring breakdown
          - Show divergence metrics
        """
        results = self.arbitration.build_results(outputs, scores)

        # Single Model Focus: return that model's output directly
        if request.selected_model:
            solo_output = outputs[0] if outputs else None
            solo_score = scores[0] if scores else None

            return OrchestratorResponse(
                session_id=session.session_id,
                chat_id=request.chat_id or "",
                mode=OperatingMode.STANDARD,
                sub_mode=request.sub_mode,
                aggregated_answer=solo_output.raw_output if solo_output else "",
                winning_model=solo_output.model_name if solo_output else request.selected_model,
                winning_score=solo_score.final_score if solo_score else 0.0,
                all_results=results,
                knowledge_bundle=knowledge_bundle,
                drift_score=session.drift_score,
                volatility_score=session.volatility_score,
                refinement_cycles=session.refinement_cycles,
            )

        if request.mode == OperatingMode.STANDARD:
            # Select winner
            winner_output, winner_score = self.arbitration.select_winner(
                outputs, scores
            )

            return OrchestratorResponse(
                session_id=session.session_id,
                chat_id=request.chat_id or "",
                mode=OperatingMode.STANDARD,
                sub_mode=request.sub_mode,
                aggregated_answer=winner_output.raw_output,
                winning_model=winner_output.model_name,
                winning_score=winner_score.final_score,
                all_results=results,
                knowledge_bundle=knowledge_bundle,
                drift_score=session.drift_score,
                volatility_score=session.volatility_score,
                refinement_cycles=session.refinement_cycles,
            )

        else:
            # Experimental: expose everything
            # Still compute best for convenience
            winner_output, winner_score = self.arbitration.select_winner(
                outputs, scores
            )

            divergence = self.arbitration.compute_divergence_metrics(
                outputs, scores
            )

            return OrchestratorResponse(
                session_id=session.session_id,
                chat_id=request.chat_id or "",
                mode=OperatingMode.EXPERIMENTAL,
                sub_mode=request.sub_mode,
                aggregated_answer=winner_output.raw_output,
                winning_model=winner_output.model_name,
                winning_score=winner_score.final_score,
                all_results=results,
                knowledge_bundle=knowledge_bundle,
                drift_score=session.drift_score,
                volatility_score=session.volatility_score,
                refinement_cycles=session.refinement_cycles,
                divergence_metrics=divergence,
                scoring_breakdown=scores,
            )

    async def _step9_update_persistence(
        self,
        session: SessionState,
        outputs: List[CognitiveGatewayOutput],
        scores: List[ArbitrationScore],
        query_embedding: List[float],
    ):
        """
        Step 9: Update persistence layer.
        
        - Store embeddings
        - Update topic centroid
        - Add graph edges
        - Log behavioral metrics
        - Increment refinement_cycles
        """
        # Update centroid with all output embeddings
        for output in outputs:
            if output.success and output.raw_output:
                out_emb = embed_text(output.raw_output)
                self.session_engine.update_topic_centroid(
                    session.session_id, out_emb
                )

        # Log behavioral records
        for score in scores:
            record = BehavioralRecord(
                model_name=score.model_name,
                final_score=score.final_score,
                topic_alignment=score.topic_alignment,
                grounding_score=score.knowledge_grounding,
                specificity=score.specificity,
                confidence_calibration=score.confidence_calibration,
                drift_penalty=score.drift_penalty,
            )
            self.session_engine.log_behavior(session.session_id, record)

        # Increment turn
        self.session_engine.increment_turn(session.session_id)

        # Extract and register concepts from winning output
        best_score = max(scores, key=lambda s: s.final_score, default=None)
        if best_score:
            best_output = next(
                (o for o in outputs if o.model_name == best_score.model_name),
                None,
            )
            if best_output and best_output.raw_output:
                concepts = self._extract_concepts(best_output.raw_output)
                for concept in concepts[:10]:
                    self.knowledge_graph.register_concept(
                        session.session_id, concept
                    )

        # Persist to Redis
        await self.session_engine.persist_to_redis(session.session_id)

    def _step10_background_scheduling(
        self,
        session: SessionState,
        request: OrchestratorRequest,
    ):
        """
        Step 10: Schedule background daemon if unresolved questions exist.
        Actual daemon runs independently (see background_daemon.py).
        """
        unresolved = self.session_engine.get_unresolved_questions(
            session.session_id
        )

        if unresolved:
            logger.info(
                f"Background daemon: {len(unresolved)} unresolved questions "
                f"pending for session {session.session_id}"
            )

        # Detect if this query introduces new unresolved questions
        if "?" in request.query and session.volatility_score > 0.5:
            # Complex volatile question → may need async refinement
            self.session_engine.add_unresolved_question(
                session.session_id,
                question=request.query,
                priority=session.volatility_score,
            )
            logger.info("Unresolved question scheduled for daemon refinement")

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def _detect_task_type(
        self,
        query: str,
        patterns: List[str],
    ) -> bool:
        """Detect if query matches task type patterns."""
        query_lower = query.lower()
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        return False

    def _detect_domain(self, query: str) -> str:
        """Detect domain from query text."""
        domain_signals = {
            "technology": ["software", "hardware", "programming", "ai", "ml", "algorithm"],
            "science": ["physics", "chemistry", "biology", "research", "experiment"],
            "finance": ["stock", "market", "investment", "trading", "economy", "financial"],
            "law": ["legal", "court", "law", "regulation", "statute", "compliance"],
            "medicine": ["medical", "health", "disease", "treatment", "clinical"],
            "engineering": ["engineering", "design", "architecture", "system", "infrastructure"],
        }

        query_lower = query.lower()
        scores = {}
        for domain, keywords in domain_signals.items():
            score = sum(1 for k in keywords if k in query_lower)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for graph registration."""
        # Simple: capitalized phrases and technical terms
        concepts = set()

        # Capitalized multi-word phrases
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            concepts.add(match.group(0))

        # Quoted terms
        for match in re.finditer(r'"([^"]{3,50})"', text):
            concepts.add(match.group(1))

        # Technical terms (hyphenated, etc.)
        for match in re.finditer(r'\b([a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*)\b', text):
            if len(match.group(0)) > 5:
                concepts.add(match.group(0))

        return list(concepts)[:15]
