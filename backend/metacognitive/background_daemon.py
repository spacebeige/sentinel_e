"""
============================================================
Async Background Daemon — Silent Session Evolution
============================================================
Runs independently. Monitors active sessions for:
  - New relevant data
  - Unresolved questions
  - Memory compression
  - Topic centroid recomputation

Rules:
  ✗ No raw pass — only refined pass
  ✗ Only if new relevant data exists
  ✓ Silent evolution (no user notification)

Pseudo-loop:
  while True:
    for session in active_sessions:
      if new_data_relevant(session.topic_centroid):
        bundle = retrieval_api(session.topic_centroid)
        stabilized = stabilize(session, bundle)
        refined_output = call_primary_model(stabilized)
        update_session(session, refined_output)
      compress_memory(session)
      recompute_topic_centroid(session)
    sleep(interval)
============================================================
"""

import asyncio
import logging
import time
from typing import Optional

from metacognitive.schemas import (
    KnowledgeRetrievalInput,
    CognitiveGatewayInput,
    QueryMode,
)
from metacognitive.cognitive_gateway import CognitiveModelGateway
from metacognitive.knowledge_engine import KnowledgeRetrievalEngine
from metacognitive.session_engine import SessionPersistenceEngine
from metacognitive.embedding import (
    embed_text,
    cosine_similarity,
)

logger = logging.getLogger("MCO-Daemon")


# ============================================================
# Configuration
# ============================================================

DEFAULT_INTERVAL_SECONDS = 300  # 5 minutes
MAX_DAEMON_ITERATIONS = 100     # Safety limit
RELEVANCE_THRESHOLD = 0.4      # New data must be this relevant


class BackgroundDaemon:
    """
    Async Background Daemon for silent session evolution.

    Operates independently of the request cycle.
    Only performs refined passes — never raw.
    Only acts when new relevant data is discovered.
    """

    def __init__(
        self,
        cognitive_gateway: CognitiveModelGateway,
        knowledge_engine: KnowledgeRetrievalEngine,
        session_engine: SessionPersistenceEngine,
        interval: int = DEFAULT_INTERVAL_SECONDS,
    ):
        self.gateway = cognitive_gateway
        self.knowledge = knowledge_engine
        self.sessions = session_engine
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._iterations = 0

    # ── Lifecycle ────────────────────────────────────────────

    def start(self):
        """Start the background daemon."""
        if self._running:
            logger.warning("Daemon already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Background daemon started (interval={self.interval}s)")

    def stop(self):
        """Stop the background daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Background daemon stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Main Loop ────────────────────────────────────────────

    async def _run_loop(self):
        """Main daemon loop."""
        while self._running and self._iterations < MAX_DAEMON_ITERATIONS:
            try:
                await self._process_all_sessions()
            except asyncio.CancelledError:
                logger.info("Daemon cancelled")
                break
            except Exception as e:
                logger.error(f"Daemon cycle error: {e}")

            self._iterations += 1
            await asyncio.sleep(self.interval)

        self._running = False
        logger.info(f"Daemon exited after {self._iterations} iterations")

    async def _process_all_sessions(self):
        """Process all active sessions."""
        sessions = self.sessions._sessions

        if not sessions:
            return

        logger.debug(f"Daemon checking {len(sessions)} sessions")

        for session_id, session in list(sessions.items()):
            try:
                await self._process_session(session_id)
            except Exception as e:
                logger.warning(f"Daemon: session {session_id} error: {e}")

    async def _process_session(self, session_id: str):
        """
        Process a single session.
        
        1. Check for new relevant data
        2. If found: retrieve, stabilize, refine
        3. Compress memory
        4. Recompute centroid
        """
        session = self.sessions.get_session(session_id)
        if not session or not session.topic_centroid_embedding:
            return

        # 1. Check for unresolved questions
        unresolved = self.sessions.get_unresolved_questions(session_id)
        if not unresolved:
            # No work to do — just compress and recompute
            self.sessions.suppress_low_signal_blocks(session_id, threshold=0.1)
            return

        # 2. Check for new relevant data
        retrieval_input = KnowledgeRetrievalInput(
            query_embedding=session.topic_centroid_embedding,
            query_text=unresolved[0].question if unresolved else "",
            volatility_score=session.volatility_score,
            domain="",
            concept_expansion_depth=0,
        )

        result = await self.knowledge.retrieve(retrieval_input)

        # Check relevance
        if not result.knowledge_bundle:
            return

        new_relevant = any(
            cosine_similarity(
                b.embedding, session.topic_centroid_embedding
            ) > RELEVANCE_THRESHOLD
            for b in result.knowledge_bundle
            if b.embedding
        )

        if not new_relevant:
            logger.debug(f"No new relevant data for session {session_id}")
            return

        # 3. Refine — ONLY refined pass (never raw)
        logger.info(
            f"Daemon: new data found for session {session_id}. "
            f"Performing refined pass."
        )

        session_summary = self.sessions.get_session_summary(session_id)

        gateway_input = CognitiveGatewayInput(
            stabilized_context={
                "daemon_refinement": True,
                "drift_score": session.drift_score,
                "volatility_score": session.volatility_score,
            },
            knowledge_bundle=result.knowledge_bundle,
            session_summary=session_summary,
            user_query=unresolved[0].question,
            mode=QueryMode.REFINED,
        )

        # Use primary model (Llama for conceptual by default)
        refined_output = await self.gateway.invoke_model(
            "llama-3.3", gateway_input
        )

        if refined_output.success and refined_output.raw_output:
            # 4. Update session with refined data
            refined_emb = embed_text(refined_output.raw_output)
            self.sessions.add_memory_block(
                session_id,
                content=f"[Daemon Refinement] {refined_output.raw_output[:500]}",
                source="daemon",
                embedding=refined_emb,
            )

            # Update centroid
            self.sessions.update_topic_centroid(session_id, refined_emb)

            # Mark question as attempted
            unresolved[0].attempts += 1
            unresolved[0].last_attempt = (
                __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat()
            )

            # Resolve if confidence is high enough
            if result.retrieval_confidence > 0.7:
                self.sessions.resolve_question(session_id, unresolved[0].id)
                logger.info(
                    f"Daemon resolved question {unresolved[0].id} "
                    f"(confidence={result.retrieval_confidence:.3f})"
                )

        # 5. Compress memory
        self.sessions.suppress_low_signal_blocks(session_id, threshold=0.1)

        # 6. Persist
        await self.sessions.persist_to_redis(session_id)
