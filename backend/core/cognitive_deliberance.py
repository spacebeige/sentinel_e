import asyncio
import numpy as np
from typing import Dict, List, Callable
from sklearn.metrics.pairwise import cosine_similarity


class CognitiveDeliberationEngine:
    """
    Simulates human-like internal debate among LLM agents.

    Properties:
    - Iterative disagreement
    - Explicit critique + self-defense
    - No forced convergence
    - Stops when epistemic stabilization occurs
    """

    def __init__(self, embed_model):
        """
        embed_model must expose:
        - encode(List[str], normalize_embeddings=True) -> np.ndarray
        """
        self.embed_model = embed_model

    # ============================================================
    # AGREEMENT METRIC
    # ============================================================

    def _agreement(self, texts: List[str]) -> float:
        """
        Compute mean pairwise cosine similarity across agent responses.
        Acts as a proxy for epistemic convergence.
        """
        if len(texts) < 2:
            return 0.0

        # Handle LangChain Embeddings wrapper vs SentenceTransformer
        if hasattr(self.embed_model, 'embed_documents'):
            embeddings = self.embed_model.embed_documents(texts)
        else:
            embeddings = self.embed_model.encode(
                texts,
                normalize_embeddings=True
            )

        sim_matrix = cosine_similarity(embeddings)

        scores = [
            sim_matrix[i][j]
            for i in range(len(texts))
            for j in range(i + 1, len(texts))
        ]

        return float(np.mean(scores)) if scores else 0.0

    # ============================================================
    # MAIN DELIBERATION LOOP
    # ============================================================

    async def deliberate(
        self,
        query: str,
        agents: Dict[str, Callable[[str], asyncio.Future]],
        max_rounds: int = 6,
        stop_delta: float = 0.04,
        min_rounds: int = 2,
    ) -> List[Dict]:
        """
        Parameters
        ----------
        query : str
            User question
        agents : Dict[str, async_fn]
            Mapping of agent name → async LLM call
        max_rounds : int
            Hard upper bound on cognition
        stop_delta : float
            Minimum agreement change to continue
        min_rounds : int
            Prevent premature convergence

        Returns
        -------
        history : List[Dict]
            Full deliberation trace
        """

        history: List[Dict] = []
        prev_agreement = 0.0

        # ========================================================
        # ROUND 0 — INITIAL INDEPENDENT BELIEFS
        # ========================================================

        current: Dict[str, str] = {}
        for name, fn in agents.items():
            current[name] = await fn(query)

        agreement = self._agreement(list(current.values()))

        history.append({
            "round": 0,
            "phase": "initial_beliefs",
            "responses": current,
            "agreement": round(agreement, 4)
        })

        # ========================================================
        # ITERATIVE COGNITION
        # ========================================================

        for r in range(1, max_rounds + 1):
            critiques: Dict[str, str] = {}

            # ----------------------------
            # CRITIQUE PHASE
            # ----------------------------
            for name, answer in current.items():
                others = "\n\n".join(
                    f"{k}:\n{v}" for k, v in current.items() if k != name
                )

                critique_prompt = f"""
You are {name}.

Your current position:
{answer}

Other agents argue:
{others}

Task:
- Identify logical flaws or weak assumptions
- Point out missing considerations
- Explicitly state where and why you disagree

Do NOT be polite. Be intellectually honest.
Respond concisely but sharply.
"""

                critiques[name] = await agents[name](critique_prompt)

            # ----------------------------
            # REVISION / DEFENSE PHASE
            # ----------------------------
            revised: Dict[str, str] = {}
            for name in current:
                revise_prompt = f"""
Original position:
{current[name]}

Your critique of others:
{critiques[name]}

Now reflect:
- Revise your answer if convinced
- Otherwise, defend your position explicitly

Important:
Do NOT force agreement. Preserve uncertainty if unresolved.
"""

                revised[name] = await agents[name](revise_prompt)

            texts = list(revised.values())
            agreement = self._agreement(texts)

            history.append({
                "round": r,
                "phase": "critique_and_revision",
                "responses": revised,
                "agreement": round(agreement, 4)
            })

            # ----------------------------
            # STOP CONDITION (HUMAN-LIKE)
            # ----------------------------
            delta = abs(agreement - prev_agreement)

            if r >= min_rounds and delta < stop_delta:
                history.append({
                    "round": r,
                    "phase": "stabilization_reached",
                    "reason": "agreement_change_below_threshold",
                    "delta": round(delta, 4)
                })
                break

            prev_agreement = agreement
            current = revised

        return history
