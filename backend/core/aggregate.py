import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity


class AggregateUnderstanding:
    """
    Produces a confidence-aware semantic consensus from multiple model responses.

    IMPORTANT:
    - Style, verbosity, and formatting are IGNORED
    - Only semantic meaning is evaluated
    """

    def __init__(self, embedder):
        self.embedder = embedder

    # ---------------------------------------------------------
    def aggregate(self, texts: List[str]) -> Dict:
        """
        Returns:
        - aggregate_text (semantic centroid)
        - confidence (semantic agreement)
        - agreement_type
        - epistemic_warning
        """

        if not texts:
            return {
                "aggregate_text": "",
                "confidence": 0.0,
                "agreement_type": "none",
                "epistemic_warning": "No valid model responses."
            }

        # === Semantic agreement only ===
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True
        )
        sim_matrix = cosine_similarity(embeddings)

        pairwise_sims = [
            sim_matrix[i][j]
            for i in range(len(texts))
            for j in range(i + 1, len(texts))
        ]

        semantic_agreement = (
            float(np.mean(pairwise_sims))
            if pairwise_sims else 1.0
        )

        # === Semantic centroid selection ===
        avg_sim_per_response = sim_matrix.mean(axis=1)
        best_idx = int(np.argmax(avg_sim_per_response))
        aggregate_text = texts[best_idx]

        confidence = round(float(semantic_agreement), 3)

        # === Epistemic classification ===
        if semantic_agreement >= 0.75:
            agreement_type = "strong_semantic_consensus"
            warning = None
        elif semantic_agreement >= 0.55:
            agreement_type = "partial_semantic_overlap"
            warning = (
                "Models broadly agree on meaning, "
                "but some divergence exists."
            )
        else:
            agreement_type = "semantic_disagreement"
            warning = (
                "⚠️ Models do NOT agree semantically. "
                "This result may not be true."
            )

        return {
            "aggregate_text": aggregate_text,
            "confidence": confidence,
            "semantic_agreement": round(semantic_agreement, 3),
            "agreement_type": agreement_type,
            "epistemic_warning": warning
        }
