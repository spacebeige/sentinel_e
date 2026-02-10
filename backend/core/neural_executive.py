import numpy as np
from typing import List, Dict


class NeuralExecutive:
    """
    Cognitive control layer.

    Responsibilities:
    - Detect semantic instability
    - Decide escalation vs stabilization
    - NEVER generate content
    """

    def __init__(
        self,
        disagreement_threshold: float = 0.55,
        sentiment_threshold: float = 0.40,
        min_agents: int = 3
    ):
        self.disagreement_threshold = disagreement_threshold
        self.sentiment_threshold = sentiment_threshold
        self.min_agents = min_agents

    def evaluate(
        self,
        similarities: List[float],
        sentiment_divergence: float
    ) -> Dict:
        """
        Decide whether to escalate reasoning.

        similarities:
            pairwise cosine similarities (semantic ONLY)

        sentiment_divergence:
            affect spread (used only as instability signal)
        """

        if not similarities or len(similarities) < self.min_agents - 1:
            return {
                "avg_agreement": 0.0,
                "sentiment_divergence": round(float(sentiment_divergence), 3),
                "escalate": True,
                "note": "Insufficient agents for reliable agreement"
            }

        avg_agreement = float(np.mean(similarities))

        escalate = False
        notes = []

        # Semantic instability
        if avg_agreement < self.disagreement_threshold:
            escalate = True
            notes.append("Low semantic agreement detected")

        # Affective instability (secondary signal)
        if sentiment_divergence > self.sentiment_threshold:
            notes.append("High affective divergence detected")

        return {
            "avg_agreement": round(avg_agreement, 3),
            "sentiment_divergence": round(float(sentiment_divergence), 3),
            "escalate": escalate,
            "note": " | ".join(notes) if notes else "Stable deliberation state"
        }
