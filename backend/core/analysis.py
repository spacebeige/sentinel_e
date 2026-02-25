# import numpy as np
# from typing import Dict, List

# from sklearn.cluster import DBSCAN
# from sklearn.ensemble import IsolationForest

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ============================================================
# # NEURAL SENTIMENT / DECISION NETWORK
# # ============================================================

# class SimpleSentimentNet(nn.Module):
#     """
#     Lightweight neural decision head operating on sentence embeddings.

#     IMPORTANT:
#     - This is NOT a language model
#     - It approximates a cognitive "approval / rejection" surface
#     """

#     def __init__(self, input_dim: int, hidden_dim: int):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)

#         # Xavier init → stable random projection
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         return x


# class NeuralAnalyzer:
#     """
#     Neural executive-style analyzer.

#     Purpose:
#     - Map semantic embeddings → soft decision signal
#     - Approximate internal "confidence / approval" dynamics

#     This mimics:
#     - prefrontal cortex gating
#     - NOT emotion or factual correctness
#     """

#     def __init__(self, embedding_dim: int = 384):
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = 64

#         self.model = SimpleSentimentNet(
#             input_dim=self.embedding_dim,
#             hidden_dim=self.hidden_dim
#         )

#         # Frozen random projection + nonlinear readout
#         self.model.eval()

#     def analyze_embedding(
#         self,
#         embedding: np.ndarray
#     ) -> Dict[str, float]:
#         """
#         Analyze embedding → cognitive approval signal.

#         Returns:
#         - score ∈ [0,1]
#         - confidence ∈ [0,1]
#         """

#         if not isinstance(embedding, np.ndarray):
#             embedding = np.array(embedding, dtype=np.float32)

#         if embedding.shape != (self.embedding_dim,):
#             raise ValueError(
#                 f"Expected embedding shape ({self.embedding_dim},), "
#                 f"got {embedding.shape}"
#             )

#         with torch.no_grad():
#             x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
#             score = float(self.model(x).item())

#         # Distance from decision boundary → confidence proxy
#         confidence = abs(score - 0.5) * 2.0

#         return {
#             "score": round(score, 4),
#             "confidence": round(confidence, 4),
#             "sentiment": "positive" if score >= 0.5 else "negative",
#             "decision": "approve" if score >= 0.6 else "reject"
#         }

#     def map_symbolic_patterns(
#         self,
#         patterns: List[str]
#     ) -> Dict[str, str]:
#         """
#         Map symbolic patterns → abstract executive actions.

#         Keeps:
#         - symbolic reasoning
#         - neural reasoning
#         STRICTLY separated (research hygiene)
#         """

#         decisions = {}

#         for p in patterns:
#             p_lower = p.lower()

#             if any(k in p_lower for k in ["risk", "harm", "unsafe"]):
#                 decisions[p] = "mitigate"
#             elif any(k in p_lower for k in ["benefit", "advantage", "safe"]):
#                 decisions[p] = "reinforce"
#             elif any(k in p_lower for k in ["uncertain", "unknown", "ambiguous"]):
#                 decisions[p] = "defer"
#             else:
#                 decisions[p] = "observe"

#         return decisions


# # ============================================================
# # CLUSTERING & ANOMALY DETECTION
# # ============================================================

# class ClusterEngine:
#     """
#     Behavioral clustering + anomaly detection.

#     Tracks epistemic state over time.

#     Input vector format (FIXED):
#     [
#         avg_agreement,
#         sentiment_confidence,
#         disagreement_entropy,
#         response_length_norm
#     ]
#     """

#     def __init__(
#         self,
#         contamination: float = 0.12,
#         eps: float = 0.6,
#         min_samples: int = 3
#     ):
#         self.anomaly_detector = IsolationForest(
#             contamination=contamination,
#             random_state=42
#         )

#         self.cluster_model = DBSCAN(
#             eps=eps,
#             min_samples=min_samples,
#             metric="euclidean"
#         )

#         self.vector_history: List[np.ndarray] = []

#     def process_state_vector(
#         self,
#         vector: List[float]
#     ) -> Dict[str, float]:
#         """
#         Add epistemic state vector → anomaly + cluster assignment.
#         """

#         vector = np.asarray(vector, dtype=np.float32)

#         if vector.ndim != 1 or vector.shape[0] != 4:
#             raise ValueError(
#                 "State vector must be 1D with 4 elements: "
#                 "[agreement, confidence, entropy, length]"
#             )

#         # Warm-up phase (brain calibration)
#         if len(self.vector_history) < 5:
#             self.vector_history.append(vector)
#             return {
#                 "status": "calibrating",
#                 "data_points": len(self.vector_history)
#             }

#         X = np.vstack(self.vector_history + [vector])

#         # Anomaly detection
#         self.anomaly_detector.fit(X)
#         is_anomaly = self.anomaly_detector.predict([vector])[0] == -1

#         # Behavioral clustering
#         labels = self.cluster_model.fit_predict(X)
#         cluster_id = int(labels[-1])

#         self.vector_history.append(vector)

#         # Confidence heuristic
#         confidence = 0.85 if not is_anomaly else 0.4

#         return {
#             "cluster_id": cluster_id,
#             "is_anomaly": bool(is_anomaly),
#             "confidence": round(confidence, 3)
#         }
import numpy as np
from typing import Dict, List

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

import os as _os

# ── Guard torch behind feature flag (not needed for cloud deployment) ──
_USE_NEURAL = _os.getenv("USE_LOCAL_INGESTION", "false").lower() == "true"

try:
    if _USE_NEURAL:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _TORCH_AVAILABLE = True
    else:
        _TORCH_AVAILABLE = False
except ImportError:
    _TORCH_AVAILABLE = False


# ============================================================
# NEURAL DECISION HEAD
# ============================================================

if _TORCH_AVAILABLE:
    class SimpleDecisionNet(nn.Module):
        """
        Tiny neural decision surface.

        This is NOT a language model.
        It operates on embeddings only.
        """

        def __init__(self, input_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return torch.sigmoid(self.fc2(x))


class NeuralAnalyzer:
    """
    Neural fallback evaluator.

    Purpose:
    - Estimate internal approval
    - Produce confidence proxy
    Falls back to heuristic scoring when torch is unavailable.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.model = None
        if _TORCH_AVAILABLE:
            self.model = SimpleDecisionNet(embedding_dim)
            self.model.eval()

    def analyze_embedding(self, embedding: np.ndarray) -> Dict:
        """
        embedding:
            sentence-transformer vector
        """

        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embedding.shape}"
            )

        if _TORCH_AVAILABLE and self.model is not None:
            with torch.no_grad():
                x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                score = float(self.model(x).item())
        else:
            # Heuristic fallback: use mean of embedding as proxy score
            score = float(np.clip(np.mean(embedding) + 0.5, 0, 1))

        # Distance from boundary → confidence
        confidence = abs(score - 0.5) * 2.0

        return {
            "approval_score": round(score, 4),
            "confidence": round(float(confidence), 4),
            "decision": "approve" if score >= 0.6 else "reject",
            "note": "Neural fallback evaluation (non-semantic)" if _TORCH_AVAILABLE else "Heuristic evaluation (torch unavailable)"
        }


# ============================================================
# CLUSTERING + ANOMALY DETECTION
# ============================================================

class ClusterEngine:
    """
    Epistemic stability detector.

    Vector format (recommended):
    [
        avg_semantic_agreement,
        neural_confidence,
        sentiment_divergence,
        response_length_norm
    ]
    """

    def __init__(
        self,
        contamination: float = 0.15,
        eps: float = 0.7,
        min_samples: int = 3
    ):
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        self.cluster_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean"
        )

        self.history: List[np.ndarray] = []

    def process(self, vector: List[float]) -> Dict:
        """
        Add vector → detect anomaly → cluster behavior
        """

        vector = np.array(vector, dtype=np.float32)

        if vector.ndim != 1:
            raise ValueError("Cluster vector must be 1D")

        # Warm-up phase
        if len(self.history) < 5:
            self.history.append(vector)
            return {
                "status": "calibrating",
                "samples": len(self.history)
            }

        X = np.vstack(self.history + [vector])

        # --- Anomaly detection ---
        self.anomaly_detector.fit(X)
        is_anomaly = self.anomaly_detector.predict([vector])[0] == -1

        # --- Clustering ---
        cluster_labels = self.cluster_model.fit_predict(X)
        cluster_id = int(cluster_labels[-1])

        self.history.append(vector)

        return {
            "cluster_id": cluster_id,
            "is_anomaly": bool(is_anomaly),
            "confidence": 0.85 if not is_anomaly else 0.4,
            "note": "Clustering-based epistemic assessment"
        }
