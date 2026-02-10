from typing import List, Dict
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("Aggregator")

class Aggregator:
    """
    Standard Mode Aggregator.
    Synthesizes multiple model outputs into a single coherent response.
    """
    def __init__(self, embedding_model=None):
        # We can pass the shared embedding model from Ingestion or Utils
        self.embedder = embedding_model

    def aggregate(self, responses: List[str]) -> Dict[str, str]:
        """
        Selects the most representative response based on semantic similarity.
        """
        if not responses:
             return {"text": "No responses generated.", "confidence": 0.0, "warning": "System Error"}
             
        if len(responses) == 1:
            return {"text": responses[0], "confidence": 1.0, "warning": None}

        if not self.embedder:
             # Fallback if no embedder: just return the first one or longest?
             # For Standard mode, returning the longest is a heuristic for "most detailed"
             # but strictly we should probably fail or log warning.
             return {"text": responses[0], "confidence": 0.0, "warning": "Aggregation skipped (No Embedder)"}

        try:
            # Assumes self.embedder has .embed_documents (LangChain) or .encode (SentenceTransformers)
            if hasattr(self.embedder, 'embed_documents'):
                embeddings = self.embedder.embed_documents(responses)
            else:
                embeddings = self.embedder.encode(responses)
                
            sim_matrix = cosine_similarity(embeddings)
            
            # Extract pairwise similarities for Neural Executive
            pairwise_sims = [
                sim_matrix[i][j]
                for i in range(len(responses))
                for j in range(i + 1, len(responses))
            ]
            
            # Find the response with highest average similarity to others (Centroid)
            avg_sim = sim_matrix.mean(axis=1)
            best_idx = int(np.argmax(avg_sim))
            
            consensus_score = float(np.mean(sim_matrix))
            
            # Detailed logging for debugging
            logger.info(f"[AGGREGATION DEBUG] Response count: {len(responses)}")
            logger.info(f"[AGGREGATION DEBUG] Similarity matrix:\n{sim_matrix}")
            logger.info(f"[AGGREGATION DEBUG] Average similarity per model: {avg_sim}")
            logger.info(f"[AGGREGATION DEBUG] Consensus score: {consensus_score}")
            logger.info(f"[AGGREGATION DEBUG] Selected response index: {best_idx}")
            
            warning = None
            if consensus_score < 0.6:
                warning = "Low consensus among internal models. Verify this information."
                logger.warning(f"[AGGREGATION DEBUG] {warning} Score: {consensus_score}")

            return {
                "text": responses[best_idx],
                "confidence": round(consensus_score, 2),
                "pairwise_similarities": pairwise_sims,
                "warning": warning
            }
        except Exception as e:
            return {"text": responses[0], "confidence": 0.0, "warning": f"Aggregation Error: {str(e)}"}
