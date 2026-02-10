from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_agreement_score(embeddings):
    """
    Calculate the mean cosine similarity between multiple response vectors.
    Returns: float (0.0 to 1.0)
    """
    if len(embeddings) < 2:
        return 1.0
    
    matrix = cosine_similarity(embeddings)
    # We want the average of the upper triangle (excluding diagonal)
    mask = np.ones(matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    return matrix[mask].mean()
