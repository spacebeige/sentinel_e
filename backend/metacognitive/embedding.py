"""
============================================================
Embedding Utilities — Vector Operations for Meta-Cognitive System
============================================================
Provides:
- Text embedding via sentence-transformers (local, fast)
- Cosine similarity
- Topic centroid computation
- Drift/volatility scoring
- Variance computation for stability checks

No external API dependency for embeddings.
Uses lightweight local model for speed.
"""

import logging
import math
import hashlib
from typing import List, Optional, Tuple, Dict

import numpy as np

logger = logging.getLogger("MCO-Embeddings")

# ============================================================
# Lazy Model Loading
# ============================================================

_embedding_model = None
_embedding_dim: int = 384


def _get_model():
    """Lazy-load sentence-transformers model."""
    global _embedding_model, _embedding_dim
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            _embedding_dim = _embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded: all-MiniLM-L6-v2 (dim={_embedding_dim})")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to hash-based pseudo-embeddings."
            )
            _embedding_model = "fallback"
            _embedding_dim = 384
    return _embedding_model


def get_embedding_dim() -> int:
    """Return embedding dimensionality."""
    _get_model()
    return _embedding_dim


# ============================================================
# Embedding Generation
# ============================================================

def embed_text(text: str) -> List[float]:
    """
    Generate embedding vector for text.
    Uses sentence-transformers if available, else hash-based fallback.
    """
    model = _get_model()
    if model == "fallback":
        return _hash_embed(text)
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return _hash_embed(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch embed multiple texts."""
    model = _get_model()
    if model == "fallback":
        return [_hash_embed(t) for t in texts]
    try:
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.tolist()
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        return [_hash_embed(t) for t in texts]


def _hash_embed(text: str, dim: int = 384) -> List[float]:
    """
    Deterministic hash-based pseudo-embedding.
    Not semantically meaningful, but provides consistent vectors
    for cosine similarity when sentence-transformers is unavailable.
    """
    h = hashlib.sha512(text.encode("utf-8")).digest()
    # Expand hash to fill dim
    repeats = (dim * 8 + len(h) * 8 - 1) // (len(h) * 8)
    expanded = (h * (repeats + 1))[:dim]
    raw = [float(b) / 255.0 - 0.5 for b in expanded]
    # Normalize
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


# ============================================================
# Vector Operations
# ============================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    dot = float(np.dot(va, vb))
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_centroid(vectors: List[List[float]]) -> List[float]:
    """Compute the mean centroid of a set of vectors."""
    if not vectors:
        return []
    mat = np.array(vectors, dtype=np.float32)
    centroid = np.mean(mat, axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 1e-10:
        centroid = centroid / norm
    return centroid.tolist()


def centroid_variance(vectors: List[List[float]], centroid: List[float]) -> float:
    """
    Compute variance of vectors around a centroid.
    Used for stability detection: stop stabilization when < epsilon.
    """
    if not vectors or not centroid:
        return 1.0
    c = np.array(centroid, dtype=np.float32)
    distances = []
    for v in vectors:
        va = np.array(v, dtype=np.float32)
        distances.append(1.0 - float(np.dot(va, c) / (np.linalg.norm(va) * np.linalg.norm(c) + 1e-10)))
    return float(np.mean(distances))


def drift_score(output_embedding: List[float], session_context_embedding: List[float]) -> float:
    """
    D = 1 - cosine(output_embedding, session_context_embedding)
    Higher = more drift from context.
    """
    sim = cosine_similarity(output_embedding, session_context_embedding)
    return max(0.0, 1.0 - sim)


def volatility_score(query_embedding: List[float], topic_centroid: List[float]) -> float:
    """
    Measures how far a new query is from the established topic centroid.
    High volatility → retrieval is mandatory.
    """
    if not topic_centroid:
        return 1.0  # No centroid = maximum volatility (always retrieve)
    sim = cosine_similarity(query_embedding, topic_centroid)
    return max(0.0, 1.0 - sim)


# ============================================================
# Text Analysis Utilities (for Specificity scoring)
# ============================================================

# Common technical term patterns
_TECHNICAL_PATTERNS = None


def count_named_entities(text: str) -> int:
    """
    Simple named entity approximation.
    Counts capitalized multi-word sequences (proper nouns).
    """
    import re
    # Match sequences of capitalized words
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'
    matches = re.findall(pattern, text)
    # Also count acronyms
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    return len(matches) + len(acronyms)


def count_technical_terms(text: str) -> int:
    """
    Count technical/domain-specific terms.
    Uses a lightweight heuristic approach.
    """
    import re
    indicators = [
        r'\b\w+(?:tion|sion|ment|ness|ity|ism|ist|ous|ive|ical|ology|ography)\b',
        r'\b(?:API|SDK|HTTP|REST|SQL|CPU|GPU|RAM|DNS|TCP|UDP|JSON|XML|YAML)\b',
        r'\b\w+[-_]\w+\b',  # Hyphenated/underscored terms
        r'\b\d+(?:\.\d+)+\b',  # Version numbers
        r'\b0x[0-9a-fA-F]+\b',  # Hex values
    ]
    count = 0
    for pattern in indicators:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def token_count_approx(text: str) -> int:
    """Approximate token count (words ≈ 0.75 tokens)."""
    words = text.split()
    return max(1, int(len(words) / 0.75))
