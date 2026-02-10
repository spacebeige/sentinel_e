import json
import os
import hashlib
from datetime import datetime
from typing import Optional, Dict, List


class KnowledgeBase:
    """
    Persistent cognitive memory for deliberative AI.

    Purpose:
    - Store high-agreement reasoning outcomes
    - Enable recall of prior successful deliberations
    - Act as a long-term epistemic memory
    """

    def __init__(self, storage_file: str = "data/knowledge_base.json"):
        self.storage_file = storage_file
        self.memory: List[Dict] = []
        self._load_memory()

    # ============================================================
    # INTERNAL UTILITIES
    # ============================================================

    def _safe_hash(self, text: str) -> str:
        """
        Deterministic hash (Python hash() is NOT stable across runs).
        """
        if not isinstance(text, str):
            text = str(text)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _ensure_dir(self):
        directory = os.path.dirname(self.storage_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

    # ============================================================
    # LOAD / SAVE
    # ============================================================

    def _load_memory(self):
        """
        Load memory safely.
        Corruption or schema drift should never crash cognition.
        """
        self._ensure_dir()

        if not os.path.exists(self.storage_file):
            self.memory = []
            return

        try:
            with open(self.storage_file, "r") as f:
                data = json.load(f)

            # Schema guard
            if isinstance(data, list):
                self.memory = data
            else:
                self.memory = []

        except Exception:
            # Corrupt memory → reset, but do not crash
            self.memory = []

    def _persist(self):
        """
        Atomic-ish write to reduce corruption risk.
        """
        tmp_path = self.storage_file + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.memory, f, indent=2)
        os.replace(tmp_path, self.storage_file)

    # ============================================================
    # WRITE (LEARNING)
    # ============================================================

    def save_interaction(
        self,
        context: str,
        cloud_insight: str,
        patterns: List[str],
        agreement_score: float,
        epistemic_note: Optional[str] = None
    ):
        """
        Store a high-quality deliberative outcome.

        IMPORTANT:
        - This is selective memory (learning)
        - NOT raw logging
        """

        if not context or not cloud_insight:
            return  # Ignore empty or degenerate entries

        context_hash = self._safe_hash(context)

        # Deduplicate by stable context hash
        for entry in self.memory:
            if entry.get("context_hash") == context_hash:
                return

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "context_snippet": context[:500],
            "context_hash": context_hash,
            "cloud_insight": cloud_insight,
            "patterns": patterns or [],
            "agreement_score": round(float(agreement_score), 4),
            "epistemic_note": epistemic_note,
        }

        self.memory.append(entry)
        self._persist()

    # ============================================================
    # READ (RECALL)
    # ============================================================

    def retrieve_similar_context(
        self,
        current_context: str,
        min_similarity: float = 0.08
    ) -> Optional[Dict]:
        """
        Retrieve the most similar past interaction using
        token-level Jaccard similarity.

        NOTE:
        - Intentionally simple
        - Replaceable with FAISS / Chroma later
        """

        if not self.memory or not current_context:
            return None

        query_tokens = set(current_context.lower().split())
        if not query_tokens:
            return None

        best_match = None
        highest_score = 0.0

        for entry in self.memory:
            snippet = entry.get("context_snippet", "")
            memory_tokens = set(snippet.lower().split())

            if not memory_tokens:
                continue

            intersection = query_tokens & memory_tokens
            union = query_tokens | memory_tokens

            score = len(intersection) / (len(union) + 1e-9)

            if score > highest_score:
                highest_score = score
                best_match = entry

        if highest_score >= min_similarity:
            return best_match

        return None

    # ============================================================
    # ANALYTICAL ACCESS (COGNITIVE)
    # ============================================================

    def get_high_confidence_memories(
        self,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Return memories with strong model agreement.
        """
        return [
            m for m in self.memory
            if m.get("agreement_score", 0.0) >= threshold
        ]

    def summarize_memory(self) -> Dict[str, float]:
        """
        Lightweight epistemic summary of memory state.
        """

        if not self.memory:
            return {
                "total_entries": 0,
                "avg_agreement": 0.0,
                "max_agreement": 0.0,
                "min_agreement": 0.0,
            }

        agreements = [
            float(m.get("agreement_score", 0.0))
            for m in self.memory
        ]

        return {
            "total_entries": len(self.memory),
            "avg_agreement": round(sum(agreements) / len(agreements), 4),
            "max_agreement": round(max(agreements), 4),
            "min_agreement": round(min(agreements), 4),
        }
