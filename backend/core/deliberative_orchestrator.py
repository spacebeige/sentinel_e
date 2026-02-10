import asyncio
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Data structures (research-grade)
# -------------------------------

@dataclass
class ModelResponse:
    model_name: str
    role: str
    raw_text: str
    normalized_text: str
    embedding: np.ndarray


@dataclass
class Disagreement:
    pair: tuple
    similarity: float
    type: str   # semantic | emphasis | structural | contradiction


# -------------------------------
# Utilities
# -------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\n•\-*]", " ", text)
    text = re.sub(r"\d+\.", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_disagreement(sim: float) -> str:
    if sim > 0.75:
        return "stylistic / emphasis"
    elif sim > 0.55:
        return "partial semantic overlap"
    elif sim > 0.35:
        return "semantic divergence"
    else:
        return "potential contradiction"


# -------------------------------
# Deliberative Orchestrator
# -------------------------------

class DeliberativeOrchestrator:

    def __init__(self, cloud_client):
        self.cloud = cloud_client
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ---------------------------------
    # Parallel role-conditioned prompts
    # ---------------------------------

    async def _query_model(self, model_fn, model_name, role, base_prompt):
        role_prompt = f"""
You are acting as: {role}

Respond to the query below from this perspective.
Be honest and explicit about uncertainty.

QUERY:
{base_prompt}
"""
        text = await model_fn(role_prompt)
        norm = normalize_text(text)
        emb = self.embedder.encode([norm], normalize_embeddings=True)[0]

        return ModelResponse(
            model_name=model_name,
            role=role,
            raw_text=text,
            normalized_text=norm,
            embedding=emb
        )

    # ---------------------------------
    # Main deliberation pipeline
    # ---------------------------------

    async def deliberate(self, query: str) -> Dict:

        roles = {
            "Groq": "Fast intuitive responder",
            "Mistral": "Analytical and cautious reasoner",
            "Qwen": "Detail-oriented explainer"
        }

        responses: List[ModelResponse] = await asyncio.gather(
            self._query_model(self.cloud.call_groq, "Groq", roles["Groq"], query),
            self._query_model(self.cloud.call_mistral, "Mistral", roles["Mistral"], query),
            self._query_model(self.cloud.call_qwen, "Qwen", roles["Qwen"], query),
        )

        # ---------------------------------
        # Pairwise disagreement analysis
        # ---------------------------------

        disagreements: List[Disagreement] = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = float(cosine_similarity(
                    responses[i].embedding.reshape(1, -1),
                    responses[j].embedding.reshape(1, -1)
                )[0][0])

                disagreements.append(
                    Disagreement(
                        pair=(responses[i].model_name, responses[j].model_name),
                        similarity=sim,
                        type=classify_disagreement(sim)
                    )
                )

        # ---------------------------------
        # Deliberation (model quarrel)
        # ---------------------------------

        conflict_prompt = self._build_conflict_prompt(responses)
        arbitrated_answer = await self.cloud.call_mistral(conflict_prompt)

        # ---------------------------------
        # Output package
        # ---------------------------------

        return {
            "query": query,
            "individual_responses": [
                {
                    "model": r.model_name,
                    "role": r.role,
                    "answer": r.raw_text
                }
                for r in responses
            ],
            "disagreements": [
                {
                    "pair": d.pair,
                    "similarity": round(d.similarity, 3),
                    "type": d.type
                }
                for d in disagreements
            ],
            "arbitrated_synthesis": arbitrated_answer,
            "epistemic_note": (
                "Differences above reflect divergent reasoning paths, "
                "not necessarily incorrectness. User discretion advised."
            )
        }

    # ---------------------------------
    # Conflict synthesis prompt
    # ---------------------------------

    def _build_conflict_prompt(self, responses: List[ModelResponse]) -> str:
        prompt = "Multiple AI models produced differing answers.\n\n"

        for r in responses:
            prompt += f"{r.model_name} ({r.role}):\n{r.raw_text}\n\n"

        prompt += """
Your task:
- Identify points of agreement
- Identify points of disagreement
- Resolve conflicts if possible
- If unresolved, explicitly state uncertainty

Produce a balanced synthesis.
"""

        return prompt
