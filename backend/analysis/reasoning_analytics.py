"""
============================================================
Reasoning Analytics Engine — Sentinel-E v3
============================================================
Advanced reasoning behavior analytics for AI evaluation.

Extends the base MetricsEngine with:
  - Cross-model reasoning pattern comparison
  - Reasoning step extraction (transparency layer)
  - Model performance tracking over time
  - Disagreement analysis (where and why models diverge)

Visualization outputs:
  - Reasoning heatmap (model × metric matrix)
  - Conflict graph (pairwise disagreement edges)
  - Agreement matrix (cosine similarity of stances)
  - Model performance chart (composite scores)

No external LLM judge. Pure structural analysis.
============================================================
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ReasoningAnalytics")


# ── Reasoning Step Patterns ──────────────────────────────────
_STEP_PATTERNS = [
    (r"\b(identify|identifying|recognize)\b.*\b(claim|issue|problem|question)\b", "identify_claim"),
    (r"\b(analyz\w*|examin\w*|evaluat\w*|assess\w*)\b.*\b(evidence|data|source|argument)\b", "analyze_evidence"),
    (r"\b(compar\w*|contrast\w*|weigh\w*|consider\w*)\b.*\b(alternative|option|approach|perspective)\b", "compare_alternatives"),
    (r"\b(conclud\w*|therefore|thus|hence|as a result)\b", "produce_conclusion"),
    (r"\b(assum\w*|presuppos\w*|presum\w*|take for granted)\b", "identify_assumptions"),
    (r"\b(risk|vulnerability|weakness|flaw|limitation)\b", "assess_risks"),
    (r"\b(counter\w*|rebut\w*|challeng\w*|disput\w*|object\w*)\b", "counter_argument"),
    (r"\b(synthes\w*|integrat\w*|reconcil\w*|merg\w*)\b", "synthesize"),
]

_STEP_RES = [(re.compile(p, re.IGNORECASE), label) for p, label in _STEP_PATTERNS]


class ReasoningStepExtractor:
    """
    Extract structured reasoning steps from model output.

    Does NOT show raw chain-of-thought. Instead, produces a
    structured summary of reasoning patterns detected.
    """

    @staticmethod
    def extract_steps(text: str) -> List[Dict[str, Any]]:
        """
        Extract reasoning steps from text.

        Returns a list of:
            {"step": "identify_claim", "confidence": 0.8, "excerpt": "..."}
        """
        if not text:
            return []

        steps = []
        seen = set()
        # Split on sentence-ending punctuation AND commas/semicolons
        # to handle comma-separated reasoning clauses
        clauses = [s.strip() for s in re.split(r'[.!?;,]+', text) if len(s.strip()) > 8]

        for clause in clauses:
            for pattern, label in _STEP_RES:
                if label in seen:
                    continue
                if pattern.search(clause):
                    steps.append({
                        "step": label,
                        "confidence": 0.8,
                        "excerpt": clause[:120],
                    })
                    seen.add(label)
                    break

        # Order by canonical reasoning sequence
        step_order = [
            "identify_claim", "identify_assumptions", "analyze_evidence",
            "compare_alternatives", "counter_argument", "assess_risks",
            "synthesize", "produce_conclusion",
        ]
        steps.sort(key=lambda s: step_order.index(s["step"]) if s["step"] in step_order else 99)

        return steps

    @staticmethod
    def format_for_display(steps: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format extracted steps for the transparency panel.

        Returns human-readable step descriptions.
        """
        labels = {
            "identify_claim": "Identify claim or question",
            "identify_assumptions": "Surface assumptions",
            "analyze_evidence": "Analyze evidence",
            "compare_alternatives": "Compare alternatives",
            "counter_argument": "Engage counterarguments",
            "assess_risks": "Assess risks and limitations",
            "synthesize": "Synthesize perspectives",
            "produce_conclusion": "Produce conclusion",
        }

        return [
            {
                "number": str(i + 1),
                "label": labels.get(s["step"], s["step"]),
                "step_id": s["step"],
                "excerpt": s.get("excerpt", ""),
            }
            for i, s in enumerate(steps)
        ]


class ReasoningAnalyticsEngine:
    """
    Cross-model reasoning analytics.

    Produces:
      - Per-model reasoning step profiles
      - Reasoning heatmap data (model × metric)
      - Disagreement analysis (where models diverge)
      - Composite performance chart data
    """

    def __init__(self):
        self._step_extractor = ReasoningStepExtractor()

    def analyze_debate(
        self,
        model_outputs: List[Dict[str, Any]],
        metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Full reasoning analytics for a completed debate.

        Args:
            model_outputs: List of {"model_id", "model_name", "position", "reasoning"}
            metrics: List of MetricsEngine output dicts

        Returns:
            Complete analytics payload for frontend rendering.
        """
        # Per-model reasoning steps
        model_steps = {}
        for output in model_outputs:
            model_id = output.get("model_id", "unknown")
            full_text = f"{output.get('position', '')} {output.get('reasoning', '')}"
            steps = self._step_extractor.extract_steps(full_text)
            model_steps[model_id] = ReasoningStepExtractor.format_for_display(steps)

        # Build heatmap matrix (model × metric)
        heatmap = self._build_reasoning_heatmap(metrics)

        # Build disagreement analysis
        disagreements = self._analyze_disagreements(model_outputs)

        # Performance chart data
        performance = self._build_performance_chart(metrics)

        return {
            "reasoning_steps": model_steps,
            "reasoning_heatmap": heatmap,
            "disagreement_analysis": disagreements,
            "performance_chart": performance,
            "model_count": len(model_outputs),
        }

    def _build_reasoning_heatmap(
        self, metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build heatmap data: rows = models, columns = metric dimensions.

        Output format:
            {
                "models": ["llama31-8b", "gemma2-9b", ...],
                "metrics": ["reasoning_score", "evidence_density", ...],
                "matrix": [[0.84, 0.62, ...], ...]
            }
        """
        metric_keys = [
            "reasoning_score", "evidence_density", "argument_depth",
            "logical_consistency", "contradiction_rate",
            "confidence_alignment", "token_efficiency",
        ]

        models = []
        matrix = []
        for m in metrics:
            models.append(m.get("model_name", m.get("model", "unknown")))
            row = [float(m.get(k, 0.0)) for k in metric_keys]
            matrix.append(row)

        return {
            "models": models,
            "metrics": metric_keys,
            "matrix": matrix,
        }

    def _analyze_disagreements(
        self, model_outputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify key areas where models disagree.

        Uses simple keyword-based stance detection to find
        contradictory positions between model pairs.
        """
        disagreements = []

        for i, a in enumerate(model_outputs):
            for j, b in enumerate(model_outputs):
                if j <= i:
                    continue

                pos_a = a.get("position", "").lower()
                pos_b = b.get("position", "").lower()

                # Detect opposing stances
                opposition_score = self._compute_opposition(pos_a, pos_b)
                if opposition_score > 0.3:
                    disagreements.append({
                        "model_a": a.get("model_id", "unknown"),
                        "model_b": b.get("model_id", "unknown"),
                        "opposition_score": round(opposition_score, 3),
                        "summary": f"{a.get('model_name', '')} vs {b.get('model_name', '')}",
                    })

        disagreements.sort(key=lambda d: d["opposition_score"], reverse=True)
        return disagreements[:10]

    def _compute_opposition(self, text_a: str, text_b: str) -> float:
        """
        Compute a simple opposition score between two model positions.

        Higher = more disagreement.
        """
        negation_words = {"not", "no", "never", "false", "incorrect", "wrong",
                          "disagree", "contrary", "however", "but", "although"}

        words_a = set(text_a.split())
        words_b = set(text_b.split())

        # Common negation + uncommon content = likely opposition
        neg_a = words_a & negation_words
        neg_b = words_b & negation_words
        negation_diff = abs(len(neg_a) - len(neg_b)) / max(len(neg_a | neg_b), 1)

        # Content overlap (low overlap + negation = opposition)
        content_a = words_a - negation_words
        content_b = words_b - negation_words
        if not content_a or not content_b:
            return 0.0
        overlap = len(content_a & content_b) / max(len(content_a | content_b), 1)
        divergence = 1.0 - overlap

        return min(1.0, 0.5 * negation_diff + 0.5 * divergence)

    def _build_performance_chart(
        self, metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build model performance chart data.

        Returns sorted list of {model, model_name, score, rank}.
        """
        chart = []
        for m in metrics:
            chart.append({
                "model": m.get("model", "unknown"),
                "model_name": m.get("model_name", m.get("model", "unknown")),
                "score": float(m.get("reasoning_score", 0.0)),
            })
        chart.sort(key=lambda c: c["score"], reverse=True)
        for i, c in enumerate(chart):
            c["rank"] = i + 1
        return chart
