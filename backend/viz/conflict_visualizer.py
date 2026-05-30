"""
viz/conflict_visualizer.py — Legacy ConflictVisualizer

Server-safe wrapper (non-interactive).
Uses Agg backend — no display environment required.
Returns base64-encoded PNG strings instead of calling plt.show().
"""
from __future__ import annotations

import base64
import io
from typing import Any, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False


class ConflictVisualizer:
    """
    Server-safe conflict visualizer.
    All methods return a base64-encoded PNG string (or None if viz unavailable).
    Never calls plt.show() — not safe in headless server environments.
    """

    def plot_similarity_heatmap(
        self,
        model_names: List[str],
        similarity_matrix: Any,
    ) -> Optional[str]:
        """Render a similarity heatmap. Returns base64 PNG or None."""
        if not _VIZ_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            similarity_matrix,
            xticklabels=model_names,
            yticklabels=model_names,
            annot=True,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title("Inter-Agent Semantic Agreement")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def plot_conflict_graph(self, disagreements: List[Any]) -> Optional[str]:
        """Render a conflict bar chart. Returns base64 PNG or None."""
        if not _VIZ_AVAILABLE:
            return None

        labels = [f"{a}-{b}" for a, b in [d.pair for d in disagreements]]
        values = [d.similarity for d in disagreements]

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=labels, y=values, ax=ax)
        ax.axhline(0.5, linestyle="--", color="red", label="Uncertainty Threshold")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Pairwise Agent Disagreement")
        ax.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

