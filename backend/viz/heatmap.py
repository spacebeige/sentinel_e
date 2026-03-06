"""
viz/heatmap.py — Inter-Agent Semantic Agreement Heatmap

Renders an annotated heatmap of model-to-model similarity scores.
Supports both interactive display and base64 PNG export for API use.
"""
from __future__ import annotations

import base64
import io
from typing import List, Optional

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False


def plot_agreement_heatmap(
    similarity_matrix: List[List[float]],
    labels: List[str],
    display: bool = False,
    title: str = "Inter-Agent Semantic Agreement",
) -> Optional[str]:
    """
    Render an agreement heatmap.

    Args:
        similarity_matrix: 2-D list of float [0, 1], shape (n, n).
        labels:            Model ID / name labels, length n.
        display:           If True, call plt.show(). If False, return base64 PNG.
        title:             Chart title.

    Returns:
        base64-encoded PNG string (display=False), or None (display=True).
    """
    if not _VIZ_AVAILABLE:
        return None

    mat = np.array(similarity_matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 1.0)))

    # Truncate long labels
    short_labels = [_truncate(l, 16) for l in labels]

    sns.heatmap(
        mat,
        xticklabels=short_labels,
        yticklabels=short_labels,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#eeeeee",
        ax=ax,
        cbar_kws={"label": "Similarity"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()

    if display:
        plt.show()
        return None

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len - 1] + "…"
