"""
viz/conflict_graph.py — Cognitive Conflict Network

Produces a NetworkX conflict graph where edges connect models
that have low semantic similarity (disagreement > threshold).

Supports two output modes:
  - display=True : renders with matplotlib (interactive / notebook)
  - display=False: returns base64-encoded PNG string (API mode)

Edge weight = 1 - similarity (thicker edge = stronger conflict).
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

try:
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False


def plot_conflict_graph(
    models: List[str],
    similarities: Dict[Tuple[str, str], float],
    threshold: float = 0.6,
    display: bool = False,
) -> Optional[str]:
    """
    Build and optionally render the cognitive conflict network.

    Args:
        models:       List of model IDs / names.
        similarities: Dict mapping (model_a, model_b) → similarity score [0, 1].
        threshold:    Pairs with similarity < threshold become conflict edges.
        display:      If True, call plt.show(). If False, return base64 PNG.

    Returns:
        base64-encoded PNG string (display=False), or None (display=True).
    """
    if not _VIZ_AVAILABLE:
        return None

    G = nx.Graph()
    for m in models:
        G.add_node(m)

    for (a, b), sim in similarities.items():
        if sim < threshold:
            G.add_edge(a, b, weight=round(1.0 - sim, 3), similarity=round(sim, 3))

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Node colouring by degree (more connections = darker)
    degrees = dict(G.degree())
    max_deg = max(degrees.values(), default=1)
    node_colors = [
        cm.YlOrRd(0.2 + 0.6 * degrees.get(m, 0) / max_deg)
        for m in G.nodes()
    ]

    # Edge width proportional to conflict intensity
    edge_weights = [G[u][v].get("weight", 0.5) * 4 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos, width=edge_weights, edge_color="#d62728",
        alpha=0.7, ax=ax,
    )

    # Edge labels: similarity score
    edge_labels = {(u, v): f"{G[u][v].get('similarity', 0):.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.set_title("Cognitive Conflict Network", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()

    if display:
        plt.show()
        return None

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_conflict_edges(
    models: List[str],
    similarities: Dict[Tuple[str, str], float],
    threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Return structured conflict edge data (no rendering).
    Used by the battle platform API to build frontend graph data.

    Returns:
        List of {source, target, weight, similarity, type}
    """
    edges = []
    for (a, b), sim in similarities.items():
        if sim < threshold:
            edges.append({
                "source": a,
                "target": b,
                "weight": round(1.0 - sim, 3),
                "similarity": round(sim, 3),
                "type": "conflict" if sim < 0.30 else "divergence",
            })
    return edges
