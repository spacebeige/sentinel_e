"""
============================================================
Knowledge Graph — Hybrid Design
============================================================
Nodes:
  - Session
  - ModelOutput
  - KnowledgeSource
  - Concept
  - Claim
  - UserGoal

Edges:
  - SESSION_HAS_CONCEPT
  - MODEL_GENERATED_OUTPUT
  - CLAIM_SUPPORTED_BY_SOURCE
  - OUTPUT_MENTIONS_CONCEPT
  - SESSION_HAS_GOAL

Used for:
  - Long-tail retrieval
  - Concept expansion
  - Drift correction
  - Session evolution

In-memory graph with optional persistence to PostgreSQL.
============================================================
"""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

from metacognitive.embedding import embed_text, cosine_similarity

logger = logging.getLogger("MCO-KnowledgeGraph")


# ============================================================
# Node & Edge Types
# ============================================================

class NodeType(str, Enum):
    SESSION = "session"
    MODEL_OUTPUT = "model_output"
    KNOWLEDGE_SOURCE = "knowledge_source"
    CONCEPT = "concept"
    CLAIM = "claim"
    USER_GOAL = "user_goal"


class EdgeType(str, Enum):
    SESSION_HAS_CONCEPT = "session_has_concept"
    MODEL_GENERATED_OUTPUT = "model_generated_output"
    CLAIM_SUPPORTED_BY_SOURCE = "claim_supported_by_source"
    OUTPUT_MENTIONS_CONCEPT = "output_mentions_concept"
    SESSION_HAS_GOAL = "session_has_goal"
    CONCEPT_RELATED_TO = "concept_related_to"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.CONCEPT
    label: str = ""
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class GraphEdge:
    """A directed edge in the knowledge graph."""
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.CONCEPT_RELATED_TO
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Knowledge Graph
# ============================================================

class KnowledgeGraph:
    """
    In-memory hybrid knowledge graph for session evolution.

    Supports:
      ✓ Node CRUD by type
      ✓ Edge CRUD
      ✓ Concept search by embedding similarity
      ✓ Subgraph extraction (per session)
      ✓ Concept expansion traversal
      ✓ Drift correction via centroid anchoring
    """

    def __init__(self):
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        # Adjacency index: node_id → list of (edge, neighbor_id)
        self._adj: Dict[str, List[Tuple[GraphEdge, str]]] = defaultdict(list)
        # Type index: NodeType → set of node_ids
        self._type_index: Dict[NodeType, Set[str]] = defaultdict(set)

    # ── Node Operations ──────────────────────────────────────

    def add_node(
        self,
        node_type: NodeType,
        label: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> GraphNode:
        """Add a node to the graph."""
        node = GraphNode(
            node_type=node_type,
            label=label,
            embedding=embedding or [],
            metadata=metadata or {},
        )
        if node_id:
            node.id = node_id

        self._nodes[node.id] = node
        self._type_index[node_type].add(node.id)
        return node

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        return [
            self._nodes[nid]
            for nid in self._type_index.get(node_type, set())
            if nid in self._nodes
        ]

    def find_concept(self, label: str) -> Optional[GraphNode]:
        """Find a concept node by exact label match."""
        for nid in self._type_index.get(NodeType.CONCEPT, set()):
            node = self._nodes.get(nid)
            if node and node.label.lower() == label.lower():
                return node
        return None

    def find_similar_concepts(
        self,
        embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[GraphNode, float]]:
        """Find concept nodes similar to an embedding."""
        results = []
        for nid in self._type_index.get(NodeType.CONCEPT, set()):
            node = self._nodes.get(nid)
            if node and node.embedding:
                sim = cosine_similarity(embedding, node.embedding)
                if sim >= threshold:
                    results.append((node, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── Edge Operations ──────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[GraphEdge]:
        """Add a directed edge."""
        if source_id not in self._nodes or target_id not in self._nodes:
            logger.warning(
                f"Cannot add edge: source={source_id} or target={target_id} not found"
            )
            return None

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
        self._edges.append(edge)
        self._adj[source_id].append((edge, target_id))
        self._adj[target_id].append((edge, source_id))
        return edge

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
    ) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get neighboring nodes, optionally filtered by edge type."""
        results = []
        for edge, neighbor_id in self._adj.get(node_id, []):
            if edge_type and edge.edge_type != edge_type:
                continue
            neighbor = self._nodes.get(neighbor_id)
            if neighbor:
                results.append((neighbor, edge))
        return results

    # ── Session Subgraph ─────────────────────────────────────

    def get_session_subgraph(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Extract the subgraph for a specific session.
        Returns nodes and edges connected to the session.
        """
        session_node = self.get_node(session_id)
        if not session_node:
            return {"nodes": [], "edges": []}

        visited = set()
        nodes = []
        edges = []

        # BFS from session node, depth 2
        queue = [(session_id, 0)]
        while queue:
            nid, depth = queue.pop(0)
            if nid in visited or depth > 2:
                continue
            visited.add(nid)

            node = self._nodes.get(nid)
            if node:
                nodes.append({
                    "id": node.id,
                    "type": node.node_type.value,
                    "label": node.label,
                })

            for edge, neighbor_id in self._adj.get(nid, []):
                if neighbor_id not in visited:
                    edges.append({
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "type": edge.edge_type.value,
                        "weight": edge.weight,
                    })
                    queue.append((neighbor_id, depth + 1))

        return {"nodes": nodes, "edges": edges}

    # ── Concept Expansion ────────────────────────────────────

    def expand_concepts(
        self,
        seed_concepts: List[str],
        depth: int = 2,
    ) -> List[GraphNode]:
        """
        Traverse graph from seed concepts to find related concepts.
        Used for long-tail retrieval and drift correction.
        """
        visited = set()
        result = []

        # Find seed nodes
        queue = []
        for label in seed_concepts:
            node = self.find_concept(label)
            if node:
                queue.append((node.id, 0))

        while queue:
            nid, d = queue.pop(0)
            if nid in visited or d > depth:
                continue
            visited.add(nid)

            node = self._nodes.get(nid)
            if node and node.node_type == NodeType.CONCEPT:
                result.append(node)

            # Follow CONCEPT_RELATED_TO and OUTPUT_MENTIONS_CONCEPT edges
            for neighbor, edge in self.get_neighbors(nid):
                if edge.edge_type in (
                    EdgeType.CONCEPT_RELATED_TO,
                    EdgeType.OUTPUT_MENTIONS_CONCEPT,
                    EdgeType.SESSION_HAS_CONCEPT,
                ):
                    if neighbor.id not in visited:
                        queue.append((neighbor.id, d + 1))

        return result

    # ── Graph Building Helpers ───────────────────────────────

    def register_session(self, session_id: str, label: str = "") -> GraphNode:
        """Register a session as a graph node."""
        return self.add_node(
            NodeType.SESSION,
            label=label or f"Session {session_id[:8]}",
            node_id=session_id,
        )

    def register_output(
        self,
        session_id: str,
        model_name: str,
        output_text: str,
        output_embedding: Optional[List[float]] = None,
    ) -> GraphNode:
        """Register a model output and link to session."""
        node = self.add_node(
            NodeType.MODEL_OUTPUT,
            label=f"{model_name} output",
            embedding=output_embedding or embed_text(output_text),
            metadata={"model": model_name, "length": len(output_text)},
        )
        self.add_edge(
            session_id, node.id,
            EdgeType.MODEL_GENERATED_OUTPUT,
        )
        return node

    def register_concept(
        self,
        session_id: str,
        concept_label: str,
        embedding: Optional[List[float]] = None,
    ) -> GraphNode:
        """Register a concept and link to session."""
        existing = self.find_concept(concept_label)
        if existing:
            # Just add edge if concept already exists
            self.add_edge(
                session_id, existing.id,
                EdgeType.SESSION_HAS_CONCEPT,
            )
            return existing

        node = self.add_node(
            NodeType.CONCEPT,
            label=concept_label,
            embedding=embedding or embed_text(concept_label),
        )
        self.add_edge(
            session_id, node.id,
            EdgeType.SESSION_HAS_CONCEPT,
        )
        return node

    def register_claim(
        self,
        output_id: str,
        claim_text: str,
        source_id: Optional[str] = None,
    ) -> GraphNode:
        """Register a claim, link to output and optionally to source."""
        node = self.add_node(
            NodeType.CLAIM,
            label=claim_text[:100],
            embedding=embed_text(claim_text),
        )
        # Link claim to output that generated it
        self.add_edge(
            output_id, node.id,
            EdgeType.MODEL_GENERATED_OUTPUT,
        )
        # Link claim to supporting source
        if source_id:
            self.add_edge(
                node.id, source_id,
                EdgeType.CLAIM_SUPPORTED_BY_SOURCE,
            )
        return node

    def register_goal(
        self,
        session_id: str,
        goal_text: str,
    ) -> GraphNode:
        """Register a user goal and link to session."""
        node = self.add_node(
            NodeType.USER_GOAL,
            label=goal_text[:100],
            embedding=embed_text(goal_text),
        )
        self.add_edge(
            session_id, node.id,
            EdgeType.SESSION_HAS_GOAL,
        )
        return node

    # ── Statistics ───────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "nodes_by_type": {
                nt.value: len(ids)
                for nt, ids in self._type_index.items()
            },
        }
