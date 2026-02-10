import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class HypothesisGraph:
    def __init__(self, embedder=None):
        self.graph = nx.Graph()
        self.embedder = embedder
        self.node_embeddings = {} # {node_id: vector}
        self.similarity_threshold = 0.85 # Threshold for merging hypotheses

    def add_hypotheses(self, source_model: str, hypotheses: List[str]):
        """
        Adds hypotheses as nodes. Connects them to the source model.
        Performs semantic deduplication if embedder is present.
        """
        self.graph.add_node(source_model, type="model")
        
        for h_text in hypotheses:
            target_node = h_text
            
            # --- Semantic Deduplication ---
            if self.embedder:
                # 1. Embed current hypothesis
                # Handle both LangChain embeddings and raw sentence-transformers
                if hasattr(self.embedder, 'embed_query'):
                    h_vec = self.embedder.embed_query(h_text)
                else: 
                     # Fallback or mock
                    h_vec = np.random.rand(384) 

                h_vec = np.array(h_vec).reshape(1, -1)
                
                # 2. Check against existing distinct hypotheses
                best_match = None
                best_score = -1.0
                
                existing_hyps = [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "hypothesis"]
                
                for eh in existing_hyps:
                    if eh in self.node_embeddings:
                        eh_vec = self.node_embeddings[eh]
                        score = cosine_similarity(h_vec, eh_vec)[0][0]
                        if score > best_score:
                            best_score = score
                            best_match = eh
                
                # 3. Merge if similar enough
                if best_match and best_score > self.similarity_threshold:
                    target_node = best_match # Reuse existing node
                else:
                    self.node_embeddings[h_text] = h_vec # Register new
            
            # --- Graph Construction ---
            self.graph.add_node(target_node, type="hypothesis") # Idempotent
            self.graph.add_edge(source_model, target_node, weight=1.0)
            
    def compute_intersection(self) -> Dict[str, Any]:
        """
        Finds hypotheses shared by multiple models.
        """
        # In a bipartite graph (models <-> hypotheses), shared hypotheses have degree > 1 (if we look at hypothesis nodes)
        intersection = []
        hypothesis_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "hypothesis"]
        
        for h in hypothesis_nodes:
            neighbors = list(self.graph.neighbors(h))
            # If connected to more than 1 model
            model_neighbors = [n for n in neighbors if self.graph.nodes[n].get("type") == "model"]
            if len(model_neighbors) > 1:
                intersection.append({
                    "hypothesis": h,
                    "shared_by": model_neighbors
                })
                
        return {"shared_hypotheses": intersection, "count": len(intersection)}

    def get_graph_data(self):
        return nx.node_link_data(self.graph)
