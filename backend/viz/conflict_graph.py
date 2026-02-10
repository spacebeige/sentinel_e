import networkx as nx
import matplotlib.pyplot as plt
def plot_conflict_graph(models, similarities, threshold=0.6):
    G = nx.Graph()

    for m in models:
        G.add_node(m)

    for (a, b), sim in similarities.items():
        if sim < threshold:
            G.add_edge(a, b, weight=1-sim)

    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        edge_color="red"
    )
    plt.title("Cognitive Conflict Network")
    plt.show()
