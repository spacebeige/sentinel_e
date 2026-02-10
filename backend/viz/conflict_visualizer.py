import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ConflictVisualizer:

    def plot_similarity_heatmap(self, model_names, similarity_matrix):
        plt.figure(figsize=(6,5))
        sns.heatmap(
            similarity_matrix,
            xticklabels=model_names,
            yticklabels=model_names,
            annot=True,
            cmap="coolwarm",
            vmin=0,
            vmax=1
        )
        plt.title("Inter-Agent Semantic Agreement")
        plt.show()

    def plot_conflict_graph(self, disagreements):
        labels = [f"{a}-{b}" for a,b in [d.pair for d in disagreements]]
        values = [d.similarity for d in disagreements]

        plt.figure(figsize=(7,4))
        sns.barplot(x=labels, y=values)
        plt.axhline(0.5, linestyle="--", color="red", label="Uncertainty Threshold")
        plt.ylabel("Cosine Similarity")
        plt.title("Pairwise Agent Disagreement")
        plt.legend()
        plt.show()
