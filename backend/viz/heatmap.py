import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_agreement_heatmap(similarity_matrix, labels):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        cmap="coolwarm",
        vmin=0,
        vmax=1
    )
    plt.title("Inter-Agent Semantic Agreement")
    plt.show()
