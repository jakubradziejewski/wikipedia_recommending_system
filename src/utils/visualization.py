import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def plot_histogram(data, title, xlabel, ylabel, bins=30, color="steelblue"):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color=color, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_barh(df: pd.DataFrame, x_col: str, y_col: str, title: str, color="coral"):
    plt.figure(figsize=(8, 5))
    plt.barh(df[y_col], df[x_col], color=color)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.show()




def visualize_strategy_comparison(engine, random_indices, similar_indices,
                                  random_recs, similar_recs,
                                  random_coherence, similar_coherence):
    """Create visualization comparing the two strategies"""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Query collection similarity heatmaps
    ax1 = fig.add_subplot(gs[0, 0])
    random_matrix = engine.tfidf_matrix[random_indices]
    random_sim = cosine_similarity(random_matrix)
    im1 = ax1.imshow(random_sim, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Random Collection\nInternal Similarity', fontweight='bold')
    ax1.set_xlabel('Article Index')
    ax1.set_ylabel('Article Index')
    plt.colorbar(im1, ax=ax1, label='Similarity')
    ax1.text(0.5, -0.15, f'Avg: {random_coherence:.4f}',
             ha='center', transform=ax1.transAxes, fontsize=10, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    similar_matrix = engine.tfidf_matrix[similar_indices]
    similar_sim = cosine_similarity(similar_matrix)
    im2 = ax2.imshow(similar_sim, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Similar Collection\nInternal Similarity', fontweight='bold')
    ax2.set_xlabel('Article Index')
    ax2.set_ylabel('Article Index')
    plt.colorbar(im2, ax=ax2, label='Similarity')
    ax2.text(0.5, -0.15, f'Avg: {similar_coherence:.4f}',
             ha='center', transform=ax2.transAxes, fontsize=10, fontweight='bold')

    # 2. Coherence comparison
    ax3 = fig.add_subplot(gs[0, 2])
    coherence_data = [random_coherence, similar_coherence]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax3.bar(['Random', 'Similar'], coherence_data, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Average Similarity Score', fontweight='bold')
    ax3.set_title('Query Collection Coherence', fontweight='bold')
