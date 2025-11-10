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



def create_top_similarities_table(df, tfidf, n_articles):
    """Create visualizations for top similar article pairs"""

    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf[:n_articles])

    # Extract all pairs and their similarities
    pairs = []
    for i in range(n_articles):
        for j in range(i + 1, n_articles):
            pairs.append({
                'idx1': i,
                'idx2': j,
                'similarity': similarity_matrix[i, j],
                'title1': df.iloc[i].get('title', f'Article {i + 1}'),
                'title2': df.iloc[j].get('title', f'Article {j + 1}')
            })

    # Sort by similarity
    pairs_sorted = sorted(pairs, key=lambda x: x['similarity'], reverse=True)
    top_pairs = pairs_sorted[:15]  # Top 15 pairs

    # Create table visualization
    _create_pairs_table(top_pairs)

    # Create focused heatmap
    _create_top_articles_heatmap(df, similarity_matrix, pairs_sorted)


def _create_pairs_table(top_pairs):
    """Create a visual table showing top similar article pairs"""

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    for rank, pair in enumerate(top_pairs, 1):
        title1 = pair['title1'][:40] + '...' if len(str(pair['title1'])) > 40 else pair['title1']
        title2 = pair['title2'][:40] + '...' if len(str(pair['title2'])) > 40 else pair['title2']

        table_data.append([
            f"#{rank}",
            f"[{pair['idx1'] + 1}]",
            title1,
            "↔",
            f"[{pair['idx2'] + 1}]",
            title2,
            f"{pair['similarity']:.3f}"
        ])

    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Rank', 'ID', 'Article 1', '', 'ID', 'Article 2', 'Similarity'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.06, 0.05, 0.35, 0.04, 0.05, 0.35, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(7):
        cell = table[(0, i)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white', fontsize=10)

    # Style rows with color gradient
    for i, row in enumerate(table_data, 1):
        color_intensity = 1 - (i - 1) / len(table_data)
        row_color = plt.cm.RdYlGn(0.3 + color_intensity * 0.6)

        for j in range(7):
            cell = table[(i, j)]
            cell.set_facecolor(row_color)
            cell.set_alpha(0.7)

            # Bold similarity scores
            if j == 6:
                cell.set_text_props(weight='bold', fontsize=10)

    plt.title('🏆 Top 15 Most Similar Article Pairs\n(Sorted by cosine similarity)',
              fontsize=16, fontweight='bold', pad=20)

    plt.savefig('../plots/top_similar_pairs.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\n✓ Top similar pairs table saved as 'top_similar_pairs.png'")
    plt.close()


def _create_top_articles_heatmap(df, similarity_matrix, pairs_sorted):
    """Create a focused heatmap of the 15 most similar articles"""

    # Get unique article indices from top pairs
    unique_indices = set()
    for pair in pairs_sorted[:20]:  # Check top 20 pairs to get ~15 unique articles
        unique_indices.add(pair['idx1'])
        unique_indices.add(pair['idx2'])
        if len(unique_indices) >= 15:
            break

    # Convert to sorted list
    article_indices = sorted(list(unique_indices))[:15]

    # Extract submatrix
    submatrix = similarity_matrix[np.ix_(article_indices, article_indices)]

    # Get article titles
    titles = []
    for idx in article_indices:
        title = df.iloc[idx].get('title', f'Article {idx + 1}')
        if isinstance(title, str):
            title = title[:50] + '...' if len(title) > 50 else title
        else:
            title = f'Article {idx + 1}'
        titles.append(f"[{idx + 1}] {title}")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Mask diagonal
    mask = np.eye(len(article_indices), dtype=bool)
    submatrix_masked = submatrix.copy()
    submatrix_masked[mask] = np.nan

    # Create heatmap
    im = ax.imshow(submatrix_masked, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=1, interpolation='nearest')

    # Add titles
    ax.set_xticks(range(len(article_indices)))
    ax.set_yticks(range(len(article_indices)))
    ax.set_xticklabels(titles, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(titles, fontsize=9)

    # Annotate all cells with similarity values
    for i in range(len(article_indices)):
        for j in range(len(article_indices)):
            if i != j:
                sim_val = submatrix[i, j]
                # Color text based on background
                text_color = 'white' if sim_val > 0.6 else 'black'
                ax.text(j, i, f'{sim_val:.2f}',
                        ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color=text_color)

    ax.set_title(f'🎯 Top 15 Most Similar Articles - Detailed Heatmap\n(Articles appearing in highest similarity pairs)',
                 fontsize=15, fontweight='bold', pad=20)

    # Grid
    ax.set_xticks(np.arange(len(article_indices)) - .5, minor=True)
    ax.set_yticks(np.arange(len(article_indices)) - .5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=11, fontweight='bold', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig('../plots/top_15_articles_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Top 15 articles heatmap saved as 'top_15_articles_heatmap.png'")
    plt.close()