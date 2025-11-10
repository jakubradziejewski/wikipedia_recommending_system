# engine/statistics.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def generate_database_statistics(engine):
    """Generate comprehensive statistics and visualizations for the article corpus"""
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS & ANALYSIS")
    print("=" * 80)

    if engine.tfidf_matrix is None:
        print("⚠ TF-IDF model not built. Building now...")
        engine.build_tfidf_model()

    tfidf = engine.tfidf_matrix
    df = engine.df

    # 1. TF-IDF Statistics
    print("\n" + "-" * 80)
    print("1. TF-IDF MODEL STATISTICS")
    print("-" * 80)
    print(f"Total documents: {tfidf.shape[0]:,}")
    print(f"Vocabulary size: {tfidf.shape[1]:,}")
    density = tfidf.nnz / (tfidf.shape[0] * tfidf.shape[1])
    print(f"Matrix density: {density:.4%}")
    print(f"Non-zero elements: {tfidf.nnz:,}")

    # 2. Top TF-IDF terms across corpus
    print("\n" + "-" * 80)
    print("2. TOP TF-IDF TERMS (Corpus-wide)")
    print("-" * 80)
    tfidf_sums = np.array(tfidf.sum(axis=0)).flatten()
    top_indices = np.argsort(tfidf_sums)[-20:][::-1]
    print("\nTop 20 terms by cumulative TF-IDF score:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i:2d}. {engine.feature_names[idx]:20s} (score: {tfidf_sums[idx]:.2f})")

    # 3. Document length distribution
    print("\n" + "-" * 80)
    print("3. DOCUMENT LENGTH DISTRIBUTION")
    print("-" * 80)
    lengths = df['token_count']
    print(f"Mean: {lengths.mean():.0f} tokens")
    print(f"Median: {lengths.median():.0f} tokens")
    print(f"Std Dev: {lengths.std():.0f} tokens")
    print(f"Min: {lengths.min()} tokens")
    print(f"Max: {lengths.max()} tokens")

    # 4. Sample similarity matrix
    print("\n" + "-" * 80)
    print("4. SIMILARITY MATRIX PREVIEW (First 10 articles)")
    print("-" * 80)
    sample_size = min(10, len(df))
    sample_matrix = tfidf[:sample_size]
    sample_similarity = cosine_similarity(sample_matrix)
    print("\nSimilarity scores (0.0 = different, 1.0 = identical):")
    print("Articles:", ", ".join(df.head(sample_size)['title'].str[:20].tolist()))
    upper_indices = np.triu_indices_from(sample_similarity, k=1)
    print(f"\nAverage similarity: {sample_similarity[upper_indices].mean():.4f}")
    print(f"Max similarity (non-diagonal): {sample_similarity[upper_indices].max():.4f}")

    # 5. Visualizations
    _create_statistics_visualizations(engine)
    print("\n" + "=" * 80)


def _create_statistics_visualizations(engine):
    """Create visualization plots for database statistics"""
    tfidf = engine.tfidf_matrix
    df = engine.df

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Token count distribution
    ax1 = axes[0, 0]
    token_counts = df['token_count']
    ax1.hist(token_counts, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(token_counts.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {token_counts.mean():.0f}')
    ax1.axvline(token_counts.median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {token_counts.median():.0f}')
    ax1.set_xlabel('Token Count', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Articles', fontsize=11, fontweight='bold')
    ax1.set_title('Article Length Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Top TF-IDF terms
    ax2 = axes[0, 1]
    tfidf_sums = np.array(tfidf.sum(axis=0)).flatten()
    top_indices = np.argsort(tfidf_sums)[-15:][::-1]
    top_terms = [engine.feature_names[i] for i in top_indices]
    top_scores = [tfidf_sums[i] for i in top_indices]
    ax2.barh(range(len(top_terms)), top_scores, color='coral', alpha=0.7)
    ax2.set_yticks(range(len(top_terms)))
    ax2.set_yticklabels(top_terms)
    ax2.set_xlabel('Cumulative TF-IDF Score', fontsize=11, fontweight='bold')
    ax2.set_title('Top 15 Terms by TF-IDF', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    # 3. Similarity heatmap (sample)
    ax3 = axes[1, 0]
    sample_size = min(15, len(df))
    sample_matrix = tfidf[:sample_size]
    sample_similarity = cosine_similarity(sample_matrix)
    im = ax3.imshow(sample_similarity, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(sample_size))
    ax3.set_yticks(range(sample_size))
    ax3.set_xticklabels([f"A{i+1}" for i in range(sample_size)], fontsize=8)
    ax3.set_yticklabels([f"A{i+1}" for i in range(sample_size)], fontsize=8)
    ax3.set_title(f'Similarity Heatmap (First {sample_size} Articles)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Cosine Similarity')

    # 4. TF-IDF sparsity visualization
    ax4 = axes[1, 1]
    sample_matrix_sparse = tfidf[:50, :100].toarray()
    im2 = ax4.imshow(sample_matrix_sparse, cmap='Blues', aspect='auto', interpolation='nearest')
    ax4.set_xlabel('Features (first 100)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Documents (first 50)', fontsize=11, fontweight='bold')
    ax4.set_title('TF-IDF Matrix Sparsity Pattern', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax4, label='TF-IDF Score')

    plt.tight_layout()
    plt.savefig('../plots/database_statistics.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'database_statistics.png'")
    plt.show()
