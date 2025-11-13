import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def analyze_query_size_impact(engine, seed_article=None, max_query_size=1000):
    """
    Analyze how recommendation quality changes with query size.

    Args:
        engine: ArticleSimilarityEngine instance
        seed_article: Starting article (random if None)
        max_query_size: Maximum number of articles in query
    """

    if engine.tfidf_matrix is None:
        raise ValueError("TF-IDF model not built.")

    # Select seed article
    if seed_article is None:
        seed_idx = np.random.choice(len(engine.df))
    else:
        seed_idx = engine._find_article_index(seed_article)

    seed_title = engine.df.iloc[seed_idx]['title']
    print(f"\n{'=' * 80}")
    print(f"QUERY SIZE IMPACT ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Seed article: {seed_title}\n")

    # Find similar articles to seed
    seed_vector = engine.tfidf_matrix[seed_idx]
    similarities = cosine_similarity(seed_vector, engine.tfidf_matrix).flatten()
    similar_indices = np.argsort(similarities)[::-1][1:max_query_size + 1]

    # Metrics storage
    results = {
        'query_size': [],
        'avg_similarity': [],
        'max_similarity': [],
        'min_similarity': [],
        'std_similarity': [],
        'internal_coherence': [],
        'top_recommendation': [],
        'recommendation_diversity': []  # How different are top recommendations
    }

    query_sizes = [1, 3, 5, 10, 15, 25, 50, 100]

    for size in query_sizes:
        if size > len(similar_indices):
            break

        # Select query articles
        query_indices = similar_indices[:size]
        query_titles = engine.df.iloc[query_indices]['title'].tolist()

        # Get recommendations
        recs = engine.find_similar_articles(query_titles, top_k=10, exclude_query=True)

        if recs.empty:
            continue

        # Calculate internal coherence of query
        query_matrix = engine.tfidf_matrix[query_indices]
        if size > 1:
            query_sim_matrix = cosine_similarity(query_matrix)
            upper_indices = np.triu_indices_from(query_sim_matrix, k=1)
            internal_coherence = query_sim_matrix[upper_indices].mean()
        else:
            internal_coherence = 1.0

        # Calculate diversity of recommendations
        if len(recs) > 1:
            top_rec_indices = [engine._find_article_index(title) for title in recs['title'].head(5)]
            top_rec_indices = [i for i in top_rec_indices if i is not None]
            if len(top_rec_indices) > 1:
                rec_matrix = engine.tfidf_matrix[top_rec_indices]
                rec_sim = cosine_similarity(rec_matrix)
                rec_upper = np.triu_indices_from(rec_sim, k=1)
                diversity = 1 - rec_sim[rec_upper].mean()  # Lower similarity = higher diversity
            else:
                diversity = 0
        else:
            diversity = 0

        # Store metrics
        results['query_size'].append(size)
        results['avg_similarity'].append(recs['similarity_score'].mean())
        results['max_similarity'].append(recs['similarity_score'].max())
        results['min_similarity'].append(recs['similarity_score'].min())
        results['std_similarity'].append(recs['similarity_score'].std())
        results['internal_coherence'].append(internal_coherence)
        results['top_recommendation'].append(recs.iloc[0]['title'])
        results['recommendation_diversity'].append(diversity)

        # Print summary for this size
        print(f"\n{'─' * 80}")
        print(f"Query Size: {size} article{'s' if size > 1 else ''}")
        print(f"{'─' * 80}")
        print(f"Query articles: {', '.join(query_titles[:3])}" +
              (f" + {size - 3} more" if size > 3 else ""))
        print(f"\nInternal coherence: {internal_coherence:.4f}")
        print(f"Top 3 recommendations:")
        for idx, row in recs.head(3).iterrows():
            print(f"  • {row['title'][:60]:60s} | Score: {row['similarity_score']:.4f}")
        print(f"\nRecommendation metrics:")
        print(f"  Average similarity: {recs['similarity_score'].mean():.4f}")
        print(f"  Max similarity: {recs['similarity_score'].max():.4f}")
        print(f"  Diversity: {diversity:.4f}")

    # Create visualizations
    _visualize_query_size_impact(results, seed_title)

    return pd.DataFrame(results)
def _visualize_query_size_impact(results, seed_title):
    """Create adaptive visualization of query size impact (correct scaling)."""

    import matplotlib.ticker as ticker

    df = pd.DataFrame(results).sort_values('query_size')
    query_sizes = df['query_size'].to_numpy()
    max_q = int(np.max(query_sizes))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Impact of Query Size on Recommendations\nSeed: {seed_title}',
        fontsize=16, fontweight='bold', y=0.995
    )

    # Helper to fix axis scaling properly
    def fix_axis(ax):
        ax.set_xlim(0, max_q * 1.05)
        ax.set_xticks(query_sizes)
        ax.xaxis.set_major_locator(ticker.FixedLocator(query_sizes))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    # Plot 1: Similarity scores vs query size
    ax1 = axes[0, 0]
    ax1.plot(query_sizes, df['avg_similarity'], 'o-', linewidth=2,
             markersize=8, label='Average', color='steelblue')
    ax1.plot(query_sizes, df['max_similarity'], 's--', linewidth=2,
             markersize=6, label='Maximum', color='coral', alpha=0.8)
    ax1.fill_between(query_sizes, df['min_similarity'], df['max_similarity'],
                     alpha=0.2, color='steelblue', label='Range')
    ax1.set_xlabel('Query Size (number of articles)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Similarity Score', fontsize=11, fontweight='bold')
    ax1.set_title('Recommendation Quality vs Query Size', fontsize=12, fontweight='bold')
    fix_axis(ax1)
    ax1.legend(fontsize=10)

    # Plot 2: Internal coherence
    ax2 = axes[0, 1]
    ax2.plot(query_sizes, df['internal_coherence'], 'o-',
             linewidth=2, markersize=8, color='darkgreen')
    ax2.set_xlabel('Query Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Internal Coherence', fontsize=11, fontweight='bold')
    ax2.set_title('Query Collection Coherence', fontsize=12, fontweight='bold')
    fix_axis(ax2)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax2.legend()

    # Plot 3: Stability (standard deviation)
    ax3 = axes[1, 0]
    ax3.plot(query_sizes, df['std_similarity'], 'o-',
             linewidth=2, markersize=8, color='purple')
    ax3.set_xlabel('Query Size', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
    ax3.set_title('Recommendation Stability', fontsize=12, fontweight='bold')
    fix_axis(ax3)

    # Plot 4: Diversity of recommendations
    ax4 = axes[1, 1]
    if 'recommendation_diversity' in df:
        ax4.plot(query_sizes, df['recommendation_diversity'], 'o-',
                 linewidth=2, markersize=8, color='darkorange')
    else:
        ax4.plot(query_sizes, np.zeros_like(query_sizes), 'o-', color='gray', alpha=0.5)
    ax4.set_xlabel('Query Size', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Diversity Score', fontsize=11, fontweight='bold')
    ax4.set_title('Recommendation Diversity', fontsize=12, fontweight='bold')
    fix_axis(ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('../plots/query_size_impact.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Query size impact visualization saved to '../plots/query_size_impact.png'")
    plt.close()

