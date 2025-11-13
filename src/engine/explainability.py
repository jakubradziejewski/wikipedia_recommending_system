import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


def explain_similarity(engine, query_identifiers, target_article, top_terms=15, verbose=True):
    """
    Explain why a target article is similar to given query articles.

    Args:
        engine: ArticleSimilarityEngine instance (must have tfidf_matrix, feature_names, df)
        query_identifiers (List[str]): List of article titles or URLs (queries)
        target_article (str): Title or URL of the target article
        top_terms (int): Number of top contributing terms to display
        verbose (bool): Whether to print detailed explanation in console

    Returns:
        dict: Explanation with top terms and similarity details
    """
    if engine.tfidf_matrix is None:
        raise ValueError("TF-IDF model not built. Call build_tfidf_model() first.")

    # Resolve article indices
    query_indices = [engine._find_article_index(q) for q in query_identifiers]
    query_indices = [i for i in query_indices if i is not None]
    target_idx = engine._find_article_index(target_article)

    if not query_indices or target_idx is None:
        print("⚠ Could not find matching articles for explanation.")
        return {}

    # Compute averaged query and target vectors
    query_vec = np.asarray(engine.tfidf_matrix[query_indices].mean(axis=0)).flatten()
    target_vec = np.asarray(engine.tfidf_matrix[target_idx].toarray()).flatten()

    # Overall cosine similarity
    similarity = cosine_similarity(query_vec.reshape(1, -1), target_vec.reshape(1, -1))[0][0]

    # Term contributions (element-wise product)
    contributions = query_vec * target_vec
    top_indices = np.argsort(contributions)[-top_terms:][::-1]

    # Collect top term details
    top_terms_data = []
    for idx in top_indices:
        if contributions[idx] > 0:
            top_terms_data.append({
                "term": engine.feature_names[idx],
                "contribution": contributions[idx],
                "query_tfidf": query_vec[idx],
                "target_tfidf": target_vec[idx],
            })

    explanation = {
        "query_articles": [engine.df.iloc[i]["title"] for i in query_indices],
        "target_article": engine.df.iloc[target_idx]["title"],
        "similarity_score": similarity,
        "top_terms": top_terms_data,
    }

    # Console output (mirroring similarities_mathing.py style)
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"RECOMMENDATION EXPLANATION")
        print(f"{'=' * 80}")
        print(f"\n📄 Target Article: '{explanation['target_article']}'")
        print(f"🎯 Overall Similarity Score: {explanation['similarity_score']:.4f}")
        print(f"\n📚 Based on {len(query_indices)} query articles")
        print(f"\n🔑 Top {len(top_terms_data)} Contributing Terms:")
        for i, term_data in enumerate(explanation["top_terms"][:top_terms], 1):
            print(f"  {i:2d}. '{term_data['term']:20s}' → {term_data['contribution']:.5f}")

    return explanation


def compare_recommendations_with_insights(engine, query_identifiers, candidate_articles, top_k=5):
    """
    Compare multiple candidate articles and explain why they rank differently.

    Args:
        engine: ArticleSimilarityEngine instance
        query_identifiers: List of query article titles/URLs
        candidate_articles: List of candidate article titles/URLs to compare
        top_k: Number of top articles to analyze in detail

    Returns:
        dict: Detailed comparison results
    """
    query_indices = [engine._find_article_index(q) for q in query_identifiers]
    query_indices = [i for i in query_indices if i is not None]

    if not query_indices:
        return {}

    # Get query vector
    query_vec = np.asarray(engine.tfidf_matrix[query_indices].mean(axis=0)).flatten()

    # Analyze each candidate
    results = []
    for candidate in candidate_articles[:top_k]:
        candidate_idx = engine._find_article_index(candidate)
        if candidate_idx is None:
            continue

        target_vec = np.asarray(engine.tfidf_matrix[candidate_idx].toarray()).flatten()
        similarity = cosine_similarity(query_vec.reshape(1, -1), target_vec.reshape(1, -1))[0][0]

        # Get term contributions
        contributions = query_vec * target_vec

        # Find top positive and negative contributors
        top_pos_indices = np.argsort(contributions)[-10:][::-1]
        non_zero_mask = contributions > 0

        top_terms = []
        for idx in top_pos_indices:
            if contributions[idx] > 0:
                top_terms.append({
                    'term': engine.feature_names[idx],
                    'contribution': contributions[idx],
                    'query_tfidf': query_vec[idx],
                    'target_tfidf': target_vec[idx]
                })

        # Calculate feature coverage
        query_nonzero = np.count_nonzero(query_vec)
        target_nonzero = np.count_nonzero(target_vec)
        overlap = np.count_nonzero(non_zero_mask)

        results.append({
            'article': candidate,
            'similarity': similarity,
            'top_terms': top_terms,
            'query_features': query_nonzero,
            'target_features': target_nonzero,
            'overlap_features': overlap,
            'overlap_ratio': overlap / query_nonzero if query_nonzero > 0 else 0
        })

    # Sort by similarity
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    return {
        'query_articles': query_identifiers,
        'comparisons': results
    }


def visualize_recommendation_breakdown(comparison_results, save_path=None, show=True):
    """
    Create a comprehensive visualization breaking down why articles rank differently.
    Similar to prediction breakdown visualizations in ML interpretability tools.

    Args:
        comparison_results: Output from compare_recommendations_with_insights()
        save_path: Optional path to save figure
        show: Whether to display the figure
    """
    if not comparison_results or 'comparisons' not in comparison_results:
        print("⚠ No comparison results to visualize.")
        return

    comparisons = comparison_results['comparisons']
    if not comparisons:
        return

    n_articles = len(comparisons)
    fig = plt.figure(figsize=(16, 4 + 3 * n_articles))

    # Create grid for subplots
    gs = fig.add_gridspec(n_articles + 1, 3, hspace=0.4, wspace=0.3,
                          height_ratios=[1] + [3] * n_articles)

    # Title section
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    query_text = ', '.join(comparison_results['query_articles'][:2])
    if len(comparison_results['query_articles']) > 2:
        query_text += f' + {len(comparison_results["query_articles"]) - 2} more'
    title_ax.text(0.5, 0.5, f'Recommendation Breakdown\nQuery: {query_text}',
                  ha='center', va='center', fontsize=14, fontweight='bold')

    # For each article, create detailed breakdown
    for idx, result in enumerate(comparisons):
        row = idx + 1

        # 1. Similarity score gauge
        ax_gauge = fig.add_subplot(gs[row, 0])
        score = result['similarity']

        # Create semi-circle gauge
        theta = np.linspace(0, np.pi, 100)
        radius = 1

        # Background arc
        ax_gauge.fill_between(theta, 0, radius, alpha=0.2, color='gray')

        # Score arc (colored by performance)
        score_theta = np.linspace(0, score * np.pi, 100)
        color = 'green' if score > 0.3 else 'orange' if score > 0.15 else 'red'
        ax_gauge.fill_between(score_theta, 0, radius, alpha=0.7, color=color)

        # Add needle
        needle_theta = score * np.pi
        ax_gauge.plot([0, radius * np.cos(needle_theta)],
                      [0, radius * np.sin(needle_theta)],
                      'k-', linewidth=3)
        ax_gauge.plot(0, 0, 'ko', markersize=10)

        # Labels
        ax_gauge.text(0, -0.3, f'{score:.4f}', ha='center', fontsize=16, fontweight='bold')
        ax_gauge.text(0, -0.5, f'Rank #{idx + 1}', ha='center', fontsize=10, style='italic')
        ax_gauge.set_xlim(-1.2, 1.2)
        ax_gauge.set_ylim(-0.6, 1.2)
        ax_gauge.axis('off')

        article_title = result['article'][:40] + '...' if len(result['article']) > 40 else result['article']
        ax_gauge.set_title(article_title, fontsize=10, fontweight='bold', pad=10)

        # 2. Feature overlap visualization
        ax_overlap = fig.add_subplot(gs[row, 1])

        overlap_pct = result['overlap_ratio'] * 100
        query_only = result['query_features'] - result['overlap_features']
        target_only = result['target_features'] - result['overlap_features']

        # Venn-like representation using bars
        categories = ['Query\nOnly', 'Shared\nFeatures', 'Target\nOnly']
        values = [query_only, result['overlap_features'], target_only]
        colors_bar = ['#ff9999', '#66b266', '#9999ff']

        bars = ax_overlap.barh(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                ax_overlap.text(val / 2, i, f'{val}', ha='center', va='center',
                                fontweight='bold', fontsize=10)

        ax_overlap.set_xlabel('Number of Features', fontweight='bold')
        ax_overlap.set_title(f'Feature Overlap: {overlap_pct:.1f}%', fontweight='bold')
        ax_overlap.grid(axis='x', alpha=0.3)

        # 3. Top contributing terms waterfall
        ax_waterfall = fig.add_subplot(gs[row, 2])

        if result['top_terms']:
            terms_df = pd.DataFrame(result['top_terms'][:8])  # Top 8 for visibility

            # Create waterfall chart
            cumsum = np.cumsum([0] + terms_df['contribution'].tolist())

            colors_waterfall = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(terms_df)))

            for i, (idx, row_data) in enumerate(terms_df.iterrows()):
                ax_waterfall.barh(i, row_data['contribution'],
                                  left=cumsum[i],
                                  color=colors_waterfall[i],
                                  alpha=0.8,
                                  edgecolor='black',
                                  linewidth=0.5)

                # Add term label
                term_label = row_data['term'][:15]
                ax_waterfall.text(cumsum[i] + row_data['contribution'] / 2, i,
                                  term_label,
                                  ha='center', va='center',
                                  fontsize=8, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3',
                                            facecolor='white',
                                            alpha=0.7,
                                            edgecolor='none'))

            ax_waterfall.set_yticks([])
            ax_waterfall.set_xlabel('Cumulative Contribution', fontweight='bold')
            ax_waterfall.set_title('Top Contributing Terms (Cumulative)', fontweight='bold')
            ax_waterfall.grid(axis='x', alpha=0.3)

            # Add total line
            total = cumsum[-1]
            ax_waterfall.axvline(total, color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax_waterfall.text(total, len(terms_df) - 0.5, f'Total: {total:.4f}',
                              ha='left', va='top', fontsize=9,
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.suptitle('Why These Articles Were Recommended',
                 fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Recommendation breakdown saved to {save_path}")

    if show:
        plt.show()

