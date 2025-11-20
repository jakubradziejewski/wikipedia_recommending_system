import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.visualization import plot_contribution_analysis, plot_distinctive_term_frequency



def analyze_term_distinctiveness(engine, query_indices, distinctive_term_indices):
    """
    OPTIMIZED: Analyze distinctiveness for multiple terms at once using vectorized operations.

    Args:
        engine: ArticleSimilarityEngine instance
        query_indices: List of query article indices
        distinctive_term_indices: Array of term indices to analyze

    Returns:
        list of dicts with term statistics
    """
    # Get all term vectors at once (vectorized)
    term_vectors = engine.tfidf_matrix[:, distinctive_term_indices].toarray()

    # Query statistics (vectorized)
    query_vectors = term_vectors[query_indices, :]
    query_freqs = query_vectors.mean(axis=0)
    query_presence = (query_vectors > 0).sum(axis=0)

    # Corpus statistics (vectorized)
    corpus_freqs = term_vectors.mean(axis=0)
    doc_freqs = (term_vectors > 0).sum(axis=0)

    # Enrichment ratios (vectorized, avoid division by zero)
    enrichments = np.divide(
        query_freqs,
        corpus_freqs,
        out=np.zeros_like(query_freqs),
        where=corpus_freqs > 0
    )

    total_docs = term_vectors.shape[0]

    # Build results
    results = []
    for i, term_idx in enumerate(distinctive_term_indices):
        results.append({
            'term': engine.feature_names[term_idx],
            'term_idx': term_idx,
            'query_frequency': query_freqs[i],
            'corpus_frequency': corpus_freqs[i],
            'enrichment_ratio': enrichments[i],
            'query_presence': query_presence[i],
            'document_frequency': doc_freqs[i],
            'total_docs': total_docs
        })

    return results


def explain_prediction_contributions(query_vec, rec_vec, similarity, feature_names, top_n=5):
    """
    UNIFIED: Now calculates percentages using similarity score (matching visualization).
    OPTIMIZED: Removed engine parameter, uses only what's needed.

    Explain which terms contributed most to the similarity between query and recommendation.

    Args:
        query_vec: Query vector (averaged TF-IDF from query articles)
        rec_vec: Recommendation article TF-IDF vector
        similarity: Pre-calculated cosine similarity score
        feature_names: Array of feature names
        top_n: Number of top contributors to return

    Returns:
        list: Top N terms with their contribution to the match
    """
    # Element-wise product gives the contribution
    contributions = query_vec * rec_vec

    # Get top contributing terms (vectorized)
    top_indices = np.argpartition(contributions, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(contributions[top_indices])[::-1]]

    # Filter only positive contributions
    top_indices = top_indices[contributions[top_indices] > 0]

    # Build contributors list
    contributors = []
    for idx in top_indices:
        contrib_value = contributions[idx]
        # ⭐ UNIFIED: Now uses similarity as denominator (same as visualization)
        contrib_pct = (contrib_value / similarity * 100) if similarity > 0 else 0

        contributors.append({
            'term': feature_names[idx],
            'contribution': contrib_value,
            'contribution_pct': contrib_pct,  # Now matches visualization!
            'query_tfidf': query_vec[idx],
            'rec_tfidf': rec_vec[idx]
        })

    return contributors


def deep_explainability_analysis(
        engine,
        query_identifiers,
        top_recommendations,
        strategy_name="Strategy",
        min_enrichment=2.0,
        show_contribution_plot=True,
        top_n_terms=5,
        top_distinctive_terms=20
):
    """
    OPTIMIZED: Vectorized operations, reduced redundant calculations, clearer logic.
    UNIFIED: Uses same percentage calculation as visualization.

    Deep analysis of why specific articles are recommended, focusing on rare distinctive terms.

    Args:
        engine: ArticleSimilarityEngine instance
        query_identifiers: List of query article titles/URLs
        top_recommendations: DataFrame with recommended articles
        strategy_name: Name of the strategy for labeling
        min_enrichment: Minimum enrichment ratio to consider a term distinctive
        show_contribution_plot: Whether to show visualization of term contributions
        top_n_terms: Number of top contributing terms to analyze per article
        top_distinctive_terms: Number of distinctive terms to track

    Returns:
        dict: Detailed analysis results
    """
    if engine.tfidf_matrix is None:
        raise ValueError("TF-IDF model not built. Call build_tfidf_model() first.")

    # Resolve query indices
    query_indices = [engine._find_article_index(q) for q in query_identifiers]
    query_indices = [i for i in query_indices if i is not None]

    if not query_indices or top_recommendations.empty:
        print("⚠ Could not perform deep analysis - missing data.")
        return {}

    print(f"\n{'=' * 80}")
    print(f"EXPLAINABILITY ANALYSIS: {strategy_name}")
    print(f"{'=' * 80}")
    print(f"Query articles: {len(query_indices)}")
    print(f"Recommendations to analyze: {len(top_recommendations)}")

    # Compute query vector once
    query_vec = np.asarray(engine.tfidf_matrix[query_indices].mean(axis=0)).flatten()

    # Find distinctive terms (OPTIMIZED: vectorized)
    print("\n[1/3] Identifying distinctive terms in query set...")
    nonzero_indices = np.where(query_vec > 0)[0]

    if len(nonzero_indices) == 0:
        print("⚠ No terms found in query vector!")
        return {}

    # Analyze all nonzero terms at once (vectorized)
    distinctive_terms = analyze_term_distinctiveness(engine, query_indices, nonzero_indices)

    # Filter and sort by enrichment
    distinctive_terms = [t for t in distinctive_terms if t['enrichment_ratio'] >= min_enrichment]
    distinctive_terms.sort(key=lambda x: x['enrichment_ratio'], reverse=True)

    print(f"✓ Found {len(distinctive_terms)} distinctive terms (enrichment ≥ {min_enrichment}x)")

    if distinctive_terms:
        print(f"\nTop 10 most distinctive terms:")
        for i, term_data in enumerate(distinctive_terms[:10], 1):
            print(f"  {i:2d}. '{term_data['term']:20s}' → "
                  f"{term_data['enrichment_ratio']:.1f}x enriched, "
                  f"present in {term_data['query_presence']}/{len(query_indices)} query articles, "
                  f"rare in corpus ({term_data['document_frequency']}/{term_data['total_docs']} docs)")

    # Analyze recommendations (OPTIMIZED: batch operations where possible)
    print(f"\n[2/3] Analyzing top {len(top_recommendations)} recommendations...")

    analyses = []
    # Pre-extract distinctive term indices for faster lookup
    distinctive_term_indices = {t['term_idx']: t for t in distinctive_terms[:top_distinctive_terms]}

    for rank, (_, rec_row) in enumerate(top_recommendations.iterrows(), 1):
        rec_idx = engine._find_article_index(rec_row['title'])
        if rec_idx is None:
            continue

        rec_vec = np.asarray(engine.tfidf_matrix[rec_idx].toarray()).flatten()

        # Compute similarity once
        similarity = cosine_similarity(
            query_vec.reshape(1, -1),
            rec_vec.reshape(1, -1)
        )[0][0]

        # Get top contributing terms (UNIFIED: now uses similarity in percentage calculation)
        term_contributions = explain_prediction_contributions(
            query_vec,
            rec_vec,
            similarity,
            engine.feature_names,
            top_n=top_n_terms
        )

        # Analyze distinctive term matches (OPTIMIZED: vectorized)
        term_matches = []
        element_wise_contributions = query_vec * rec_vec

        for term_idx, term_data in distinctive_term_indices.items():
            rec_tfidf = rec_vec[term_idx]

            if rec_tfidf > 0:
                contribution = element_wise_contributions[term_idx]
                term_matches.append({
                    'term': term_data['term'],
                    'enrichment': term_data['enrichment_ratio'],
                    'rec_tfidf': rec_tfidf,
                    'contribution': contribution
                })

        # Sort by contribution
        term_matches.sort(key=lambda x: x['contribution'], reverse=True)

        coverage_ratio = (
            len(term_matches) / min(top_distinctive_terms, len(distinctive_terms))
            if distinctive_terms else 0
        )

        analyses.append({
            'rank': rank,
            'title': rec_row['title'],
            'similarity': similarity,
            'distinctive_matches': term_matches,
            'term_contributions': term_contributions,
            'coverage_ratio': coverage_ratio,
            'num_matches': len(term_matches)
        })

    # Print detailed results
    print(f"\n[3/3] Results Summary")
    print("=" * 80)
    print("WHY ARTICLES RANK IN THIS ORDER:")
    print("=" * 80)

    for analysis in analyses:
        print(f"\n#{analysis['rank']} - {analysis['title']}")
        print(f"  Similarity Score: {analysis['similarity']:.4f}")
        print(f"  Distinctive Term Coverage: {analysis['coverage_ratio'] * 100:.1f}% "
              f"({analysis['num_matches']}/{min(top_distinctive_terms, len(distinctive_terms))} "
              f"distinctive terms matched)")

        # Top contributing terms
        print(f"\n  Top {len(analysis['term_contributions'])} Contributing Terms:")
        if analysis['term_contributions']:
            for j, contrib in enumerate(analysis['term_contributions'], 1):
                # ⭐ UNIFIED: Now prints the same percentage as shown in plot!
                print(f"    {j}. '{contrib['term']:20s}' → {contrib['contribution']:.5f} "
                      f"({contrib['contribution_pct']:.1f}% of similarity score)")
                print(f"       Query TF-IDF: {contrib['query_tfidf']:.4f} | "
                      f"Article TF-IDF: {contrib['rec_tfidf']:.4f}")
        else:
            print(f"    ⚠ No significant contributors found")

        # Distinctive terms present
        if analysis['distinctive_matches']:
            print(f"\n  Key Query-Distinctive Terms Present:")
            for j, match in enumerate(analysis['distinctive_matches'][:5], 1):
                print(f"    {j}. '{match['term']}' "
                      f"(enriched {match['enrichment']:.1f}x in query) "
                      f"→ contribution: {match['contribution']:.5f}")
        else:
            print(f"  ⚠ No distinctive query terms found - "
                  f"recommendation based on generic overlap")

    # Generate visualizations
    print(f"\n{'=' * 80}")
    print("GENERATING VISUALIZATIONS...")
    print(f"{'=' * 80}")

    plot_contribution_analysis(
        analyses,
        strategy_name=strategy_name,
        save_path=f"../plots/contribution_plot_{strategy_name.replace(' ', '_')}.png"
    )


    plot_distinctive_term_frequency(
            analyses,
            distinctive_terms,
            strategy_name=strategy_name,
            save_path=f"../plots/rare_terms_contribution_{strategy_name.replace(' ', '_')}.png"
        )


    return {
            'query_identifiers': query_identifiers,
            'distinctive_terms': distinctive_terms,
            'analyses': analyses,
            'strategy_name': strategy_name
        }

