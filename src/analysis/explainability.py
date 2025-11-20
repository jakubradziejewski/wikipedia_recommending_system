import numpy as np
from src.utils.visualization import plot_contribution_analysis, plot_distinctive_term_frequency


def analyze_term_distinctiveness(engine, query_indices, distinctive_term_indices):
    """Analyze distinctiveness for multiple terms at once using vectorized operations."""

    # Get subset of tfidf matrix matching columns with distinctive terms
    term_vectors = engine.tfidf_matrix[:, distinctive_term_indices].toarray()

    # Statistics for query set (vectorized)
    query_vectors = term_vectors[query_indices, :]
    query_freqs = query_vectors.mean(axis=0)

    # Statistics for entire corpus (vectorized)
    corpus_freqs = term_vectors.mean(axis=0)
    doc_freqs = (term_vectors > 0).sum(axis=0)

    # Enrichment ratios, safe division
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
            'enrichment_ratio': enrichments[i],
            'document_frequency': doc_freqs[i],
            'total_docs': total_docs
        })

    return results


def explain_prediction_contributions(query_vec, rec_vec, feature_names, top_n=5):
    """Explain which terms contributed most to the similarity between query and recommendation."""

    # Combination of the impact of term importance in both vectors - query and recommendation
    contributions = query_vec * rec_vec

    # Get top contributing terms
    top_indices = np.argpartition(contributions, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(contributions[top_indices])[::-1]]

    # Filter only positive contributions
    top_indices = top_indices[contributions[top_indices] > 0]

    contributors = []
    for idx in top_indices:
        contrib_value = contributions[idx]

        contributors.append({
            'term': feature_names[idx],
            'contribution': contrib_value
        })

    return contributors


def explainability_analysis(
        engine,
        query_identifiers,
        top_recommendations,
        strategy_name="Strategy",
        min_enrichment=2.0,
        top_n_terms=5,
        top_distinctive_terms=20
):
    """Performs explainability analysis on recommendations."""
    if engine.tfidf_matrix is None:
        raise ValueError("TF-IDF model not built. Call build_tfidf_model() first.")

    # Resolve query indices
    query_indices = [engine._find_article_index(q) for q in query_identifiers]
    query_indices = [i for i in query_indices if i is not None]

    if not query_indices or top_recommendations.empty:
        print("⚠ Could not perform explainability analysis - missing data.")
        return {}

    print(f"Explainability analysis using {strategy_name}")
    print(f"Query articles: {len(query_indices)}")
    print(f"Recommendations to analyze: {len(top_recommendations)}")

    # Compute vector for all articles used in query
    query_vec = np.asarray(engine.tfidf_matrix[query_indices].mean(axis=0)).flatten()

    # Find distinctive terms
    print("\nIdentifying distinctive terms in query set...")
    nonzero_indices = np.where(query_vec > 0)[0]

    if len(nonzero_indices) == 0:
        print("⚠ No terms found in query vector!")
        return {}

    # Find distinctive terms, specific to this query set
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
                  f"rare in corpus ({term_data['document_frequency']}/{term_data['total_docs']} docs)")

    # Analyze recommendations
    print(f"\nAnalyzing top {len(top_recommendations)} recommendations...")

    analyses = []
    # Pre-extract distinctive term indices for faster lookup
    distinctive_term_indices = {t['term_idx']: t for t in distinctive_terms[:top_distinctive_terms]}

    for rank, (_, rec_row) in enumerate(top_recommendations.iterrows(), 1):
        rec_idx = engine._find_article_index(rec_row['title'])
        if rec_idx is None:
            continue

        rec_vec = np.asarray(engine.tfidf_matrix[rec_idx].toarray()).flatten()

        # Compute similarity once
        similarity = rec_row['similarity_score']

        # Get top contributing terms
        term_contributions = explain_prediction_contributions(
            query_vec,
            rec_vec,
            engine.feature_names,
            top_n=top_n_terms
        )

        # Analyze distinctive term matches
        term_matches = []
        element_wise_contributions = query_vec * rec_vec

        for term_idx, term_data in distinctive_term_indices.items():
            rec_tfidf = rec_vec[term_idx]

            if rec_tfidf > 0:
                contribution = element_wise_contributions[term_idx]
                term_matches.append({
                    'term': term_data['term'],
                    'contribution': contribution
                })

        # Sort by contribution
        term_matches.sort(key=lambda x: x['contribution'], reverse=True)

        analyses.append({
            'rank': rank,
            'title': rec_row['title'],
            'similarity': similarity,
            'distinctive_matches': term_matches,
            'term_contributions': term_contributions
        })

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
