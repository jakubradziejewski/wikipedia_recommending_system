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

    # Console output
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


def analyze_term_distinctiveness(engine, query_indices, term_idx):
    """
    Analyze how distinctive a term is for the query set vs. the entire corpus.

    Returns:
        dict with:
        - query_frequency: avg TF-IDF in query articles
        - corpus_frequency: avg TF-IDF across all articles
        - enrichment_ratio: how much more frequent in query vs corpus
        - document_frequency: in how many articles the term appears
    """
    term_vector = engine.tfidf_matrix[:, term_idx].toarray().flatten()

    # Query statistics
    query_values = term_vector[query_indices]
    query_freq = query_values.mean()
    query_presence = (query_values > 0).sum()

    # Corpus statistics
    corpus_freq = term_vector.mean()
    doc_freq = (term_vector > 0).sum()

    # Enrichment ratio (how overrepresented in query vs corpus)
    enrichment = query_freq / corpus_freq if corpus_freq > 0 else 0

    return {
        'query_frequency': query_freq,
        'corpus_frequency': corpus_freq,
        'enrichment_ratio': enrichment,
        'query_presence': query_presence,
        'document_frequency': doc_freq,
        'total_docs': len(term_vector)
    }


def deep_explainability_analysis(engine, query_identifiers, top_recommendations, strategy_name="Strategy",
                                 min_enrichment=2.0):
    """
    Deep analysis of why specific articles are recommended, focusing on rare distinctive terms.

    Args:
        engine: ArticleSimilarityEngine instance
        query_identifiers: List of query article titles/URLs
        top_recommendations: DataFrame with recommended articles (from find_similar_articles)
        strategy_name: Name of the strategy for labeling
        min_enrichment: Minimum enrichment ratio to consider a term distinctive (default: 2.0x)

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

    print("\n" + "=" * 80)
    print(f"DEEP EXPLAINABILITY ANALYSIS: {strategy_name}")
    print("=" * 80)

    # Compute query vector
    query_vec = np.asarray(engine.tfidf_matrix[query_indices].mean(axis=0)).flatten()

    # Identify distinctive terms in query set
    print("\n🔍 Identifying distinctive terms in query set...")

    distinctive_terms = []
    for term_idx in np.where(query_vec > 0)[0]:
        stats = analyze_term_distinctiveness(engine, query_indices, term_idx)

        # Focus on rare but important terms
        if (stats['enrichment_ratio'] >= min_enrichment and
                stats['document_frequency'] < 0.3 * stats['total_docs']):  # Not too common
            distinctive_terms.append({
                'term': engine.feature_names[term_idx],
                'term_idx': term_idx,
                **stats
            })

    # Sort by enrichment ratio
    distinctive_terms = sorted(distinctive_terms, key=lambda x: x['enrichment_ratio'], reverse=True)

    print(f"\n📊 Found {len(distinctive_terms)} distinctive terms (enrichment ≥ {min_enrichment}x)")
    print("\nTop 10 most distinctive terms in query set:")
    for i, term_data in enumerate(distinctive_terms[:10], 1):
        print(f"  {i:2d}. '{term_data['term']:20s}' → "
              f"{term_data['enrichment_ratio']:.1f}x enriched, "
              f"present in {term_data['query_presence']}/{len(query_indices)} query articles, "
              f"rare in corpus ({term_data['document_frequency']}/{term_data['total_docs']} docs)")

    # Analyze each recommendation
    print(f"\n📝 Analyzing top {len(top_recommendations)} recommendations...")

    analyses = []
    for rank, (_, rec_row) in enumerate(top_recommendations.iterrows(), 1):
        rec_idx = engine._find_article_index(rec_row['title'])
        if rec_idx is None:
            continue

        rec_vec = np.asarray(engine.tfidf_matrix[rec_idx].toarray()).flatten()

        # Compute similarity
        similarity = cosine_similarity(query_vec.reshape(1, -1), rec_vec.reshape(1, -1))[0][0]

        # Analyze distinctive term coverage
        term_matches = []
        for term_data in distinctive_terms[:20]:  # Focus on top 20 distinctive terms
            term_idx = term_data['term_idx']
            rec_tfidf = rec_vec[term_idx]

            if rec_tfidf > 0:
                contribution = query_vec[term_idx] * rec_tfidf
                term_matches.append({
                    'term': term_data['term'],
                    'enrichment': term_data['enrichment_ratio'],
                    'rec_tfidf': rec_tfidf,
                    'contribution': contribution
                })

        # Sort by contribution
        term_matches = sorted(term_matches, key=lambda x: x['contribution'], reverse=True)

        coverage_ratio = len(term_matches) / min(20, len(distinctive_terms)) if distinctive_terms else 0

        analyses.append({
            'rank': rank,
            'title': rec_row['title'],
            'similarity': similarity,
            'distinctive_matches': term_matches,
            'coverage_ratio': coverage_ratio,
            'num_matches': len(term_matches)
        })

    # Print detailed analysis
    print("\n" + "=" * 80)
    print("WHY ARTICLES RANK IN THIS ORDER:")
    print("=" * 80)

    for i, analysis in enumerate(analyses):
        print(f"\n#{analysis['rank']} - {analysis['title']}")
        print(f"  Similarity Score: {analysis['similarity']:.4f}")
        print(f"  Distinctive Term Coverage: {analysis['coverage_ratio'] * 100:.1f}% "
              f"({analysis['num_matches']}/{min(20, len(distinctive_terms))} top distinctive terms matched)")

        if analysis['distinctive_matches']:
            print(f"  Key query-distinctive terms present:")
            for j, match in enumerate(analysis['distinctive_matches'][:5], 1):
                print(f"    {j}. '{match['term']}' (enriched {match['enrichment']:.1f}x in query) "
                      f"→ contribution: {match['contribution']:.5f}")
        else:
            print(f"  ⚠ No distinctive query terms found - recommendation based on generic overlap")

        # Comparison with higher ranked article
        if i > 0:
            prev = analyses[i - 1]
            print(f"\n  Why ranked lower than #{prev['rank']}?")

            sim_diff = analysis['similarity'] - prev['similarity']
            coverage_diff = analysis['coverage_ratio'] - prev['coverage_ratio']

            if sim_diff < 0:
                print(f"    • Lower overall similarity: {abs(sim_diff):.4f} lower")

            if coverage_diff < 0:
                print(f"    • Fewer distinctive terms: {abs(coverage_diff) * 100:.1f}% less coverage")

            # Find terms present in higher-ranked but not in current
            prev_terms = {m['term'] for m in prev['distinctive_matches'][:5]}
            curr_terms = {m['term'] for m in analysis['distinctive_matches'][:5]}
            unique_prev = prev_terms - curr_terms

            if unique_prev:
                print(
                    f"    • Higher-ranked article has these query-distinctive terms: {', '.join(list(unique_prev)[:3])}")

        print()


    return {
        'query_identifiers': query_identifiers,
        'distinctive_terms': distinctive_terms,
        'analyses': analyses,
        'strategy_name': strategy_name
    }


def compare_recommendations_with_insights(engine, query_identifiers, candidate_articles, top_k=5):
    """
    Compare multiple candidate articles and explain why they rank differently.
    (Legacy function - consider using deep_explainability_analysis for more insights)

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

        # Find top positive contributors
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