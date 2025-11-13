import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
def analyze_query_size_impact(engine, max_query_size=1000):
    """
    Analyze how recommendation quality changes with query size.
    Query articles are selected *randomly*.
    """

    if engine.tfidf_matrix is None:
        raise ValueError("TF-IDF model not built.")

    print(f"\n{'=' * 80}")
    print(f"QUERY SIZE IMPACT ANALYSIS (Random Selection Only)")
    print(f"{'=' * 80}")

    # --- Create a random ordering of all article indices
    candidate_indices = np.arange(len(engine.df))
    np.random.shuffle(candidate_indices)

    # --- Metrics storage
    results = {
        'query_size': [],
        'avg_similarity': [],
        'max_similarity': [],
        'min_similarity': [],
        'std_similarity': [],
        'internal_coherence': [],
        'recommendation_diversity': [],
    }

    # --- Range of query sizes
    query_sizes = [1, 3, 5, 10, 25, 50, 100, 250, 500]

    for size in query_sizes:
        actual_size = min(size, len(candidate_indices))
        if actual_size == 0:
            continue

        # Choose the first N shuffled indices → purely random
        query_indices = candidate_indices[:actual_size]
        query_titles = engine.df.iloc[query_indices]['title'].tolist()

        # Get recommendations
        recs = engine.find_similar_articles(query_titles, top_k=10, exclude_query=True)
        if recs.empty:
            continue

        # --- Internal coherence
        if actual_size > 1:
            query_matrix = engine.tfidf_matrix[query_indices]
            query_sim = cosine_similarity(query_matrix)
            upper = np.triu_indices_from(query_sim, k=1)
            internal_coherence = query_sim[upper].mean()
        else:
            internal_coherence = 1.0

        # --- Diversity of top recommendations
        top_rec_indices = [engine._find_article_index(t) for t in recs['title'].head(5)]
        top_rec_indices = [i for i in top_rec_indices if i is not None]
        if len(top_rec_indices) > 1:
            rec_matrix = engine.tfidf_matrix[top_rec_indices]
            rec_sim = cosine_similarity(rec_matrix)
            upper = np.triu_indices_from(rec_sim, k=1)
            diversity = 1 - rec_sim[upper].mean()
        else:
            diversity = 0

        # --- Store metrics
        results['query_size'].append(actual_size)
        results['avg_similarity'].append(recs['similarity_score'].mean())
        results['max_similarity'].append(recs['similarity_score'].max())
        results['min_similarity'].append(recs['similarity_score'].min())
        results['std_similarity'].append(recs['similarity_score'].std())
        results['internal_coherence'].append(internal_coherence)
        results['recommendation_diversity'].append(diversity)
    df = pd.DataFrame(results)
    df.to_csv('data/query_size_analysis.csv', index=False)
    print(f"Results saved to 'data/query_size_analysis.csv'")
    return df
