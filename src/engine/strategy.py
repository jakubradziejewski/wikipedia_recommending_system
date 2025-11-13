import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.visualization import visualize_strategy_comparison
from src.engine.explainability import compare_recommendations_with_insights, visualize_recommendation_breakdown

def compare_recommendation_strategies(engine, num_articles=10, deep_explainability=True):
    """
    Compare recommendations from multiple query selection strategies:
      1. Random Article Collection
      2. Similar (connected) Article Collection
      3. Weighted Query Strategy
      4. Recursive Query Expansion

    Args:
        engine: ArticleSimilarityEngine instance
        num_articles: Number of articles to use in query
        deep_explainability: If True, perform detailed explainability analysis
    """
    print("\n" + "-" * 80)
    print("Approaches of recommendation based on different query article selection strategies:")
    print("-" * 80)

    # ========================================================================
    # Strategy 1: Random Article Collection
    # ========================================================================
    print("\n1. Random Article Collection")
    random_indices = np.random.choice(len(engine.df), size=num_articles, replace=False)
    random_titles = engine.df.iloc[random_indices]['title'].tolist()

    print(f"\nQuery articles (randomly selected):")
    for i, title in enumerate(random_titles, 1):
        print(f"  {i:2d}. {title}")

    random_matrix = engine.tfidf_matrix[random_indices]
    random_sim_matrix = cosine_similarity(random_matrix)
    random_avg_sim = random_sim_matrix[np.triu_indices_from(random_sim_matrix, k=1)].mean()

    print(f"\nInternal coherence (avg similarity): {random_avg_sim:.4f}")

    random_recs = engine.find_similar_articles(random_titles, top_k=10)
    print(f"\nTop 10 Recommendations:")
    for idx, row in random_recs.iterrows():
        print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")

    # ========================================================================
    # Strategy 2: Similar Article Collection
    # ========================================================================
    print("\n2. Similar Article Collection based on randomly chosen seed article")
    seed_idx = np.random.choice(len(engine.df))
    seed_title = engine.df.iloc[seed_idx]['title']
    print(f"\nSeed article: {seed_title}")

    seed_vector = engine.tfidf_matrix[seed_idx]
    similarities = cosine_similarity(seed_vector, engine.tfidf_matrix).flatten()
    similar_indices = np.argsort(similarities)[::-1][1:num_articles + 1]
    similar_titles = engine.df.iloc[similar_indices]['title'].tolist()

    print(f"\nQuery articles (similar to seed):")
    for i, (idx, title) in enumerate(zip(similar_indices, similar_titles), 1):
        sim_score = similarities[idx]
        print(f"  {i:2d}. {title} (similarity to seed: {sim_score:.4f})")

    similar_matrix = engine.tfidf_matrix[similar_indices]
    similar_sim_matrix = cosine_similarity(similar_matrix)
    similar_avg_sim = similar_sim_matrix[np.triu_indices_from(similar_sim_matrix, k=1)].mean()

    print(f"\nInternal coherence (avg similarity): {similar_avg_sim:.4f}")

    similar_recs = engine.find_similar_articles(similar_titles, top_k=10)
    print(f"\nTop 10 Recommendations:")
    for idx, row in similar_recs.iterrows():
        print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")

    # ========================================================================
    # Strategy 3: Weighted Query
    # ========================================================================
    print("\n3. Weighted Query Strategy (recent articles have higher importance)")
    last_articles = engine.df.tail(num_articles)
    weighted_titles = last_articles['title'].tolist()
    weights = list(range(1, len(weighted_titles) + 1))

    print("\nWeighted query articles (Weight: Title):")
    for w, title in zip(weights, weighted_titles):
        print(f"  {w:2d}: {title}")

    weighted_matrix = engine.tfidf_matrix[engine.df.tail(num_articles).index]
    weighted_sim_matrix = cosine_similarity(weighted_matrix)
    weighted_avg_sim = weighted_sim_matrix[np.triu_indices_from(weighted_sim_matrix, k=1)].mean()

    print(f"\nInternal coherence (avg similarity): {weighted_avg_sim:.4f}")

    weighted_recs = engine.find_similar_articles(
        query_identifiers=weighted_titles,
        top_k=10,
        weights=weights
    )
    print(f"\nTop 10 Recommendations (Weighted Query):")
    for idx, row in weighted_recs.iterrows():
        print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")

    # ========================================================================
    # Strategy 4: Recursive Query Expansion
    # ========================================================================
    print("\n4. Recursive Query Expansion Strategy")
    print("Building query list step-by-step based on most similar previous selections.")

    seed_idx = np.random.choice(len(engine.df))
    recursive_indices = [seed_idx]
    recursive_titles = [engine.df.iloc[seed_idx]['title']]
    print(f"Step 1: Seed -> {recursive_titles[-1]}")

    for step in range(2, num_articles + 1):
        current_matrix = engine.tfidf_matrix[recursive_indices]
        avg_vector = np.asarray(current_matrix.mean(axis=0))
        sims = cosine_similarity(avg_vector.reshape(1, -1), engine.tfidf_matrix).flatten()
        sims[recursive_indices] = -1
        next_idx = np.argmax(sims)
        recursive_indices.append(next_idx)
        next_title = engine.df.iloc[next_idx]['title']
        recursive_titles.append(next_title)
        print(f"  Step {step}: Added -> {next_title} (similarity {sims[next_idx]:.4f})")

    recursive_matrix = engine.tfidf_matrix[recursive_indices]
    recursive_sim_matrix = cosine_similarity(recursive_matrix)
    recursive_avg_sim = recursive_sim_matrix[np.triu_indices_from(recursive_sim_matrix, k=1)].mean()

    print(f"Internal coherence (avg similarity): {recursive_avg_sim:.4f}")

    recursive_recs = engine.find_similar_articles(recursive_titles, top_k=10)
    print(f"\nTop 10 Recommendations (Recursive Query):")
    for idx, row in recursive_recs.iterrows():
        print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")

    # ========================================================================
    # DEEP EXPLAINABILITY SECTION (only for Random strategy)
    # ========================================================================
    if deep_explainability:
        print("\n" + "=" * 80)
        print("DEEP EXPLAINABILITY ANALYSIS - Random Strategy")
        print("=" * 80)

        if random_recs is not None and not random_recs.empty:
            print("\n🔍 Performing deep analysis of recommendation decisions...")

            # Get top 5 recommendations for detailed comparison
            top_candidates = random_recs.head(5)['title'].tolist()

            # Compare recommendations with deep insights
            comparison = compare_recommendations_with_insights(
                engine,
                query_identifiers=random_titles,
                candidate_articles=top_candidates,
                top_k=5
            )

            # Print detailed insights
            print("\n" + "-" * 80)
            print("WHY ARTICLES RANK IN THIS ORDER:")
            print("-" * 80)

            for i, comp in enumerate(comparison['comparisons'], 1):
                print(f"\n#{i} - {comp['article']}")
                print(f"  Similarity Score: {comp['similarity']:.4f}")
                print(f"  Feature Overlap: {comp['overlap_ratio'] * 100:.1f}% "
                      f"({comp['overlap_features']}/{comp['query_features']} query features)")
                print(f"  Key differentiators:")

                if comp['top_terms']:
                    for j, term in enumerate(comp['top_terms'][:3], 1):
                        print(f"    {j}. '{term['term']}' → contribution: {term['contribution']:.5f}")

                # Explain ranking differences
                if i > 1:
                    prev_comp = comparison['comparisons'][i - 2]
                    sim_diff = comp['similarity'] - prev_comp['similarity']
                    overlap_diff = comp['overlap_ratio'] - prev_comp['overlap_ratio']

                    print(f"\n  Why ranked lower than #{i - 1}?")
                    if sim_diff < 0:
                        print(f"    • Lower similarity by {abs(sim_diff):.4f}")
                    if overlap_diff < 0:
                        print(f"    • Less feature overlap ({overlap_diff * 100:.1f}% fewer shared features)")

                    # Find unique strong terms in higher-ranked article
                    prev_terms = {t['term'] for t in prev_comp['top_terms'][:5]}
                    curr_terms = {t['term'] for t in comp['top_terms'][:5]}
                    unique_prev = prev_terms - curr_terms
                    if unique_prev:
                        print(
                            f"    • Higher-ranked article has unique strong terms: {', '.join(list(unique_prev)[:3])}")

            # Generate comprehensive visualizations
            print("\n📊 Generating visualization 1: Recommendation Breakdown...")
            visualize_recommendation_breakdown(
                comparison,
                save_path='../plots/explainability_breakdown_random.png',
                show=False
            )


            print("\n✓ Deep explainability analysis complete!")
            print("✓ Visualizations saved to ../plots/")

    # ========================================================================
    # STRATEGY COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("Analysis of Recommendation Strategies")
    print("=" * 80)

    print(f"\nInternal Coherence within query articles:")
    print(f"  Random:    {random_avg_sim:.4f}")
    print(f"  Similar:   {similar_avg_sim:.4f}")
    print(f"  Weighted:  {weighted_avg_sim:.4f}")
    print(f"  Recursive: {recursive_avg_sim:.4f}")

    print(f"\nMaximum Similarity Score: (first recommendation)")
    print(f"  Random   : {random_recs['similarity_score'].max():.4f}")
    print(f"  Similar  : {similar_recs['similarity_score'].max():.4f}")
    print(f"  Weighted : {weighted_recs['similarity_score'].max():.4f}")
    print(f"  Recursive: {recursive_recs['similarity_score'].max():.4f}")

    print(f"\nRecommendation Quality (average score of recommendations):")
    print(f"  Random   : {random_recs['similarity_score'].mean():.4f}")
    print(f"  Similar  : {similar_recs['similarity_score'].mean():.4f}")
    print(f"  Weighted : {weighted_recs['similarity_score'].mean():.4f}")
    print(f"  Recursive: {recursive_recs['similarity_score'].mean():.4f}")

    # Original comparison visualization
    visualize_strategy_comparison(
        engine, random_indices, similar_indices,
        random_recs, similar_recs,
        random_avg_sim, similar_avg_sim
    )

    return {
        'random_titles': random_titles,
        'similar_titles': similar_titles,
        'weighted_titles': weighted_titles,
        'recursive_titles': recursive_titles,
        'random_recommendations': random_recs,
        'similar_recommendations': similar_recs,
        'weighted_recommendations': weighted_recs,
        'recursive_recommendations': recursive_recs,
        'random_coherence': random_avg_sim,
        'similar_coherence': similar_avg_sim,
        'weighted_coherence': weighted_avg_sim,
        'recursive_coherence': recursive_avg_sim
    }