import numpy as np
from src.utils.visualization import visualize_strategy_comparison
from src.engine.explainability import compare_recommendations_with_insights, visualize_recommendation_breakdown
from sklearn.metrics.pairwise import cosine_similarity


def compute_internal_coherence(matrix):
    """Compute average cosine similarity within a collection of article vectors."""
    sim_matrix = cosine_similarity(matrix)
    return sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean()


def print_article_list(titles, similarities=None, label="Articles"):
    """Print formatted list of article titles, optionally with similarity scores."""
    print(f"\n{label}:")
    for i, title in enumerate(titles, 1):
        if similarities is not None:
            print(f"  {i:2d}. {title} (similarity: {similarities[i - 1]:.4f})")
        else:
            print(f"  {i:2d}. {title}")


def generate_recommendations(engine, titles, label="Top Recommendations"):
    """Generate and print top recommendations for a given query set."""
    recs = engine.find_similar_articles(titles, top_k=10)
    print(f"\n{label}:")
    for _, row in recs.iterrows():
        print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")
    return recs


def random_strategy(engine, num_articles):
    """Strategy 1: Random Article Collection"""
    print("\n1. Random Article Collection")
    indices = np.random.choice(len(engine.df), size=num_articles, replace=False)
    titles = engine.df.iloc[indices]['title'].tolist()

    print_article_list(titles, label="Query articles (randomly selected)")
    avg_sim = compute_internal_coherence(engine.tfidf_matrix[indices])
    print(f"\nInternal coherence (avg similarity): {avg_sim:.4f}")

    recs = generate_recommendations(engine, titles)
    return {"titles": titles, "indices": indices, "recs": recs, "coherence": avg_sim}


def similar_strategy(engine, num_articles):
    """Strategy 2: Similar Article Collection"""
    print("\n2. Similar Article Collection (based on randomly chosen seed article)")
    seed_idx = np.random.choice(len(engine.df))
    seed_title = engine.df.iloc[seed_idx]['title']
    print(f"\nSeed article: {seed_title}")

    seed_vector = engine.tfidf_matrix[seed_idx]
    similarities = cosine_similarity(seed_vector, engine.tfidf_matrix).flatten()
    similar_indices = np.argsort(similarities)[::-1][1:num_articles + 1]
    similar_titles = engine.df.iloc[similar_indices]['title'].tolist()

    query_titles = [seed_title] + similar_titles

    print_article_list(query_titles, label="Query articles (seed + similar ones)")


    avg_sim = compute_internal_coherence(engine.tfidf_matrix[similar_indices])
    print(f"\nInternal coherence (avg similarity): {avg_sim:.4f}")

    recs = generate_recommendations(engine, query_titles)

    return {"titles": similar_titles, "indices": similar_indices, "recs": recs, "coherence": avg_sim}


def recursive_strategy(engine, num_articles):
    """Strategy 3: Recursive Query Expansion"""
    print("\n3. Recursive Query Expansion Strategy")
    print("Building query list step-by-step based on most similar previous selections.")

    seed_idx = np.random.choice(len(engine.df))
    indices = [seed_idx]
    titles = [engine.df.iloc[seed_idx]['title']]
    print(f"Step 1: Seed -> {titles[-1]}")

    for step in range(2, num_articles + 1):
        current_matrix = engine.tfidf_matrix[indices]
        avg_vector = np.asarray(current_matrix.mean(axis=0))
        sims = cosine_similarity(avg_vector.reshape(1, -1), engine.tfidf_matrix).flatten()
        sims[indices] = -1
        next_idx = np.argmax(sims)
        indices.append(next_idx)
        next_title = engine.df.iloc[next_idx]['title']
        titles.append(next_title)
        print(f"  Step {step}: Added -> {next_title} (similarity {sims[next_idx]:.4f})")

    avg_sim = compute_internal_coherence(engine.tfidf_matrix[indices])
    print(f"\nInternal coherence (avg similarity): {avg_sim:.4f}")

    recs = generate_recommendations(engine, titles, label="Top 10 Recommendations (Recursive Query)")
    return {"titles": titles, "indices": indices, "recs": recs, "coherence": avg_sim}


def deep_explainability_analysis(engine, data):
    """Optional deep explainability step for random strategy results."""
    print("\n" + "=" * 80)
    print(f"DEEP EXPLAINABILITY ANALYSIS FOR {data}")
    print("=" * 80)

    recs = data["recs"]
    titles = data["titles"]

    if recs is not None and not recs.empty:
        print("\n🔍 Performing deep analysis of recommendation decisions...")

        top_candidates = recs.head(5)['title'].tolist()
        comparison = compare_recommendations_with_insights(
            engine,
            query_identifiers=titles,
            candidate_articles=top_candidates,
            top_k=5
        )

        print("\n" + "-" * 80)
        print("WHY ARTICLES RANK IN THIS ORDER:")
        print("-" * 80)

        for i, comp in enumerate(comparison['comparisons'], 1):
            print(f"\n#{i} - {comp['article']}")
            print(f"  Similarity Score: {comp['similarity']:.4f}")
            print(f"  Feature Overlap: {comp['overlap_ratio'] * 100:.1f}% "
                  f"({comp['overlap_features']}/{comp['query_features']} query features)")
            print(f"  Key differentiators:")

            for j, term in enumerate(comp['top_terms'][:3], 1):
                print(f"    {j}. '{term['term']}' → contribution: {term['contribution']:.5f}")

            if i > 1:
                prev = comparison['comparisons'][i - 2]
                sim_diff = comp['similarity'] - prev['similarity']
                overlap_diff = comp['overlap_ratio'] - prev['overlap_ratio']
                print(f"\n  Why ranked lower than #{i - 1}?")
                if sim_diff < 0:
                    print(f"    • Lower similarity by {abs(sim_diff):.4f}")
                if overlap_diff < 0:
                    print(f"    • Less feature overlap ({overlap_diff * 100:.1f}% fewer shared features)")
                unique_prev = {t['term'] for t in prev['top_terms'][:5]} - {t['term'] for t in comp['top_terms'][:5]}
                if unique_prev:
                    print(f"    • Higher-ranked article has unique strong terms: {', '.join(list(unique_prev)[:3])}")

        print("\nGenerating visualization 1: Recommendation Breakdown...")
        visualize_recommendation_breakdown(
            comparison,
            save_path='../plots/explainability_breakdown_random.png',
            show=False
        )

        print("\n✓ Deep explainability analysis complete!")
        print("✓ Visualizations saved to ../plots/")


def compare_recommendation_strategies(engine, num_articles=10, deep_explainability=True):
    """
    Compare recommendations from multiple query selection strategies.
    """
    print("Approaches of recommendation based on different query article selection strategies:")

    # Run each strategy modularly
    random_data = random_strategy(engine, num_articles)
    similar_data = similar_strategy(engine, num_articles)
    recursive_data = recursive_strategy(engine, num_articles)

    # Deep explainability (optional)
    if deep_explainability:
        deep_explainability_analysis(engine, random_data)

    # Final summary comparison
    print("\n" + "=" * 80)
    print("Analysis of Recommendation Strategies")
    print("=" * 80)

    print(f"\nInternal Coherence within query articles:")
    print(f"  Random:    {random_data['coherence']:.4f}")
    print(f"  Similar:   {similar_data['coherence']:.4f}")
    print(f"  Recursive: {recursive_data['coherence']:.4f}")

    print(f"\nMaximum Similarity Score (first recommendation):")
    print(f"  Random   : {random_data['recs']['similarity_score'].max():.4f}")
    print(f"  Similar  : {similar_data['recs']['similarity_score'].max():.4f}")
    print(f"  Recursive: {recursive_data['recs']['similarity_score'].max():.4f}")

    print(f"\nRecommendation Quality (average score of recommendations):")
    print(f"  Random   : {random_data['recs']['similarity_score'].mean():.4f}")
    print(f"  Similar  : {similar_data['recs']['similarity_score'].mean():.4f}")
    print(f"  Recursive: {recursive_data['recs']['similarity_score'].mean():.4f}")

    # Visualization
    visualize_strategy_comparison(
        engine,
        random_data["indices"], similar_data["indices"],
        random_data["recs"], similar_data["recs"],
        random_data["coherence"], similar_data["coherence"]
    )

    return {
        'random': random_data,
        'similar': similar_data,
        'recursive': recursive_data
    }
