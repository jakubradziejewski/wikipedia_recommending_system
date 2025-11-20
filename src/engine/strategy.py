import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.analysis.explainability import deep_explainability_analysis

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


def compare_recommendation_strategies(engine, num_articles=10, run_explainability=True):
    """
    Compare recommendations from multiple query selection strategies.

    Args:
        engine: ArticleSimilarityEngine instance
        num_articles: Number of articles to use in query set
        run_explainability: Whether to run deep explainability analysis on all strategies

    Returns:
        dict: Results from all strategies
    """
    print("\nApproaches of recommendation based on different query article selection strategies:")

    # Run each strategy modularly
    random_data = random_strategy(engine, num_articles)
    similar_data = similar_strategy(engine, num_articles)
    recursive_data = recursive_strategy(engine, num_articles)

    # Optional: Run deep explainability on all strategies
    if run_explainability:
        deep_explainability_analysis(
            engine,
            random_data["titles"],
            random_data["recs"],
            strategy_name="Random Strategy",
            min_enrichment=2.0
        )

        deep_explainability_analysis(
            engine,
            similar_data["titles"],
            similar_data["recs"],
            strategy_name="Similar Strategy",
            min_enrichment=2.0
        )

        deep_explainability_analysis(
            engine,
            recursive_data["titles"],
            recursive_data["recs"],
            strategy_name="Recursive Strategy",
            min_enrichment=2.0
        )

    # Final summary comparison
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
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



    return {
        'random': random_data,
        'similar': similar_data,
        'recursive': recursive_data
    }

