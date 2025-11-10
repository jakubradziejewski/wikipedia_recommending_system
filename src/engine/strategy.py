import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.visualization import visualize_strategy_comparison

def compare_recommendation_strategies(engine, num_articles=10):
    """
    Compare recommendations from random vs similar article collections

    Args:
        engine: ArticleSimilarityEngine instance
        num_articles: Number of articles to use in each strategy
    """
    print("\n" + "-" * 80)
    print("Approaches of recommendation based on different query article selection strategies:")
    print("-" * 80)
    # Strategy 1: Random articles
    print("1. Random Article Collection")

    random_indices = np.random.choice(len(engine.df), size=num_articles, replace=False)
    random_titles = engine.df.iloc[random_indices]['title'].tolist()

    print(f"\nQuery articles (randomly selected):")
    for i, title in enumerate(random_titles, 1):
        print(f"  {i:2d}. {title}")
    # Get pairwise similarities within random collection
    random_matrix = engine.tfidf_matrix[random_indices]
    random_sim_matrix = cosine_similarity(random_matrix)
    random_avg_sim = random_sim_matrix[np.triu_indices_from(random_sim_matrix, k=1)].mean()

    print(f"\nAvg similarity within sample articles: {random_avg_sim:.4f}")

    random_recs = engine.find_similar_articles(random_titles, top_k=10)

    print(f"\nTop 10 Recommendations:")
    for idx, row in random_recs.iterrows():
        print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")

    # Strategy 2: Similar (connected) articles
    print("2. Similar Article Collection based on randomly chosen seed article, imitating somehow user interests and possible past readings")

    # Start with a random article and find similar ones
    seed_idx = np.random.choice(len(engine.df))
    seed_title = engine.df.iloc[seed_idx]['title']

    print(f"\nSeed article: {seed_title}")

    # Find similar articles to the seed
    seed_vector = engine.tfidf_matrix[seed_idx]
    similarities = cosine_similarity(seed_vector, engine.tfidf_matrix).flatten()
    similar_indices = np.argsort(similarities)[::-1][1:num_articles + 1]
    similar_titles = engine.df.iloc[similar_indices]['title'].tolist()

    print(f"\nQuery articles (similar to seed):")
    for i, (idx, title) in enumerate(zip(similar_indices, similar_titles), 1):
        sim_score = similarities[idx]
        print(f"  {i:2d}. {title} (similarity to seed: {sim_score:.4f})")

    # Get pairwise similarities within similar collection
    similar_matrix = engine.tfidf_matrix[similar_indices]
    similar_sim_matrix = cosine_similarity(similar_matrix)
    similar_avg_sim = similar_sim_matrix[np.triu_indices_from(similar_sim_matrix, k=1)].mean()

    print(f"\nInternal coherence (avg similarity): {similar_avg_sim:.4f}")

    similar_recs = engine.find_similar_articles(similar_titles, top_k=10)

    print(f"\nTop 10 Recommendations:")
    for idx, row in similar_recs.iterrows():
        print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")

    # Comparison Analysis
    print("\n" + "=" * 80)
    print("Analysis of Recommendation Strategies")
    print("=" * 80)

    print(f"\n📈 Internal Coherence:")
    print(f"  Random collection:  {random_avg_sim:.4f}")
    print(f"  Similar collection: {similar_avg_sim:.4f}")
    print(
        f"  Difference:         {similar_avg_sim - random_avg_sim:.4f} ({(similar_avg_sim / random_avg_sim - 1) * 100:+.1f}%)")

    print(f"\n📈 Recommendation Quality:")
    print(f"  Random - Avg score:  {random_recs['similarity_score'].mean():.4f}")
    print(f"  Similar - Avg score: {similar_recs['similarity_score'].mean():.4f}")
    print(f"  Random - Max score:  {random_recs['similarity_score'].max():.4f}")
    print(f"  Similar - Max score: {similar_recs['similarity_score'].max():.4f}")

    # Check overlap in recommendations
    random_rec_set = set(random_recs['title'].tolist())
    similar_rec_set = set(similar_recs['title'].tolist())
    overlap = random_rec_set.intersection(similar_rec_set)

    print(f"\n📊 Recommendation Overlap:")
    print(f"  Common recommendations: {len(overlap)}/10")
    if overlap:
        print(f"  Overlapping articles:")
        for title in overlap:
            print(f"    • {title}")

    # Visualize comparison
    visualize_strategy_comparison(engine, random_indices, similar_indices,
                                  random_recs, similar_recs,
                                  random_avg_sim, similar_avg_sim)

    return {
        'random_titles': random_titles,
        'similar_titles': similar_titles,
        'random_recommendations': random_recs,
        'similar_recommendations': similar_recs,
        'random_coherence': random_avg_sim,
        'similar_coherence': similar_avg_sim
    }

