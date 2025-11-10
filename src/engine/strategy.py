import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.visualization import visualize_strategy_comparison
from src.engine.explainability import explain_similarity, visualize_similarity_explanation


def compare_recommendation_strategies(engine, num_articles=10):
    """
    Compare recommendations from multiple query selection strategies:
      1. Random Article Collection
      2. Similar (connected) Article Collection
      3. Weighted Query Strategy
      4. Recursive Query Expansion
    """
    print("\n" + "-" * 80)
    print("Approaches of recommendation based on different query article selection strategies:")
    print("-" * 80)

    print("\n11. Random Article Collection")
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

        # =====================================================================
        # EXPLAINABILITY SECTION (for each strategy)
        # =====================================================================
    print("\n" + "=" * 80)
    print("EXPLAINABILITY COMPARISONS")
    print("=" * 80)

    explain_cases = [
        ("Random", random_titles, random_recs),
        ("Similar", similar_titles, similar_recs),
        ("Weighted", weighted_titles, weighted_recs),
        ("Recursive", recursive_titles, recursive_recs),
    ]

    for label, query_list, rec_df in explain_cases:
        if rec_df is not None and not rec_df.empty:
            target_title = rec_df.iloc[0]["title"]
            print(f"\n{'-' * 80}")
            print(f"🔍 Explainability for {label} Strategy")
            print(f"Target article: {target_title}")
            print(f"Based on {len(query_list)} query articles")
            print(f"{'-' * 80}")

            explanation = explain_similarity(
                engine,
                query_identifiers=query_list,
                target_article=target_title,
                top_terms=15,
                verbose=True
            )

            # Save visualization per strategy
            save_name = f"../plots/explainability_{label.lower()}.png"
            visualize_similarity_explanation(explanation, save_path=save_name)
        else:
            print(f"\n⚠ No recommendations found for {label} strategy — skipping explainability.")

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
