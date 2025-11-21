import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.analysis.explainability import explainability_analysis


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


def generate_recommendations(engine, titles, label="Top Recommendations", verbose=True):
    """Generate and print top recommendations for a given query set."""
    recs = engine.find_similar_articles(titles, top_k=10)

    if verbose:
        print(f"\n{label}:")
        for _, row in recs.iterrows():
            print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")
    return recs


def random_strategy(engine, num_articles, verbose=True):
    """Strategy 1: Random Article Collection"""
    if verbose:
        print("\n1. Random Article Collection - selecting articles randomly from the corpus\n")
    indices = np.random.choice(len(engine.df), size=num_articles, replace=False)
    titles = engine.df.iloc[indices]['title'].tolist()

    if verbose:
        print_article_list(titles, label="Query articles (randomly selected)")
    avg_sim = compute_internal_coherence(engine.tfidf_matrix[indices])
    if verbose:
        print(f"\nInternal coherence (avg similarity): {avg_sim:.4f}")

    recs = generate_recommendations(engine, titles, verbose=verbose)
    return {"titles": titles, "indices": indices, "recs": recs, "coherence": avg_sim}


def similar_strategy(engine, num_articles, verbose=True):
    """Strategy 2: Similar Article Collection"""
    if verbose:
        print("\n2. Similar Article Collection (based on randomly chosen seed article)")
        print(
            "Building query list based on most similar articles to the seed, imitating a collection user might saw during the search.\n")
    seed_idx = np.random.choice(len(engine.df))
    seed_title = engine.df.iloc[seed_idx]['title']
    if verbose:
        print(f"\nSeed article: {seed_title}")

    seed_vector = engine.tfidf_matrix[seed_idx]
    similarities = cosine_similarity(seed_vector, engine.tfidf_matrix).flatten()
    similar_indices = np.argsort(similarities)[::-1][1:num_articles + 1]
    similar_titles = engine.df.iloc[similar_indices]['title'].tolist()

    query_titles = [seed_title] + similar_titles

    if verbose:
        print_article_list(query_titles, label="Query articles (seed + similar ones)")

    avg_sim = compute_internal_coherence(engine.tfidf_matrix[similar_indices])
    if verbose:
        print(f"\nInternal coherence (avg similarity): {avg_sim:.4f}")

    recs = generate_recommendations(engine, query_titles, verbose=verbose)

    return {"titles": similar_titles, "indices": similar_indices, "recs": recs, "coherence": avg_sim}


def recursive_strategy(engine, num_articles, verbose=True):
    """Strategy 3: Recursive Query Expansion"""
    if verbose:
        print("\n3. Recursive Query Expansion Strategy")
        print(
            "Building query list step-by-step based on most similar previous selections. Every step adds the most similar article to the already selected articles simulating user actions.\n")

    # Randomly select the first article - seed
    seed_idx = np.random.choice(len(engine.df))
    indices = [seed_idx]
    titles = [engine.df.iloc[seed_idx]['title']]
    if verbose:
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
        if verbose:
            print(f"  Step {step}: Added -> {next_title} (similarity {sims[next_idx]:.4f})")

    avg_sim = compute_internal_coherence(engine.tfidf_matrix[indices])
    if verbose:
        print(f"\nInternal coherence (avg similarity): {avg_sim:.4f}")

    recs = generate_recommendations(engine, titles, label="Top 10 Recommendations (Recursive Query)", verbose=verbose)
    return {"titles": titles, "indices": indices, "recs": recs, "coherence": avg_sim}


def compare_recommendation_strategies(engine, num_articles=10, run_explainability=True, num_trials=1):
    """Compare recommendations from multiple query selection strategies."""
    print("\nApproaches of recommendation based on different query article selection strategies:")

    # Storage for averaging
    if num_trials > 1:
        stats = {
            'random': {'coherence': [], 'max_sim': [], 'mean_sim': []},
            'similar': {'coherence': [], 'max_sim': [], 'mean_sim': []},
            'recursive': {'coherence': [], 'max_sim': [], 'mean_sim': []}
        }

    # Run trials
    for trial in range(num_trials):
        verbose = (trial == 0)  # Only verbose on first run

        random_data = random_strategy(engine, num_articles, verbose=verbose)
        similar_data = similar_strategy(engine, num_articles - 1, verbose=verbose)
        recursive_data = recursive_strategy(engine, num_articles, verbose=verbose)
        if run_explainability and verbose:
            explainability_analysis(
                engine,
                random_data["titles"],
                random_data["recs"],
                strategy_name="Random Strategy",
                min_enrichment=2.0
            )

            explainability_analysis(
                engine,
                similar_data["titles"],
                similar_data["recs"],
                strategy_name="Similar Strategy",
                min_enrichment=2.0
            )

            explainability_analysis(
                engine,
                recursive_data["titles"],
                recursive_data["recs"],
                strategy_name="Recursive Strategy",
                min_enrichment=2.0
            )

        if num_trials > 1:
            stats['random']['coherence'].append(random_data['coherence'])
            stats['random']['max_sim'].append(random_data['recs']['similarity_score'].max())
            stats['random']['mean_sim'].append(random_data['recs']['similarity_score'].mean())

            stats['similar']['coherence'].append(similar_data['coherence'])
            stats['similar']['max_sim'].append(similar_data['recs']['similarity_score'].max())
            stats['similar']['mean_sim'].append(similar_data['recs']['similarity_score'].mean())

            stats['recursive']['coherence'].append(recursive_data['coherence'])
            stats['recursive']['max_sim'].append(recursive_data['recs']['similarity_score'].max())
            stats['recursive']['mean_sim'].append(recursive_data['recs']['similarity_score'].mean())

    # Final summary comparison
    print("\n" + "=" * 80)
    print("Comparison of use cases -  different recommendation strategies")
    print("=" * 80)

    if num_trials > 1:
        print(f"\n(Averaged over {num_trials} trials)")
        print(f"\nInternal Coherence within query articles:")
        print(f"  Random:    {np.mean(stats['random']['coherence']):.4f} ± {np.std(stats['random']['coherence']):.4f}")
        print(
            f"  Similar:   {np.mean(stats['similar']['coherence']):.4f} ± {np.std(stats['similar']['coherence']):.4f}")
        print(
            f"  Recursive: {np.mean(stats['recursive']['coherence']):.4f} ± {np.std(stats['recursive']['coherence']):.4f}")

        print(f"\nMaximum Similarity Score (first recommendation):")
        print(f"  Random   : {np.mean(stats['random']['max_sim']):.4f} ± {np.std(stats['random']['max_sim']):.4f}")
        print(f"  Similar  : {np.mean(stats['similar']['max_sim']):.4f} ± {np.std(stats['similar']['max_sim']):.4f}")
        print(
            f"  Recursive: {np.mean(stats['recursive']['max_sim']):.4f} ± {np.std(stats['recursive']['max_sim']):.4f}")

        print(f"\nRecommendation Quality (average score of recommendations):")
        print(f"  Random   : {np.mean(stats['random']['mean_sim']):.4f} ± {np.std(stats['random']['mean_sim']):.4f}")
        print(f"  Similar  : {np.mean(stats['similar']['mean_sim']):.4f} ± {np.std(stats['similar']['mean_sim']):.4f}")
        print(
            f"  Recursive: {np.mean(stats['recursive']['mean_sim']):.4f} ± {np.std(stats['recursive']['mean_sim']):.4f}")
    else:
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