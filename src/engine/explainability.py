import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


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

    # Console output (mirroring similarities_mathing.py style)
    if verbose:
        print(f"\nWhy is '{explanation['target_article']}' recommended?")
        print(f"Overall similarity score: {explanation['similarity_score']:.4f}\n")
        print("\nTop contributing terms:")
        for i, term_data in enumerate(explanation["top_terms"][:top_terms], 1):
            print(f"  {i:2d}. '{term_data['term']}' - contribution: {term_data['contribution']:.5f}")

    return explanation


def visualize_similarity_explanation(explanation, save_path=None, show=True):
    """
    Visualize the output of explain_similarity() with contribution and TF-IDF comparisons.

    Args:
        explanation (dict): Output from explain_similarity()
        save_path (str): Optional file path to save the figure
        show (bool): Whether to display the figure interactively
    """
    if not explanation or "top_terms" not in explanation:
        print("⚠ No explanation data to visualize.")
        return

    terms_df = pd.DataFrame(explanation["top_terms"])
    if terms_df.empty:
        print("⚠ No terms to visualize.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # --- Plot 1: Term contributions ---
    ax1 = axes[0]
    ax1.barh(range(len(terms_df)), terms_df["contribution"], color="steelblue", alpha=0.7)
    ax1.set_yticks(range(len(terms_df)))
    ax1.set_yticklabels(terms_df["term"])
    ax1.set_xlabel("Contribution to Similarity", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Top Contributing Terms\nSimilarity Score: {explanation['similarity_score']:.4f}",
        fontsize=13, fontweight="bold", pad=15
    )
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()

    for i, (idx, row) in enumerate(terms_df.iterrows()):
        ax1.text(row["contribution"], i, f" {row['contribution']:.4f}", va="center", fontsize=9)

    # --- Plot 2: TF-IDF comparison ---
    ax2 = axes[1]
    x = np.arange(len(terms_df))
    width = 0.35

    ax2.bar(x - width/2, terms_df["query_tfidf"], width, label="Query Articles", color="coral", alpha=0.7)
    ax2.bar(x + width/2, terms_df["target_tfidf"], width, label="Target Article", color="lightgreen", alpha=0.7)

    ax2.set_xlabel("Terms", fontsize=11, fontweight="bold")
    ax2.set_ylabel("TF-IDF Score", fontsize=11, fontweight="bold")
    ax2.set_title("TF-IDF Scores Comparison", fontsize=13, fontweight="bold", pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(terms_df["term"], rotation=45, ha="right")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # Info text
    query_text = "Query: " + ", ".join(explanation["query_articles"][:2])
    if len(explanation["query_articles"]) > 2:
        query_text += f" + {len(explanation['query_articles']) - 2} more"
    target_text = f"Target: {explanation['target_article']}"

    fig.text(0.5, 0.98, query_text, ha="center", fontsize=10, style="italic", wrap=True)
    fig.text(0.5, 0.96, target_text, ha="center", fontsize=10, style="italic", color="darkgreen", wrap=True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Visualization saved to {save_path}")

    if show:
        plt.show()
