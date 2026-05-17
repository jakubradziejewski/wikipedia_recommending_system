"""TF-IDF matrix statistics + corpus-wide similarity distribution."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from wiki_recommender.engine.similarity import ArticleSimilarityEngine
from wiki_recommender.errors import ModelNotBuiltError
from wiki_recommender.viz import plot_similarity_distribution

log = logging.getLogger(__name__)

_TOP_TERMS = 20


def generate_model_statistics(
    engine: ArticleSimilarityEngine,
    plots_dir: Path,
    dpi: int = 300,
) -> tuple[float, float]:
    """Print TF-IDF matrix density / size + corpus-similarity stats; save plot.

    Returns (mean, median) pairwise cosine similarity over the full corpus.
    The full upper-triangle matrix is materialized — at ~5k articles this is
    well under 100 MB of float64. For much larger corpora prefer sampling.
    """
    if not engine.is_built:
        log.warning("TF-IDF model not built; building now.")
        engine.build_tfidf_model()

    matrix = engine.tfidf_matrix
    if matrix is None:
        raise ModelNotBuiltError("TF-IDF matrix unavailable after build.")

    num_docs, num_terms = matrix.shape
    density = matrix.nnz / (num_docs * num_terms)

    print("\n" + "-" * 80)
    print("TF-IDF MODEL STATISTICS")
    print("-" * 80)
    print(f"Number of documents (articles): {num_docs}")
    print(f"Number of unique features (terms/n-grams): {num_terms}")
    print(f"Matrix density: {density:.4%}")
    print(f"Non-zero elements: {matrix.nnz:,}")

    print("\nCalculating full corpus pairwise similarity matrix (might take a moment)...")
    similarity_matrix = cosine_similarity(matrix)

    plots_dir.mkdir(parents=True, exist_ok=True)
    _fig, mean_sim, median_sim = plot_similarity_distribution(
        similarity_matrix,
        save_path=plots_dir / "similarity_distribution.png",
        dpi=dpi,
    )

    print("\nCorpus Similarity Statistics:")
    print(f"  Mean pairwise similarity: {mean_sim:.4f}")
    print(f"  Median pairwise similarity: {median_sim:.4f}")

    print("\n" + "-" * 80)
    print("TOP TF-IDF TERMS (Corpus-wide)")
    print("-" * 80)
    tfidf_sums = np.array(matrix.sum(axis=0)).flatten()
    top_indices = np.argsort(tfidf_sums)[-_TOP_TERMS:][::-1]
    print(f"\nTop {_TOP_TERMS} terms by cumulative TF-IDF score:")
    feature_names = engine.feature_names
    assert feature_names is not None  # implied by is_built
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {feature_names[idx]:20s} (score: {tfidf_sums[idx]:.2f})")

    return mean_sim, median_sim
