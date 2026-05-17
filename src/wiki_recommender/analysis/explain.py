"""Recommendation explainability: distinctive query terms + element-wise contributions.

The math: for each query article, the TF-IDF vector is averaged. We then
identify *distinctive terms* — terms whose mean weight in the query set is
``min_enrichment`` × higher than in the corpus, indicating the query is unusually
focused on them. For each recommendation, the element-wise product of the
query and recommendation vectors gives the per-term contribution to the
cosine similarity score (modulo a constant scaling).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wiki_recommender.engine.similarity import ArticleSimilarityEngine
from wiki_recommender.viz import plot_contribution_analysis, plot_distinctive_term_frequency

log = logging.getLogger(__name__)


def _analyze_term_distinctiveness(
    engine: ArticleSimilarityEngine,
    query_indices: list[int],
    distinctive_term_indices: NDArray[np.integer],
) -> list[dict[str, Any]]:
    """Vectorized enrichment computation for a candidate term set."""
    matrix = engine._require_built()
    term_vectors = matrix[:, distinctive_term_indices].toarray()

    query_freqs = term_vectors[query_indices, :].mean(axis=0)
    corpus_freqs = term_vectors.mean(axis=0)
    doc_freqs = (term_vectors > 0).sum(axis=0)

    enrichments = np.divide(
        query_freqs, corpus_freqs,
        out=np.zeros_like(query_freqs),
        where=corpus_freqs > 0,
    )
    total_docs = term_vectors.shape[0]

    feature_names = engine.feature_names
    assert feature_names is not None
    return [
        {
            "term": str(feature_names[term_idx]),
            "term_idx": int(term_idx),
            "enrichment_ratio": float(enrichments[i]),
            "document_frequency": int(doc_freqs[i]),
            "total_docs": int(total_docs),
        }
        for i, term_idx in enumerate(distinctive_term_indices)
    ]


def _top_contribution_terms(
    query_vec: NDArray[np.floating],
    rec_vec: NDArray[np.floating],
    feature_names: NDArray,
    top_n: int,
) -> list[dict[str, Any]]:
    """Top-N positive contributors to the query × recommendation dot product."""
    contributions = query_vec * rec_vec
    top_indices = np.argpartition(contributions, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(contributions[top_indices])[::-1]]
    top_indices = top_indices[contributions[top_indices] > 0]
    return [
        {"term": str(feature_names[idx]), "contribution": float(contributions[idx])}
        for idx in top_indices
    ]


def explainability_analysis(
    engine: ArticleSimilarityEngine,
    query_identifiers: list[str],
    top_recommendations: pd.DataFrame,
    strategy_name: str,
    plots_dir: Path,
    min_enrichment: float = 2.0,
    top_n_terms: int = 5,
    top_distinctive_terms: int = 20,
    dpi: int = 300,
) -> dict[str, Any]:
    """Run distinctive-term analysis for a single strategy's recommendations.

    Side effect: writes two PNGs into ``plots_dir``. Returns the structured
    analysis dict so the caller can drop it into a report or a notebook.
    """
    matrix = engine._require_built()

    query_indices = engine.resolve_indices(query_identifiers)
    if not query_indices or top_recommendations.empty:
        log.warning("explainability_analysis: missing query indices or recommendations; skipping.")
        return {}

    print(f"Explainability analysis using {strategy_name}")
    print(f"Query articles: {len(query_indices)}")
    print(f"Recommendations to analyze: {len(top_recommendations)}")

    query_vec = np.asarray(matrix[query_indices].mean(axis=0)).flatten()
    nonzero_indices = np.where(query_vec > 0)[0]
    if len(nonzero_indices) == 0:
        log.warning("Query vector has no non-zero entries — nothing to explain.")
        return {}

    distinctive = _analyze_term_distinctiveness(engine, query_indices, nonzero_indices)
    distinctive = [t for t in distinctive if t["enrichment_ratio"] >= min_enrichment]
    distinctive.sort(key=lambda t: t["enrichment_ratio"], reverse=True)

    print(f"✓ Found {len(distinctive)} distinctive terms (enrichment ≥ {min_enrichment}x)")
    if distinctive:
        print("\nTop 10 most distinctive terms:")
        for rank, t in enumerate(distinctive[:10], 1):
            print(
                f"  {rank:2d}. '{t['term']:20s}' → "
                f"{t['enrichment_ratio']:.1f}x enriched, "
                f"rare in corpus ({t['document_frequency']}/{t['total_docs']} docs)"
            )

    feature_names = engine.feature_names
    assert feature_names is not None
    distinctive_lookup = {t["term_idx"]: t for t in distinctive[:top_distinctive_terms]}

    print(f"\nAnalyzing top {len(top_recommendations)} recommendations...")
    analyses: list[dict[str, Any]] = []
    for rank, (_, rec_row) in enumerate(top_recommendations.iterrows(), 1):
        rec_idx = engine.find_index(rec_row["title"])
        if rec_idx is None:
            continue

        rec_vec = np.asarray(matrix[rec_idx].toarray()).flatten()
        element_wise = query_vec * rec_vec

        term_matches = []
        for term_idx, meta in distinctive_lookup.items():
            if rec_vec[term_idx] > 0:
                term_matches.append({
                    "term": meta["term"],
                    "contribution": float(element_wise[term_idx]),
                })
        term_matches.sort(key=lambda m: m["contribution"], reverse=True)

        analyses.append({
            "rank": rank,
            "title": rec_row["title"],
            "similarity": float(rec_row["similarity_score"]),
            "distinctive_matches": term_matches,
            "term_contributions": _top_contribution_terms(
                query_vec, rec_vec, feature_names, top_n=top_n_terms,
            ),
        })

    safe_name = strategy_name.replace(" ", "_")
    plot_contribution_analysis(
        analyses,
        strategy_name=strategy_name,
        save_path=plots_dir / f"contribution_plot_{safe_name}.png",
        dpi=dpi,
    )
    plot_distinctive_term_frequency(
        analyses,
        distinctive,
        strategy_name=strategy_name,
        save_path=plots_dir / f"rare_terms_contribution_{safe_name}.png",
        dpi=dpi,
    )

    return {
        "query_identifiers": query_identifiers,
        "distinctive_terms": distinctive,
        "analyses": analyses,
        "strategy_name": strategy_name,
    }
