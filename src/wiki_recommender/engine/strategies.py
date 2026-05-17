"""Query-construction strategies for the recommender.

Three strategies that model different ways a user might assemble a query set:

- ``random``: pick N articles at random — a baseline for how *unrelated*
  recommendations look.
- ``similar``: pick a seed, then N-1 of its nearest neighbors — models a user
  who's already browsing topically related material.
- ``recursive``: pick a seed, then iteratively add the article most similar to
  the current averaged query — models a user whose path through the corpus
  drifts but coheres.

Each returns a :class:`StrategyResult` so the reporting/explainability layers
can consume the same shape regardless of how the query was constructed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from wiki_recommender.config import StrategyConfig
from wiki_recommender.engine.similarity import ArticleSimilarityEngine

log = logging.getLogger(__name__)


@dataclass(slots=True)
class StrategyResult:
    """Single-strategy single-trial output.

    ``query_titles`` carries the article list expected by downstream
    explainability — that is, the list a user (or analyst) would name as
    "the query". For the ``similar`` strategy the seed is intentionally
    excluded from this list, matching the original analysis behavior.
    """

    name: str
    query_titles: list[str]
    coherence: float
    recommendations: pd.DataFrame
    max_sim: float = field(init=False)
    mean_sim: float = field(init=False)

    def __post_init__(self) -> None:
        if self.recommendations.empty:
            self.max_sim = float("nan")
            self.mean_sim = float("nan")
        else:
            scores = self.recommendations["similarity_score"]
            self.max_sim = float(scores.max())
            self.mean_sim = float(scores.mean())


def compute_internal_coherence(matrix) -> float:
    """Mean off-diagonal cosine similarity of a set of TF-IDF vectors.

    Uses only the strict upper triangle so each pair is counted once and the
    self-similarity diagonal (always 1.0) is excluded.
    """
    sim = cosine_similarity(matrix)
    return float(sim[np.triu_indices_from(sim, k=1)].mean())


def random_strategy(
    engine: ArticleSimilarityEngine,
    num_articles: int,
    rng: np.random.Generator | None = None,
) -> StrategyResult:
    """Strategy 1: sample ``num_articles`` titles uniformly at random."""
    rng = rng or np.random.default_rng()
    matrix = engine._require_built()

    indices = rng.choice(len(engine.df), size=num_articles, replace=False)
    titles = engine.df.iloc[indices]["title"].tolist()
    coherence = compute_internal_coherence(matrix[indices])
    recs = engine.find_similar_articles(titles)

    return StrategyResult(
        name="Random Strategy",
        query_titles=titles,
        coherence=coherence,
        recommendations=recs,
    )


def similar_strategy(
    engine: ArticleSimilarityEngine,
    num_articles: int,
    rng: np.random.Generator | None = None,
) -> StrategyResult:
    """Strategy 2: pick a random seed, expand with its top-``num_articles`` neighbors.

    The recommendation query is the seed *plus* the neighbors (so the request
    feels like "more like this cluster"); coherence and explainability are
    computed on the neighbors alone, since that's the part of the query that
    isn't trivially correlated by construction.
    """
    rng = rng or np.random.default_rng()
    matrix = engine._require_built()

    seed_idx = int(rng.integers(len(engine.df)))
    seed_title = engine.df.iloc[seed_idx]["title"]
    seed_vector = matrix[seed_idx]
    sims = cosine_similarity(seed_vector, matrix).flatten()
    # argsort ascending; reverse and skip index 0 which is the seed itself.
    similar_indices = np.argsort(sims)[::-1][1 : num_articles + 1]
    similar_titles = engine.df.iloc[similar_indices]["title"].tolist()

    recs = engine.find_similar_articles([seed_title, *similar_titles])
    coherence = compute_internal_coherence(matrix[similar_indices])

    return StrategyResult(
        name="Similar Strategy",
        query_titles=similar_titles,
        coherence=coherence,
        recommendations=recs,
    )


def recursive_strategy(
    engine: ArticleSimilarityEngine,
    num_articles: int,
    rng: np.random.Generator | None = None,
) -> StrategyResult:
    """Strategy 3: seed-then-grow.

    Each step adds whichever unseen article best fits the running average
    query vector — a path-dependent walk that can drift topic over time.
    """
    rng = rng or np.random.default_rng()
    matrix = engine._require_built()

    seed_idx = int(rng.integers(len(engine.df)))
    indices: list[int] = [seed_idx]
    titles: list[str] = [engine.df.iloc[seed_idx]["title"]]

    for _ in range(num_articles - 1):
        current = matrix[indices]
        avg_vec = np.asarray(current.mean(axis=0))
        sims = cosine_similarity(avg_vec.reshape(1, -1), matrix).flatten()
        sims[indices] = -1.0  # forbid picking an already-selected article
        next_idx = int(np.argmax(sims))
        indices.append(next_idx)
        titles.append(engine.df.iloc[next_idx]["title"])

    coherence = compute_internal_coherence(matrix[indices])
    recs = engine.find_similar_articles(titles)

    return StrategyResult(
        name="Recursive Strategy",
        query_titles=titles,
        coherence=coherence,
        recommendations=recs,
    )


def compare_strategies(
    engine: ArticleSimilarityEngine,
    cfg: StrategyConfig,
    rng: np.random.Generator | None = None,
) -> list[list[StrategyResult]]:
    """Run all three strategies for ``cfg.num_trials`` trials.

    Returns a list-of-trials where each trial is the three
    :class:`StrategyResult` objects in [random, similar, recursive] order.
    """
    rng = rng or np.random.default_rng()

    # similar_strategy is called with num_articles-1 so the *total* query —
    # seed plus neighbors — has the same size as the other strategies'.
    similar_size = max(cfg.num_articles - 1, 1)

    trials: list[list[StrategyResult]] = []
    for trial_idx in range(cfg.num_trials):
        log.debug("Strategy comparison trial %d/%d", trial_idx + 1, cfg.num_trials)
        trials.append([
            random_strategy(engine, cfg.num_articles, rng=rng),
            similar_strategy(engine, similar_size, rng=rng),
            recursive_strategy(engine, cfg.num_articles, rng=rng),
        ])
    return trials
