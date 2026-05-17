"""TF-IDF similarity engine and query-strategy implementations."""

from __future__ import annotations

from wiki_recommender.engine.similarity import ArticleSimilarityEngine
from wiki_recommender.engine.strategies import (
    StrategyResult,
    compare_strategies,
    random_strategy,
    recursive_strategy,
    similar_strategy,
)

__all__ = [
    "ArticleSimilarityEngine",
    "StrategyResult",
    "compare_strategies",
    "random_strategy",
    "recursive_strategy",
    "similar_strategy",
]
