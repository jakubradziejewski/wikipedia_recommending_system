"""TF-IDF + cosine-similarity engine over the article corpus."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wiki_recommender.config import EngineConfig
from wiki_recommender.errors import (
    EmptyCorpusError,
    MissingDataError,
    ModelNotBuiltError,
    QueryNotFoundError,
)

log = logging.getLogger(__name__)


class ArticleSimilarityEngine:
    """Loads the parquet corpus and computes article-to-article similarities.

    Build order: ``ArticleSimilarityEngine.load(path, cfg)`` runs the full
    setup in one call; for stepwise control instantiate then call
    :meth:`build_tfidf_model` explicitly.
    """

    def __init__(self, df: pd.DataFrame, cfg: EngineConfig | None = None) -> None:
        if df.empty:
            raise EmptyCorpusError("Corpus DataFrame has zero rows.")

        self.df: pd.DataFrame = df.reset_index(drop=True)
        self.cfg: EngineConfig = cfg or EngineConfig()
        self.title_to_idx: dict[str, int] = {
            str(title).lower(): idx for idx, title in enumerate(self.df["title"])
        }

        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: csr_matrix | None = None
        self.feature_names: np.ndarray | None = None

    @classmethod
    def load(cls, parquet_path: Path, cfg: EngineConfig | None = None) -> "ArticleSimilarityEngine":
        """Read parquet, instantiate, and fit the TF-IDF model in one step."""
        if not parquet_path.exists():
            raise MissingDataError(
                f"Corpus parquet not found at {parquet_path}. Run `wikirec scrape` first."
            )
        df = pd.read_parquet(parquet_path)
        log.info("Loaded %d articles from %s", len(df), parquet_path)
        engine = cls(df, cfg=cfg)
        engine.build_tfidf_model()
        return engine

    def build_tfidf_model(self) -> None:
        """Fit the TfidfVectorizer on ``self.cfg.text_column``."""
        column = self.cfg.text_column
        if column not in self.df.columns:
            raise EmptyCorpusError(
                f"Text column '{column}' missing from corpus. Available: {list(self.df.columns)}"
            )

        texts = self.df[column].fillna("")
        log.info(
            "Fitting TF-IDF on column '%s' (n_docs=%d, ngram_range=%s, min_df=%s, max_df=%s)",
            column, len(texts), self.cfg.ngram_range, self.cfg.min_df, self.cfg.max_df,
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.cfg.ngram_range,
            min_df=self.cfg.min_df,
            max_df=self.cfg.max_df,
            sublinear_tf=self.cfg.sublinear_tf,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        log.info(
            "TF-IDF matrix shape: %s, density: %.4f%%",
            self.tfidf_matrix.shape, 100 * self.tfidf_matrix.nnz / np.prod(self.tfidf_matrix.shape),
        )

    @property
    def is_built(self) -> bool:
        return self.tfidf_matrix is not None

    def _require_built(self) -> csr_matrix:
        if self.tfidf_matrix is None:
            raise ModelNotBuiltError("TF-IDF model not built. Call build_tfidf_model() first.")
        return self.tfidf_matrix

    def find_index(self, identifier: str) -> int | None:
        """Resolve an article title to its row index. Case-insensitive."""
        return self.title_to_idx.get(identifier.lower())

    def resolve_indices(self, identifiers: Iterable[str]) -> list[int]:
        """Resolve a batch of titles, silently dropping unknowns."""
        resolved = []
        for ident in identifiers:
            idx = self.find_index(ident)
            if idx is not None:
                resolved.append(idx)
        return resolved

    def find_similar_articles(
        self,
        query_identifiers: list[str],
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Return the top-k articles most similar to the averaged query vector.

        Returns:
            DataFrame with columns ``title``, ``similarity_score``,
            ``token_count``. Query articles themselves are excluded from the
            result, even if they would otherwise rank highest.
        """
        matrix = self._require_built()
        k = top_k if top_k is not None else self.cfg.top_k

        query_indices = self.resolve_indices(query_identifiers)
        if not query_indices:
            raise QueryNotFoundError(
                f"None of the requested articles were found in the corpus: {query_identifiers}"
            )

        # Averaging a stack of sparse TF-IDF vectors keeps multi-article queries
        # tractable; for a single-article query this collapses to that vector.
        query_vectors = matrix[query_indices]
        avg_query_vector = np.asarray(query_vectors.mean(axis=0))

        similarities = cosine_similarity(avg_query_vector.reshape(1, -1), matrix).flatten()

        results = self.df.copy()
        results["similarity_score"] = similarities
        results = results.drop(query_indices)
        results = results.sort_values("similarity_score", ascending=False).head(k)

        return results[["title", "similarity_score", "token_count"]]
