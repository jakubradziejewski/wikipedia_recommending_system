import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class ArticleSimilarityEngine:
    """
    Wikipedia Article Similarity Engine using TF-IDF

    This engine computes article similarities using Term Frequency-Inverse Document Frequency (TF-IDF)
    vectorization and cosine similarity metrics.
    """

    def __init__(self, parquet_file='wikipedia_articles.parquet'):
        """
        Initialize the similarity engine and load data

        Args:
            parquet_file: Path to the parquet file containing scraped articles
        """

        self.df = pd.read_parquet(parquet_file)
        print(f"Loaded {len(self.df)} articles from {parquet_file}")

        # Create URL to index mapping for quick lookups
        self.url_to_idx = {url: idx for idx, url in enumerate(self.df['url'])}
        self.title_to_idx = {title.lower(): idx for idx, title in enumerate(self.df['title'])}

        # Initialize TF-IDF vectorizer
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None

        print("Engine initialized successfully\n")

    def build_tfidf_model(self, max_features=10000):
        """
        Build TF-IDF model from the article corpus

        Args:
            max_features: Maximum number of features to extract
        """

        print("Building TF-IDF Model")

        # Choose text column, we use lemmatized text
        text_column = 'lemmas'
        texts = self.df[text_column].fillna('')

        print(f"  - Max features: {max_features}")
        print(f"  - Documents: {len(texts)}")

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=15,  # Ignore terms that appear in less than 15 documents
            max_df=0.60,  # Ignore terms that appear in more than 60% of documents
            sublinear_tf=True  # Use sublinear term frequency scaling
        )

        # Fit and transform
        print("\nComputing TF-IDF matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()


    def _find_article_index(self, identifier: str) -> Optional[int]:
        """Find article index by URL or title"""
        # Try as URL first
        if identifier in self.url_to_idx:
            return self.url_to_idx[identifier]

        # Try as title (case-insensitive)
        identifier_lower = identifier.lower()
        if identifier_lower in self.title_to_idx:
            return self.title_to_idx[identifier_lower]

        # Try partial match on title
        for title, idx in self.title_to_idx.items():
            if identifier_lower in title:
                return idx

        return None


    def find_similar_articles(self,
                              query_identifiers: List[str],
                              top_k=10,
                              exclude_query=True,
                              weights: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Find similar articles given a collection of article titles or URLs

        Args:
            query_identifiers: List of article titles or URLs
            top_k: Number of recommendations to return
            exclude_query: Whether to exclude query articles from results
            weights: Optional list of weights corresponding to query_identifiers
                     (must be same length as query_identifiers)

        Returns:
            DataFrame with recommended articles and similarity scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF model not built. Call build_tfidf_model() first.")

        # Find indices of query articles
        query_indices = []
        found_articles = []

        for identifier in query_identifiers:
            idx = self._find_article_index(identifier)
            if idx is not None:
                query_indices.append(idx)
                found_articles.append(self.df.iloc[idx]['title'])

        if not query_indices:
            print("⚠ No valid articles found in the query!")
            return pd.DataFrame()

        # Get the vectors
        query_vectors = self.tfidf_matrix[query_indices]

        # Handle weights
        if weights:
            if len(weights) != len(query_indices):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number of found query articles ({len(query_indices)}).")


            # Create a numpy array for weights, shaped for broadcasting
            weights_arr = np.array(weights).reshape(-1, 1)  # Shape (n_queries, 1)

            # Use .multiply for element-wise multiplication with broadcasting
            weighted_vectors = query_vectors.multiply(weights_arr)

            # Sum the weighted vectors
            weighted_sum_vector = weighted_vectors.sum(axis=0)

            # Normalize by the sum of weights
            sum_of_weights = np.sum(weights_arr)
            avg_query_vector = np.asarray(weighted_sum_vector / sum_of_weights)
        else:
            # Simple mean
            avg_query_vector = np.asarray(query_vectors.mean(axis=0))

        # Compute similarities
        similarities = cosine_similarity(avg_query_vector.reshape(1, -1), self.tfidf_matrix).flatten()

        # Create results DataFrame
        results_df = self.df.copy()
        results_df['similarity_score'] = similarities

        # Exclude query articles if requested
        if exclude_query:
            results_df = results_df[~results_df.index.isin(query_indices)]

        # Sort by similarity and get top-k
        results_df = results_df.sort_values('similarity_score', ascending=False).head(top_k)

        return results_df[['title', 'url', 'similarity_score', 'token_count']]

