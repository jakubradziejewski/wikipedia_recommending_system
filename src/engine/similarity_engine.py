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
        print("=" * 80)
        print("INITIALIZING ARTICLE SIMILARITY ENGINE")
        print("=" * 80)

        self.df = pd.read_parquet(parquet_file)
        print(f"✓ Loaded {len(self.df)} articles from {parquet_file}")

        # Create URL to index mapping for quick lookups
        self.url_to_idx = {url: idx for idx, url in enumerate(self.df['url'])}
        self.title_to_idx = {title.lower(): idx for idx, title in enumerate(self.df['title'])}

        # Initialize TF-IDF vectorizer
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None

        print("✓ Engine initialized successfully\n")

    def build_tfidf_model(self, use_lemmas=True, max_features=10000, ngram_range=(1, 2)):
        """
        Build TF-IDF model from the article corpus

        Args:
            use_lemmas: Whether to use lemmatized text (True) or regular tokens (False)
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to consider (1,1) for unigrams, (1,2) for unigrams+bigrams
        """
        print("-" * 80)
        print("BUILDING TF-IDF MODEL")
        print("-" * 80)

        # Choose text column
        text_column = 'lemmas' if use_lemmas else 'tokens'
        texts = self.df[text_column].fillna('')

        print(f"Configuration:")
        print(f"  - Text type: {'Lemmatized' if use_lemmas else 'Tokenized'}")
        print(f"  - Max features: {max_features}")
        print(f"  - N-gram range: {ngram_range}")
        print(f"  - Documents: {len(texts)}")

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            sublinear_tf=True  # Use sublinear term frequency scaling
        )

        # Fit and transform
        print("\n⏳ Computing TF-IDF matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"✓ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"✓ Vocabulary size: {len(self.feature_names)}")
        print(
            f"✓ Matrix density: {self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]):.4%}")
        print()

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

    # ============================================================================
    # RECOMMENDATION METHODS
    # ============================================================================

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

        print(f"\n🔍 Query articles found: {len(query_indices)}")
        for i, title in enumerate(found_articles):
            if weights:
                print(f"  • {title} (weight: {weights[i]:.2f})")
            else:
                print(f"  • {title}")

        # Get the vectors
        query_vectors = self.tfidf_matrix[query_indices]

        # Handle weights
        if weights:
            if len(weights) != len(query_indices):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number of found query articles ({len(query_indices)}).")

            print(f"\n📊 Applying custom weights")

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
            print("\n📊 Applying uniform weights (standard average)")
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

