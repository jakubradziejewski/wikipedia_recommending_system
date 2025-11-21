import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class ArticleSimilarityEngine:
    """
    Wikipedia Article Similarity engine computes article similarities using Term Frequency-Inverse Document Frequency (TF-IDF)
    vectorization and cosine similarity metrics.
    """

    def __init__(self, parquet_file='wikipedia_articles.parquet'):
        """Initialize the similarity engine and load data"""

        self.df = pd.read_parquet(parquet_file)
        print(f"Loaded {len(self.df)} articles from {parquet_file}")

        self.title_to_idx = {title.lower(): idx for idx, title in enumerate(self.df['title'])}

        # Initialize TF-IDF vectorizer
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None

        print("Engine initialized successfully\n")

    def build_tfidf_model(self, max_features):
        """Build TF-IDF model from the article corpus"""

        print("Building TF-IDF Model")

        # Using lemmatized text, here it can be changed to 'tokens' or 'stems' if needed
        text_column = 'lemmas'
        texts = self.df[text_column].fillna('')

        print(f"Max features: {max_features}")
        print(f"Documents: {len(texts)}")

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2), # also includes double word phrases, can be changed to (1,1) for single words only
            min_df=15,  # Ignore terms that appear in less than 15 documents
            max_df=0.60,  # Ignore terms that appear in more than 60% of documents
            sublinear_tf=True  # logarithmic scaling so long articles don't dominate
        )

        # Create TF-IDF matrix (article-term matrix)
        print("\nComputing TF-IDF matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def _find_article_index(self, identifier: str) -> Optional[int]:
        """Find article index by title"""
        identifier_lower = identifier.lower()
        if identifier_lower in self.title_to_idx:
            return self.title_to_idx[identifier_lower]
        return None

    def find_similar_articles(self,
                              query_identifiers: List[str],
                              top_k=10) -> pd.DataFrame:
        """Find similar articles given a collection of article titles"""

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
            print("No valid articles found in the query!")
            return pd.DataFrame()

        # Get the vectors for the query articles
        query_vectors = self.tfidf_matrix[query_indices]
        # Take average vector of the query articles
        avg_query_vector = np.asarray(query_vectors.mean(axis=0))

        # Compute similarities using cosine similarity
        similarities = cosine_similarity(avg_query_vector.reshape(1, -1), self.tfidf_matrix).flatten()

        results_df = self.df.copy()
        results_df['similarity_score'] = similarities
        # Exclude query articles from results
        results_df = results_df.drop(query_indices)
        # Sort by similarity and get top-k
        results_df = results_df.sort_values('similarity_score', ascending=False).head(top_k)

        return results_df[['title', 'similarity_score', 'token_count']]
