import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CORE ENGINE CLASS
# ============================================================================

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

    # ============================================================================
    # EXPLAINABILITY METHODS
    # ============================================================================

    def explain_similarity(self,
                           query_identifiers: List[str],
                           target_article: str,
                           top_terms=15) -> Dict:
        """
        Explain why a target article is similar to query articles

        Args:
            query_identifiers: List of article titles or URLs (query)
            target_article: Title or URL of article to explain
            top_terms: Number of top contributing terms to show

        Returns:
            Dictionary containing explanation data
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF model not built. Call build_tfidf_model() first.")

        # Find query indices
        query_indices = [self._find_article_index(id) for id in query_identifiers]
        query_indices = [idx for idx in query_indices if idx is not None]

        # Find target index
        target_idx = self._find_article_index(target_article)

        if not query_indices or target_idx is None:
            return {}

        # Get vectors
        query_vectors = self.tfidf_matrix[query_indices]
        avg_query_vector = np.asarray(query_vectors.mean(axis=0)).flatten()
        target_vector = np.asarray(self.tfidf_matrix[target_idx].toarray()).flatten()

        # Compute overall similarity
        similarity = cosine_similarity(
            avg_query_vector.reshape(1, -1),
            target_vector.reshape(1, -1)
        )[0][0]

        # Find top contributing terms (element-wise product)
        contributions = avg_query_vector * target_vector
        top_indices = np.argsort(contributions)[-top_terms:][::-1]

        top_terms_data = []
        for idx in top_indices:
            if contributions[idx] > 0:
                top_terms_data.append({
                    'term': self.feature_names[idx],
                    'contribution': contributions[idx],
                    'query_tfidf': avg_query_vector[idx],
                    'target_tfidf': target_vector[idx]
                })

        return {
            'query_articles': [self.df.iloc[i]['title'] for i in query_indices],
            'target_article': self.df.iloc[target_idx]['title'],
            'similarity_score': similarity,
            'top_terms': top_terms_data
        }

    def visualize_similarity_explanation(self, explanation: Dict, save_path=None):
        """
        Create visualization of similarity explanation

        Args:
            explanation: Output from explain_similarity()
            save_path: Optional path to save the figure
        """
        if not explanation or 'top_terms' not in explanation:
            print("⚠ No explanation data to visualize")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Prepare data
        terms_df = pd.DataFrame(explanation['top_terms'])

        if len(terms_df) == 0:
            print("⚠ No terms to visualize")
            return

        # Plot 1: Term contributions
        ax1 = axes[0]
        bars = ax1.barh(range(len(terms_df)), terms_df['contribution'],
                        color='steelblue', alpha=0.7)
        ax1.set_yticks(range(len(terms_df)))
        ax1.set_yticklabels(terms_df['term'])
        ax1.set_xlabel('Contribution to Similarity', fontsize=11, fontweight='bold')
        ax1.set_title(f'Top Contributing Terms\nSimilarity Score: {explanation["similarity_score"]:.4f}',
                      fontsize=13, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()

        # Add value labels
        for i, (idx, row) in enumerate(terms_df.iterrows()):
            ax1.text(row['contribution'], i, f' {row["contribution"]:.4f}',
                     va='center', fontsize=9)

        # Plot 2: TF-IDF comparison
        ax2 = axes[1]
        x = np.arange(len(terms_df))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, terms_df['query_tfidf'], width,
                        label='Query Articles', color='coral', alpha=0.7)
        bars2 = ax2.bar(x + width / 2, terms_df['target_tfidf'], width,
                        label='Target Article', color='lightgreen', alpha=0.7)

        ax2.set_xlabel('Terms', fontsize=11, fontweight='bold')
        ax2.set_ylabel('TF-IDF Score', fontsize=11, fontweight='bold')
        ax2.set_title('TF-IDF Scores Comparison', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(terms_df['term'], rotation=45, ha='right')
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)

        # Add info text
        query_text = "Query: " + ", ".join(explanation['query_articles'][:2])
        if len(explanation['query_articles']) > 2:
            query_text += f" + {len(explanation['query_articles']) - 2} more"
        target_text = f"Target: {explanation['target_article']}"

        fig.text(0.5, 0.98, query_text, ha='center', fontsize=10,
                 style='italic', wrap=True)
        fig.text(0.5, 0.96, target_text, ha='center', fontsize=10,
                 style='italic', color='darkgreen', wrap=True)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")

        plt.show()

    # ============================================================================
    # STATISTICS & ANALYSIS METHODS
    # ============================================================================

    def generate_database_statistics(self):
        """Generate comprehensive statistics about the article database"""
        print("\n" + "=" * 80)
        print("DATABASE STATISTICS & ANALYSIS")
        print("=" * 80)

        if self.tfidf_matrix is None:
            print("⚠ TF-IDF model not built. Building now...")
            self.build_tfidf_model()

        # 1. TF-IDF Statistics
        print("\n" + "-" * 80)
        print("1. TF-IDF MODEL STATISTICS")
        print("-" * 80)
        print(f"Total documents: {self.tfidf_matrix.shape[0]:,}")
        print(f"Vocabulary size: {self.tfidf_matrix.shape[1]:,}")
        print(
            f"Matrix density: {self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]):.4%}")
        print(f"Non-zero elements: {self.tfidf_matrix.nnz:,}")

        # 2. Top TF-IDF terms across corpus
        print("\n" + "-" * 80)
        print("2. TOP TF-IDF TERMS (Corpus-wide)")
        print("-" * 80)

        # Sum TF-IDF scores across all documents
        tfidf_sums = np.array(self.tfidf_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(tfidf_sums)[-20:][::-1]

        print("\nTop 20 terms by cumulative TF-IDF score:")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i:2d}. {self.feature_names[idx]:20s} (score: {tfidf_sums[idx]:.2f})")

        # 3. Document length distribution
        print("\n" + "-" * 80)
        print("3. DOCUMENT LENGTH DISTRIBUTION")
        print("-" * 80)
        lengths = self.df['token_count']
        print(f"Mean: {lengths.mean():.0f} tokens")
        print(f"Median: {lengths.median():.0f} tokens")
        print(f"Std Dev: {lengths.std():.0f} tokens")
        print(f"Min: {lengths.min()} tokens")
        print(f"Max: {lengths.max()} tokens")

        # 4. Sample similarity matrix
        print("\n" + "-" * 80)
        print("4. SIMILARITY MATRIX PREVIEW (First 10 articles)")
        print("-" * 80)
        sample_size = min(10, len(self.df))
        sample_matrix = self.tfidf_matrix[:sample_size]
        sample_similarity = cosine_similarity(sample_matrix)

        print("\nSimilarity scores (0.0 = different, 1.0 = identical):")
        print("Articles:", ", ".join(self.df.head(sample_size)['title'].str[:20].tolist()))
        print(f"\nAverage similarity: {sample_similarity[np.triu_indices_from(sample_similarity, k=1)].mean():.4f}")
        print(
            f"Max similarity (non-diagonal): {sample_similarity[np.triu_indices_from(sample_similarity, k=1)].max():.4f}")

        # 5. Visualizations
        self._create_statistics_visualizations()

        print("\n" + "=" * 80)

    def _create_statistics_visualizations(self):
        """Create visualization plots for database statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Token count distribution
        ax1 = axes[0, 0]
        token_counts = self.df['token_count']
        ax1.hist(token_counts, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(token_counts.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {token_counts.mean():.0f}')
        ax1.axvline(token_counts.median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {token_counts.median():.0f}')
        ax1.set_xlabel('Token Count', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Articles', fontsize=11, fontweight='bold')
        ax1.set_title('Article Length Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Top terms bar chart
        ax2 = axes[0, 1]
        tfidf_sums = np.array(self.tfidf_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(tfidf_sums)[-15:][::-1]
        top_terms = [self.feature_names[i] for i in top_indices]
        top_scores = [tfidf_sums[i] for i in top_indices]

        ax2.barh(range(len(top_terms)), top_scores, color='coral', alpha=0.7)
        ax2.set_yticks(range(len(top_terms)))
        ax2.set_yticklabels(top_terms)
        ax2.set_xlabel('Cumulative TF-IDF Score', fontsize=11, fontweight='bold')
        ax2.set_title('Top 15 Terms by TF-IDF', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

        # 3. Similarity heatmap (sample)
        ax3 = axes[1, 0]
        sample_size = min(15, len(self.df))
        sample_matrix = self.tfidf_matrix[:sample_size]
        sample_similarity = cosine_similarity(sample_matrix)

        im = ax3.imshow(sample_similarity, cmap='YlOrRd', aspect='auto')
        ax3.set_xticks(range(sample_size))
        ax3.set_yticks(range(sample_size))
        ax3.set_xticklabels([f"A{i + 1}" for i in range(sample_size)], fontsize=8)
        ax3.set_yticklabels([f"A{i + 1}" for i in range(sample_size)], fontsize=8)
        ax3.set_title(f'Similarity Heatmap (First {sample_size} Articles)', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax3, label='Cosine Similarity')

        # 4. TF-IDF sparsity visualization
        ax4 = axes[1, 1]
        sample_matrix = self.tfidf_matrix[:50, :100].toarray()
        im2 = ax4.imshow(sample_matrix, cmap='Blues', aspect='auto', interpolation='nearest')
        ax4.set_xlabel('Features (first 100)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Documents (first 50)', fontsize=11, fontweight='bold')
        ax4.set_title('TF-IDF Matrix Sparsity Pattern', fontsize=13, fontweight='bold')
        plt.colorbar(im2, ax=ax4, label='TF-IDF Score')

        plt.tight_layout()
        plt.savefig('../plots/database_statistics.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'database_statistics.png'")
        plt.show()


# ============================================================================
# STRATEGY COMPARISON FUNCTIONS
# ============================================================================

def compare_recommendation_strategies(engine, num_articles=10):
    """
    Compare recommendations from random vs similar article collections

    Args:
        engine: ArticleSimilarityEngine instance
        num_articles: Number of articles to use in each strategy
    """
    print("\n" + "=" * 80)
    print("COMPARING RECOMMENDATION STRATEGIES")
    print("=" * 80)

    # Strategy 1: Random articles
    print("\n" + "-" * 80)
    print("STRATEGY 1: Random Article Collection")
    print("-" * 80)

    random_indices = np.random.choice(len(engine.df), size=num_articles, replace=False)
    random_titles = engine.df.iloc[random_indices]['title'].tolist()

    print(f"\n📋 Query articles (randomly selected):")
    for i, title in enumerate(random_titles, 1):
        print(f"  {i:2d}. {title}")

    # Get pairwise similarities within random collection
    random_matrix = engine.tfidf_matrix[random_indices]
    random_sim_matrix = cosine_similarity(random_matrix)
    random_avg_sim = random_sim_matrix[np.triu_indices_from(random_sim_matrix, k=1)].mean()

    print(f"\n📊 Internal coherence (avg similarity): {random_avg_sim:.4f}")

    random_recs = engine.find_similar_articles(random_titles, top_k=10)

    print(f"\n🎯 Top 10 Recommendations:")
    for idx, row in random_recs.iterrows():
        print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")

    # Strategy 2: Similar (connected) articles
    print("\n" + "-" * 80)
    print("STRATEGY 2: Similar (Connected) Article Collection")
    print("-" * 80)

    # Start with a random article and find similar ones
    seed_idx = np.random.choice(len(engine.df))
    seed_title = engine.df.iloc[seed_idx]['title']

    print(f"\n🌱 Seed article: {seed_title}")

    # Find similar articles to the seed
    seed_vector = engine.tfidf_matrix[seed_idx]
    similarities = cosine_similarity(seed_vector, engine.tfidf_matrix).flatten()
    similar_indices = np.argsort(similarities)[::-1][1:num_articles + 1]
    similar_titles = engine.df.iloc[similar_indices]['title'].tolist()

    print(f"\n📋 Query articles (similar to seed):")
    for i, (idx, title) in enumerate(zip(similar_indices, similar_titles), 1):
        sim_score = similarities[idx]
        print(f"  {i:2d}. {title} (similarity to seed: {sim_score:.4f})")

    # Get pairwise similarities within similar collection
    similar_matrix = engine.tfidf_matrix[similar_indices]
    similar_sim_matrix = cosine_similarity(similar_matrix)
    similar_avg_sim = similar_sim_matrix[np.triu_indices_from(similar_sim_matrix, k=1)].mean()

    print(f"\n📊 Internal coherence (avg similarity): {similar_avg_sim:.4f}")

    similar_recs = engine.find_similar_articles(similar_titles, top_k=10)

    print(f"\n🎯 Top 10 Recommendations:")
    for idx, row in similar_recs.iterrows():
        print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")

    # Comparison Analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    print(f"\n📈 Internal Coherence:")
    print(f"  Random collection:  {random_avg_sim:.4f}")
    print(f"  Similar collection: {similar_avg_sim:.4f}")
    print(
        f"  Difference:         {similar_avg_sim - random_avg_sim:.4f} ({(similar_avg_sim / random_avg_sim - 1) * 100:+.1f}%)")

    print(f"\n📈 Recommendation Quality:")
    print(f"  Random - Avg score:  {random_recs['similarity_score'].mean():.4f}")
    print(f"  Similar - Avg score: {similar_recs['similarity_score'].mean():.4f}")
    print(f"  Random - Max score:  {random_recs['similarity_score'].max():.4f}")
    print(f"  Similar - Max score: {similar_recs['similarity_score'].max():.4f}")

    # Check overlap in recommendations
    random_rec_set = set(random_recs['title'].tolist())
    similar_rec_set = set(similar_recs['title'].tolist())
    overlap = random_rec_set.intersection(similar_rec_set)

    print(f"\n📊 Recommendation Overlap:")
    print(f"  Common recommendations: {len(overlap)}/10")
    if overlap:
        print(f"  Overlapping articles:")
        for title in overlap:
            print(f"    • {title}")

    # Visualize comparison
    visualize_strategy_comparison(engine, random_indices, similar_indices,
                                  random_recs, similar_recs,
                                  random_avg_sim, similar_avg_sim)

    return {
        'random_titles': random_titles,
        'similar_titles': similar_titles,
        'random_recommendations': random_recs,
        'similar_recommendations': similar_recs,
        'random_coherence': random_avg_sim,
        'similar_coherence': similar_avg_sim
    }


def visualize_strategy_comparison(engine, random_indices, similar_indices,
                                  random_recs, similar_recs,
                                  random_coherence, similar_coherence):
    """Create visualization comparing the two strategies"""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Query collection similarity heatmaps
    ax1 = fig.add_subplot(gs[0, 0])
    random_matrix = engine.tfidf_matrix[random_indices]
    random_sim = cosine_similarity(random_matrix)
    im1 = ax1.imshow(random_sim, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Random Collection\nInternal Similarity', fontweight='bold')
    ax1.set_xlabel('Article Index')
    ax1.set_ylabel('Article Index')
    plt.colorbar(im1, ax=ax1, label='Similarity')
    ax1.text(0.5, -0.15, f'Avg: {random_coherence:.4f}',
             ha='center', transform=ax1.transAxes, fontsize=10, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    similar_matrix = engine.tfidf_matrix[similar_indices]
    similar_sim = cosine_similarity(similar_matrix)
    im2 = ax2.imshow(similar_sim, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Similar Collection\nInternal Similarity', fontweight='bold')
    ax2.set_xlabel('Article Index')
    ax2.set_ylabel('Article Index')
    plt.colorbar(im2, ax=ax2, label='Similarity')
    ax2.text(0.5, -0.15, f'Avg: {similar_coherence:.4f}',
             ha='center', transform=ax2.transAxes, fontsize=10, fontweight='bold')

    # 2. Coherence comparison
    ax3 = fig.add_subplot(gs[0, 2])
    coherence_data = [random_coherence, similar_coherence]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax3.bar(['Random', 'Similar'], coherence_data, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Average Similarity Score', fontweight='bold')
    ax3.set_title('Query Collection Coherence', fontweight='bold')


def demo_similarity_system():
    """Demonstrate the similarity system with examples"""
    print("\n" + "=" * 80)
    print("WIKIPEDIA ARTICLE SIMILARITY SYSTEM - DEMO")
    print("=" * 80)
    print("\n📌 NOTE: Similarity computation uses the 'lemmas' column from wikipedia_articles.parquet")
    print("   This column contains lemmatized tokens (root forms of words)")
    print("   Example: 'running', 'runs', 'ran' → 'run'")
    print("=" * 80)

    # Initialize engine
    engine = ArticleSimilarityEngine('wikipedia_articles.parquet')

    # Build TF-IDF model
    print("\n🔧 Building TF-IDF model using 'lemmas' column...")
    engine.build_tfidf_model(use_lemmas=True, max_features=20000, ngram_range=(1, 2))

    # Generate database statistics
    engine.generate_database_statistics()

    # NEW: Compare strategies
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON EXPERIMENT")
    print("=" * 80)
    comparison_results = compare_recommendation_strategies(engine, num_articles=10)

    # Example 1: Explain similarity for random strategy
    if len(comparison_results['random_recommendations']) > 0:
        print("\n" + "=" * 80)
        print("EXPLAINABILITY: Random Strategy Recommendation")
        print("=" * 80)

        target = comparison_results['random_recommendations'].iloc[0]['title']
        explanation = engine.explain_similarity(
            comparison_results['random_titles'][:3],
            target,
            top_terms=15
        )

        print(f"\nWhy is '{target}' recommended?")
        print("-" * 80)
        print(f"Overall similarity score: {explanation['similarity_score']:.4f}")
        print(f"\nTop contributing terms:")
        for i, term_data in enumerate(explanation['top_terms'][:10], 1):
            print(f"  {i:2d}. '{term_data['term']}' - contribution: {term_data['contribution']:.5f}")

        engine.visualize_similarity_explanation(explanation,
                                                save_path='similarity_explanation_random.png')

    # Example 2: Explain similarity for similar strategy
    if len(comparison_results['similar_recommendations']) > 0:
        print("\n" + "=" * 80)
        print("EXPLAINABILITY: Similar Strategy Recommendation")
        print("=" * 80)

        target = comparison_results['similar_recommendations'].iloc[0]['title']
        explanation = engine.explain_similarity(
            comparison_results['similar_titles'][:3],
            target,
            top_terms=15
        )

        print(f"\nWhy is '{target}' recommended?")
        print("-" * 80)
        print(f"Overall similarity score: {explanation['similarity_score']:.4f}")
        print(f"\nTop contributing terms:")
        for i, term_data in enumerate(explanation['top_terms'][:10], 1):
            print(f"  {i:2d}. '{term_data['term']}' - contribution: {term_data['contribution']:.5f}")

        engine.visualize_similarity_explanation(explanation,
                                                save_path='similarity_explanation_similar.png')
    # Example 3: Weighted query
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Weighted Query (Order Matters)")
    print("=" * 80)
    print("Finding recommendations for the 10 most recently scraped articles.")
    print("The query will be weighted, giving the *last* article the highest importance.")

        # Get 10 last articles from the dataframe
    last_10_articles = engine.df.tail(10)
    query_titles = last_10_articles['title'].tolist()

        # Create a linear weight scheme: [1, 2, 3, ..., 10]
        # The first article in this list (index 0) gets weight 1
        # The last article (index 9) gets weight 10
    query_weights = list(range(1, len(query_titles) + 1))

    print("\n📋 Weighted query articles (Weight: Title):")
    for weight, title in zip(query_weights, query_titles):
        print(f"  {weight:2d}: {title}")

        # Call the modified function with the weights parameter
    weighted_recs = engine.find_similar_articles(
            query_identifiers=query_titles,
            top_k=10,
            weights=query_weights
        )

    print(f"\n🎯 Top 10 Recommendations (from weighted query):")
    for idx, row in weighted_recs.iterrows():
            print(f"  {row['title'][:55]:55s} | Score: {row['similarity_score']:.4f}")
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\n📁 Files generated:")
    print("  • database_statistics.png - Database overview visualizations")
    print("  • strategy_comparison.png - Random vs Similar strategy comparison")
    print("  • similarity_explanation_random.png - Random strategy explanation")
    print("  • similarity_explanation_similar.png - Similar strategy explanation")
    print("\n💡 Column used for similarity: 'lemmas' (lemmatized text)")
    print("=" * 80)


if __name__ == '__main__':
    demo_similarity_system()