# engine/statistics.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_database_statistics(engine):
    """Generate comprehensive statistics and visualizations for the article corpus"""

    if engine.tfidf_matrix is None:
        print("⚠ TF-IDF model not built. Building now...")
        engine.build_tfidf_model()

    tfidf = engine.tfidf_matrix
    df = engine.df

    # 1. TF-IDF Statistics
    print("\n" + "-" * 80)
    print("TF-IDF MODEL STATISTICS")
    print("-" * 80)
    print(f"Total documents: {tfidf.shape[0]:,}")
    print(f"Vocabulary size: {tfidf.shape[1]:,}")
    density = tfidf.nnz / (tfidf.shape[0] * tfidf.shape[1])
    print(f"Matrix density: {density:.4%}")
    print(f"Non-zero elements: {tfidf.nnz:,}")

    # 2. Top TF-IDF terms across corpus
    print("\n" + "-" * 80)
    print("TOP TF-IDF TERMS (Corpus-wide)")
    print("-" * 80)
    tfidf_sums = np.array(tfidf.sum(axis=0)).flatten()
    top_indices = np.argsort(tfidf_sums)[-20:][::-1]
    print("\nTop 20 terms by cumulative TF-IDF score:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i:2d}. {engine.feature_names[idx]:20s} (score: {tfidf_sums[idx]:.2f})")

    # 4. Sample similarity matrix
    print("\n" + "-" * 80)
    print("SIMILARITY MATRIX PREVIEW (First 5 articles)")
    print("-" * 80)
    sample_size = min(5, len(df))
    sample_matrix = tfidf[:sample_size]
    sample_similarity = cosine_similarity(sample_matrix)
    print("\nSimilarity scores (0.0 = different, 1.0 = identical):")
    print("Articles:", ", ".join(df.head(sample_size)['title'].str[:20].tolist()))
    upper_indices = np.triu_indices_from(sample_similarity, k=1)
    print(f"\nAverage similarity: {sample_similarity[upper_indices].mean():.4f}")
    print(f"Max similarity (non-diagonal): {sample_similarity[upper_indices].max():.4f}")



