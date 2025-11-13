# engine/statistics.py
import numpy as np
import os
import pandas as pd
from collections import Counter


def perform_text_analysis(parquet_file="data/wikipedia_articles.parquet"):
    """Perform comprehensive text analysis on scraped data"""

    print("Statistics about scraped Wikipedia Article Database")

    if not os.path.exists(parquet_file):
        print(f"✗ Error: {parquet_file} not found!")
        return

    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} articles from {parquet_file}")

    # Basic statistics
    print("\n" + "-" * 70)
    print("1. Basic Statistics")
    print("-" * 70)
    print(f"Total articles: {len(df)}")
    print(f"Total tokens: {df['token_count'].sum():,}")
    print(f"Total characters: {df['text_length'].sum():,}")
    print(f"\nTokens per article:")
    print(f"  - Mean: {df['token_count'].mean():.0f}")
    print(f"  - Median: {df['token_count'].median():.0f}")
    print(f"  - Min: {df['token_count'].min()}")
    print(f"  - Max: {df['token_count'].max()}")
    print(f"\nText length per article:")
    print(f"  - Mean: {df['text_length'].mean():.0f} characters")
    print(f"  - Median: {df['text_length'].median():.0f} characters")

    # Vocabulary analysis
    print("\n" + "-" * 70)
    print("2. Vocabulary Analysis")
    print("-" * 70)

    all_tokens = ' '.join(df['tokens'].dropna()).split()
    all_stems = ' '.join(df['stems'].dropna()).split()
    all_lemmas = ' '.join(df['lemmas'].dropna()).split()

    print(f"Unique tokens (vocabulary size): {len(set(all_tokens)):,}")
    print(f"Unique stems: {len(set(all_stems)):,}")
    print(f"Unique lemmas: {len(set(all_lemmas)):,}")
    print(f"Total word occurrences: {len(all_tokens):,}")


    print(f"\n Top 12 most frequent Tokens, Stems, and Lemmas with their Count:")
    token_freq = Counter(all_tokens)
    stem_freq = Counter(all_stems)
    lemma_freq = Counter(all_lemmas)

    # Take top 12 from each
    top_n = 12
    top_tokens = token_freq.most_common(top_n)
    top_stems = stem_freq.most_common(top_n)
    top_lemmas = lemma_freq.most_common(top_n)

    # Combine into a table
    data = []
    for i in range(top_n):
        token, token_count = top_tokens[i] if i < len(top_tokens) else ("", 0)
        stem, stem_count = top_stems[i] if i < len(top_stems) else ("", 0)
        lemma, lemma_count = top_lemmas[i] if i < len(top_lemmas) else ("", 0)
        data.append([i + 1, token, token_count, i + 1, stem, stem_count, i + 1, lemma, lemma_count])

    df = pd.DataFrame(
        data,
        columns=["#", "Token", "Token Count", "#", "Stem", "Stem Count", "#", "Lemma", "Lemma Count"]
    )
    print(df.to_string(index=False))

    print("\n" + "-" * 70)
    print("3. Text Reduction Analysis")
    print("-" * 70)

    vocab_reduction_stem = (1 - len(set(all_stems)) / len(set(all_tokens))) * 100
    vocab_reduction_lemma = (1 - len(set(all_lemmas)) / len(set(all_tokens))) * 100

    print(f"Vocabulary reduction through stemming: {vocab_reduction_stem:.1f}%")
    print(f"Vocabulary reduction through lemmatization: {vocab_reduction_lemma:.1f}%")

def generate_model_statistics(engine):
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
