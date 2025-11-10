import os
import pandas as pd
from collections import Counter


def perform_text_analysis(parquet_file="data/wikipedia_articles.parquet"):
    """Perform comprehensive text analysis on scraped data"""

    print("\n" + "=" * 70)
    print("TEXT PROCESSING & ANALYSIS")
    print("=" * 70)

    if not os.path.exists(parquet_file):
        print(f"✗ Error: {parquet_file} not found!")
        return

    # Load data
    print(f"\n📊 Loading data from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    print(f"✓ Loaded {len(df)} articles")

    # Basic statistics
    print("\n" + "-" * 70)
    print("1. BASIC STATISTICS")
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
    print("2. VOCABULARY ANALYSIS")
    print("-" * 70)

    all_tokens = ' '.join(df['tokens'].dropna()).split()
    all_stems = ' '.join(df['stems'].dropna()).split()
    all_lemmas = ' '.join(df['lemmas'].dropna()).split()

    print(f"Unique tokens (vocabulary size): {len(set(all_tokens)):,}")
    print(f"Unique stems: {len(set(all_stems)):,}")
    print(f"Unique lemmas: {len(set(all_lemmas)):,}")
    print(f"Total word occurrences: {len(all_tokens):,}")

    # Most common words
    token_freq = Counter(all_tokens)
    print(f"\nTop 20 most frequent tokens:")
    for i, (word, count) in enumerate(token_freq.most_common(20), 1):
        print(f"  {i:2d}. {word:15s} ({count:5d} occurrences)")

    # Stem analysis
    print("\n" + "-" * 70)
    print("3. STEMMING ANALYSIS")
    print("-" * 70)
    stem_freq = Counter(all_stems)
    print(f"Top 15 most frequent stems:")
    for i, (stem, count) in enumerate(stem_freq.most_common(15), 1):
        print(f"  {i:2d}. {stem:15s} ({count:5d} occurrences)")

    # Lemma analysis
    print("\n" + "-" * 70)
    print("4. LEMMATIZATION ANALYSIS")
    print("-" * 70)
    lemma_freq = Counter(all_lemmas)
    print(f"Top 15 most frequent lemmas:")
    for i, (lemma, count) in enumerate(lemma_freq.most_common(15), 1):
        print(f"  {i:2d}. {lemma:15s} ({count:5d} occurrences)")

    # Compression analysis
    print("\n" + "-" * 70)
    print("5. TEXT NORMALIZATION EFFICIENCY")
    print("-" * 70)
    vocab_reduction_stem = (1 - len(set(all_stems)) / len(set(all_tokens))) * 100
    vocab_reduction_lemma = (1 - len(set(all_lemmas)) / len(set(all_tokens))) * 100
    print(f"Vocabulary reduction through stemming: {vocab_reduction_stem:.1f}%")
    print(f"Vocabulary reduction through lemmatization: {vocab_reduction_lemma:.1f}%")

    # Article samples
    print("\n" + "-" * 70)
    print("6. SAMPLE ARTICLES")
    print("-" * 70)
    print("First 10 articles:")
    for i, (idx, row) in enumerate(df.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['title']} ({row['token_count']} tokens)")

    print("\nLast 10 articles:")
    for i, (idx, row) in enumerate(df.tail(10).iterrows(), 1):
        print(f"  {len(df) - 10 + i:2d}. {row['title']} ({row['token_count']} tokens)")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
