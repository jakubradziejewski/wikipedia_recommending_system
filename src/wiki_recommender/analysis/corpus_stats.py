"""Text-level corpus statistics — operates directly on the parquet."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from wiki_recommender.errors import EmptyCorpusError, MissingDataError

log = logging.getLogger(__name__)

_TOP_N = 12


def perform_text_analysis(parquet_path: Path) -> None:
    """Print corpus token / vocab / reduction statistics.

    Output is intentionally print-based: this is user-facing CLI text, not
    structured logging, and the matplotlib analysis modules follow the same
    convention.
    """
    if not parquet_path.exists():
        raise MissingDataError(f"Corpus parquet not found at {parquet_path}.")

    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise EmptyCorpusError(f"Corpus parquet at {parquet_path} is empty.")

    print("Statistics about scraped Wikipedia Article Database")
    print(f"Loaded {len(df)} articles from {parquet_path}")

    print("\n" + "-" * 70)
    print("1. Basic Statistics")
    print("-" * 70)
    print(f"Total articles: {len(df)}")
    print(f"Total tokens: {df['token_count'].sum():,}")
    print(f"Total characters: {df['text_length'].sum():,}")
    print("\nTokens per article:")
    print(f"  - Mean: {df['token_count'].mean():.0f}")
    print(f"  - Median: {df['token_count'].median():.0f}")
    print(f"  - Min: {df['token_count'].min()}")
    print(f"  - Max: {df['token_count'].max()}")
    print("\nText length per article:")
    print(f"  - Mean: {df['text_length'].mean():.0f} characters")
    print(f"  - Median: {df['text_length'].median():.0f} characters")

    print("\n" + "-" * 70)
    print("2. Vocabulary Analysis")
    print("-" * 70)

    # The parquet stores space-joined strings per article — split once,
    # share the lists across the four downstream computations.
    all_tokens = " ".join(df["tokens"].dropna()).split()
    all_stems = " ".join(df["stems"].dropna()).split()
    all_lemmas = " ".join(df["lemmas"].dropna()).split()

    unique_tokens = len(set(all_tokens))
    unique_stems = len(set(all_stems))
    unique_lemmas = len(set(all_lemmas))

    print(f"Unique tokens (vocabulary size): {unique_tokens:,}")
    print(f"Unique stems: {unique_stems:,}")
    print(f"Unique lemmas: {unique_lemmas:,}")
    print(f"Total word occurrences: {len(all_tokens):,}")

    print("\n Top 12 most frequent Tokens, Stems, and Lemmas with their Count:")
    top_tokens = Counter(all_tokens).most_common(_TOP_N)
    top_stems = Counter(all_stems).most_common(_TOP_N)
    top_lemmas = Counter(all_lemmas).most_common(_TOP_N)

    rows = []
    for i in range(_TOP_N):
        tok, tok_c = top_tokens[i] if i < len(top_tokens) else ("", 0)
        stem, stem_c = top_stems[i] if i < len(top_stems) else ("", 0)
        lem, lem_c = top_lemmas[i] if i < len(top_lemmas) else ("", 0)
        rows.append([i + 1, tok, tok_c, i + 1, stem, stem_c, i + 1, lem, lem_c])

    table = pd.DataFrame(
        rows,
        columns=["#", "Token", "Token Count", "#", "Stem", "Stem Count", "#", "Lemma", "Lemma Count"],
    )
    print(table.to_string(index=False))

    print("\n" + "-" * 70)
    print("3. Text Reduction Analysis")
    print("-" * 70)
    if unique_tokens > 0:
        reduction_stem = (1 - unique_stems / unique_tokens) * 100
        reduction_lemma = (1 - unique_lemmas / unique_tokens) * 100
        print(f"Vocabulary reduction through stemming: {reduction_stem:.1f}%")
        print(f"Vocabulary reduction through lemmatization: {reduction_lemma:.1f}%")
