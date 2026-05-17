"""User-facing print formatting for strategy comparison.

Kept separate from :mod:`wiki_recommender.engine.strategies` so the strategies
remain pure: they return data, this module turns data into stdout.
"""

from __future__ import annotations

import urllib.parse

import numpy as np
import pandas as pd

from wiki_recommender.engine.strategies import StrategyResult

_DIVIDER = "=" * 80


def _decode_title(title: str) -> str:
    return urllib.parse.unquote(title)


def print_article_list(titles: list[str], label: str = "Articles") -> None:
    print(f"\n{label}:")
    for i, title in enumerate(titles, 1):
        print(f"  {i:2d}. {_decode_title(title)}")


def print_recommendations(recs: pd.DataFrame, label: str = "Top Recommendations") -> None:
    print(f"\n{label}:")
    for _, row in recs.iterrows():
        title = _decode_title(row["title"])
        print(f"  {title[:55]:55s} | Score: {row['similarity_score']:.4f}")


def describe_strategy(result: StrategyResult) -> None:
    """Render a single trial of a single strategy as the original verbose output."""
    print(f"\n{result.name}")
    print_article_list(result.query_titles, label="Query articles")
    print(f"\nInternal coherence (avg similarity): {result.coherence:.4f}")
    print_recommendations(result.recommendations)


def summarize_comparison(trials: list[list[StrategyResult]]) -> None:
    """Print the cross-strategy comparison table aggregated across trials.

    For a single trial, raw values are printed. For multiple trials, each
    metric is reported as mean ± standard deviation so the variation between
    random seeds is visible.
    """
    if not trials:
        return

    num_trials = len(trials)
    strategy_names = [s.name for s in trials[0]]

    # trials[t][s] → StrategyResult; transpose to per-strategy series.
    by_strategy: dict[str, dict[str, list[float]]] = {
        name: {"coherence": [], "max_sim": [], "mean_sim": []} for name in strategy_names
    }
    for trial in trials:
        for result in trial:
            metrics = by_strategy[result.name]
            metrics["coherence"].append(result.coherence)
            metrics["max_sim"].append(result.max_sim)
            metrics["mean_sim"].append(result.mean_sim)

    print("\n" + _DIVIDER)
    print("Comparison of use cases -  different recommendation strategies")
    print(_DIVIDER)

    if num_trials > 1:
        print(f"\n(Averaged over {num_trials} trials)")

        _print_block("Internal Coherence within query articles", by_strategy, "coherence", with_std=True)
        _print_block("Maximum Similarity Score (first recommendation)", by_strategy, "max_sim", with_std=True)
        _print_block("Recommendation Quality (average score of recommendations)", by_strategy, "mean_sim", with_std=True)
    else:
        _print_block("Internal Coherence within query articles", by_strategy, "coherence", with_std=False)
        _print_block("Maximum Similarity Score (first recommendation)", by_strategy, "max_sim", with_std=False)
        _print_block("Recommendation Quality (average score of recommendations)", by_strategy, "mean_sim", with_std=False)


def _print_block(
    heading: str,
    by_strategy: dict[str, dict[str, list[float]]],
    metric: str,
    *,
    with_std: bool,
) -> None:
    print(f"\n{heading}:")
    for name, metrics in by_strategy.items():
        values = metrics[metric]
        label = name.replace(" Strategy", "")
        if with_std:
            print(f"  {label:10s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        else:
            print(f"  {label:10s}: {values[0]:.4f}")
