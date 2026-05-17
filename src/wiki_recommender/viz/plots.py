"""Matplotlib output for the analysis pipeline.

Each function returns the produced ``Figure`` so callers can embed it
elsewhere (a future Streamlit dashboard, a Jupyter notebook); the side
effect of saving to disk is opt-in via ``save_path``.
"""

from __future__ import annotations

import logging
import textwrap
import urllib.parse
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# Labels alternate between three vertical offsets to keep adjacent bar
# segments' text from colliding. Three positions is enough for the typical
# top-5 / top-20 segment counts we draw here.
_LABEL_OFFSETS: tuple[float, ...] = (0.35, -0.35, 0.0)


def _label_offset(cycle_idx: int) -> float:
    return _LABEL_OFFSETS[cycle_idx % len(_LABEL_OFFSETS)]


def _wrapped_label(rank: int, title: str, width: int = 50) -> str:
    decoded = urllib.parse.unquote(title)
    return "\n".join(textwrap.wrap(f"#{rank}  {decoded}", width=width))


def _save_figure(fig: Figure, save_path: Path | None, dpi: int) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    log.info("Saved figure to %s", save_path)


def plot_contribution_analysis(
    articles_analysis: Sequence[Mapping[str, Any]],
    strategy_name: str = "Strategy",
    save_path: Path | None = None,
    dpi: int = 300,
) -> Figure:
    """Stacked horizontal bars showing each recommendation's top-5 contributing terms.

    Bar widths are percentages of the recommendation's total similarity score,
    so the entire bar reads as "how much of why this article was recommended
    is explained by these five terms".
    """
    num_articles = len(articles_analysis)
    fig, ax = plt.subplots(figsize=(18, 15))

    base_colors = sns.color_palette("muted", 10)
    y_positions = np.arange(num_articles)[::-1] * 1.2

    # Find the widest x extent so the right-side annotations have room without
    # padding the entire axis with empty space when bars are narrow.
    cumulative_percentages: list[float] = []
    for analysis in articles_analysis:
        contributors = analysis.get("term_contributions", [])
        similarity = analysis["similarity"]
        if contributors and similarity > 0:
            cumulative_pct = sum(c["contribution"] / similarity * 100 for c in contributors[:5])
        else:
            cumulative_pct = 0.0
        cumulative_percentages.append(cumulative_pct)
    max_cumulative = max(cumulative_percentages, default=0.0)
    x_limit = max_cumulative * 1.25 if max_cumulative > 0 else 1.0

    for idx, analysis in enumerate(articles_analysis):
        y_pos = y_positions[idx]
        contributors = analysis.get("term_contributions", [])
        similarity = analysis["similarity"]

        if not contributors or similarity <= 0:
            continue

        left = 0.0
        for term_idx, contrib in enumerate(contributors[:5]):
            width_pct = (contrib["contribution"] / similarity) * 100
            color = base_colors[term_idx % len(base_colors)]

            ax.barh(
                y_pos, width_pct, left=left, height=0.8,
                label=contrib["term"] if idx == 0 else "",
                color=color, edgecolor="white", linewidth=2,
            )

            ax.text(
                left + width_pct / 2,
                y_pos + _label_offset(term_idx),
                f"{contrib['term']}\n{width_pct:.1f}%",
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.85,
                          edgecolor="gray", linewidth=1),
            )
            left += width_pct

        ax.text(
            left + max_cumulative * 0.02, y_pos, f"{left:.1f}%",
            ha="left", va="center", fontsize=10, fontweight="bold", color="darkgreen",
        )

        cumulative_raw = sum(c["contribution"] for c in contributors[:5])
        ax.text(
            left + max_cumulative * 0.09, y_pos,
            f"{cumulative_raw:.4f} / {similarity:.4f}",
            ha="left", va="center", fontsize=9, fontweight="bold", color="red",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="lightyellow", alpha=0.8, edgecolor="red"),
        )

    y_labels = [_wrapped_label(a["rank"], a["title"]) for a in articles_analysis]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9, linespacing=1.3)

    ax.set_xlabel("% of Terms Contribution to Total Similarity Score", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Top 5 Contributing Terms for Each Recommendation using {strategy_name}\n"
        "Each bar shows how much of the similarity score is explained by top 5 most impactful terms",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.set_xlim(0, x_limit)
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05)
    _save_figure(fig, save_path, dpi)
    return fig


def plot_distinctive_term_frequency(
    analyses: Sequence[Mapping[str, Any]],
    distinctive_terms: Sequence[Mapping[str, Any]],
    strategy_name: str = "Strategy",
    save_path: Path | None = None,
    dpi: int = 300,
) -> Figure | None:
    """Per-recommendation contribution of the 20 rarest query terms.

    Returns ``None`` if there's nothing to plot — empty inputs short-circuit
    before any axes setup so callers can detect "nothing visualised".
    """
    if not analyses or not distinctive_terms:
        log.warning("plot_distinctive_term_frequency: no data, skipping.")
        return None

    top_20_rare = list(distinctive_terms[:20])
    rare_term_names = [t["term"] for t in top_20_rare]
    term_meta = {t["term"]: t for t in top_20_rare}
    colors = sns.color_palette("muted", 20)

    fig, ax = plt.subplots(figsize=(18, 14))
    y_positions = np.arange(len(analyses))[::-1] * 1.2
    article_labels: list[str] = []

    for i, analysis in enumerate(analyses):
        y_pos = y_positions[i]
        article_labels.append(_wrapped_label(analysis["rank"], analysis["title"]))

        current_matches = {m["term"]: m["contribution"] for m in analysis["distinctive_matches"]}

        left = 0.0
        cycle = 0
        for term_idx, term in enumerate(rare_term_names):
            contrib = current_matches.get(term, 0)
            if contrib <= 0:
                continue

            meta = term_meta[term]
            rarity_pct = (meta["document_frequency"] / meta["total_docs"]) * 100
            enrichment = meta["enrichment_ratio"]

            ax.barh(
                y_pos, contrib, left=left, height=0.9,
                color=colors[term_idx], edgecolor="white", linewidth=2,
            )

            label_text = f"{term}\n{rarity_pct:.1f}% | {enrichment:.0f}x"
            ax.text(
                left + contrib / 2, y_pos + _label_offset(cycle),
                label_text, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", alpha=0.9,
                          edgecolor="gray", linewidth=0.5),
            )
            cycle += 1
            left += contrib

        if left > 0:
            ax.text(
                left * 1.02, y_pos, f"{left:.4f}",
                ha="left", va="center", fontsize=10, fontweight="bold", color="darkred",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="lightyellow", alpha=0.9,
                          edgecolor="red", linewidth=1.5),
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(article_labels, fontsize=9, linespacing=1.3)
    ax.set_xlabel(
        "Cumulative Contribution for Recommended Article using 20 Rarest Terms from Query Set",
        fontsize=13, fontweight="bold",
    )
    ax.set_title(
        f"Impact of Rarest Query Terms on Each Recommendation — {strategy_name}\n"
        "Labels show: term name | corpus frequency % | enrichment ratio",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.20, right=0.95, top=0.95, bottom=0.05)
    _save_figure(fig, save_path, dpi)
    return fig


def plot_similarity_distribution(
    similarity_matrix: NDArray[np.floating],
    save_path: Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, float, float]:
    """Histogram + KDE of the corpus pairwise cosine similarities.

    Returns the figure along with mean and median for convenience — the
    caller almost always wants those numbers in addition to the plot.
    """
    similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    mean_sim = float(np.mean(similarities))
    median_sim = float(np.median(similarities))
    max_sim = float(np.max(similarities))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(similarities, bins=100, kde=True, ax=ax, color="skyblue", edgecolor="black")

    ax.axvline(mean_sim, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_sim:.4f}")
    ax.axvline(median_sim, color="green", linestyle="-", linewidth=1.5,
               label=f"Median: {median_sim:.4f}")

    ax.set_title("Distribution of All Pairwise Cosine Similarities in Corpus",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()

    summary = (
        f"Total pairs: {len(similarities):,}\n"
        f"Mean pairwise similarity: {mean_sim:.4f}\n"
        f"Median pairwise similarity: {median_sim:.4f}\n"
        f"Maximum similarity: {max_sim:.4f}"
    )
    ax.text(
        0.95, 0.95, summary, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )
    fig.tight_layout()
    _save_figure(fig, save_path, dpi)
    return fig, mean_sim, median_sim
