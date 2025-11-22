import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import urllib.parse
import textwrap

def plot_contribution_analysis(articles_analysis, strategy_name="Strategy", save_path=None):
    """
    Visualize the most impactful 5 terms which made an article be recommended using horizontal stacked bars.
    Each recommended article gets its own stacked bar showing its top 5 contributing terms as a percentage of total similarity score.
    """

    num_of_articles = len(articles_analysis)
    # Create a large figure to accommodate all bars and labels
    fig, ax = plt.subplots(figsize=(18, 15))

    # Color palette - use a distinctive color scheme, possible showing up to 10 different colours
    base_colors = sns.color_palette("muted", 10)
    # Y positions for each article (reversed, so top article is at the top)
    y_positions = np.arange(num_of_articles)[::-1] * 1.2

    # Compute max cumulative percentage for x-axis limit (otherwise 80% of the plot space is empty)
    max_cumulative_percentage = 0
    cumulative_percentages = []

    for analysis in articles_analysis:
        contributors = analysis.get('term_contributions', [])
        similarity = analysis['similarity']

        if contributors and similarity > 0:
            # Contributions converted to percentage, so it's easier to understand visually
            cumulative_pct = sum(c['contribution'] / similarity * 100 for c in contributors[:5])
            cumulative_percentages.append(cumulative_pct)
            max_cumulative_percentage = max(max_cumulative_percentage, cumulative_pct)
        else:
            cumulative_percentages.append(0)

    # 25% padding to the right
    x_limit = max_cumulative_percentage * 1.25

    # Stacked bars for each article
    for idx, analysis in enumerate(articles_analysis):
        y_pos = y_positions[idx]
        contributors = analysis.get('term_contributions', [])
        similarity = analysis['similarity']

        # Top 5 most important terms for this recommendation
        left = 0
        for term_idx, contrib in enumerate(contributors[:5]):
            # Contributions converted to percentage, so it's easier to understand visually
            width_pct = (contrib['contribution'] / similarity) * 100
            color = base_colors[term_idx % len(base_colors)]

            # Create the bar segment
            ax.barh(y_pos, width_pct, left=left, height=0.8,
                    label=contrib['term'] if idx == 0 else "",
                    color=color, edgecolor='white', linewidth=2)

            # Add term label, alternating the place so the labels don't overlap
            cycle = term_idx % 3
            if cycle == 0:
                vertical_offset = +0.35
            elif cycle == 1:
                vertical_offset = -0.35
            else:
                vertical_offset = 0

            ax.text(
                left + width_pct / 2,
                y_pos + vertical_offset,
                f"{contrib['term']}\n{width_pct:.1f}%",
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.85,
                          edgecolor='gray', linewidth=1)
            )
            left += width_pct

        # Add total percentage at the end of the bar
        ax.text(left + max_cumulative_percentage * 0.02, y_pos, f'{left:.1f}%',
                ha='left', va='center', fontsize=10, fontweight='bold',
                color='darkgreen')

        # Calculate cumulative raw contribution sum
        cumulative_raw = sum(c['contribution'] for c in contributors[:5])

        # Add cumulative contribution and similarity score on the right side
        ax.text(left + max_cumulative_percentage * 0.09, y_pos,
                f'{cumulative_raw:.4f} / {similarity:.4f}',
                ha='left', va='center', fontsize=9, fontweight='bold',
                color='red',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='lightyellow', alpha=0.8, edgecolor='red'))

    # Set y-axis labels with full article titles (wrapped to multiple lines)
    y_labels = []
    for a in articles_analysis:
        title = urllib.parse.unquote(a['title'])
        # Wrap title to ~50 characters per line
        wrapped = '\n'.join(textwrap.wrap(f"#{a['rank']}  {title}", width=50))
        y_labels.append(wrapped)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9, linespacing=1.3)

    # Formatting
    ax.set_xlabel('% of Terms Contribution to Total Similarity Score ', fontsize=13, fontweight='bold')
    ax.set_title(f'Top 5 Contributing Terms for Each Recommendation using {strategy_name}\n' +
                 'Each bar shows how much of the similarity score is explained by top 5 most impactful terms',
                 fontsize=14, fontweight='bold', pad=20)

    # X-axis limit based on max cumulative percentage
    ax.set_xlim(0, x_limit)

    # Slight vertical grid lines
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout with extra space for wrapped titles
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nContribution plot saved to: {save_path}")


def plot_distinctive_term_frequency(analyses, distinctive_terms, strategy_name="Strategy", save_path=None):
    """
    Visualize contribution of 20 rarest terms to each recommended article.
    """
    if not analyses or not distinctive_terms:
        print("No data to visualize.")
        return

    # Top 20 most distinctive (rarest) terms in query set
    top_20_rare = distinctive_terms[:20]
    rare_term_names = [t['term'] for t in top_20_rare]

    # Lookup map for term metadata
    term_map = {t['term']: t for t in top_20_rare}

    colors = sns.color_palette("muted", 20)

    num_articles = len(analyses)
    fig, ax = plt.subplots(figsize=(18, 14))

    # Y positions (reversed so the most similar article is at top)
    y_positions = np.arange(num_articles)[::-1] * 1.2
    article_labels = []

    # Iterate through each recommended article
    for i, analysis in enumerate(analyses):
        y_pos = y_positions[i]

        title = urllib.parse.unquote(analysis['title'])
        wrapped_title = '\n'.join(textwrap.wrap(f"#{analysis['rank']} {title}", width=50))
        article_labels.append(wrapped_title)

        # Lookup for current article's distinctive term contributions
        current_matches = {m['term']: m['contribution'] for m in analysis['distinctive_matches']}

        left = 0
        cycle = 0
        for term_idx, term in enumerate(rare_term_names):
            # Get contribution (default to 0 if not found)
            contrib = current_matches.get(term, 0)
            if contrib > 0:
                # Get metadata for labels
                metadata = term_map[term]
                doc_freq = metadata['document_frequency']
                total_docs = metadata['total_docs']
                enrichment = metadata['enrichment_ratio']
                rarity_pct = (doc_freq / total_docs) * 100

                ax.barh(y_pos, contrib, left=left, height=0.9,
                        color=colors[term_idx],
                        edgecolor='white', linewidth=2)

                if cycle == 0:
                    vertical_offset = +0.35
                elif cycle == 1:
                    vertical_offset = -0.35
                else:
                    vertical_offset = 0

                # Change the position of label for next term, so they don't overlap
                cycle = (cycle + 1) % 3
                label_text = f'{term}\n{rarity_pct:.1f}% | {enrichment:.0f}x'

                ax.text(left + contrib / 2, y_pos + vertical_offset,
                        label_text,
                        ha='center', va='center', fontsize=6.5,
                        fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.9,
                                  edgecolor='gray', linewidth=0.5))

                left += contrib

        # Add the total sum number at the end of the bar
        if left > 0:
            ax.text(left * 1.02, y_pos, f'{left:.4f}',
                    ha='left', va='center', fontsize=10,
                    fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='lightyellow', alpha=0.9,
                              edgecolor='red', linewidth=1.5))

    ax.set_yticks(y_positions)
    ax.set_yticklabels(article_labels, fontsize=9, linespacing=1.3)

    ax.set_xlabel('Cumulative Contribution for Recommended Article using 20 Rarest Terms from Query Set',
                  fontsize=13, fontweight='bold')

    ax.set_title(f'Impact of Rarest Query Terms on Each Recommendation — {strategy_name}\n'
                 f'Labels show: term name | corpus frequency % | enrichment ratio',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplots_adjust(left=0.20, right=0.95, top=0.95, bottom=0.05)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nRare terms contribution plot saved to: {save_path}")

def plot_similarity_distribution(similarity_matrix, save_path=None):
    """
    Visualize the distribution of pairwise cosine similarities in the corpus.
    The distribution is calculated from the upper triangle of the similarity matrix
    (excluding the diagonal).
    """
    # Extract pairwise similarities from the upper triangle (k=1 excludes diagonal)
    similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    max_sim = np.max(similarities)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(similarities, bins=100, kde=True, ax=ax, color='skyblue', edgecolor='black')

    ax.axvline(mean_sim, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_sim:.4f}')
    ax.axvline(median_sim, color='green', linestyle='-', linewidth=1.5,
               label=f'Median: {median_sim:.4f}')

    ax.set_title('Distribution of All Pairwise Cosine Similarities in Corpus',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Cosine Similarity Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()

    text_summary = (
        f"Total pairs: {len(similarities):,}\n"
        f"Mean pairwise similarity: {mean_sim:.4f}\n"
        f"Median pairwise similarity: {median_sim:.4f}\n"
        f"Maximum similarity: {max_sim:.4f}"
    )
    ax.text(0.95, 0.95, text_summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\nSimilarity distribution plot saved to: {save_path}")

    return mean_sim, median_sim