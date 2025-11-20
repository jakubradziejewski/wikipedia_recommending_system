import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.parse
import textwrap


def plot_distinctive_term_frequency(analyses, distinctive_terms, strategy_name="Strategy", save_path=None):
    """
    Visualize contribution of 10 rarest terms to each recommended article.
    Uses horizontal stacked bars where:
    - Y-axis = recommended articles
    - X-axis = stacked contributions from the 10 rarest/most distinctive terms

    This answers: "How much did rare query terms contribute to each article's match?"

    Args:
        analyses: List of analysis dicts from deep_explainability_analysis
        distinctive_terms: List of distinctive term dicts
        strategy_name: Name of the strategy for labeling
        save_path: Optional path to save the figure
    """
    if not analyses or not distinctive_terms:
        print("No data to visualize.")
        return

    # Take top 20 most distinctive (rarest) terms
    top_20_rare = distinctive_terms[:20]
    rare_term_names = [t['term'] for t in top_20_rare]

    # Create color palette for the 20 rare terms
    colors = sns.color_palette("Spectral", 20)

    # Prepare data structure
    num_articles = len(analyses)

    # Build contribution matrix and get term metadata
    contribution_matrix = []
    article_labels = []

    # Create mapping of term -> metadata for easy lookup
    term_metadata = {}
    for term_data in top_20_rare:
        term_metadata[term_data['term']] = {
            'doc_freq': term_data['document_frequency'],
            'total_docs': term_data['total_docs'],
            'enrichment': term_data['enrichment_ratio']
        }

    for analysis in analyses:
        # Create dict of term -> contribution for this article
        term_to_contrib = {m['term']: m['contribution'] for m in analysis['distinctive_matches']}

        # Get contributions for each of the 20 rarest terms
        row_contributions = []
        for term in rare_term_names:
            contrib = term_to_contrib.get(term, 0)
            row_contributions.append(contrib)

        contribution_matrix.append(row_contributions)

        # Prepare article label
        title = urllib.parse.unquote(analysis['title'])
        wrapped = '\n'.join(textwrap.wrap(f"#{analysis['rank']} {title}", width=50))
        article_labels.append(wrapped)

    contribution_matrix = np.array(contribution_matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 14))

    # Y positions for each article (reversed so #1 is at top)
    y_positions = np.arange(num_articles)[::-1] * 1.2

    # Create horizontal stacked bars
    for i, analysis in enumerate(analyses):
        y_pos = y_positions[i]
        left = 0

        # Stack each rare term's contribution
        cycle = 0
        for term_idx, term in enumerate(rare_term_names):
            contrib = contribution_matrix[i][term_idx]

            if contrib > 0:
                # Get metadata for this term
                metadata = term_metadata[term]
                doc_freq = metadata['doc_freq']
                total_docs = metadata['total_docs']
                enrichment = metadata['enrichment']

                # Calculate rarity percentage (lower = rarer)
                rarity_pct = (doc_freq / total_docs) * 100
                # Add term label, alternating the place so the labels don't overlap

                # Create bar segment
                ax.barh(y_pos, contrib, left=left, height=0.9,
                        color=colors[term_idx],
                        edgecolor='white', linewidth=2,
                        label=term if i == 0 else "")

                if cycle == 0:
                    vertical_offset = +0.35
                elif cycle == 1:
                    vertical_offset = -0.35
                else:
                    vertical_offset = 0
                cycle = (cycle + 1) % 3
                # Add term label on the segment with rarity info
                # Only label if segment is visible enough
                label_text = f'{term}\n{rarity_pct:.1f}% | {enrichment:.0f}x'
                ax.text(left + contrib / 2, y_pos + vertical_offset,
                        label_text,
                        ha='center', va='center', fontsize=6.5,
                        fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.9,
                                  edgecolor='gray', linewidth=0.5))

                left += contrib

        # Add total rare term contribution at the end of the bar
        if left > 0:
            ax.text(left * 1.02, y_pos, f'{left:.4f}',
                    ha='left', va='center', fontsize=10,
                    fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='lightyellow', alpha=0.9,
                              edgecolor='red', linewidth=1.5))

    # Set y-axis with article names
    ax.set_yticks(y_positions)
    ax.set_yticklabels(article_labels, fontsize=9, linespacing=1.3)

    # X-axis formatting
    ax.set_xlabel('Cumulative Contribution from 20 Rarest Query Terms',
                  fontsize=13, fontweight='bold')
    ax.set_title(f'Impact of Rarest Query Terms on Each Recommendation — {strategy_name}\n'
                 f'How much did the 20 most distinctive/rare terms contribute to matching each article?\n'
                 f'Labels show: term name | corpus frequency % | enrichment ratio',
                 fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend - show the 20 rare terms with their rarity
    legend_labels = []
    for term, term_data in zip(rare_term_names, top_20_rare):
        doc_freq = term_data['document_frequency']
        total_docs = term_data['total_docs']
        rarity_pct = (doc_freq / total_docs) * 100
        enrichment = term_data['enrichment_ratio']
        legend_labels.append(f"{term} ({rarity_pct:.1f}% corpus | {enrichment:.0f}x)")

    # Create custom legend
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels,
              loc='lower right', fontsize=8,
              title='20 Rarest Terms (corpus freq | enrichment)',
              title_fontsize=9,
              framealpha=0.95, edgecolor='black',
              bbox_to_anchor=(1.0, 0.0))

    # Add explanation
    fig.text(0.02, 0.01,
             'Reading: Longer bars = more contribution from rare terms | '
             'Segments = individual rare terms | '
             'Number at end = total contribution from all 10 rarest terms',
             ha='left', va='bottom', fontsize=9, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    plt.subplots_adjust(left=0.20, right=0.88, top=0.95, bottom=0.05)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nRare terms contribution plot saved to: {save_path}")


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

    # Set y-axis labels with FULL article titles (wrapped to multiple lines)
    y_labels = []
    for a in articles_analysis:
        title = urllib.parse.unquote(a['title'])
        # Wrap title to ~80 characters per line
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
