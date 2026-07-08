import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

CLASS_COLORS = {
    'Closed': '#E69F00',
    'Closed with explanation': '#56B4E9',
    'Closed with monetary relief': '#009E73',
    'Closed with non-monetary relief': '#F0E442',
    'Untimely response': '#D55E00',
    }

DEFAULT_COLOR = '#999999'

def class_distribution_plot(df: pd.DataFrame, target: str = 'Company response to consumer'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#D55E00']
    counts = df[target].value_counts()
    axes[0].bar(counts.index.astype(str), counts.values, color=['green', 'coral'], edgecolor='white', width=0.5)
    axes[0].set_title('Class Distribution (Counts)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_yscale('log')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')
    
    proportions = df[target].value_counts(normalize=True)
    axes[1].bar(proportions.index.astype(str), proportions.values, color=['green', 'coral'], edgecolor='white', width=0.5)
    axes[1].set_title('Class Distribution (Proportions)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Proportion')
    axes[1].set_xticklabels(proportions.index.astype(str), rotation=30, ha='right')
    for i, v in enumerate(proportions.values):
        axes[1].text(i, v + 0.001, f'{v:.1%}', ha='center', fontweight='bold')
    
    plt.suptitle(f'Target Variable Class Distribution: "{target}"', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def average_word_length(text):
    if pd.isna(text) or text.strip() == "":
        return 0
    words = text.split()
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)


def class_distribution_plot(df: pd.DataFrame, target: str = 'Company response to consumer'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot of counts
    counts = df[target].value_counts()
    bar_colors = [CLASS_COLORS.get(c, DEFAULT_COLOR) for c in counts.index]

    axes[0].bar(counts.index.astype(str), counts.values, color=bar_colors, edgecolor='white', width=0.6)
    axes[0].set_title('Class Distribution (Counts)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_yscale('log')
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(counts.index.astype(str), rotation=30, ha='right')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v * 1.1, f'{v:,}', ha='center', fontweight='bold', fontsize=9)

    # Bar plot of proportions
    proportions = df[target].value_counts(normalize=True)
    prop_colors = [CLASS_COLORS.get(c, DEFAULT_COLOR) for c in proportions.index]

    axes[1].bar(proportions.index.astype(str), proportions.values, color=prop_colors, edgecolor='white', width=0.6)
    axes[1].set_title('Class Distribution (Proportions)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Proportion')
    axes[1].set_xticks(range(len(proportions)))
    axes[1].set_xticklabels(proportions.index.astype(str), rotation=30, ha='right')
    for i, v in enumerate(proportions.values):
        axes[1].text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold', fontsize=9)

    plt.suptitle(f'Target Variable Class Distribution: "{target}"', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def class_distribution_over_time(df: pd.DataFrame, target: str = 'Company response to consumer', date_col: str = 'Date received'):
    df = df.copy()
    df['year'] = df[date_col].dt.year

    class_by_year = (
        df.groupby('year')[target]
        .value_counts(normalize=True)
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    for col in class_by_year.columns:
        color = CLASS_COLORS.get(col, DEFAULT_COLOR)
        ax.plot(
            class_by_year.index,
            class_by_year[col],
            marker='o',
            markersize=5,
            color=color,
            label=col,
            linewidth=2.2
        )

    ax.set_title(f'{target}: Trend by Year', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Proportion of Complaints', fontsize=12)
    ax.legend(title='Response Type', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(class_by_year.index)
    ax.set_xticklabels(class_by_year.index, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    return class_by_year

def sub_product_missing_values(df: pd.DataFrame, product_col: str = 'Product', sub_col: str = 'Sub-product', show_zero: bool = False):
    df = df.copy()
    df['is_missing'] = df[sub_col].isna()

    pct_missing = (
        df.groupby(product_col)['is_missing']
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    if not show_zero:
        pct_missing = pct_missing[pct_missing > 0]

    fig, ax = plt.subplots(figsize=(9, max(2, len(pct_missing) * 0.6)))

    colors = ['#D55E00' if v == 100 else '#F0E442' for v in pct_missing.values]
    ax.barh(pct_missing.index, pct_missing.values, color=colors, edgecolor='white')
    ax.set_title(f'Products with Missing {sub_col} (>0%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('% Missing')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, v in enumerate(pct_missing.values):
        ax.text(v + 1.5, i, f'{v:.0f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    return pct_missing

def top_companies_by_volume(df: pd.DataFrame, company_col: str = 'Company', top_n: int = 15):
    """Plots the top N companies by complaint count, and reports concentration stats."""

    counts = df[company_col].value_counts()

    fig, ax = plt.subplots(figsize=(10, 8))
    top = counts.head(top_n).sort_values()

    ax.barh(top.index, top.values, color='#56B4E9', edgecolor='white')
    ax.set_title(f'Top {top_n} Companies by Complaint Volume', fontsize=14, fontweight='bold')
    ax.set_xlabel('Complaint Count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, v in enumerate(top.values):
        ax.text(v * 1.01, i, f'{v:,}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    total = len(df)
    top_n_share = counts.head(top_n).sum() / total * 100
    n_companies_total = len(counts)
    n_single_complaint = (counts == 1).sum()

    print(f"Total unique companies: {n_companies_total:,}")
    print(f"Top {top_n} companies account for {top_n_share:.1f}% of all complaints")
    print(f"Companies with only 1 complaint: {n_single_complaint:,} ({n_single_complaint/n_companies_total*100:.1f}% of all companies)")

    return counts

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def top_companies_by_complaint(df: pd.DataFrame, company_col: str = 'Company', top_n: int = 15):
    counts = df[company_col].value_counts()
    top = counts.head(top_n).sort_values(ascending=True)

    norm = mcolors.Normalize(vmin=top.min(), vmax=top.max())
    cmap = cm.Blues
    bar_colors = [cmap(0.3 + 0.7 * norm(v)) for v in top.values]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color=bar_colors, edgecolor='white')

    ax.set_title(f'Top {top_n} Companies by Complaint Volume', fontsize=11, fontweight='bold')
    ax.set_xlabel('Complaint Count', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, v in enumerate(top.values):
        ax.text(v * 1.01, i, f'{v:,}', va='center', fontsize=7)

    plt.tight_layout()
    plt.show()

    return top


# N-grams for text features
def top_ngrams(text_series, n=2, top_k=20):
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    ngram_matrix = vectorizer.fit_transform(text_series.dropna())

    ngram_counts = ngram_matrix.sum(axis=0)
    ngrams_freq = [(word, ngram_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    ngrams_freq = sorted(ngrams_freq, key=lambda x: x[1], reverse=True)

    top_ngrams_list = []
    for i, item in enumerate(ngrams_freq):
        if i >= top_k:
            break
        top_ngrams_list.append(item)

    return top_ngrams_list

# plot the n-grams
def plot_top_ngrams(text_series, n=2, top_k=15):
    ngrams_freq = top_ngrams(text_series, n=n, top_k=top_k)
    phrases, counts = zip(*ngrams_freq[::-1])  # reverse so most frequent lands at top

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.barh(phrases, counts, color='#56B4E9', edgecolor='white')
    ax.set_title(f'Top {top_k} {"Bigrams" if n == 2 else "Trigrams"}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, v in enumerate(counts):
        ax.text(v * 1.01, i, f'{v:,}', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()

def narrative_length_by_target(df: pd.DataFrame, text_col: str = 'Consumer complaint narrative', target_col: str = 'Company response to consumer'):
    df = df.copy()
    df['word_count'] = df[text_col].str.split().str.len()
    order = df.groupby(target_col)['word_count'].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, 6))
    data_by_class = [df[df[target_col] == cls]['word_count'].dropna() for cls in order]
    box_colors = [CLASS_COLORS.get(cls, DEFAULT_COLOR) for cls in order]
    bp = ax.boxplot(data_by_class, labels=order, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title('Narrative Length by Company Response', fontsize=13, fontweight='bold')
    ax.set_ylabel('Word Count')
    ax.set_xticklabels(order, rotation=30, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    print(df.groupby(target_col)['word_count'].median().sort_values(ascending=False))
