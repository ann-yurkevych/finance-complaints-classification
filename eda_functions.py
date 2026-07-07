import pandas as pd
import matplotlib.pyplot as plt

def class_distribution_plot(df: pd.DataFrame, target: str = 'Company response to consumer'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#D55E00']

    # Bar plot of counts
    counts = df[target].value_counts()
    axes[0].bar(counts.index.astype(str), counts.values, color=['green', 'coral'], edgecolor='white', width=0.5)
    axes[0].set_title('Class Distribution (Counts)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_yscale('log')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')
    
    # Bar plot of proportions
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

