"""
Visualization utilities.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

from ..config import TARGET_FEATURE, RESULTS_DIR, DPI

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = DPI


def plot_target_distribution(df: pd.DataFrame, save: bool = False) -> None:
    """
    Plot the distribution of the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    save : bool, default=False
        Whether to save the plot
    """
    plt.figure(figsize=(8, 6))

    # Count plot
    ax = sns.countplot(data=df, x=TARGET_FEATURE)
    plt.title(f'Distribution of {TARGET_FEATURE}')
    plt.xlabel(TARGET_FEATURE)
    plt.ylabel('Count')

    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.,
            height + 10,
            f'{height/total*100:.1f}%',
            ha='center'
        )

    plt.tight_layout()

    if save:
        plt.savefig(RESULTS_DIR / 'target_distribution.png', dpi=DPI)

    plt.show()


def plot_missing_rate(df: pd.DataFrame, top_n: int = 20, save: bool = False) -> None:
    """
    Plot missing value rates for top N columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    top_n : int
        Number of top columns to display
    save : bool
        Whether to save the plot
    """
    missing_rate = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    missing_rate.head(top_n).plot(kind='bar')
    plt.title(f'Top {top_n} Columns with Missing Values')
    plt.xlabel('Columns')
    plt.ylabel('Missing Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save:
        plt.savefig(RESULTS_DIR / 'missing_rate.png', dpi=DPI)

    plt.show()
