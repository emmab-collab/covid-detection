"""
Visualization functions for exploratory data analysis.
Creates comprehensive plots for understanding the data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional

from ..config import (
    TARGET_FEATURE,
    FIGURE_SIZE,
    HEATMAP_SIZE,
    RESULTS_DIR,
    DPI
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = DPI


def plot_target_distribution(
    df: pd.DataFrame,
    save: bool = False
) -> None:
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


def plot_missing_values_heatmap(
    df: pd.DataFrame,
    save: bool = False
) -> None:
    """
    Plot heatmap of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    save : bool, default=False
        Whether to save the plot
    """
    plt.figure(figsize=HEATMAP_SIZE)
    sns.heatmap(df.isna(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap (yellow = missing)')
    plt.tight_layout()

    if save:
        plt.savefig(RESULTS_DIR / 'missing_values_heatmap.png', dpi=DPI)

    plt.show()


def plot_missing_values_bar(
    df: pd.DataFrame,
    threshold: float = 0.5,
    save: bool = False
) -> None:
    """
    Plot bar chart of missing value percentages.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    threshold : float, default=0.5
        Only show columns with missing rate above this threshold
    save : bool, default=False
        Whether to save the plot
    """
    missing_rate = (df.isna().sum() / df.shape[0]).sort_values(ascending=False)
    missing_rate = missing_rate[missing_rate > threshold]

    if len(missing_rate) > 0:
        plt.figure(figsize=(12, max(6, len(missing_rate) * 0.3)))
        missing_rate.plot(kind='barh')
        plt.xlabel('Missing Value Rate')
        plt.title(f'Columns with >{threshold*100}% Missing Values')
        plt.tight_layout()

        if save:
            plt.savefig(RESULTS_DIR / 'missing_values_bar.png', dpi=DPI)

        plt.show()
    else:
        print(f"No columns with missing rate > {threshold}")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson',
    save: bool = False
) -> None:
    """
    Plot correlation heatmap for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str], optional
        Specific columns to include. If None, uses all numeric columns.
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'
    save : bool, default=False
        Whether to save the plot
    """
    if columns is None:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[columns]

    if numeric_df.shape[1] < 2:
        print("Need at least 2 numeric columns for correlation")
        return

    plt.figure(figsize=FIGURE_SIZE)
    correlation = numeric_df.corr(method=method)

    sns.heatmap(
        correlation,
        annot=True if correlation.shape[0] < 15 else False,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        square=True,
        linewidths=0.5
    )

    plt.title(f'{method.capitalize()} Correlation Heatmap')
    plt.tight_layout()

    if save:
        plt.savefig(RESULTS_DIR / 'correlation_heatmap.png', dpi=DPI)

    plt.show()


def plot_feature_distributions(
    df: pd.DataFrame,
    columns: List[str],
    ncols: int = 3,
    save: bool = False
) -> None:
    """
    Plot distributions for multiple features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of columns to plot
    ncols : int, default=3
        Number of columns in subplot grid
    save : bool, default=False
        Whether to save the plot
    """
    nrows = int(np.ceil(len(columns) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten() if nrows > 1 else [axes]

    for idx, col in enumerate(columns):
        if idx < len(axes):
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                # Categorical or few unique values
                df[col].value_counts().plot(kind='bar', ax=axes[idx])
            else:
                # Continuous
                df[col].hist(bins=30, ax=axes[idx], edgecolor='black')

            axes[idx].set_title(col)
            axes[idx].set_xlabel('')

    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save:
        plt.savefig(RESULTS_DIR / 'feature_distributions.png', dpi=DPI)

    plt.show()


def plot_feature_vs_target(
    df: pd.DataFrame,
    feature: str,
    target: str = TARGET_FEATURE,
    plot_type: str = 'auto',
    save: bool = False
) -> None:
    """
    Plot relationship between a feature and the target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature : str
        Feature column name
    target : str, default=TARGET_FEATURE
        Target column name
    plot_type : str, default='auto'
        Type of plot: 'auto', 'box', 'violin', 'count', 'dist'
    save : bool, default=False
        Whether to save the plot
    """
    plt.figure(figsize=FIGURE_SIZE)

    # Determine plot type
    if plot_type == 'auto':
        if df[feature].dtype == 'object' or df[feature].nunique() < 10:
            plot_type = 'count'
        else:
            plot_type = 'dist'

    # Create plot
    if plot_type == 'count':
        sns.countplot(data=df, x=feature, hue=target)
        plt.xticks(rotation=45)
    elif plot_type == 'box':
        sns.boxplot(data=df, x=target, y=feature)
    elif plot_type == 'violin':
        sns.violinplot(data=df, x=target, y=feature)
    elif plot_type == 'dist':
        for category in df[target].unique():
            subset = df[df[target] == category]
            sns.kdeplot(data=subset[feature].dropna(), label=str(category), fill=True, alpha=0.5)
        plt.legend(title=target)
    else:
        print(f"Unknown plot type: {plot_type}")
        return

    plt.title(f'{feature} vs {target}')
    plt.tight_layout()

    if save:
        filename = f'{feature}_vs_{target}.png'.replace(' ', '_')
        plt.savefig(RESULTS_DIR / filename, dpi=DPI)

    plt.show()


def plot_multiple_features_vs_target(
    df: pd.DataFrame,
    features: List[str],
    target: str = TARGET_FEATURE,
    ncols: int = 2,
    save: bool = False
) -> None:
    """
    Plot multiple features against the target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    features : List[str]
        List of feature column names
    target : str, default=TARGET_FEATURE
        Target column name
    ncols : int, default=2
        Number of columns in subplot grid
    save : bool, default=False
        Whether to save the plot
    """
    nrows = int(np.ceil(len(features) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    axes = axes.flatten() if nrows > 1 else [axes]

    # Get target categories for color coding
    target_categories = df[target].unique()

    for idx, feature in enumerate(features):
        if idx < len(axes):
            ax = axes[idx]

            # Plot distribution for each target category
            for category in target_categories:
                subset = df[df[target] == category]
                subset[feature].dropna().hist(
                    bins=20,
                    alpha=0.6,
                    label=str(category),
                    ax=ax
                )

            ax.set_title(f'{feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend(title=target)

    # Hide empty subplots
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Feature Distributions by {target}', y=1.00, fontsize=14)
    plt.tight_layout()

    if save:
        plt.savefig(RESULTS_DIR / 'features_vs_target.png', dpi=DPI)

    plt.show()
