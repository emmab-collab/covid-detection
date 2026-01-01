"""
EDA Analyzer class for comprehensive exploratory data analysis.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from typing import List, Tuple, Dict

from ..config import TARGET_FEATURE, BLOOD_MISSING_RATE_MIN, BLOOD_MISSING_RATE_MAX, VIRAL_MISSING_RATE_MIN, VIRAL_MISSING_RATE_MAX


class EDAAnalyzer:
    """
    Comprehensive Exploratory Data Analysis class.

    Provides methods for analyzing dataset structure, missing values,
    target distribution, feature relationships, and statistical tests.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA Analyzer.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to analyze
        """
        self.df = df.copy()
        self.target = TARGET_FEATURE
        self.missing_rate = None
        self.blood_columns = None
        self.viral_columns = None

    def analyze_shape(self) -> Dict:
        """
        Analyze dataset shape and basic information.

        Returns
        -------
        Dict
            Dictionary with shape information
        """
        info = {
            'n_rows': self.df.shape[0],
            'n_columns': self.df.shape[1],
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'n_numeric': (self.df.dtypes == 'float64').sum() + (self.df.dtypes == 'int64').sum(),
            'n_categorical': (self.df.dtypes == 'object').sum()
        }

        print(f"Nombre de lignes: {info['n_rows']:,}")
        print(f"Nombre de colonnes: {info['n_columns']}")
        print(f"Mémoire utilisée: {info['memory_mb']:.2f} MB")
        print(f"\nVariables numériques: {info['n_numeric']}")
        print(f"Variables catégorielles: {info['n_categorical']}")

        return info

    def analyze_missing_values(self, top_n: int = 20) -> pd.Series:
        """
        Analyze missing values patterns.

        Parameters
        ----------
        top_n : int
            Number of top missing columns to display

        Returns
        -------
        pd.Series
            Missing rate for each column
        """
        self.missing_rate = (self.df.isna().sum() / len(self.df) * 100).sort_values(ascending=False)

        print(f"Top {top_n} colonnes avec le plus de valeurs manquantes:\n")
        print(self.missing_rate.head(top_n))

        print(f"\nColonnes avec >90% de NaN: {(self.missing_rate > 90).sum()}")
        print(f"Colonnes avec >50% de NaN: {(self.missing_rate > 50).sum()}")
        print(f"Colonnes sans NaN: {(self.missing_rate == 0).sum()}")

        return self.missing_rate

    def plot_missing_heatmap(self, figsize: Tuple[int, int] = (20, 10)):
        """
        Visualize missing values with heatmap.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size
        """
        plt.figure(figsize=figsize)
        sns.heatmap(self.df.isna(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Heatmap des Valeurs Manquantes (jaune = manquant)', fontsize=16)
        plt.xlabel('Features')
        plt.tight_layout()
        plt.show()

    def analyze_target(self) -> Dict:
        """
        Analyze target variable distribution.

        Returns
        -------
        Dict
            Target distribution information
        """
        target_counts = self.df[self.target].value_counts()
        target_pct = self.df[self.target].value_counts(normalize=True) * 100

        print(f"Distribution de {self.target}:\n")
        for cat in target_counts.index:
            print(f"{cat}: {target_counts[cat]:,} ({target_pct[cat]:.1f}%)")

        return {
            'counts': target_counts.to_dict(),
            'percentages': target_pct.to_dict()
        }

    def plot_target_distribution(self, figsize: Tuple[int, int] = (14, 5)):
        """
        Visualize target distribution.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size
        """
        target_counts = self.df[self.target].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Countplot
        sns.countplot(data=self.df, x=self.target, ax=ax1, palette='Set2')
        ax1.set_title('Distribution de la Target', fontsize=14)
        ax1.set_ylabel('Nombre')
        for p in ax1.patches:
            height = p.get_height()
            ax1.text(p.get_x() + p.get_width()/2., height,
                     f'{int(height):,}\n({height/len(self.df)*100:.1f}%)',
                     ha='center', va='bottom')

        # Pie chart
        ax2.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%',
                colors=sns.color_palette('Set2'), startangle=90)
        ax2.set_title('Proportion de la Target', fontsize=14)

        plt.tight_layout()
        plt.show()

    def identify_feature_groups(self,
                                blood_min: float = None,
                                blood_max: float = None,
                                viral_min: float = None,
                                viral_max: float = None) -> Tuple[List[str], List[str]]:
        """
        Identify blood and viral feature groups based on missing rates.

        Parameters
        ----------
        blood_min, blood_max : float, optional
            Missing rate range for blood features (default: from config)
        viral_min, viral_max : float, optional
            Missing rate range for viral features (default: from config)

        Returns
        -------
        Tuple[List[str], List[str]]
            (blood_columns, viral_columns)
        """
        if self.missing_rate is None:
            self.analyze_missing_values()

        # Use config values if not specified
        blood_min = blood_min or BLOOD_MISSING_RATE_MIN
        blood_max = blood_max or BLOOD_MISSING_RATE_MAX
        viral_min = viral_min or VIRAL_MISSING_RATE_MIN
        viral_max = viral_max or VIRAL_MISSING_RATE_MAX

        missing_rate_fraction = self.missing_rate / 100

        self.blood_columns = list(
            self.df.columns[(missing_rate_fraction > blood_min) &
                           (missing_rate_fraction < blood_max)]
        )

        self.viral_columns = list(
            self.df.columns[(missing_rate_fraction > viral_min) &
                           (missing_rate_fraction < viral_max)]
        )

        print(f"Blood features: {len(self.blood_columns)}")
        print(f"Viral features: {len(self.viral_columns)}")

        return self.blood_columns, self.viral_columns

    def plot_feature_distributions(self, features: List[str],
                                   n_cols: int = 4,
                                   figsize: Tuple[int, int] = (16, 12)):
        """
        Plot distributions of multiple features.

        Parameters
        ----------
        features : List[str]
            List of feature names to plot
        n_cols : int
            Number of columns in subplot grid
        figsize : Tuple[int, int]
            Figure size
        """
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, col in enumerate(features[:min(n_features, len(axes))]):
            self.df[col].hist(bins=30, ax=axes[idx], edgecolor='black', alpha=0.7)
            axes[idx].set_title(col, fontsize=10)
            axes[idx].set_xlabel('')

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Distribution des Features', fontsize=16, y=1.00)
        plt.tight_layout()
        plt.show()

    def compare_distributions_by_target(self, features: List[str],
                                       figsize: Tuple[int, int] = (14, 10)):
        """
        Compare feature distributions between positive and negative cases.

        Parameters
        ----------
        features : List[str]
            Features to compare
        figsize : Tuple[int, int]
            Figure size
        """
        positive_df = self.df[self.df[self.target] == 'positive']
        negative_df = self.df[self.df[self.target] == 'negative']

        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, col in enumerate(features[:min(n_features, len(axes))]):
            if col in self.df.columns:
                positive_df[col].dropna().hist(bins=30, alpha=0.5, label='Positive',
                                              ax=axes[idx], color='red', edgecolor='black')
                negative_df[col].dropna().hist(bins=30, alpha=0.5, label='Negative',
                                              ax=axes[idx], color='blue', edgecolor='black')
                axes[idx].set_title(f'Distribution de {col}', fontsize=12)
                axes[idx].set_xlabel('Valeur')
                axes[idx].set_ylabel('Fréquence')
                axes[idx].legend()

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Comparaison Features : Positifs vs Négatifs', fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, features: List[str],
                                figsize: Tuple[int, int] = (14, 12)):
        """
        Plot correlation matrix for given features.

        Parameters
        ----------
        features : List[str]
            Features to include in correlation matrix
        figsize : Tuple[int, int]
            Figure size
        """
        plt.figure(figsize=figsize)
        corr_matrix = self.df[features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Matrice de Corrélation', fontsize=16)
        plt.tight_layout()
        plt.show()

        # Identify high correlations
        high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix.abs() < 1.0)
        print("\nPaires avec corrélation > 0.8:")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if high_corr.iloc[i, j]:
                    print(f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")

    def statistical_tests(self, features: List[str], alpha: float = 0.02) -> List[str]:
        """
        Perform t-tests to identify significant features.

        Parameters
        ----------
        features : List[str]
            Features to test
        alpha : float
            Significance level

        Returns
        -------
        List[str]
            List of significant features
        """
        positive_df = self.df[self.df[self.target] == 'positive']
        negative_df = self.df[self.df[self.target] == 'negative']

        significant_features = []

        print(f"Test t de Student (H0: moyennes égales)\n")
        print(f"{'Feature':<50} {'p-value':<12} {'Résultat'}")
        print("-" * 75)

        for col in features:
            pos_data = positive_df[col].dropna()
            neg_data = negative_df[col].dropna()

            if len(pos_data) > 5 and len(neg_data) > 5:
                stat, p_value = ttest_ind(pos_data, neg_data)
                result = "H0 REJETÉE" if p_value < alpha else "H0 acceptée"

                if p_value < alpha:
                    significant_features.append(col)

                print(f"{col:<50} {p_value:<12.6f} {result}")

        print(f"\nFeatures significativement différentes (p < {alpha}): {len(significant_features)}")
        return significant_features

    def generate_summary_report(self) -> Dict:
        """
        Generate complete EDA summary report.

        Returns
        -------
        Dict
            Complete summary of EDA findings
        """
        report = {
            'shape_info': self.analyze_shape(),
            'missing_summary': {
                'n_cols_90pct': (self.missing_rate > 90).sum() if self.missing_rate is not None else None,
                'n_cols_50pct': (self.missing_rate > 50).sum() if self.missing_rate is not None else None,
                'n_cols_no_missing': (self.missing_rate == 0).sum() if self.missing_rate is not None else None
            },
            'target_distribution': self.analyze_target(),
            'feature_groups': {
                'blood_features': len(self.blood_columns) if self.blood_columns else 0,
                'viral_features': len(self.viral_columns) if self.viral_columns else 0
            }
        }

        return report
