"""
Feature engineering functions for COVID-19 detection.
Creates new features from existing ones to improve model performance.
"""

import pandas as pd
import numpy as np
from typing import List


def feature_engineering(df: pd.DataFrame, viral_columns: List[str]) -> pd.DataFrame:
    """
    Create engineered features from viral test results.

    Creates a binary feature 'est malade' (is sick) which indicates
    if the patient tested positive for any viral infection.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with viral test columns
    viral_columns : List[str]
        List of column names containing viral test results

    Returns
    -------
    pd.DataFrame
        Dataframe with new engineered feature and viral columns removed
    """
    df_engineered = df.copy()

    # Create 'est malade' feature: True if any viral test is positive
    # Sum across viral columns and check if >= 1
    df_engineered['est malade'] = df_engineered[viral_columns].sum(axis=1) >= 1

    # Drop original viral columns as they're now encoded in 'est malade'
    df_engineered = df_engineered.drop(viral_columns, axis=1)

    return df_engineered


def create_age_bins(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Create age group bins from age quantile.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    n_bins : int, default=5
        Number of age bins to create

    Returns
    -------
    pd.DataFrame
        Dataframe with added age_group column
    """
    df_binned = df.copy()

    df_binned['age_group'] = pd.cut(
        df_binned['Patient age quantile'],
        bins=n_bins,
        labels=[f'group_{i}' for i in range(n_bins)]
    )

    return df_binned


def create_blood_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio features from blood test results.

    Creates useful ratios like:
    - Neutrophil-to-Lymphocyte ratio (NLR)
    - Platelet-to-Lymphocyte ratio (PLR)
    - Monocyte-to-Lymphocyte ratio (MLR)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with blood test columns

    Returns
    -------
    pd.DataFrame
        Dataframe with added ratio features
    """
    df_ratios = df.copy()

    # Only create ratios if the required columns exist
    if 'Lymphocytes' in df_ratios.columns:
        if 'Platelets' in df_ratios.columns:
            df_ratios['PLR'] = df_ratios['Platelets'] / (df_ratios['Lymphocytes'] + 1e-6)

        if 'Monocytes' in df_ratios.columns:
            df_ratios['MLR'] = df_ratios['Monocytes'] / (df_ratios['Lymphocytes'] + 1e-6)

    return df_ratios


def create_interaction_features(
    df: pd.DataFrame,
    feature_pairs: List[tuple] = None
) -> pd.DataFrame:
    """
    Create interaction features between specified feature pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_pairs : List[tuple], optional
        List of tuples specifying feature pairs to interact

    Returns
    -------
    pd.DataFrame
        Dataframe with added interaction features
    """
    df_interactions = df.copy()

    if feature_pairs is None:
        # Default pairs based on domain knowledge
        feature_pairs = [
            ('Leukocytes', 'Lymphocytes'),
            ('Platelets', 'Monocytes'),
        ]

    for feat1, feat2 in feature_pairs:
        if feat1 in df_interactions.columns and feat2 in df_interactions.columns:
            interaction_name = f'{feat1}_x_{feat2}'
            df_interactions[interaction_name] = (
                df_interactions[feat1] * df_interactions[feat2]
            )

    return df_interactions
