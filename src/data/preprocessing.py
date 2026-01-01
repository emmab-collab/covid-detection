"""
Data preprocessing functions for COVID-19 detection.
Handles data loading, cleaning, encoding, and imputation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

from ..config import (
    DATASET_PATH,
    TARGET_FEATURE,
    BLOOD_MISSING_RATE_MIN,
    BLOOD_MISSING_RATE_MAX,
    VIRAL_MISSING_RATE_MIN,
    VIRAL_MISSING_RATE_MAX,
    KEY_COLUMNS,
    ENCODING_MAP
)


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the COVID-19 dataset from Excel file.

    Parameters
    ----------
    filepath : str, optional
        Path to the dataset. If None, uses default from config.

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    if filepath is None:
        filepath = DATASET_PATH

    data = pd.read_excel(filepath)
    df = data.copy()

    # Set display options for better visualization
    pd.set_option('display.max_row', 111)
    pd.set_option('display.max_column', 111)

    return df


def get_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify blood and viral feature columns based on missing rate.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    Tuple[List[str], List[str]]
        (blood_columns, viral_columns)
    """
    missing_rate = df.isna().sum() / df.shape[0]

    blood_columns = list(
        df.columns[(missing_rate > BLOOD_MISSING_RATE_MIN) &
                   (missing_rate < BLOOD_MISSING_RATE_MAX)]
    )

    viral_columns = list(
        df.columns[(missing_rate > VIRAL_MISSING_RATE_MIN) &
                   (missing_rate < VIRAL_MISSING_RATE_MAX)]
    )

    return blood_columns, viral_columns


def select_features_by_missing_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features based on missing value patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with selected features (key columns + blood + viral)
    """
    blood_columns, viral_columns = get_feature_groups(df)
    selected_columns = KEY_COLUMNS + blood_columns + viral_columns

    return df[selected_columns]


def encodage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables to numerical values.

    Maps:
    - positive/detected -> 1
    - negative/not_detected -> 0

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Encoded dataframe
    """
    df_encoded = df.copy()

    for col in df_encoded.select_dtypes('object'):
        df_encoded[col] = df_encoded[col].map(ENCODING_MAP)

    return df_encoded


def imputation(df: pd.DataFrame, method: str = 'dropna') -> pd.DataFrame:
    """
    Handle missing values in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    method : str, default='dropna'
        Imputation method: 'dropna', 'fillna', or 'flag'

    Returns
    -------
    pd.DataFrame
        Dataframe with handled missing values
    """
    df_imputed = df.copy()

    if method == 'dropna':
        df_imputed = df_imputed.dropna(axis=0)
    elif method == 'fillna':
        df_imputed = df_imputed.fillna(-999)
    elif method == 'flag':
        # Create a flag column for NaN presence
        df_imputed['has_missing'] = df_imputed.isna().any(axis=1).astype(int)
        df_imputed = df_imputed.fillna(-999)

    return df_imputed


def preprocessing(
    df: pd.DataFrame,
    viral_columns: List[str] = None,
    return_viral_columns: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete preprocessing pipeline: encoding, feature engineering, imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    viral_columns : List[str], optional
        List of viral columns for feature engineering
    return_viral_columns : bool, default=False
        Whether to return viral_columns list

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series] or Tuple[pd.DataFrame, pd.Series, List[str]]
        (X, y) or (X, y, viral_columns)
    """
    from ..features.engineering import feature_engineering

    df_processed = df.copy()

    # Get viral columns if not provided
    if viral_columns is None:
        _, viral_columns = get_feature_groups(df_processed)

    # Apply preprocessing steps
    df_processed = encodage(df_processed)
    df_processed = feature_engineering(df_processed, viral_columns)
    df_processed = imputation(df_processed)

    # Split features and target
    X = df_processed.drop(TARGET_FEATURE, axis=1)
    y = df_processed[TARGET_FEATURE]

    print(f"Target distribution:\n{y.value_counts()}\n")

    if return_viral_columns:
        return X, y, viral_columns
    return X, y
