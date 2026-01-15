"""
Data preprocessing functions - matching the original notebook approach.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import (
    TARGET_FEATURE,
    BLOOD_MISSING_RATE,
    VIRAL_MISSING_RATE,
    TEST_SIZE,
    RANDOM_STATE
)


def create_subsets(df):
    """
    Create feature subsets based on missing rates.

    Returns
    -------
    tuple
        (df_subset, blood_columns, viral_columns)
    """
    missing_rate = df.isna().sum() / df.shape[0]

    blood_columns = list(df.columns[
        (missing_rate > BLOOD_MISSING_RATE[0]) &
        (missing_rate < BLOOD_MISSING_RATE[1])
    ])

    viral_columns = list(df.columns[
        (missing_rate > VIRAL_MISSING_RATE[0]) &
        (missing_rate < VIRAL_MISSING_RATE[1])
    ])

    key_columns = ['Patient age quantile', TARGET_FEATURE]

    df_subset = df[key_columns + blood_columns + viral_columns].copy()

    return df_subset, blood_columns, viral_columns


def encodage(df):
    """
    Encode categorical variables.
    positive/detected -> 1, negative/not_detected -> 0
    """
    code = {
        'positive': 1,
        'negative': 0,
        'detected': 1,
        'not_detected': 0
    }

    for col in df.select_dtypes('object'):
        df[col] = df[col].map(code)

    return df


def feature_engineering(df, viral_columns):
    """
    Create 'est malade' feature from viral tests.
    """
    df['est malade'] = df[viral_columns].sum(axis=1) >= 1
    df = df.drop(viral_columns, axis=1)
    return df


def imputation(df):
    """
    Handle missing values by dropping rows with NaN.
    """
    df = df.dropna(axis=0)
    return df


def preprocessing(df, viral_columns):
    """
    Full preprocessing pipeline.

    Returns
    -------
    tuple
        (X, y)
    """
    df = encodage(df)
    df = feature_engineering(df, viral_columns)
    df = imputation(df)

    X = df.drop(TARGET_FEATURE, axis=1)
    y = df[TARGET_FEATURE]

    print(f"Shape après preprocessing: {X.shape}")
    print(f"Distribution target:\n{y.value_counts()}")

    return X, y


def prepare_data(df):
    """
    Complete data preparation pipeline.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Create subsets
    df_subset, blood_columns, viral_columns = create_subsets(df)

    print(f"Dataset après sélection: {df_subset.shape}")
    print(f"Blood columns: {len(blood_columns)}")
    print(f"Viral columns: {len(viral_columns)}")

    # Train/test split AVANT preprocessing (comme dans le notebook original)
    trainset, testset = train_test_split(
        df_subset,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"\n--- Trainset (avant preprocessing) ---")
    print(f"Shape: {trainset.shape}")
    print(f"Proportions de {TARGET_FEATURE}:")
    print(trainset[TARGET_FEATURE].value_counts())

    print(f"\n--- Testset (avant preprocessing) ---")
    print(f"Shape: {testset.shape}")
    print(f"Proportions de {TARGET_FEATURE}:")
    print(testset[TARGET_FEATURE].value_counts())

    # Preprocessing (avec dropna)
    print("\n--- Preprocessing trainset ---")
    X_train, y_train = preprocessing(trainset.copy(), viral_columns)

    print("\n--- Preprocessing testset ---")
    X_test, y_test = preprocessing(testset.copy(), viral_columns)

    # Résumé final
    print("\n--- Résumé après dropna ---")
    print(f"X_train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"X_test: {X_test.shape[0]} samples")
    print(f"y_train: {y_train.value_counts().to_dict()}")
    print(f"y_test: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test
