"""
Data Preprocessor class for comprehensive data preprocessing.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split

from ..config import (
    TARGET_FEATURE,
    TEST_SIZE,
    RANDOM_STATE,
    BLOOD_MISSING_RATE_MIN,
    BLOOD_MISSING_RATE_MAX,
    VIRAL_MISSING_RATE_MIN,
    VIRAL_MISSING_RATE_MAX
)


class DataPreprocessor:
    """
    Comprehensive Data Preprocessing class.

    Handles data loading, feature selection, encoding, feature engineering,
    imputation, and train/test splitting.
    """

    def __init__(self, df: pd.DataFrame = None, target: str = None):
        """
        Initialize Data Preprocessor.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Input dataframe
        target : str, optional
            Target column name (default: from config)
        """
        self.df_original = df.copy() if df is not None else None
        self.df = df.copy() if df is not None else None
        self.target = target or TARGET_FEATURE

        self.df_selected = None
        self.df_encoded = None
        self.df_engineered = None
        self.df_imputed = None

        self.blood_columns = []
        self.viral_columns = []
        self.selected_features = []

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def select_features_by_missing_rate(self, threshold: float = 0.9) -> pd.DataFrame:
        """
        Select features based on missing value threshold.

        Parameters
        ----------
        threshold : float
            Maximum allowed missing rate (0.9 = 90%)

        Returns
        -------
        pd.DataFrame
            Dataframe with selected features
        """
        print(f"=== Sélection des Features (seuil: <{threshold*100:.0f}% NaN) ===")
        print(f"Shape avant: {self.df.shape}")

        missing_rate = self.df.isna().sum() / len(self.df)
        self.selected_features = list(missing_rate[missing_rate < threshold].index)

        self.df_selected = self.df[self.selected_features].copy()

        print(f"Shape après: {self.df_selected.shape}")
        print(f"Features retenues: {len(self.selected_features)}")

        return self.df_selected

    def identify_feature_groups(self) -> Tuple[List[str], List[str]]:
        """
        Identify blood and viral feature groups based on missing rates.

        Returns
        -------
        Tuple[List[str], List[str]]
            (blood_columns, viral_columns)
        """
        if self.df_selected is None:
            raise ValueError("Exécutez d'abord select_features_by_missing_rate()")

        print(f"\n=== Identification des Groupes de Features ===")

        missing_rate = self.df_selected.isna().sum() / len(self.df_selected)

        self.blood_columns = list(
            self.df_selected.columns[(missing_rate > BLOOD_MISSING_RATE_MIN) &
                                     (missing_rate < BLOOD_MISSING_RATE_MAX)]
        )

        self.viral_columns = list(
            self.df_selected.columns[(missing_rate > VIRAL_MISSING_RATE_MIN) &
                                     (missing_rate < VIRAL_MISSING_RATE_MAX)]
        )

        print(f"Blood features: {len(self.blood_columns)}")
        print(f"Viral features: {len(self.viral_columns)}")

        return self.blood_columns, self.viral_columns

    def encode_categorical(self) -> pd.DataFrame:
        """
        Encode categorical variables.

        Maps:
        - positive/detected -> 1
        - negative/not_detected -> 0

        Returns
        -------
        pd.DataFrame
            Encoded dataframe
        """
        if self.df_selected is None:
            raise ValueError("Exécutez d'abord select_features_by_missing_rate()")

        print(f"\n=== Encodage des Variables Catégorielles ===")

        self.df_encoded = self.df_selected.copy()

        # Encoder les valeurs positives/négatives
        mapping = {
            'positive': 1,
            'negative': 0,
            'detected': 1,
            'not_detected': 0
        }

        for col in self.df_encoded.columns:
            if self.df_encoded[col].dtype == 'object':
                self.df_encoded[col] = self.df_encoded[col].map(mapping)

        print(f"Encodage effectué: positive/detected -> 1, negative/not_detected -> 0")
        print(f"\nDistribution de la target après encodage:")
        print(self.df_encoded[self.target].value_counts())

        return self.df_encoded

    def engineer_features(self) -> pd.DataFrame:
        """
        Create engineered features.

        Creates 'est malade' feature from viral test results.

        Returns
        -------
        pd.DataFrame
            Dataframe with engineered features
        """
        if self.df_encoded is None:
            raise ValueError("Exécutez d'abord encode_categorical()")

        if not self.viral_columns:
            self.identify_feature_groups()

        print(f"\n=== Feature Engineering ===")

        self.df_engineered = self.df_encoded.copy()

        # Créer la feature 'est malade' (au moins 1 test viral positif)
        if self.viral_columns:
            self.df_engineered['est malade'] = (
                self.df_engineered[self.viral_columns].sum(axis=1) >= 1
            ).astype(int)

            # Supprimer les colonnes virales d'origine
            self.df_engineered = self.df_engineered.drop(self.viral_columns, axis=1)

            print(f"Feature 'est malade' créée à partir de {len(self.viral_columns)} tests viraux")
            print(f"\nDistribution 'est malade':")
            print(self.df_engineered['est malade'].value_counts())

        print(f"Shape après feature engineering: {self.df_engineered.shape}")

        return self.df_engineered

    def impute_missing_values(self, method: str = 'fillna', fill_value: float = 0) -> pd.DataFrame:
        """
        Handle missing values.

        Parameters
        ----------
        method : str
            Imputation method: 'fillna', 'dropna', or 'median'
        fill_value : float
            Value to fill when method='fillna'

        Returns
        -------
        pd.DataFrame
            Dataframe with imputed values
        """
        if self.df_engineered is None:
            raise ValueError("Exécutez d'abord engineer_features()")

        print(f"\n=== Imputation des Valeurs Manquantes (méthode: {method}) ===")
        print(f"Valeurs manquantes avant: {self.df_engineered.isna().sum().sum()}")

        self.df_imputed = self.df_engineered.copy()

        if method == 'fillna':
            # Remplir les colonnes numériques avec la valeur spécifiée
            numeric_cols = self.df_imputed.select_dtypes(include=['float64', 'int64']).columns
            self.df_imputed[numeric_cols] = self.df_imputed[numeric_cols].fillna(fill_value)

        elif method == 'median':
            # Remplir avec la médiane de chaque colonne
            numeric_cols = self.df_imputed.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                median_val = self.df_imputed[col].median()
                self.df_imputed[col] = self.df_imputed[col].fillna(median_val)

        elif method == 'dropna':
            # Supprimer les lignes avec NaN
            self.df_imputed = self.df_imputed.dropna(axis=0)

        # Toujours supprimer les lignes où la target est manquante
        self.df_imputed = self.df_imputed.dropna(subset=[self.target])

        print(f"Valeurs manquantes après: {self.df_imputed.isna().sum().sum()}")
        print(f"Shape après imputation: {self.df_imputed.shape}")
        print(f"Taux de rétention: {len(self.df_imputed) / len(self.df_original) * 100:.1f}%")

        return self.df_imputed

    def split_data(self, test_size: float = None, random_state: int = None) -> Tuple:
        """
        Split data into train and test sets.

        Parameters
        ----------
        test_size : float, optional
            Test set proportion (default: from config)
        random_state : int, optional
            Random seed (default: from config)

        Returns
        -------
        Tuple
            (X_train, X_test, y_train, y_test)
        """
        if self.df_imputed is None:
            raise ValueError("Exécutez d'abord impute_missing_values()")

        test_size = test_size or TEST_SIZE
        random_state = random_state or RANDOM_STATE

        print(f"\n=== Split Train/Test ===")

        X = self.df_imputed.drop(self.target, axis=1)
        y = self.df_imputed[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Features (X): {X.shape}")
        print(f"Target (y): {y.shape}")
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"\nTrain target distribution:")
        print(self.y_train.value_counts(normalize=True) * 100)
        print(f"\nTest target distribution:")
        print(self.y_test.value_counts(normalize=True) * 100)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def run_full_pipeline(self,
                         threshold: float = 0.9,
                         imputation_method: str = 'fillna',
                         fill_value: float = 0,
                         test_size: float = None,
                         random_state: int = None) -> Tuple:
        """
        Run complete preprocessing pipeline.

        Parameters
        ----------
        threshold : float
            Missing rate threshold for feature selection
        imputation_method : str
            Method for imputation ('fillna', 'median', 'dropna')
        fill_value : float
            Fill value when method='fillna'
        test_size : float, optional
            Test set proportion
        random_state : int, optional
            Random seed

        Returns
        -------
        Tuple
            (X_train, X_test, y_train, y_test)
        """
        print("=" * 70)
        print("PIPELINE DE PREPROCESSING COMPLET")
        print("=" * 70)

        # 1. Sélection des features
        self.select_features_by_missing_rate(threshold)

        # 2. Identification des groupes
        self.identify_feature_groups()

        # 3. Encodage
        self.encode_categorical()

        # 4. Feature engineering
        self.engineer_features()

        # 5. Imputation
        self.impute_missing_values(method=imputation_method, fill_value=fill_value)

        # 6. Split
        self.split_data(test_size=test_size, random_state=random_state)

        print("\n" + "=" * 70)
        print("PREPROCESSING TERMINÉ")
        print("=" * 70)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of preprocessing steps.

        Returns
        -------
        Dict
            Summary information
        """
        summary = {
            'original_shape': self.df_original.shape if self.df_original is not None else None,
            'selected_shape': self.df_selected.shape if self.df_selected is not None else None,
            'final_shape': self.df_imputed.shape if self.df_imputed is not None else None,
            'n_blood_features': len(self.blood_columns),
            'n_viral_features': len(self.viral_columns),
            'train_samples': len(self.X_train) if self.X_train is not None else None,
            'test_samples': len(self.X_test) if self.X_test is not None else None,
            'n_features': self.X_train.shape[1] if self.X_train is not None else None
        }

        return summary
