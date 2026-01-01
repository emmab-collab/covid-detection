"""
Unit tests for preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    encodage,
    imputation,
    get_feature_groups,
    select_features_by_missing_rate
)
from src.features.engineering import feature_engineering


class TestEncodage:
    """Test encoding functions."""

    def test_encodage_positive_negative(self):
        """Test encoding of positive/negative values."""
        df = pd.DataFrame({
            'test': ['positive', 'negative', 'positive']
        })
        result = encodage(df)
        assert result['test'].tolist() == [1, 0, 1]

    def test_encodage_detected_not_detected(self):
        """Test encoding of detected/not_detected values."""
        df = pd.DataFrame({
            'viral': ['detected', 'not_detected', 'detected']
        })
        result = encodage(df)
        assert result['viral'].tolist() == [1, 0, 1]

    def test_encodage_preserves_numeric(self):
        """Test that numeric columns are preserved."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'categorical': ['positive', 'negative', 'positive']
        })
        result = encodage(df)
        assert result['numeric'].tolist() == [1, 2, 3]


class TestImputation:
    """Test imputation functions."""

    def test_imputation_dropna(self):
        """Test dropna imputation method."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, np.nan, 7, 8]
        })
        result = imputation(df, method='dropna')
        assert len(result) == 2
        assert not result.isna().any().any()

    def test_imputation_fillna(self):
        """Test fillna imputation method."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, np.nan, 7, 8]
        })
        result = imputation(df, method='fillna')
        assert len(result) == 4
        assert not result.isna().any().any()
        assert -999 in result.values

    def test_imputation_flag(self):
        """Test flag imputation method."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, np.nan, 7, 8]
        })
        result = imputation(df, method='flag')
        assert 'has_missing' in result.columns
        assert result['has_missing'].sum() == 2


class TestFeatureGroups:
    """Test feature group identification."""

    def test_get_feature_groups(self):
        """Test identification of blood and viral columns."""
        # Create mock dataframe with specific missing rates
        n_rows = 1000

        df = pd.DataFrame({
            'target': ['positive'] * n_rows,
            'blood1': [np.nan if i < 890 else 1 for i in range(n_rows)],  # 89% missing
            'blood2': [np.nan if i < 885 else 1 for i in range(n_rows)],  # 88.5% missing
            'viral1': [np.nan if i < 760 else 1 for i in range(n_rows)],  # 76% missing
            'viral2': [np.nan if i < 800 else 1 for i in range(n_rows)],  # 80% missing
            'other': [1] * n_rows  # 0% missing
        })

        blood_cols, viral_cols = get_feature_groups(df)

        assert 'blood1' in blood_cols
        assert 'blood2' in blood_cols
        assert 'viral1' in viral_cols
        # viral2 has 80% missing, which is not in the 75-88% range
        assert 'other' not in blood_cols
        assert 'other' not in viral_cols


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_feature_engineering_creates_est_malade(self):
        """Test that 'est malade' feature is created."""
        df = pd.DataFrame({
            'age': [10, 20, 30],
            'viral1': [1, 0, 0],
            'viral2': [0, 1, 0]
        })
        viral_cols = ['viral1', 'viral2']

        result = feature_engineering(df, viral_cols)

        assert 'est malade' in result.columns
        assert result['est malade'].tolist() == [True, True, False]

    def test_feature_engineering_drops_viral_columns(self):
        """Test that viral columns are dropped."""
        df = pd.DataFrame({
            'age': [10, 20, 30],
            'viral1': [1, 0, 0],
            'viral2': [0, 1, 0]
        })
        viral_cols = ['viral1', 'viral2']

        result = feature_engineering(df, viral_cols)

        assert 'viral1' not in result.columns
        assert 'viral2' not in result.columns

    def test_feature_engineering_preserves_other_columns(self):
        """Test that non-viral columns are preserved."""
        df = pd.DataFrame({
            'age': [10, 20, 30],
            'blood': [1.5, 2.3, 1.8],
            'viral1': [1, 0, 0]
        })
        viral_cols = ['viral1']

        result = feature_engineering(df, viral_cols)

        assert 'age' in result.columns
        assert 'blood' in result.columns
        assert result['age'].tolist() == [10, 20, 30]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
