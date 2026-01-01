#!/usr/bin/env python
"""
Prediction script for COVID-19 detection model.

This script loads a trained model and makes predictions on new data.
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

from src.config import DECISION_THRESHOLD, TARGET_FEATURE
from src.data.preprocessing import (
    load_data,
    select_features_by_missing_rate,
    preprocessing,
    get_feature_groups
)
from src.models.train import load_model
from src.models.evaluate import model_final


def predict_from_file(
    data_path: str,
    model_path: str = 'best_covid_model.pkl',
    threshold: float = DECISION_THRESHOLD,
    output_path: str = None
) -> pd.DataFrame:
    """
    Make predictions on data from a file.

    Parameters
    ----------
    data_path : str
        Path to data file (Excel or CSV)
    model_path : str, default='best_covid_model.pkl'
        Path to trained model file
    threshold : float, default=DECISION_THRESHOLD
        Decision threshold for classification
    output_path : str, optional
        Path to save predictions. If None, doesn't save.

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions
    """
    print("=" * 80)
    print("COVID-19 DETECTION MODEL - PREDICTION")
    print("=" * 80)

    # Load data
    print(f"\n[1/4] Loading data from {data_path}...")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")

    print(f"Loaded {len(df)} samples")

    # Check if target exists (for validation)
    has_target = TARGET_FEATURE in df.columns

    # Preprocess
    print("\n[2/4] Preprocessing data...")
    df_selected = select_features_by_missing_rate(df)
    _, viral_columns = get_feature_groups(df_selected)

    if has_target:
        X, y = preprocessing(df_selected.copy(), viral_columns)
    else:
        # Create temporary target for preprocessing
        df_selected[TARGET_FEATURE] = 'negative'
        X, _ = preprocessing(df_selected.copy(), viral_columns)

    print(f"Preprocessed features shape: {X.shape}")

    # Load model
    print(f"\n[3/4] Loading model from {model_path}...")
    model = load_model(model_path)

    # Make predictions
    print(f"\n[4/4] Making predictions (threshold={threshold})...")
    predictions = model_final(model, X, threshold=threshold)

    # Create results dataframe
    results = pd.DataFrame({
        'prediction': ['positive' if p else 'negative' for p in predictions],
        'prediction_numeric': predictions.astype(int)
    })

    # Add probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        results['probability_negative'] = probabilities[:, 0]
        results['probability_positive'] = probabilities[:, 1]
    elif hasattr(model, 'decision_function'):
        decision_scores = model.decision_function(X)
        results['decision_score'] = decision_scores

    # Add ground truth if available
    if has_target:
        results['actual'] = y.values
        results['correct'] = results['prediction_numeric'] == results['actual']

        accuracy = results['correct'].mean()
        print(f"\nPrediction Accuracy: {accuracy:.2%}")

        # Confusion matrix
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(results['actual'], results['prediction_numeric'],
                                    target_names=['negative', 'positive']))

    # Display summary
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal samples: {len(results)}")
    print(f"Predicted positive: {(predictions == 1).sum()} ({(predictions == 1).mean():.1%})")
    print(f"Predicted negative: {(predictions == 0).sum()} ({(predictions == 0).mean():.1%})")

    # Save results
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")

    print("=" * 80)

    return results


def predict_single(
    age_quantile: int,
    blood_features: dict,
    viral_tests: dict = None,
    model_path: str = 'best_covid_model.pkl',
    threshold: float = DECISION_THRESHOLD
) -> dict:
    """
    Make prediction for a single patient.

    Parameters
    ----------
    age_quantile : int
        Patient age quantile (0-19)
    blood_features : dict
        Dictionary of blood test results
    viral_tests : dict, optional
        Dictionary of viral test results
    model_path : str, default='best_covid_model.pkl'
        Path to trained model file
    threshold : float, default=DECISION_THRESHOLD
        Decision threshold for classification

    Returns
    -------
    dict
        Prediction results
    """
    # Create dataframe from input
    data = {'Patient age quantile': age_quantile}
    data.update(blood_features)

    if viral_tests:
        data.update(viral_tests)

    df = pd.DataFrame([data])

    # Add target (temporary)
    df[TARGET_FEATURE] = 'negative'

    # Preprocess
    df_selected = select_features_by_missing_rate(df)
    _, viral_columns = get_feature_groups(df_selected)
    X, _ = preprocessing(df_selected.copy(), viral_columns)

    # Load model and predict
    model = load_model(model_path)
    prediction = model_final(model, X, threshold=threshold)[0]

    # Get probability/score
    result = {
        'prediction': 'positive' if prediction else 'negative',
        'prediction_numeric': int(prediction)
    }

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        result['probability_negative'] = float(proba[0])
        result['probability_positive'] = float(proba[1])
    elif hasattr(model, 'decision_function'):
        score = model.decision_function(X)[0]
        result['decision_score'] = float(score)

    return result


def main():
    """Main prediction pipeline with CLI."""
    parser = argparse.ArgumentParser(
        description='Make COVID-19 predictions using trained model'
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to data file (Excel or CSV)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='best_covid_model.pkl',
        help='Path to trained model file (default: best_covid_model.pkl)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=DECISION_THRESHOLD,
        help=f'Decision threshold (default: {DECISION_THRESHOLD})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions (default: None)'
    )

    args = parser.parse_args()

    # Make predictions
    predict_from_file(
        args.data_path,
        args.model,
        args.threshold,
        args.output
    )


if __name__ == "__main__":
    main()
