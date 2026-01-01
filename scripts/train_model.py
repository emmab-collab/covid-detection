#!/usr/bin/env python
"""
Training script for COVID-19 detection model.

This script handles the complete training pipeline:
1. Load and preprocess data
2. Train multiple models
3. Optimize best model
4. Evaluate and save results
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    TARGET_FEATURE,
    TEST_SIZE,
    RANDOM_STATE,
    DECISION_THRESHOLD
)
from src.data.preprocessing import (
    load_data,
    select_features_by_missing_rate,
    preprocessing,
    get_feature_groups
)
from src.models.train import (
    build_models,
    optimize_model,
    save_model
)
from src.models.evaluate import (
    evaluation,
    evaluate_with_threshold,
    plot_precision_recall_curve,
    plot_roc_curve
)


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("COVID-19 DETECTION MODEL - TRAINING PIPELINE")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/7] Loading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    # Step 2: Select relevant features
    print("\n[2/7] Selecting features based on missing values...")
    df = select_features_by_missing_rate(df)
    print(f"Selected features shape: {df.shape}")

    # Step 3: Train/Test split
    print("\n[3/7] Splitting into train and test sets...")
    trainset, testset = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_FEATURE]
    )

    print(f"Training set size: {trainset.shape[0]}")
    print(f"Test set size: {testset.shape[0]}")
    print(f"\nTarget distribution in training set:")
    print(trainset[TARGET_FEATURE].value_counts())

    # Step 4: Preprocessing
    print("\n[4/7] Preprocessing data...")
    _, viral_columns = get_feature_groups(df)
    X_train, y_train = preprocessing(trainset.copy(), viral_columns)
    X_test, y_test = preprocessing(testset.copy(), viral_columns)

    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Step 5: Train multiple models
    print("\n[5/7] Training multiple models...")
    models = build_models()

    best_model_name = None
    best_f1 = 0

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        try:
            metrics = evaluation(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                show_plots=False
            )

            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_model_name = name

        except Exception as e:
            print(f"Error training {name}: {e}")

    print(f"\nBest performing model: {best_model_name} (F1: {best_f1:.4f})")

    # Step 6: Optimize best model (using SVM as it was the best in original notebook)
    print("\n[6/7] Optimizing SVM model...")
    svm_model = models['SVM']

    grid_search = optimize_model(
        svm_model,
        X_train,
        y_train
    )

    # Step 7: Final evaluation
    print("\n[7/7] Final model evaluation...")
    best_model = grid_search.best_estimator_

    # Evaluate with default threshold
    print("\n--- Evaluation with default threshold ---")
    evaluation(
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
        show_plots=True
    )

    # Plot precision-recall curve
    print("\nGenerating precision-recall curve...")
    plot_precision_recall_curve(best_model, X_test, y_test)

    # Plot ROC curve
    print("\nGenerating ROC curve...")
    auc_score = plot_roc_curve(best_model, X_test, y_test)
    print(f"AUC Score: {auc_score:.4f}")

    # Evaluate with custom threshold
    print(f"\n--- Evaluation with threshold={DECISION_THRESHOLD} ---")
    final_metrics = evaluate_with_threshold(
        best_model,
        X_test,
        y_test,
        threshold=DECISION_THRESHOLD
    )

    # Save model
    print("\nSaving best model...")
    save_model(best_model, 'best_covid_model.pkl')

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  F1 Score:  {final_metrics['f1_score']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  AUC:       {auc_score:.4f}")
    print(f"\nModel saved to: {PROJECT_ROOT / 'data' / 'processed' / 'best_covid_model.pkl'}")
    print(f"Plots saved to: {PROJECT_ROOT / 'data' / 'results' / '*.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
