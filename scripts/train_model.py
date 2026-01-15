#!/usr/bin/env python
"""
Training script for COVID-19 detection model.
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

from src.utils import load_data
from src.preprocessing import prepare_data
from src.modeling import (
    evaluate_all_models,
    optimize_svm,
    evaluation,
    plot_precision_recall_curve,
    final_evaluation
)
from src.config import TARGET_F1_SCORE, TARGET_RECALL


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("COVID-19 DETECTION - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Chargement des données...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    # 2. Preprocessing
    print("\n[2/5] Preprocessing...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    # 3. Train all models
    print("\n[3/5] Entraînement des modèles...")
    models = evaluate_all_models(X_train, y_train, X_test, y_test)

    # 4. Optimize SVM
    print("\n[4/5] Optimisation du SVM...")
    best_model, best_params, grid = optimize_svm(X_train, y_train, X_test, y_test)

    # 5. Final evaluation
    print("\n[5/5] Évaluation finale...")
    evaluation(best_model, X_train, y_train, X_test, y_test)
    plot_precision_recall_curve(best_model, X_test, y_test)
    results = final_evaluation(best_model, X_test, y_test, threshold=-1)

    # Summary
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"\nObjectifs:")
    print(f"  F1 >= {TARGET_F1_SCORE}: {'OUI' if results['f1'] >= TARGET_F1_SCORE else 'NON'} ({results['f1']:.4f})")
    print(f"  Recall >= {TARGET_RECALL}: {'OUI' if results['recall'] >= TARGET_RECALL else 'NON'} ({results['recall']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
