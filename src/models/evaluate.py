"""
Model evaluation functions.
Provides comprehensive model performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import learning_curve
from typing import Any, Tuple

from ..config import (
    TARGET_F1_SCORE,
    TARGET_RECALL,
    CV_FOLDS,
    SCORING_METRIC,
    FIGURE_SIZE,
    RESULTS_DIR
)


def evaluation(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    show_plots: bool = True
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with metrics and visualizations.

    Parameters
    ----------
    model : estimator
        Model to evaluate
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    show_plots : bool, default=True
        Whether to display plots

    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
    # Train model if not already trained
    try:
        y_pred = model.predict(X_test)
    except:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate metrics
    metrics = {
        'f1_score': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred)
    }

    # Learning curve
    if show_plots:
        N, train_score, val_score = learning_curve(
            model,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring=SCORING_METRIC,
            train_sizes=np.linspace(0.1, 1, 10),
            n_jobs=-1
        )

        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(N, train_score.mean(axis=1), label='Train score')
        plt.plot(N, val_score.mean(axis=1), label='Validation score')
        plt.xlabel('Training examples')
        plt.ylabel(f'{SCORING_METRIC.upper()} Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'learning_curve.png', dpi=100)
        plt.show()

    return metrics


def plot_feature_importances(
    model: Any,
    feature_names: list,
    top_n: int = 20
) -> None:
    """
    Plot feature importances for tree-based models.

    Parameters
    ----------
    model : estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, default=20
        Number of top features to display
    """
    # Extract the actual model from pipeline if needed
    if hasattr(model, 'steps'):
        # It's a pipeline, get the last step
        actual_model = model.steps[-1][1]
    else:
        actual_model = model

    if not hasattr(actual_model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return

    # Get feature importances
    importances = actual_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=FIGURE_SIZE)
    plt.title(f'Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_importances.png', dpi=100)
    plt.show()


def plot_precision_recall_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Plot precision-recall curve for the model.

    Parameters
    ----------
    model : estimator
        Trained model with decision_function or predict_proba
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    """
    # Get decision scores
    if hasattr(model, 'decision_function'):
        y_scores = model.decision_function(X_test)
    elif hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        print("Model does not support probability/decision function")
        return

    precision, recall, threshold = precision_recall_curve(y_test, y_scores)

    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(threshold, precision[:-1], label='Precision', linewidth=2)
    plt.plot(threshold, recall[:-1], label='Recall', linewidth=2)
    plt.axhline(y=TARGET_RECALL, color='r', linestyle='--',
                label=f'Target Recall ({TARGET_RECALL})')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'precision_recall_curve.png', dpi=100)
    plt.show()


def plot_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    """
    Plot ROC curve and return AUC score.

    Parameters
    ----------
    model : estimator
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target

    Returns
    -------
    float
        AUC score
    """
    # Get decision scores
    if hasattr(model, 'decision_function'):
        y_scores = model.decision_function(X_test)
    elif hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        print("Model does not support probability/decision function")
        return None

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'roc_curve.png', dpi=100)
    plt.show()

    return auc


def model_final(
    model: Any,
    X: pd.DataFrame,
    threshold: float = 0
) -> np.ndarray:
    """
    Make predictions with a custom decision threshold.

    Parameters
    ----------
    model : estimator
        Trained model
    X : pd.DataFrame
        Features for prediction
    threshold : float, default=0
        Decision threshold

    Returns
    -------
    np.ndarray
        Binary predictions
    """
    if hasattr(model, 'decision_function'):
        return model.decision_function(X) > threshold
    elif hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1] > threshold
    else:
        return model.predict(X)


def evaluate_with_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0
) -> Dict[str, float]:
    """
    Evaluate model with custom decision threshold.

    Parameters
    ----------
    model : estimator
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    threshold : float, default=0
        Decision threshold

    Returns
    -------
    Dict[str, float]
        Evaluation metrics
    """
    y_pred = model_final(model, X_test, threshold)

    metrics = {
        'f1_score': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred)
    }

    print(f"\nMetrics with threshold={threshold}:")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")

    # Check if targets are met
    if metrics['f1_score'] >= TARGET_F1_SCORE and metrics['recall'] >= TARGET_RECALL:
        print(f"\n✓ Targets met! F1 >= {TARGET_F1_SCORE}, Recall >= {TARGET_RECALL}")
    else:
        print(f"\n✗ Targets not met. Required: F1 >= {TARGET_F1_SCORE}, Recall >= {TARGET_RECALL}")

    return metrics
