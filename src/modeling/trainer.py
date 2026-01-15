"""
Model training and evaluation functions - matching the original notebook approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import learning_curve, RandomizedSearchCV
from sklearn.metrics import (
    f1_score, recall_score, confusion_matrix,
    classification_report, precision_recall_curve
)

from ..config import HYPER_PARAMS, N_ITER, CV_FOLDS, RANDOM_STATE


def build_models():
    """
    Build all classification models with preprocessing pipelines.

    Returns
    -------
    dict
        Dictionary of model names and pipelines
    """
    preprocessor = make_pipeline(
        PolynomialFeatures(2, include_bias=False),
        SelectKBest(f_classif, k=10)
    )

    models = {
        'RandomForest': make_pipeline(
            preprocessor,
            RandomForestClassifier(random_state=RANDOM_STATE)
        ),
        'AdaBoost': make_pipeline(
            preprocessor,
            AdaBoostClassifier(random_state=RANDOM_STATE, algorithm='SAMME')
        ),
        'SVM': make_pipeline(
            preprocessor,
            StandardScaler(),
            SVC(random_state=RANDOM_STATE)
        ),
        'KNN': make_pipeline(
            preprocessor,
            StandardScaler(),
            KNeighborsClassifier()
        )
    }

    return models


def evaluation(model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a model with learning curves.

    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    N, train_score, val_score = learning_curve(
        model, X_train, y_train,
        cv=CV_FOLDS,
        scoring='f1',
        train_sizes=np.linspace(0.1, 1, 10)
    )

    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.xlabel("Nombre d'échantillons d'entraînement")
    plt.ylabel('F1 Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def evaluate_all_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate all models.

    Returns
    -------
    dict
        Dictionary of trained models
    """
    models = build_models()

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"{name}")
        print('='*50)
        evaluation(model, X_train, y_train, X_test, y_test)

    return models


def optimize_svm(X_train, y_train, X_test, y_test):
    """
    Optimize SVM model using RandomizedSearchCV.

    Returns
    -------
    tuple
        (best_estimator, best_params, grid)
    """
    preprocessor = make_pipeline(
        PolynomialFeatures(2, include_bias=False),
        SelectKBest(f_classif, k=10)
    )

    SVM = make_pipeline(
        preprocessor,
        StandardScaler(),
        SVC(random_state=RANDOM_STATE)
    )

    print("Optimisation du SVM avec RandomizedSearchCV...")
    print(f"Hyperparamètres: {HYPER_PARAMS}")

    grid = RandomizedSearchCV(
        SVM,
        HYPER_PARAMS,
        scoring='recall',
        cv=CV_FOLDS,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print(f"\nMeilleurs paramètres: {grid.best_params_}")

    y_pred = grid.predict(X_test)
    print(f"\nRésultats sur le test set:")
    print(classification_report(y_test, y_pred))

    return grid.best_estimator_, grid.best_params_, grid


def plot_precision_recall_curve(model, X_test, y_test):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    model : sklearn estimator
        Trained model with decision_function
    X_test, y_test : array-like
        Test data
    """
    precision, recall, threshold = precision_recall_curve(
        y_test,
        model.decision_function(X_test)
    )

    plt.figure(figsize=(10, 6))
    plt.plot(threshold, precision[:-1], label='precision')
    plt.plot(threshold, recall[:-1], label='recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return precision, recall, threshold


def predict_with_threshold(model, X, threshold=0):
    """
    Make predictions with custom threshold.

    Parameters
    ----------
    model : sklearn estimator
        Trained model with decision_function
    X : array-like
        Features
    threshold : float
        Decision threshold

    Returns
    -------
    array
        Predictions
    """
    return model.decision_function(X) > threshold


def final_evaluation(model, X_test, y_test, threshold=-1):
    """
    Final evaluation with custom threshold.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X_test, y_test : array-like
        Test data
    threshold : float
        Decision threshold

    Returns
    -------
    dict
        F1 and recall scores
    """
    y_pred = predict_with_threshold(model, X_test, threshold)

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Threshold: {threshold}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")

    return {'f1': f1, 'recall': recall}
