"""
Model training and optimization functions.
Handles model creation, hyperparameter tuning, and training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Dict, Any, Tuple, Optional
import joblib

from ..config import (
    RANDOM_STATE,
    POLYNOMIAL_DEGREE,
    SELECT_K_BEST,
    SVM_HYPERPARAM_GRID,
    N_ITER,
    OPTIMIZATION_SCORING,
    CV_FOLDS,
    PROCESSED_DATA_DIR
)


def build_preprocessor(
    polynomial_degree: int = POLYNOMIAL_DEGREE,
    k_best: int = SELECT_K_BEST
) -> Any:
    """
    Build the preprocessing pipeline.

    Parameters
    ----------
    polynomial_degree : int, default=2
        Degree for polynomial features
    k_best : int, default=10
        Number of best features to select

    Returns
    -------
    Pipeline
        Preprocessing pipeline
    """
    preprocessor = make_pipeline(
        PolynomialFeatures(polynomial_degree, include_bias=False),
        SelectKBest(f_classif, k=k_best)
    )
    return preprocessor


def build_models() -> Dict[str, Any]:
    """
    Build multiple ML models with preprocessing.

    Returns
    -------
    Dict[str, Pipeline]
        Dictionary of model names and their pipelines
    """
    preprocessor = build_preprocessor()

    models = {
        'RandomForest': make_pipeline(
            preprocessor,
            RandomForestClassifier(random_state=RANDOM_STATE)
        ),
        'AdaBoost': make_pipeline(
            preprocessor,
            AdaBoostClassifier(random_state=RANDOM_STATE)
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


def optimize_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, Any] = None,
    cv: int = CV_FOLDS,
    n_iter: int = N_ITER,
    scoring: str = OPTIMIZATION_SCORING,
    search_type: str = 'randomized'
) -> Any:
    """
    Optimize model hyperparameters using grid or randomized search.

    Parameters
    ----------
    model : estimator
        Model to optimize
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    param_grid : Dict[str, Any], optional
        Hyperparameter grid. If None, uses default SVM grid.
    cv : int, default=4
        Number of cross-validation folds
    n_iter : int, default=40
        Number of iterations for randomized search
    scoring : str, default='recall'
        Scoring metric to optimize
    search_type : str, default='randomized'
        'randomized' or 'grid' search

    Returns
    -------
    estimator
        Fitted search object with best_estimator_ attribute
    """
    if param_grid is None:
        param_grid = SVM_HYPERPARAM_GRID

    if search_type == 'randomized':
        search = RandomizedSearchCV(
            model,
            param_grid,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
    else:
        search = GridSearchCV(
            model,
            param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best {scoring} score: {search.best_score_:.4f}")

    return search


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = 'model'
) -> Any:
    """
    Train a single model.

    Parameters
    ----------
    model : estimator
        Model to train
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    model_name : str, default='model'
        Name for saving the model

    Returns
    -------
    estimator
        Trained model
    """
    print(f"Training {model_name}...")

    model.fit(X_train, y_train)

    print(f"{model_name} training completed.")

    return model


def save_model(model: Any, filename: str = 'best_model.pkl') -> None:
    """
    Save trained model to disk.

    Parameters
    ----------
    model : estimator
        Trained model to save
    filename : str, default='best_model.pkl'
        Filename for the saved model
    """
    filepath = PROCESSED_DATA_DIR / filename
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filename: str = 'best_model.pkl') -> Any:
    """
    Load a trained model from disk.

    Parameters
    ----------
    filename : str, default='best_model.pkl'
        Filename of the saved model

    Returns
    -------
    estimator
        Loaded model
    """
    filepath = PROCESSED_DATA_DIR / filename
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
