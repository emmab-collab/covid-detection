"""
Model Trainer class for comprehensive model training and evaluation.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, learning_curve
from sklearn.metrics import (
    f1_score, recall_score, precision_score, classification_report,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
)
import joblib

from ..config import (
    RANDOM_STATE,
    POLYNOMIAL_DEGREE,
    SELECT_K_BEST,
    TARGET_F1_SCORE,
    TARGET_RECALL,
    CV_FOLDS
)


class ModelTrainer:
    """
    Comprehensive Model Training and Evaluation class.

    Handles model creation, training, optimization, evaluation,
    and comparison of multiple models.
    """

    def __init__(self, X_train=None, X_test=None, y_train=None, y_test=None):
        """
        Initialize Model Trainer.

        Parameters
        ----------
        X_train, X_test : array-like
            Training and test features
        y_train, y_test : array-like
            Training and test targets
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.optimized_model = None
        self.best_params = None
        self.best_threshold = 0

    def build_models(self) -> Dict[str, Any]:
        """
        Build all classification models with preprocessing pipelines.

        Returns
        -------
        Dict[str, Any]
            Dictionary of model names and pipelines
        """
        print("=== Construction des Modèles ===\n")

        self.models = {
            'Random Forest': make_pipeline(
                PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False),
                SelectKBest(f_classif, k=SELECT_K_BEST),
                StandardScaler(),
                RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
            ),
            'AdaBoost': make_pipeline(
                PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False),
                SelectKBest(f_classif, k=SELECT_K_BEST),
                StandardScaler(),
                AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=100, algorithm='SAMME')
            ),
            'SVM': make_pipeline(
                PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False),
                SelectKBest(f_classif, k=SELECT_K_BEST),
                StandardScaler(),
                SVC(random_state=RANDOM_STATE, kernel='rbf', probability=True)
            ),
            'KNN': make_pipeline(
                PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False),
                SelectKBest(f_classif, k=SELECT_K_BEST),
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=5)
            )
        }

        for name, model in self.models.items():
            print(f"{name}:")
            for step_name, step in model.named_steps.items():
                print(f"  - {step_name}: {step.__class__.__name__}")
            print()

        return self.models

    def train_and_evaluate_all(self) -> Dict[str, Dict]:
        """
        Train and evaluate all models.

        Returns
        -------
        Dict[str, Dict]
            Results for each model
        """
        if not self.models:
            self.build_models()

        print("=" * 70)
        print("ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
        print("=" * 70)

        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Entraînement: {name}")
            print('='*60)

            # Entraînement
            model.fit(self.X_train, self.y_train)

            # Prédictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)

            # Métriques
            train_f1 = f1_score(self.y_train, y_pred_train)
            test_f1 = f1_score(self.y_test, y_pred_test)
            train_recall = recall_score(self.y_train, y_pred_train)
            test_recall = recall_score(self.y_test, y_pred_test)

            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                       cv=CV_FOLDS, scoring='f1')

            self.results[name] = {
                'model': model,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'train_recall': train_recall,
                'test_recall': test_recall,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std()
            }

            print(f"\nTrain F1: {train_f1:.3f}")
            print(f"Test F1: {test_f1:.3f}")
            print(f"Train Recall: {train_recall:.3f}")
            print(f"Test Recall: {test_recall:.3f}")
            print(f"CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

            print(f"\nClassification Report (Test):")
            print(classification_report(self.y_test, y_pred_test,
                                       target_names=['Negative', 'Positive']))

        print("\n" + "=" * 70)
        print("ENTRAÎNEMENT TERMINÉ")
        print("=" * 70)

        return self.results

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get comparison table of all models.

        Returns
        -------
        pd.DataFrame
            Comparison dataframe sorted by Test F1
        """
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train F1': [r['train_f1'] for r in self.results.values()],
            'Test F1': [r['test_f1'] for r in self.results.values()],
            'Train Recall': [r['train_recall'] for r in self.results.values()],
            'Test Recall': [r['test_recall'] for r in self.results.values()],
            'CV F1 (mean)': [r['cv_f1_mean'] for r in self.results.values()],
            'CV F1 (std)': [r['cv_f1_std'] for r in self.results.values()]
        })

        comparison_df = comparison_df.sort_values('Test F1', ascending=False)

        self.best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = self.results[self.best_model_name]['model']

        return comparison_df

    def plot_model_comparison(self, figsize: Tuple[int, int] = (16, 6)):
        """
        Plot comparison of model performances.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size
        """
        comparison_df = self.get_comparison_dataframe()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        x = np.arange(len(comparison_df))
        width = 0.35

        # F1 Score comparison
        axes[0].bar(x - width/2, comparison_df['Train F1'], width,
                   label='Train', alpha=0.8)
        axes[0].bar(x + width/2, comparison_df['Test F1'], width,
                   label='Test', alpha=0.8)
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('F1 Score')
        axes[0].set_title('Comparaison F1 Score par Modèle')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(comparison_df['Model'], rotation=45)
        axes[0].axhline(y=TARGET_F1_SCORE, color='r', linestyle='--',
                       label=f'Target F1 = {TARGET_F1_SCORE}')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Recall comparison
        axes[1].bar(x - width/2, comparison_df['Train Recall'], width,
                   label='Train', alpha=0.8)
        axes[1].bar(x + width/2, comparison_df['Test Recall'], width,
                   label='Test', alpha=0.8)
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('Recall')
        axes[1].set_title('Comparaison Recall par Modèle')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(comparison_df['Model'], rotation=45)
        axes[1].axhline(y=TARGET_RECALL, color='r', linestyle='--',
                       label=f'Target Recall = {TARGET_RECALL}')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, figsize: Tuple[int, int] = (14, 12)):
        """
        Plot confusion matrices for all models.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for idx, (name, result) in enumerate(self.results.items()):
            model = result['model']
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            axes[idx].set_title(f'{name}\nF1={result["test_f1"]:.3f}, Recall={result["test_recall"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.suptitle('Confusion Matrices - Tous les Modèles', fontsize=16, y=1.00)
        plt.tight_layout()
        plt.show()

    def optimize_model(self, param_grid: Dict = None, n_iter: int = 50,
                      cv: int = None) -> Tuple[Any, Dict, float]:
        """
        Optimize best model using RandomizedSearchCV.

        Parameters
        ----------
        param_grid : Dict, optional
            Parameter grid for optimization
        n_iter : int
            Number of iterations
        cv : int, optional
            Number of CV folds

        Returns
        -------
        Tuple
            (optimized_model, best_params, best_score)
        """
        if self.best_model is None:
            self.get_comparison_dataframe()

        cv = cv or CV_FOLDS

        print(f"\n=== Optimisation de {self.best_model_name} ===")
        print(f"Performance initiale - F1: {self.results[self.best_model_name]['test_f1']:.3f}")

        if param_grid is None:
            # Grilles par défaut
            param_grids = {
                'Random Forest': {
                    'polynomialfeatures__degree': [2, 3, 4],
                    'selectkbest__k': [10, 20, 30, 40, 50],
                    'randomforestclassifier__n_estimators': [100, 200, 300],
                    'randomforestclassifier__max_depth': [10, 20, 30, None],
                },
                'AdaBoost': {
                    'polynomialfeatures__degree': [2, 3, 4],
                    'selectkbest__k': [10, 20, 30, 40, 50],
                    'adaboostclassifier__n_estimators': [50, 100, 200],
                    'adaboostclassifier__learning_rate': [0.01, 0.1, 1.0]
                },
                'SVM': {
                    'polynomialfeatures__degree': [2, 3, 4],
                    'selectkbest__k': [10, 20, 30, 40, 50],
                    'svc__C': [0.1, 1, 10, 100, 1000],
                    'svc__gamma': [0.001, 0.01, 0.1, 1]
                },
                'KNN': {
                    'polynomialfeatures__degree': [2, 3, 4],
                    'selectkbest__k': [10, 20, 30, 40, 50],
                    'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 11],
                    'kneighborsclassifier__weights': ['uniform', 'distance']
                }
            }
            param_grid = param_grids.get(self.best_model_name, {})

        print(f"\nLancement de RandomizedSearchCV (n_iter={n_iter}, cv={cv})...")

        random_search = RandomizedSearchCV(
            self.best_model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='f1',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        random_search.fit(self.X_train, self.y_train)

        self.optimized_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        best_score = random_search.best_score_

        print(f"\nOptimisation terminée!")
        print(f"Meilleur score CV: {best_score:.3f}")
        print(f"\nMeilleurs paramètres:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        return self.optimized_model, self.best_params, best_score

    def evaluate_model(self, model=None, show_plots: bool = True):
        """
        Comprehensive model evaluation.

        Parameters
        ----------
        model : optional
            Model to evaluate (default: optimized_model or best_model)
        show_plots : bool
            Whether to show plots
        """
        model = model or self.optimized_model or self.best_model

        if model is None:
            raise ValueError("Aucun modèle à évaluer")

        print("\n=== Évaluation du Modèle ===")

        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        print("\nPerformances Train:")
        print(classification_report(self.y_train, y_pred_train,
                                   target_names=['Negative', 'Positive']))

        print("\nPerformances Test:")
        print(classification_report(self.y_test, y_pred_test,
                                   target_names=['Negative', 'Positive']))

        if show_plots:
            # Confusion Matrix
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            cm_train = confusion_matrix(self.y_train, y_pred_train)
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            axes[0].set_title('Confusion Matrix - Train')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')

            cm_test = confusion_matrix(self.y_test, y_pred_test)
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            axes[1].set_title('Confusion Matrix - Test')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')

            plt.tight_layout()
            plt.show()

            # ROC Curve
            self.plot_roc_curve(model)

    def plot_roc_curve(self, model=None) -> float:
        """
        Plot ROC curve.

        Parameters
        ----------
        model : optional
            Model to evaluate

        Returns
        -------
        float
            AUC score
        """
        model = model or self.optimized_model or self.best_model

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(self.X_test)[:, 1]
        else:
            y_proba = model.decision_function(self.X_test)

        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc_score = roc_auc_score(self.y_test, y_proba)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return auc_score

    def tune_threshold(self, model=None, threshold_range: Tuple[float, float] = (-3, 1),
                      step: float = 0.1) -> float:
        """
        Find optimal decision threshold.

        Parameters
        ----------
        model : optional
            Model to tune
        threshold_range : Tuple[float, float]
            Range of thresholds to test
        step : float
            Step size

        Returns
        -------
        float
            Best threshold
        """
        model = model or self.optimized_model or self.best_model

        print("\n=== Tuning du Threshold de Décision ===")

        if hasattr(model, 'decision_function'):
            y_scores = model.decision_function(self.X_test)
        else:
            y_scores = model.predict_proba(self.X_test)[:, 1]

        thresholds = np.arange(threshold_range[0], threshold_range[1], step)
        threshold_results = []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            f1 = f1_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)

            threshold_results.append({
                'threshold': threshold,
                'f1': f1,
                'recall': recall,
                'precision': precision
            })

        threshold_df = pd.DataFrame(threshold_results)

        # Sélectionner le meilleur threshold
        valid_thresholds = threshold_df[threshold_df['recall'] >= TARGET_RECALL]
        if len(valid_thresholds) > 0:
            best_row = valid_thresholds.sort_values('f1', ascending=False).iloc[0]
            self.best_threshold = best_row['threshold']

            print(f"\nMeilleur threshold: {self.best_threshold:.2f}")
            print(f"F1 Score: {best_row['f1']:.3f}")
            print(f"Recall: {best_row['recall']:.3f}")
            print(f"Precision: {best_row['precision']:.3f}")
        else:
            print("Aucun threshold ne satisfait le recall cible")
            self.best_threshold = 0

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(threshold_df['threshold'], threshold_df['f1'], label='F1 Score', linewidth=2)
        plt.plot(threshold_df['threshold'], threshold_df['recall'], label='Recall', linewidth=2)
        plt.plot(threshold_df['threshold'], threshold_df['precision'], label='Precision', linewidth=2)
        plt.axhline(y=TARGET_F1_SCORE, color='r', linestyle='--', alpha=0.5,
                   label=f'Target F1 = {TARGET_F1_SCORE}')
        plt.axhline(y=TARGET_RECALL, color='g', linestyle='--', alpha=0.5,
                   label=f'Target Recall = {TARGET_RECALL}')
        plt.axvline(x=self.best_threshold, color='purple', linestyle=':', alpha=0.7,
                   label=f'Best Threshold = {self.best_threshold:.2f}')
        plt.xlabel('Decision Threshold')
        plt.ylabel('Score')
        plt.title('Impact du Threshold sur les Métriques')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return self.best_threshold

    def save_model(self, filepath: str = 'best_model.pkl'):
        """
        Save trained model to file.

        Parameters
        ----------
        filepath : str
            Path to save model
        """
        model = self.optimized_model or self.best_model

        if model is None:
            raise ValueError("Aucun modèle à sauvegarder")

        joblib.dump(model, filepath)
        print(f"\nModèle sauvegardé: {filepath}")
        print(f"Threshold recommandé: {self.best_threshold:.2f}")

    def get_training_summary(self) -> Dict:
        """
        Get summary of training results.

        Returns
        -------
        Dict
            Training summary
        """
        summary = {
            'n_models_tested': len(self.results),
            'best_model': self.best_model_name,
            'best_f1': self.results[self.best_model_name]['test_f1'] if self.best_model_name else None,
            'best_recall': self.results[self.best_model_name]['test_recall'] if self.best_model_name else None,
            'optimized': self.optimized_model is not None,
            'best_threshold': self.best_threshold
        }

        return summary
