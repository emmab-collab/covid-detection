"""
Configuration file for the COVID-19 detection project.
Contains all constants, paths, and hyperparameters.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Ensure directories exist
for directory in [PROCESSED_DATA_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
DATASET_PATH = RAW_DATA_DIR / "dataset.xlsx"

# Target and feature definitions
TARGET_FEATURE = 'SARS-Cov-2 exam result'

# Feature groups - missing rate thresholds
BLOOD_MISSING_RATE_MIN = 0.88
BLOOD_MISSING_RATE_MAX = 0.9
VIRAL_MISSING_RATE_MIN = 0.75
VIRAL_MISSING_RATE_MAX = 0.88

# Key columns to always keep
KEY_COLUMNS = ['Patient age quantile', TARGET_FEATURE]

# Preprocessing parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Encoding mappings
ENCODING_MAP = {
    'positive': 1,
    'negative': 0,
    'detected': 1,
    'not_detected': 0
}

# Model evaluation parameters
CV_FOLDS = 4
SCORING_METRIC = 'f1'

# Performance thresholds
TARGET_F1_SCORE = 0.5
TARGET_RECALL = 0.7

# Visualization settings
FIGURE_SIZE = (12, 8)
HEATMAP_SIZE = (20, 10)
DPI = 100

# Model hyperparameters for optimization
SVM_HYPERPARAM_GRID = {
    'svc__gamma': [1e-3, 1e-4],
    'svc__C': [1, 10, 100, 1000],
    'pipeline__polynomialfeatures__degree': [2, 3, 4],
    'pipeline__selectkbest__k': range(40, 60)
}

# RandomizedSearchCV parameters
N_ITER = 40
OPTIMIZATION_SCORING = 'recall'

# Feature engineering
POLYNOMIAL_DEGREE = 2
SELECT_K_BEST = 10

# Decision threshold for final model
DECISION_THRESHOLD = -1
