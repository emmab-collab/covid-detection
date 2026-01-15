"""
Configuration file for the COVID-19 detection project.
"""

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

# Data file
DATASET_PATH = RAW_DATA_DIR / "dataset.xlsx"

# Target
TARGET_FEATURE = 'SARS-Cov-2 exam result'

# Missing rate thresholds for feature groups
BLOOD_MISSING_RATE = (0.88, 0.9)
VIRAL_MISSING_RATE = (0.75, 0.80)

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model hyperparameters for GridSearch
HYPER_PARAMS = {
    'svc__gamma': [1e-3, 1e-4],
    'svc__C': [1, 10, 100, 1000],
    'pipeline__polynomialfeatures__degree': [2, 3, 4],
    'pipeline__selectkbest__k': range(40, 60)
}

# RandomizedSearchCV
N_ITER = 40
CV_FOLDS = 4

# Performance objectives
TARGET_F1_SCORE = 0.5
TARGET_RECALL = 0.7
