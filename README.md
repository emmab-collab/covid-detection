# COVID-19 Detection from Clinical Data

A machine learning project to predict COVID-19 infection from clinical and blood test data.

## Project Objective

Predict whether a patient is infected with COVID-19 based on available clinical data, achieving:
- **F1 Score ≥ 0.5**
- **Recall ≥ 0.7**

**Current Performance:**
- **F1 Score: 0.56**
- **Recall: 0.81**

## Dataset

- **Total samples:** 5,644
- **Features:** 111 (originally)
- **Target:** SARS-Cov-2 exam result (positive/negative)
- **Class distribution:** 10% positive, 90% negative (imbalanced)

### Feature Groups
- **Blood tests:** Hemoglobin, Platelets, Leukocytes, Lymphocytes, Monocytes, etc.
- **Viral tests:** Influenza A/B, Rhinovirus, Coronavirus, etc.
- **Demographics:** Patient age quantile

## Project Structure

```
covid-detection/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                   # Package installation
├── .gitignore                 # Git ignore rules
│
├── data/
│   ├── raw/                   # Original dataset
│   │   └── dataset.xlsx
│   ├── processed/             # Processed data and models
│   └── results/               # Plots and results
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_modeling.ipynb
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration and constants
│   ├── data/                  # Data processing
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/              # Feature engineering
│   │   ├── __init__.py
│   │   └── engineering.py
│   ├── models/                # Model training & evaluation
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── visualization/         # Plotting functions
│       ├── __init__.py
│       └── plots.py
│
├── scripts/                   # Executable scripts
│   ├── train_model.py         # Training pipeline
│   └── predict.py             # Prediction script
│
└── tests/                     # Unit tests
    └── test_preprocessing.py
```

## Installation

### 1. Clone the repository
```bash
cd "COVID DETECTION"
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install the package in development mode
```bash
pip install -e .
```

## Usage

### Training a Model

Run the complete training pipeline:

```bash
python scripts/train_model.py
```

This will:
1. Load and preprocess the data
2. Train multiple models (RandomForest, AdaBoost, SVM, KNN)
3. Optimize the best model (SVM)
4. Evaluate performance
5. Save the trained model and plots

### Making Predictions

#### From a file:
```bash
python scripts/predict.py path/to/data.xlsx --output predictions.csv
```

#### With custom threshold:
```bash
python scripts/predict.py data.xlsx --threshold -1.0 --output results.csv
```

#### From Python code:
```python
from scripts.predict import predict_single

# Single patient prediction
result = predict_single(
    age_quantile=15,
    blood_features={
        'Leukocytes': -0.09,
        'Platelets': -0.52,
        'Lymphocytes': 0.32,
        'Monocytes': 0.36
    }
)

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability_positive']:.2%}")
```

### Using in Jupyter Notebooks

```python
# Import modules
from src.data.preprocessing import load_data, preprocessing
from src.models.train import build_models
from src.models.evaluate import evaluation
from src.visualization.plots import plot_target_distribution

# Load data
df = load_data()

# Create visualizations
plot_target_distribution(df, save=True)

# Train models
models = build_models()
```

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Missing value analysis (>75% missing for many features)
- Target distribution (imbalanced: 10% positive)
- Feature-target relationships
- Statistical hypothesis testing (t-tests)

### 2. Preprocessing
- **Feature selection:** Based on missing value patterns
  - Blood columns: 88-90% missing rate
  - Viral columns: 75-88% missing rate
- **Encoding:** positive/detected → 1, negative/not_detected → 0
- **Imputation:** Drop rows with missing values
- **Feature engineering:** Create "est malade" (is sick) from viral tests

### 3. Modeling
- **Models tested:** RandomForest, AdaBoost, SVM, KNN
- **Best model:** SVM with polynomial features
- **Pipeline:**
  - Polynomial features (degree 4)
  - SelectKBest (56 features)
  - StandardScaler
  - SVM (C=1000, gamma=0.001)

### 4. Evaluation
- **Metrics:** F1 Score, Recall, Precision
- **Cross-validation:** 4-fold CV
- **Threshold tuning:** Adjusted to optimize recall
- **Learning curves:** Monitor overfitting

## Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| F1 Score | ≥ 0.5 | **0.56** (passed) |
| Recall | ≥ 0.7 | **0.81** (passed) |
| Precision | - | **0.71** |

### Feature Importance
Top predictive features:
1. Monocytes
2. Platelets
3. Leukocytes
4. Patient age quantile
5. Viral test results (engineered feature)

## Configuration

All parameters can be modified in [`src/config.py`](src/config.py):

```python
# Model parameters
POLYNOMIAL_DEGREE = 2
SELECT_K_BEST = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Performance targets
TARGET_F1_SCORE = 0.5
TARGET_RECALL = 0.7

# Decision threshold
DECISION_THRESHOLD = -1
```

## Development

### Adding New Features

1. Add feature engineering functions to `src/features/engineering.py`
2. Import and use in preprocessing pipeline
3. Update configuration in `src/config.py`

### Adding New Models

1. Add model to `build_models()` in `src/models/train.py`
2. Define hyperparameter grid if needed
3. Run training pipeline

### Running Tests

```bash
pytest tests/
```

## Dependencies

- **numpy** ≥ 1.21.0
- **pandas** ≥ 1.3.0
- **scikit-learn** ≥ 1.0.0
- **matplotlib** ≥ 3.4.0
- **seaborn** ≥ 0.11.0
- **scipy** ≥ 1.7.0
- **openpyxl** ≥ 3.0.0
- **joblib** ≥ 1.0.0

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Authors

- emmab-collab - Initial work

## Acknowledgments

- Dataset source: [Kaggle COVID-19 Dataset]
- Original notebook inspiration
- scikit-learn documentation

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note:** This model is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis.
