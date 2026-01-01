# Quick Start Guide

Get started with COVID-19 Detection in 5 minutes!

##  Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

##  Quick Setup

### 1. Navigate to Project Directory

```bash
cd "COVID DETECTION"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Package

```bash
pip install -e .
```

##  Three Ways to Use This Package

### Option 1: Command Line Scripts (Easiest)

#### Train a Model

```bash
python scripts/train_model.py
```

This will:
- Load the dataset
- Preprocess the data
- Train multiple models
- Optimize the best model
- Save results and plots

#### Make Predictions

```bash
python scripts/predict.py data/raw/dataset.xlsx --output predictions.csv
```

### Option 2: Jupyter Notebooks (Interactive)

#### Start Jupyter

```bash
jupyter notebook
```

#### Open Quick Start Notebook

Navigate to `notebooks/00_quick_start.ipynb` and run the cells.

### Option 3: Python Code (Advanced)

#### Complete Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Import our modules
from src.data.preprocessing import load_data, select_features_by_missing_rate, preprocessing
from src.models.train import build_models, optimize_model, save_model
from src.models.evaluate import evaluation

# 1. Load data
df = load_data()
df = select_features_by_missing_rate(df)

# 2. Split data
from src.config import TARGET_FEATURE, TEST_SIZE, RANDOM_STATE
trainset, testset = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=df[TARGET_FEATURE]
)

# 3. Preprocess
from src.data.preprocessing import get_feature_groups
_, viral_columns = get_feature_groups(df)
X_train, y_train = preprocessing(trainset, viral_columns)
X_test, y_test = preprocessing(testset, viral_columns)

# 4. Train model
models = build_models()
svm = models['SVM']

# 5. Optimize
grid = optimize_model(svm, X_train, y_train, n_iter=20)
best_model = grid.best_estimator_

# 6. Evaluate
metrics = evaluation(best_model, X_train, y_train, X_test, y_test)
print(f"F1 Score: {metrics['f1_score']:.3f}")

# 7. Save
save_model(best_model, 'my_model.pkl')
```

##  Expected Results

After training, you should see:

```
Final Results:
  F1 Score:  0.56 (Target: ≥0.5)
  Recall:    0.81 (Target: ≥0.7)
  Precision: 0.71
  AUC:       0.XX
```

##  What Gets Created

After running the training script:

```
data/
├── processed/
│   └── best_covid_model.pkl    # Trained model
└── results/
    ├── learning_curve.png       # Training curves
    ├── precision_recall_curve.png
    ├── roc_curve.png
    └── target_distribution.png
```

##  Common Tasks

### Change Model Parameters

Edit `src/config.py`:

```python
# Example: Change test size
TEST_SIZE = 0.3  # Default: 0.2

# Example: Change polynomial degree
POLYNOMIAL_DEGREE = 3  # Default: 2

# Example: Change decision threshold
DECISION_THRESHOLD = -1.5  # Default: -1
```

### Use Different Model

```python
from src.models.train import build_models

models = build_models()

# Try different models
rf_model = models['RandomForest']
ada_model = models['AdaBoost']
knn_model = models['KNN']
```

### Make Single Prediction

```python
from scripts.predict import predict_single

result = predict_single(
    age_quantile=15,
    blood_features={
        'Leukocytes': -0.09,
        'Platelets': -0.52,
        'Lymphocytes': 0.32,
        'Monocytes': 0.36,
        'Hemoglobin': -0.02
    }
)

print(f"Prediction: {result['prediction']}")
```

### Create Custom Visualization

```python
from src.visualization.plots import plot_feature_vs_target

plot_feature_vs_target(
    df,
    feature='Leukocytes',
    plot_type='dist',
    save=True
)
```

##  Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Make sure you installed the package
pip install -e .

# Or add project to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/COVID DETECTION"
```

### File Not Found

**Problem:** `FileNotFoundError: dataset.xlsx not found`

**Solution:**
- Ensure `dataset.xlsx` is in `data/raw/`
- Or specify path explicitly:
  ```python
  df = load_data('path/to/your/data.xlsx')
  ```

### Memory Issues

**Problem:** `MemoryError` during training

**Solution:**
- Reduce `n_iter` in optimization:
  ```python
  grid = optimize_model(model, X_train, y_train, n_iter=10)
  ```
- Use a simpler model (RandomForest instead of SVM)

### Low Performance

**Problem:** Model doesn't meet targets

**Solution:**
- Try different threshold:
  ```python
  from src.models.evaluate import evaluate_with_threshold
  evaluate_with_threshold(model, X_test, y_test, threshold=-2)
  ```
- Optimize different model
- Add more features

##  Next Steps

1. **Explore the notebooks:**
   - `01_exploratory_data_analysis.ipynb` - Deep dive into data
   - `02_modeling.ipynb` - Advanced modeling techniques

2. **Read the docs:**
   - [README.md](README.md) - Full project overview
   - [docs/API.md](docs/API.md) - Complete API reference
   - [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) - Contribution guide

3. **Experiment:**
   - Try different feature engineering
   - Test new models
   - Tune hyperparameters

4. **Contribute:**
   - Report bugs
   - Suggest features
   - Submit improvements

##  Tips

- **Always use `RANDOM_STATE`** for reproducible results
- **Save your models** before experimenting
- **Version control** your config changes
- **Document** custom features you add
- **Test** on a subset before full training

##  Success!

You're now ready to use the COVID-19 Detection package!

For detailed documentation, see [README.md](README.md).

For questions, open an issue on GitHub.

Happy coding! 
