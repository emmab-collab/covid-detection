# API Documentation

Complete API reference for the COVID-19 Detection package.

## Table of Contents
- [Data Processing](#data-processing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)

---

## Data Processing

Module: `src.data.preprocessing`

### `load_data(filepath=None)`

Load the COVID-19 dataset from Excel file.

**Parameters:**
- `filepath` (str, optional): Path to dataset. Defaults to config value.

**Returns:**
- `pd.DataFrame`: Loaded dataset

**Example:**
```python
from src.data.preprocessing import load_data
df = load_data()
```

---

### `get_feature_groups(df)`

Identify blood and viral feature columns based on missing rate.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `Tuple[List[str], List[str]]`: (blood_columns, viral_columns)

**Example:**
```python
blood_cols, viral_cols = get_feature_groups(df)
print(f"Blood features: {blood_cols}")
```

---

### `select_features_by_missing_rate(df)`

Select relevant features based on missing value patterns.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `pd.DataFrame`: Dataframe with selected features

---

### `encodage(df)`

Encode categorical variables to numerical values.

**Mapping:**
- positive/detected → 1
- negative/not_detected → 0

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `pd.DataFrame`: Encoded dataframe

---

### `imputation(df, method='dropna')`

Handle missing values in the dataframe.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `method` (str): 'dropna', 'fillna', or 'flag'

**Returns:**
- `pd.DataFrame`: Dataframe with handled missing values

---

### `preprocessing(df, viral_columns=None, return_viral_columns=False)`

Complete preprocessing pipeline.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `viral_columns` (List[str], optional): Viral columns for feature engineering
- `return_viral_columns` (bool): Whether to return viral_columns list

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: (X, y)

**Example:**
```python
X_train, y_train = preprocessing(trainset)
```

---

## Feature Engineering

Module: `src.features.engineering`

### `feature_engineering(df, viral_columns)`

Create engineered features from viral test results.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `viral_columns` (List[str]): Viral test column names

**Returns:**
- `pd.DataFrame`: Dataframe with engineered features

**Example:**
```python
from src.features.engineering import feature_engineering
df_engineered = feature_engineering(df, viral_columns)
```

---

### `create_blood_ratios(df)`

Create ratio features from blood test results (NLR, PLR, MLR).

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `pd.DataFrame`: Dataframe with ratio features

---

## Model Training

Module: `src.models.train`

### `build_models()`

Build multiple ML models with preprocessing.

**Returns:**
- `Dict[str, Pipeline]`: Dictionary of model names and pipelines

**Example:**
```python
from src.models.train import build_models
models = build_models()
for name, model in models.items():
    model.fit(X_train, y_train)
```

---

### `optimize_model(model, X_train, y_train, param_grid=None, cv=4, n_iter=40, scoring='recall', search_type='randomized')`

Optimize model hyperparameters.

**Parameters:**
- `model`: Model to optimize
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training target
- `param_grid` (Dict, optional): Hyperparameter grid
- `cv` (int): Cross-validation folds
- `n_iter` (int): Iterations for randomized search
- `scoring` (str): Scoring metric
- `search_type` (str): 'randomized' or 'grid'

**Returns:**
- Fitted search object with `best_estimator_` attribute

**Example:**
```python
from src.models.train import optimize_model
grid = optimize_model(model, X_train, y_train)
best_model = grid.best_estimator_
```

---

### `save_model(model, filename='best_model.pkl')`

Save trained model to disk.

**Parameters:**
- `model`: Trained model
- `filename` (str): Filename for saved model

---

### `load_model(filename='best_model.pkl')`

Load trained model from disk.

**Parameters:**
- `filename` (str): Filename of saved model

**Returns:**
- Loaded model

---

## Model Evaluation

Module: `src.models.evaluate`

### `evaluation(model, X_train, y_train, X_test, y_test, show_plots=True)`

Comprehensive model evaluation with metrics and visualizations.

**Parameters:**
- `model`: Model to evaluate
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training target
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test target
- `show_plots` (bool): Whether to display plots

**Returns:**
- `Dict[str, float]`: Evaluation metrics

**Example:**
```python
from src.models.evaluate import evaluation
metrics = evaluation(model, X_train, y_train, X_test, y_test)
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

---

### `model_final(model, X, threshold=0)`

Make predictions with custom decision threshold.

**Parameters:**
- `model`: Trained model
- `X` (pd.DataFrame): Features
- `threshold` (float): Decision threshold

**Returns:**
- `np.ndarray`: Binary predictions

---

### `evaluate_with_threshold(model, X_test, y_test, threshold=0)`

Evaluate model with custom decision threshold.

**Parameters:**
- `model`: Trained model
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test target
- `threshold` (float): Decision threshold

**Returns:**
- `Dict[str, float]`: Evaluation metrics

---

### `plot_precision_recall_curve(model, X_test, y_test)`

Plot precision-recall curve.

**Parameters:**
- `model`: Trained model
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test target

---

### `plot_roc_curve(model, X_test, y_test)`

Plot ROC curve and return AUC score.

**Parameters:**
- `model`: Trained model
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test target

**Returns:**
- `float`: AUC score

---

## Visualization

Module: `src.visualization.plots`

### `plot_target_distribution(df, save=False)`

Plot distribution of target variable.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `save` (bool): Whether to save plot

**Example:**
```python
from src.visualization.plots import plot_target_distribution
plot_target_distribution(df, save=True)
```

---

### `plot_missing_values_heatmap(df, save=False)`

Plot heatmap of missing values.

---

### `plot_correlation_heatmap(df, columns=None, method='pearson', save=False)`

Plot correlation heatmap for numerical features.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `columns` (List[str], optional): Specific columns
- `method` (str): 'pearson', 'spearman', or 'kendall'
- `save` (bool): Whether to save plot

---

### `plot_feature_vs_target(df, feature, target=TARGET_FEATURE, plot_type='auto', save=False)`

Plot relationship between feature and target.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `feature` (str): Feature column name
- `target` (str): Target column name
- `plot_type` (str): 'auto', 'box', 'violin', 'count', or 'dist'
- `save` (bool): Whether to save plot

---

## Configuration

Module: `src.config`

### Key Constants

```python
# Paths
PROJECT_ROOT      # Project root directory
DATA_DIR          # Data directory
RAW_DATA_DIR      # Raw data directory
PROCESSED_DATA_DIR # Processed data directory
RESULTS_DIR       # Results directory

# Target and features
TARGET_FEATURE    # 'SARS-Cov-2 exam result'

# Preprocessing
TEST_SIZE         # 0.2
RANDOM_STATE      # 42

# Model parameters
POLYNOMIAL_DEGREE # 2
SELECT_K_BEST     # 10
CV_FOLDS          # 4

# Performance targets
TARGET_F1_SCORE   # 0.5
TARGET_RECALL     # 0.7
```

**Example:**
```python
from src.config import TARGET_FEATURE, RANDOM_STATE
print(f"Target: {TARGET_FEATURE}")
```
