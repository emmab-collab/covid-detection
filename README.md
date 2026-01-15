# COVID-19 Detection from Clinical Data

Un projet de machine learning pour prédire l'infection au COVID-19 à partir de données cliniques et d'analyses sanguines.

## Objectif

Prédire si un patient est infecté par le COVID-19 en atteignant :
- **F1 Score >= 0.5**
- **Recall >= 0.7**

**Résultats obtenus :** F1 = 0.56, Recall = 0.81

## Dataset

- **Échantillons :** 5,644
- **Caractéristiques :** 111 (initialement)
- **Cible :** SARS-Cov-2 exam result (positif/négatif)
- **Distribution :** 10% positif, 90% négatif

### Groupes de features
- **Analyses sanguines** (88-90% NaN) : Hémoglobine, Plaquettes, Leucocytes, etc.
- **Tests viraux** (75-80% NaN) : Influenza A/B, Rhinovirus, Coronavirus, etc.
- **Démographie** : Patient age quantile

## Structure du projet

```
COVID DETECTION/
├── data/
│   └── raw/dataset.xlsx          # Données originales
├── notebooks/
│   └── 02_Preprocessing_Modeling.ipynb   # Notebook principal
├── src/
│   ├── config.py                 # Configuration
│   ├── preprocessing/            # Fonctions de preprocessing
│   │   └── preprocessor.py
│   ├── modeling/                 # Fonctions de modélisation
│   │   └── trainer.py
│   └── utils/                    # Utilitaires
│       └── data_loader.py
├── scripts/
│   └── train_model.py            # Script d'entraînement
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Utilisation

### Avec le notebook

Ouvrir `notebooks/02_Preprocessing_Modeling.ipynb` et exécuter les cellules.

### Avec le script

```bash
python scripts/train_model.py
```

### En Python

```python
from src.utils import load_data
from src.preprocessing import prepare_data
from src.modeling import evaluate_all_models, optimize_svm, final_evaluation

# Charger et préparer les données
df = load_data()
X_train, X_test, y_train, y_test = prepare_data(df)

# Entraîner les modèles
models = evaluate_all_models(X_train, y_train, X_test, y_test)

# Optimiser le SVM
best_model, best_params, grid = optimize_svm(X_train, y_train, X_test, y_test)

# Évaluation finale
results = final_evaluation(best_model, X_test, y_test, threshold=-1)
```

## Méthodologie

### 1. Preprocessing
- Sélection des features par taux de valeurs manquantes
- Encodage : positive/detected -> 1, negative/not_detected -> 0
- Feature engineering : création de "est malade" à partir des tests viraux
- Imputation : suppression des lignes avec NaN (dropna)

### 2. Modélisation
- **Modèles testés :** RandomForest, AdaBoost, SVM, KNN
- **Pipeline :** PolynomialFeatures(2) -> SelectKBest(k=10) -> StandardScaler -> Model
- **Évaluation :** Learning curves avec cross-validation (cv=4, scoring='f1')

### 3. Optimisation
- **Modèle sélectionné :** SVM (meilleure learning curve)
- **Méthode :** RandomizedSearchCV avec scoring='recall'
- **Hyperparamètres :**
  - `svc__gamma`: [1e-3, 1e-4]
  - `svc__C`: [1, 10, 100, 1000]
  - `polynomialfeatures__degree`: [2, 3, 4]
  - `selectkbest__k`: range(40, 60)

### 4. Threshold tuning
- Courbe Precision-Recall pour trouver le threshold optimal
- Threshold = -1 pour maximiser le recall tout en gardant F1 >= 0.5

## Résultats

| Métrique | Objectif | Obtenu |
|----------|----------|--------|
| F1 Score | >= 0.5   | 0.56   |
| Recall   | >= 0.7   | 0.81   |

## Dépendances

- numpy
- pandas
- scikit-learn
- matplotlib
- openpyxl

## Licence

MIT
