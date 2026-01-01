# COVID-19 Detection from Clinical Data

Un projet de machine learning pour prédire l'infection au COVID-19 à partir de données cliniques et d'analyses sanguines.

## Project Objective

Prédire si un patient est infecté par le COVID-19 sur la base des données cliniques disponibles, en atteignant :
- **F1 Score ≥ 0.5**
- **Recall ≥ 0.7**

**Performance actuelle :**
- **F1 Score : 0.56**
- **Recall : 0.81**

## Dataset

- **Échantillons totaux :** 5,644
- **Caractéristiques :** 111 (initialement)
- **Cible :** Résultat de l'examen SARS-Cov-2 (positif/négatif)
- **Distribution des classes :** 10% positif, 90% négatif (déséquilibré)

### Feature Groups
- **Analyses sanguines :** Hémoglobine, Plaquettes, Leucocytes, Lymphocytes, Monocytes, etc.
- **Tests viraux :** Influenza A/B, Rhinovirus, Coronavirus, etc.
- **Démographie :** Quantile d'âge du patient

## Project Structure

```
covid-detection/
├── README.md                   # Documentation
├── LICENSE                     # Licence MIT
├── requirements.txt            # Dépendances Python
├── setup.py                   # Installation du package
├── .gitignore                 # Règles Git
│
├── data/
│   ├── raw/                   # Données originales
│   │   └── dataset.xlsx
│   ├── processed/             # Données traitées et modèles
│   └── results/               # Graphiques et résultats
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Analyse exploratoire (avec EDAAnalyzer)
│   └── 02_Preprocessing_Modeling.ipynb  # Preprocessing & ML (avec classes)
│
├── src/                       # Code source (architecture OOP modulaire)
│   ├── __init__.py
│   ├── config.py              # Configuration centralisée
│   ├── eda/                   # Analyse exploratoire (classe)
│   │   ├── __init__.py
│   │   └── analyzer.py        # EDAAnalyzer class
│   ├── preprocessing/         # Preprocessing (classe)
│   │   ├── __init__.py
│   │   ├── preprocessor.py    # DataPreprocessor class
│   │   └── feature_engineering.py  # Fonctions de feature engineering
│   ├── modeling/              # Modeling (classe)
│   │   ├── __init__.py
│   │   └── trainer.py         # ModelTrainer class
│   └── utils/                 # Utilitaires partagés
│       ├── __init__.py
│       ├── data_loader.py     # load_data()
│       └── visualization.py   # Fonctions de visualisation
│
├── scripts/                   # Scripts exécutables
│   ├── train_model.py         # Pipeline d'entraînement
│   ├── predict.py             # Script de prédiction
│   └── check_installation.py  # Vérification installation
│
├── tests/                     # Tests unitaires
│   └── test_preprocessing.py
│
└── docs/                      # Documentation
    ├── API.md                 # Référence API
    └── CONTRIBUTING.md        # Guide de contribution
```

## Installation

### 1. Cloner le repository
```bash
cd "COVID DETECTION"
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Installer le package en mode développement
```bash
pip install -e .
```

## Usage

### Training a Model

Exécuter le pipeline complet d'entraînement :

```bash
python scripts/train_model.py
```

Ceci va :
1. Charger et prétraiter les données
2. Entraîner plusieurs modèles (RandomForest, AdaBoost, SVM, KNN)
3. Optimiser le meilleur modèle (SVM)
4. Évaluer la performance
5. Sauvegarder le modèle entraîné et les graphiques

### Making Predictions

#### À partir d'un fichier :
```bash
python scripts/predict.py chemin/vers/data.xlsx --output predictions.csv
```

#### Avec un seuil personnalisé :
```bash
python scripts/predict.py data.xlsx --threshold -1.0 --output resultats.csv
```

#### Depuis du code Python :
```python
from scripts.predict import predict_single

# Prédiction pour un patient unique
resultat = predict_single(
    age_quantile=15,
    blood_features={
        'Leukocytes': -0.09,
        'Platelets': -0.52,
        'Lymphocytes': 0.32,
        'Monocytes': 0.36
    }
)

print(f"Prédiction : {resultat['prediction']}")
print(f"Probabilité : {resultat['probability_positive']:.2%}")
```

### Using in Jupyter Notebooks

#### Approche Orientée Objet (Recommandée)

```python
# Importer les classes
from src.eda import EDAAnalyzer
from src.preprocessing import DataPreprocessor
from src.modeling import ModelTrainer
from src.utils import load_data

# 1. EDA - Analyse exploratoire
df = load_data()
eda = EDAAnalyzer(df)
eda.analyze_shape()
eda.plot_target_distribution()
blood_cols, viral_cols = eda.identify_feature_groups()
significant_features = eda.statistical_tests(blood_cols)

# 2. Preprocessing - Pipeline complet
preprocessor = DataPreprocessor(df)
X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline(
    threshold=0.9,
    imputation_method='fillna'
)

# 3. Modeling - Entraînement et évaluation
trainer = ModelTrainer(X_train, X_test, y_train, y_test)
results = trainer.train_and_evaluate_all()
comparison = trainer.get_comparison_dataframe()
trainer.plot_model_comparison()

# 4. Optimisation
optimized_model, params, score = trainer.optimize_model(n_iter=50)
trainer.evaluate_model()
best_threshold = trainer.tune_threshold()
trainer.save_model('best_model.pkl')
```

#### Utilisation des fonctions utilitaires

```python
# Import des utilitaires
from src.utils import load_data, plot_target_distribution, plot_missing_rate

# Charger les données
df = load_data()

# Créer des visualisations
plot_target_distribution(df, save=True)
plot_missing_rate(df, top_n=20)
```

## Methodology

### Architecture Modulaire

Le projet utilise une architecture orientée objet avec **3 classes principales** :

#### 1. **EDAAnalyzer** (`src/eda/analyzer.py`)
Classe pour l'analyse exploratoire des données:
- `analyze_shape()`: Informations sur la structure du dataset
- `analyze_missing_values()`: Analyse des valeurs manquantes
- `plot_missing_heatmap()`: Visualisation des NaN
- `analyze_target()`: Distribution de la cible
- `identify_feature_groups()`: Identification groupes blood/viral
- `plot_feature_distributions()`: Distributions des features
- `compare_distributions_by_target()`: Comparaison positifs vs négatifs
- `plot_correlation_matrix()`: Matrice de corrélation
- `statistical_tests()`: Tests t de Student pour features significatives
- `generate_summary_report()`: Rapport complet

#### 2. **DataPreprocessor** (`src/preprocessing/preprocessor.py`)
Classe pour le preprocessing complet:
- `select_features_by_missing_rate()`: Sélection par seuil de NaN
- `identify_feature_groups()`: Groupes blood et viral
- `encode_categorical()`: Encodage positif/négatif → 1/0
- `engineer_features()`: Création feature 'est malade'
- `impute_missing_values()`: Imputation (fillna/median/dropna)
- `split_data()`: Split train/test stratifié
- `run_full_pipeline()`: Pipeline complet automatisé
- `get_preprocessing_summary()`: Résumé des transformations

#### 3. **ModelTrainer** (`src/modeling/trainer.py`)
Classe pour l'entraînement et l'évaluation:
- `build_models()`: Construction de 4 modèles (RF, AdaBoost, SVM, KNN)
- `train_and_evaluate_all()`: Entraînement et évaluation de tous
- `get_comparison_dataframe()`: Tableau comparatif
- `plot_model_comparison()`: Graphiques de comparaison
- `plot_confusion_matrices()`: Matrices de confusion
- `optimize_model()`: Optimisation hyperparamètres (RandomizedSearchCV)
- `evaluate_model()`: Évaluation complète avec visualisations
- `plot_roc_curve()`: Courbe ROC et AUC
- `tune_threshold()`: Optimisation du seuil de décision
- `save_model()`: Sauvegarde du modèle
- `get_training_summary()`: Résumé des résultats

### Workflow

### 1. Analyse Exploratoire des Données (EDA)
Utilisation de **EDAAnalyzer**:
- Analyse des valeurs manquantes (>75% manquant pour beaucoup)
- Distribution de la cible (déséquilibrée : 10% positif)
- Relations caractéristique-cible
- Tests d'hypothèses statistiques (tests t)

### 2. Preprocessing
Utilisation de **DataPreprocessor**:
- **Sélection de caractéristiques :** Basée sur les motifs de valeurs manquantes
  - Seuil configurable (défaut: <90% NaN)
  - Colonnes sanguines : 88-90% de taux de manquants
  - Colonnes virales : 75-88% de taux de manquants
- **Encodage :** positive/detected → 1, negative/not_detected → 0
- **Feature engineering :** Création de "est malade" à partir des tests viraux
- **Imputation :** Configurable (fillna/median/dropna)
- **Split :** Train/test stratifié automatique

### 3. Modeling
Utilisation de **ModelTrainer**:
- **Modèles testés :** RandomForest, AdaBoost, SVM, KNN
- **Pipelines automatisés :** PolynomialFeatures → SelectKBest → StandardScaler → Modèle
- **Comparaison systématique :** Métriques, graphiques, matrices de confusion
- **Meilleur modèle :** SVM avec caractéristiques polynomiales
- **Optimisation :** RandomizedSearchCV sur grille d'hyperparamètres
- **Configuration typique :**
  - Caractéristiques polynomiales (degré 2-4)
  - SelectKBest (10-56 caractéristiques)
  - StandardScaler
  - SVM (C=1-1000, gamma=0.001-1)

### 4. Evaluation
- **Métriques :** F1 Score, Recall, Précision
- **Validation croisée :** 4-fold CV
- **Visualisations automatiques :** Confusion matrix, ROC curve, comparaisons
- **Ajustement du seuil :** Optimisé pour maximiser le recall tout en maintenant F1
- **Courbes d'apprentissage :** Surveillance du surapprentissage

## Key Results

| Métrique | Objectif | Obtenu |
|----------|----------|--------|
| F1 Score | ≥ 0.5 | **0.56** |
| Recall | ≥ 0.7 | **0.81** |
| Précision | - | **0.71** |

### Feature Importance
Caractéristiques les plus prédictives :
1. Monocytes
2. Plaquettes
3. Leucocytes
4. Quantile d'âge du patient
5. Résultats des tests viraux (caractéristique créée)

## Configuration

Tous les paramètres peuvent être modifiés dans `src/config.py` :

```python
# Paramètres du modèle
POLYNOMIAL_DEGREE = 2
SELECT_K_BEST = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Objectifs de performance
TARGET_F1_SCORE = 0.5
TARGET_RECALL = 0.7

# Seuil de décision
DECISION_THRESHOLD = -1
```

## Development

### Adding New Features

1. Ajouter des fonctions de feature engineering à `src/features/engineering.py`
2. Importer et utiliser dans le pipeline de prétraitement
3. Mettre à jour la configuration dans `src/config.py`

### Adding New Models

1. Ajouter le modèle à `build_models()` dans `src/models/train.py`
2. Définir la grille d'hyperparamètres si nécessaire
3. Exécuter le pipeline d'entraînement

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

1. Forker le repository
2. Créer une branche de fonctionnalité (`git checkout -b feature/fonctionnalite-geniale`)
3. Commiter vos modifications (`git commit -m 'Ajouter une fonctionnalité géniale'`)
4. Pousser vers la branche (`git push origin feature/fonctionnalite-geniale`)
5. Ouvrir une Pull Request

Voir [CONTRIBUTING.md](docs/CONTRIBUTING.md) pour plus de détails.

## License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Authors

- emmab-collab - Travail initial

## Acknowledgments

- Source du jeu de données : [Kaggle COVID-19 Dataset]
- Inspiration du notebook original
- Documentation scikit-learn

## Contact

Pour des questions ou des retours, veuillez ouvrir une issue sur GitHub.

---

**Note :** Ce modèle est uniquement à des fins de recherche et d'éducation. Il ne doit pas être utilisé comme substitut à un diagnostic médical professionnel.
