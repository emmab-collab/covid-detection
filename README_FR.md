# Détection COVID-19 à partir de Données Cliniques

Un projet de machine learning pour prédire l'infection au COVID-19 à partir de données cliniques et d'analyses sanguines.

[English version](README.md) | **Version française**

## 🎯 Objectif du Projet

Prédire si un patient est infecté par le COVID-19 sur la base des données cliniques disponibles, en atteignant :
- **F1 Score ≥ 0.5**
- **Recall ≥ 0.7**

**Performance actuelle :**
- ✅ **F1 Score : 0.56**
- ✅ **Recall : 0.81**

## 📊 Jeu de Données

- **Échantillons totaux :** 5,644
- **Caractéristiques :** 111 (initialement)
- **Cible :** Résultat de l'examen SARS-Cov-2 (positif/négatif)
- **Distribution des classes :** 10% positif, 90% négatif (déséquilibré)

### Groupes de Caractéristiques
- **Analyses sanguines :** Hémoglobine, Plaquettes, Leucocytes, Lymphocytes, Monocytes, etc.
- **Tests viraux :** Influenza A/B, Rhinovirus, Coronavirus, etc.
- **Démographie :** Quantile d'âge du patient

## 🏗️ Structure du Projet

```
covid-detection/
├── README.md                   # Documentation (EN)
├── README_FR.md               # Documentation (FR)
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
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_modeling.ipynb
│
├── src/                       # Code source
│   ├── __init__.py
│   ├── config.py              # Configuration
│   ├── data/                  # Traitement des données
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/              # Feature engineering
│   │   ├── __init__.py
│   │   └── engineering.py
│   ├── models/                # Entraînement & évaluation
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── visualization/         # Visualisations
│       ├── __init__.py
│       └── plots.py
│
├── scripts/                   # Scripts exécutables
│   ├── train_model.py         # Pipeline d'entraînement
│   └── predict.py             # Script de prédiction
│
└── tests/                     # Tests unitaires
    └── test_preprocessing.py
```

## 🚀 Installation

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

## 💻 Utilisation

### Entraîner un Modèle

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

### Faire des Prédictions

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

### Utilisation dans Jupyter Notebooks

```python
# Importer les modules
from src.data.preprocessing import load_data, preprocessing
from src.models.train import build_models
from src.models.evaluate import evaluation
from src.visualization.plots import plot_target_distribution

# Charger les données
df = load_data()

# Créer des visualisations
plot_target_distribution(df, save=True)

# Entraîner des modèles
models = build_models()
```

## 📈 Méthodologie

### 1. Analyse Exploratoire des Données (EDA)
- Analyse des valeurs manquantes (>75% manquant pour beaucoup)
- Distribution de la cible (déséquilibrée : 10% positif)
- Relations caractéristique-cible
- Tests d'hypothèses statistiques (tests t)

### 2. Prétraitement
- **Sélection de caractéristiques :** Basée sur les motifs de valeurs manquantes
  - Colonnes sanguines : 88-90% de taux de manquants
  - Colonnes virales : 75-88% de taux de manquants
- **Encodage :** positive/detected → 1, negative/not_detected → 0
- **Imputation :** Suppression des lignes avec valeurs manquantes
- **Feature engineering :** Création de "est malade" à partir des tests viraux

### 3. Modélisation
- **Modèles testés :** RandomForest, AdaBoost, SVM, KNN
- **Meilleur modèle :** SVM avec caractéristiques polynomiales
- **Pipeline :**
  - Caractéristiques polynomiales (degré 4)
  - SelectKBest (56 caractéristiques)
  - StandardScaler
  - SVM (C=1000, gamma=0.001)

### 4. Évaluation
- **Métriques :** F1 Score, Recall, Précision
- **Validation croisée :** 4-fold CV
- **Ajustement du seuil :** Optimisé pour maximiser le recall
- **Courbes d'apprentissage :** Surveillance du surapprentissage

## 📊 Résultats Clés

| Métrique | Objectif | Obtenu |
|----------|----------|--------|
| F1 Score | ≥ 0.5 | **0.56** ✅ |
| Recall | ≥ 0.7 | **0.81** ✅ |
| Précision | - | **0.71** |

### Importance des Caractéristiques
Caractéristiques les plus prédictives :
1. Monocytes
2. Plaquettes
3. Leucocytes
4. Quantile d'âge du patient
5. Résultats des tests viraux (caractéristique créée)

## 🔧 Configuration

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

## 📝 Développement

### Ajouter de Nouvelles Caractéristiques

1. Ajouter des fonctions de feature engineering à `src/features/engineering.py`
2. Importer et utiliser dans le pipeline de prétraitement
3. Mettre à jour la configuration dans `src/config.py`

### Ajouter de Nouveaux Modèles

1. Ajouter le modèle à `build_models()` dans `src/models/train.py`
2. Définir la grille d'hyperparamètres si nécessaire
3. Exécuter le pipeline d'entraînement

### Exécuter les Tests

```bash
pytest tests/
```

## 📚 Dépendances

- **numpy** ≥ 1.21.0
- **pandas** ≥ 1.3.0
- **scikit-learn** ≥ 1.0.0
- **matplotlib** ≥ 3.4.0
- **seaborn** ≥ 0.11.0
- **scipy** ≥ 1.7.0
- **openpyxl** ≥ 3.0.0
- **joblib** ≥ 1.0.0

## 🤝 Contribution

1. Forker le repository
2. Créer une branche de fonctionnalité (`git checkout -b feature/fonctionnalite-geniale`)
3. Commiter vos modifications (`git commit -m 'Ajouter une fonctionnalité géniale'`)
4. Pousser vers la branche (`git push origin feature/fonctionnalite-geniale`)
5. Ouvrir une Pull Request

Voir [CONTRIBUTING.md](docs/CONTRIBUTING.md) pour plus de détails.

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Auteurs

- Votre Nom - Travail initial

## 🙏 Remerciements

- Source du jeu de données : [Kaggle COVID-19 Dataset]
- Inspiration du notebook original
- Documentation scikit-learn

## 📧 Contact

Pour des questions ou des retours, veuillez ouvrir une issue sur GitHub.

## 📖 Documentation Complète

- **[QUICKSTART.md](QUICKSTART.md)** - Guide de démarrage rapide (5 minutes)
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Guide de démarrage détaillé
- **[docs/API.md](docs/API.md)** - Référence complète de l'API
- **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Guide de contribution
- **[GITHUB_SETUP.md](GITHUB_SETUP.md)** - Configuration GitHub

---

**Note :** Ce modèle est uniquement à des fins de recherche et d'éducation. Il ne doit pas être utilisé comme substitut à un diagnostic médical professionnel.
