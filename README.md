# Prédiction du Risque de Dépression chez les Étudiants

Modèle LightGBM entraîné sur 27 901 étudiants pour prédire le risque de dépression.

## Structure

```
.
├── app.py                  Flask backend
├── model.pkl               Modèle entraîné (généré par notebook)
├── preprocessor.pkl        Pipeline de prétraitement (généré par notebook)
├── requirements.txt
├── setup.ps1               Installation
├── run.ps1                 Lancement du serveur
├── data/
│   └── Student_Depression_Dataset.csv
├── src/
│   └── preprocessing.py    Classe Preprocessor
├── notebooks/
│   ├── 01_EDA.ipynb        Analyse exploratoire
│   └── 02_Modeling.ipynb   Entraînement du modèle
└── webapp/
    ├── templates/          Pages HTML
    └── static/             CSS, JS, images
```

## Installation

```powershell
# 1. Créer l'environnement et installer les dépendances
.\setup.ps1

# 2. Activer l'environnement
.\.venv\Scripts\Activate.ps1

# 3. Générer les graphiques EDA
jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb --output notebooks/01_EDA.ipynb --ExecutePreprocessor.timeout=300

# 4. Entraîner le modèle
jupyter nbconvert --to notebook --execute notebooks/02_Modeling.ipynb --output notebooks/02_Modeling.ipynb --ExecutePreprocessor.timeout=600

# 5. Lancer l'application
python app.py
```

## Utilisation

Ouvrir http://127.0.0.1:5000 dans un navigateur.

## Modèle

- **Algorithme** : LightGBM (binary classification)
- **Features** : 14 variables (sommeil, pression académique, CGPA, alimentation, stress financier...)
- **Métriques** : voir page `/models`
