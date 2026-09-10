import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement des données préparées

usagers = pd.read_csv(
    "data/processed/usagers_prepares.csv"
)

# Sélection des variables pour le Machine Learning

variables = [
    "age",
    "sexe",
    "catu",
    "jour",
    "mois",
    "hrmn",
    "lum",
    "agg",
    "atm",
    "catr"
]

cible = "grav"

data_ml = usagers[variables + [cible]].copy()

# Suppression des lignes avec un âge manquant

data_ml = data_ml.dropna()

# Séparation des variables d'entrée et de la cible

X = data_ml.drop(columns="grav")
y = data_ml["grav"]

# Séparation des données d'entraînement et de test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Données d'entraînement :", X_train.shape)
print("Données de test :", X_test.shape)