import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Chargement du modèle Random Forest déjà entraîné

modele = joblib.load(
    "models/modele_random_forest.pkl"
)

print("Modèle Random Forest chargé avec succès.")

# Chargement des données

usagers = pd.read_csv(
    "data/processed/usagers_prepares.csv"
)

# Sélection des variables utilisées par le modèle

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

# Suppression des lignes avec des valeurs manquantes

data_ml = data_ml.dropna()

# Extraction de l'heure à partir de hrmn

data_ml["heure"] = pd.to_datetime(
    data_ml["hrmn"],
    format="%H:%M"
).dt.hour

# Séparation des variables d'entrée et de la cible

X = data_ml.drop(columns=["grav", "hrmn"])
y = data_ml["grav"]

# Même séparation que lors de l'entraînement

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Sélection d'un usager du jeu de test

exemple = X_test.iloc[[0]]
gravite_reelle = y_test.iloc[0]

print("\n--- Usager testé ---")
print(exemple)

# Prédiction

prediction = modele.predict(exemple)[0]

# Traduction de la gravité

labels_gravite = {
    1: "Indemne",
    2: "Tué",
    3: "Blessé hospitalisé",
    4: "Blessé léger"
}

print("\n--- Résultat ---")
print("Gravité réelle :", labels_gravite[gravite_reelle])
print("Gravité prédite :", labels_gravite[prediction])

if prediction == gravite_reelle:
    print("La machine a raison.")
else:
    print("La machine s'est trompée.")