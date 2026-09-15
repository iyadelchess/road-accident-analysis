import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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

# Suppression des lignes avec des valeurs manquantes

data_ml = data_ml.dropna()

# Extraction de l'heure à partir de la colonne hrmn

data_ml["heure"] = pd.to_datetime(
    data_ml["hrmn"],
    format="%H:%M"
).dt.hour

# Séparation des variables d'entrée et de la cible

X = data_ml.drop(columns=["grav", "hrmn"])
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

# Variables catégorielles

variables_categorielles = [
    "sexe",
    "catu",
    "lum",
    "agg",
    "atm",
    "catr"
]

# Préprocesseur pour encoder les variables catégorielles

preprocesseur = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            variables_categorielles
        )
    ],
    remainder="passthrough"
)

# Création du modèle Decision Tree

modele_arbre = Pipeline(
    steps=[
        ("preprocessing", preprocesseur),
        (
            "model",
            DecisionTreeClassifier(random_state=42)
        )
    ]
)

# Entraînement du modèle

modele_arbre.fit(X_train, y_train)

print("Modèle Decision Tree entraîné avec succès.")

# Prédictions sur les données de test

predictions_arbre = modele_arbre.predict(X_test)

# Évaluation du modèle

print("Accuracy :", accuracy_score(y_test, predictions_arbre))

print("\nRapport de classification :")
print(classification_report(y_test, predictions_arbre))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, predictions_arbre))

# Création du modèle Random Forest
modele_foret = Pipeline(
    steps=[
        ("preprocessing", preprocesseur),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Entraînement du modèle
modele_foret.fit(X_train, y_train)

print("\nModèle Random Forest entraîné avec succès.")

# Prédictions
predictions_foret = modele_foret.predict(X_test)

# Évaluation
print("Accuracy :", accuracy_score(y_test, predictions_foret))

print("\nRapport de classification :")
print(classification_report(y_test, predictions_foret))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, predictions_foret))