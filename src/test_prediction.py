import pandas as pd
import joblib

# Chargement du modèle Random Forest déjà entraîné

modele = joblib.load(
    "models/modele_random_forest.pkl"
)

print("Modèle Random Forest chargé avec succès.")

# Chargement des données

usagers = pd.read_csv(
    "data/processed/usagers_prepares.csv"
)


# Sélection d'un usager pour le test

exemple = usagers.iloc[[0]].copy()

print("\n--- Usager testé ---")
print(exemple)

# Préparation des variables pour le modèle

exemple["heure"] = pd.to_datetime(
    exemple["hrmn"],
    format="%H:%M"
).dt.hour

X_exemple = exemple[
    [
        "age",
        "sexe",
        "catu",
        "jour",
        "mois",
        "heure",
        "lum",
        "agg",
        "atm",
        "catr"
    ]
]


# Prédiction

prediction = modele.predict(X_exemple)[0]

print("\nGravité prédite :", prediction)

# Traduction de la gravité

labels_gravite = {
    1: "Indemne",
    2: "Tué",
    3: "Blessé hospitalisé",
    4: "Blessé léger"
}

gravite_reelle = exemple["grav"].iloc[0]

print("\n--- Résultat ---")
print("Gravité réelle :", labels_gravite[gravite_reelle])
print("Gravité prédite :", labels_gravite[prediction])

if prediction == gravite_reelle:
    print("La machine a raison.")
else:
    print("La machine s'est trompée.")