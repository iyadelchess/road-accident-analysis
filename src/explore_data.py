import pandas as pd


# Dictionnaire contenant les noms des jeux de données
# et les chemins vers leurs fichiers CSV
files = {
    "Caractéristiques": "data/raw/caract-2024.csv",
    "Lieux": "data/raw/lieux-2024.csv",
    "Véhicules": "data/raw/vehicules-2024.csv",
    "Usagers": "data/raw/usagers-2024.csv"
}


# Parcours de chaque jeu de données
for name, path in files.items():

    # Affichage d'un séparateur pour améliorer la lisibilité
    print("\n" + "=" * 50)

    # Affichage du nom du jeu de données actuellement analysé
    print(f"DATASET : {name}")

    # Affichage du séparateur
    print("=" * 50)

    # Chargement du fichier CSV avec Pandas
    df = pd.read_csv(path, sep=";")

    # Affichage des cinq premières lignes du jeu de données
    print("\n--- Premières lignes ---")
    print(df.head())

    # Affichage des informations générales :
    # nombre de valeurs non nulles et types des colonnes
    print("\n--- Informations générales ---")
    df.info()

    # Affichage du nombre de lignes et de colonnes
    print("\n--- Dimensions ---")
    print(f"Nombre de lignes : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")

    # Affichage de la liste des colonnes
    print("\n--- Colonnes ---")
    print(df.columns.tolist())

    # Calcul et affichage du nombre de valeurs manquantes
    # pour chaque colonne
    print("\n--- Valeurs manquantes ---")
    print(df.isnull().sum())