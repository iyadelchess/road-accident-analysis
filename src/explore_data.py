import pandas as pd

# Chargement des données
df = pd.read_csv("data/raw/caract-2024.csv", sep=";")

# Afficher les premières lignes
print(df.head())

# Informations générales sur le dataset
print("\n--- Informations générales ---")
df.info()

# Dimensions du dataset
print("\n--- Dimensions ---")
print(f"Nombre de lignes : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")

# Noms des colonnes
print("\n--- Colonnes ---")
print(df.columns.tolist())

# Valeurs manquantes
print("\n--- Valeurs manquantes ---")
print(df.isnull().sum())