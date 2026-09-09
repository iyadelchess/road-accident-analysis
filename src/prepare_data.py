import pandas as pd


# Chargement des données
caract = pd.read_csv("data/raw/caract-2024.csv", sep=";")
lieux = pd.read_csv("data/raw/lieux-2024.csv", sep=";")
usagers = pd.read_csv("data/raw/usagers-2024.csv", sep=";")

# Sélection des colonnes utiles pour l'analyse

caract = caract[
    [
        "Num_Acc",
        "jour",
        "mois",
        "hrmn",
        "lum",
        "agg",
        "atm"
    ]
]

lieux = lieux[
    [
        "Num_Acc",
        "catr",
        "surf",
        "vma"
    ]
]

usagers = usagers[
    [
        "Num_Acc",
        "grav",
        "sexe",
        "an_nais",
        "catu"
    ]
]
# Création de l'âge des usagers à partir de l'année de naissance
usagers["age"] = 2024 - usagers["an_nais"]

# Vérification des premières valeurs
print("\nAperçu des âges :")
print(usagers[["an_nais", "age"]].head())

# Vérification des valeurs manquantes
print("\nValeurs manquantes pour l'âge :")
print(usagers["age"].isnull().sum())

# Création du dataset principal au niveau de l'accident

accidents = caract.copy()

print("\nInformations sur le dataset accidents :")
print(accidents.shape)

print("\nNombre d'accidents uniques :")
print(accidents["Num_Acc"].nunique())

# Ajout du type de route au dataset accidents

lieux_catr = lieux[["Num_Acc", "catr"]].drop_duplicates()

accidents = accidents.merge(
    lieux_catr,
    on="Num_Acc",
    how="left"
)

print("\nDataset accidents après ajout de catr :")
print(accidents.shape)

print("Nombre d'accidents uniques :")
print(accidents["Num_Acc"].nunique())

# Création du dataset au niveau des usagers
# Une ligne représente un usager impliqué dans un accident

usagers_prepares = usagers.copy()

# Ajout des informations générales de l'accident à chaque usager
usagers_prepares = usagers_prepares.merge(
    accidents,
    on="Num_Acc",
    how="left"
)

# Vérification du résultat
print("\nDataset usagers préparé :")
print(usagers_prepares.shape)

print("Nombre d'usagers :", len(usagers_prepares))
print("Nombre d'accidents uniques :", usagers_prepares["Num_Acc"].nunique())

# Vérification des valeurs des variables codées

variables_accidents = ["lum", "agg", "atm", "catr"]
variables_usagers = ["grav", "sexe", "catu"]

# Dictionnaire des conditions de luminosité

luminosite = {
    1: "Plein jour",
    2: "Crépuscule ou aube",
    3: "Nuit sans éclairage public",
    4: "Nuit avec éclairage public non allumé",
    5: "Nuit avec éclairage public allumé"
}

# Création d'une colonne descriptive
accidents["lum_label"] = accidents["lum"].map(luminosite)
usagers_prepares["lum_label"] = usagers_prepares["lum"].map(luminosite)

# Dictionnaire des conditions atmosphériques

atmosphere = {
    1: "Normale",
    2: "Pluie légère",
    3: "Pluie forte",
    4: "Neige ou grêle",
    5: "Brouillard ou fumée",
    6: "Vent fort ou tempête",
    7: "Temps éblouissant",
    8: "Temps couvert",
    9: "Autre"
}

# Création d'une colonne descriptive
accidents["atm_label"] = accidents["atm"].map(atmosphere)
usagers_prepares["atm_label"] = usagers_prepares["atm"].map(atmosphere)

# Dictionnaire agglomération / hors agglomération

agglomeration = {
    1: "Hors agglomération",
    2: "En agglomération"
}

# Création d'une colonne descriptive
accidents["agg_label"] = accidents["agg"].map(agglomeration)
usagers_prepares["agg_label"] = usagers_prepares["agg"].map(agglomeration)

# Dictionnaire des catégories de routes

categorie_route = {
    1: "Autoroute",
    2: "Route nationale",
    3: "Route départementale",
    4: "Voie communale",
    5: "Hors réseau public",
    6: "Parc de stationnement ouvert à la circulation publique",
    7: "Routes de métropole urbaine",
    9: "Autre"
}

# Création d'une colonne descriptive
accidents["catr_label"] = accidents["catr"].map(categorie_route)
usagers_prepares["catr_label"] = usagers_prepares["catr"].map(categorie_route)

# Dictionnaire des niveaux de gravité

gravite = {
    1: "Indemne",
    2: "Tué",
    3: "Blessé hospitalisé",
    4: "Blessé léger"
}

# Création d'une colonne descriptive
usagers_prepares["grav_label"] = usagers_prepares["grav"].map(gravite)

# Dictionnaire du sexe des usagers

sexe_label = {
    1: "Masculin",
    2: "Féminin",
    -1: "Non renseigné"
}

# Création d'une colonne descriptive
usagers_prepares["sexe_label"] = usagers_prepares["sexe"].map(sexe_label)

# Dictionnaire des catégories d'usagers

categorie_usager = {
    1: "Conducteur",
    2: "Passager",
    3: "Piéton"
}

# Création d'une colonne descriptive
usagers_prepares["catu_label"] = usagers_prepares["catu"].map(categorie_usager)

# Vérification
print("\nCatégories d'usagers :")
print(
    usagers_prepares[["catu", "catu_label"]]
    .drop_duplicates()
    .sort_values("catu")
)