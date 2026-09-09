import pandas as pd


# Chargement des données
caract = pd.read_csv("data/raw/caract-2024.csv", sep=";")
lieux = pd.read_csv("data/raw/lieux-2024.csv", sep=";")
usagers = pd.read_csv("data/raw/usagers-2024.csv", sep=";")


# Sélection des colonnes utiles
caract = caract[
    ["Num_Acc", "jour", "mois", "hrmn", "lum", "agg", "atm"]
]

lieux = lieux[
    ["Num_Acc", "catr", "surf", "vma"]
]

usagers = usagers[
    ["Num_Acc", "grav", "sexe", "an_nais", "catu"]
]


# Création de l'âge à partir de l'année de naissance
usagers["age"] = 2024 - usagers["an_nais"]


# Création du dataset des accidents
accidents = caract.copy()


# Ajout du type de route
lieux_catr = lieux[["Num_Acc", "catr"]].drop_duplicates()

accidents = accidents.merge(
    lieux_catr,
    on="Num_Acc",
    how="left"
)


# Création du dataset des usagers
usagers_prepares = usagers.copy()

# Ajout des informations de l'accident
usagers_prepares = usagers_prepares.merge(
    accidents,
    on="Num_Acc",
    how="left"
)


# Conditions de luminosité
luminosite = {
    1: "Plein jour",
    2: "Crépuscule ou aube",
    3: "Nuit sans éclairage public",
    4: "Nuit avec éclairage public non allumé",
    5: "Nuit avec éclairage public allumé"
}

accidents["lum_label"] = accidents["lum"].map(luminosite)
usagers_prepares["lum_label"] = usagers_prepares["lum"].map(luminosite)


# Conditions atmosphériques
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

accidents["atm_label"] = accidents["atm"].map(atmosphere)
usagers_prepares["atm_label"] = usagers_prepares["atm"].map(atmosphere)


# Agglomération
agglomeration = {
    1: "Hors agglomération",
    2: "En agglomération"
}

accidents["agg_label"] = accidents["agg"].map(agglomeration)
usagers_prepares["agg_label"] = usagers_prepares["agg"].map(agglomeration)


# Catégorie de route
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

accidents["catr_label"] = accidents["catr"].map(categorie_route)
usagers_prepares["catr_label"] = usagers_prepares["catr"].map(categorie_route)


# Niveau de gravité
gravite = {
    1: "Indemne",
    2: "Tué",
    3: "Blessé hospitalisé",
    4: "Blessé léger"
}

usagers_prepares["grav_label"] = usagers_prepares["grav"].map(gravite)


# Sexe des usagers
sexe_labels = {
    -1: "Non renseigné",
    1: "Masculin",
    2: "Féminin"
}

usagers_prepares["sexe_label"] = usagers_prepares["sexe"].map(sexe_labels)


# Catégorie d'usager
categorie_usager = {
    1: "Conducteur",
    2: "Passager",
    3: "Piéton"
}

usagers_prepares["catu_label"] = usagers_prepares["catu"].map(categorie_usager)


# Sauvegarde des données préparées
accidents.to_csv(
    "data/processed/accidents_prepares.csv",
    index=False
)

usagers_prepares.to_csv(
    "data/processed/usagers_prepares.csv",
    index=False
)

print("Données préparées et sauvegardées avec succès.")