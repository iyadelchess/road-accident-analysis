# Road Accident Analysis

Analyse des accidents corporels de la circulation routière en France en 2024 à l'aide de techniques de Data Analysis, de visualisation de données et de Machine Learning.

**Auteur : El Cheriaa El Manssouri Iyad**

---

## Présentation du projet

Ce projet consiste à analyser les accidents corporels de la circulation routière survenus en France en 2024.

L'objectif est dans un premier temps de comprendre les données et d'identifier différentes tendances concernant les accidents et les usagers impliqués.

Une seconde partie est consacrée au Machine Learning afin d'étudier la possibilité de prédire la gravité d'un usager impliqué dans un accident à partir de différentes caractéristiques disponibles dans les données.

Le projet couvre donc plusieurs étapes :

- exploration des données ;
- nettoyage et préparation des données ;
- analyse statistique ;
- visualisation des données ;
- entraînement de modèles de Machine Learning ;
- évaluation des performances ;
- test de prédiction.

---

## Objectifs

Les principaux objectifs du projet sont :

1. Comprendre la structure des données d'accidents routiers.
2. Explorer les principales caractéristiques des accidents et des usagers.
3. Identifier des tendances dans les données.
4. Préparer les données pour leur utilisation dans des modèles de Machine Learning.
5. Prédire la gravité d'un usager impliqué dans un accident.
6. Comparer plusieurs modèles de classification.
7. Analyser les résultats et les limites des modèles.

---

## Données

Les données utilisées correspondent aux accidents corporels de la circulation routière enregistrés en France en 2024.

Les données brutes sont conservées dans le dossier `data/raw/`.

### Données brutes

```text
data/raw/
├── caract-2024.csv
├── lieux-2024.csv
├── usagers-2024.csv
└── vehicules-2024.csv
```

Les fichiers contiennent différentes informations sur les accidents :

- `caract-2024.csv` : caractéristiques générales des accidents ;
- `lieux-2024.csv` : caractéristiques des lieux des accidents ;
- `usagers-2024.csv` : informations concernant les usagers impliqués ;
- `vehicules-2024.csv` : informations concernant les véhicules impliqués.

Les fichiers CSV utilisent `;` comme séparateur.

### Source

Les données proviennent des bases de données annuelles des accidents corporels de la circulation routière publiées par le Ministère de l'Intérieur et l'Observatoire national interministériel de la sécurité routière (ONISR).

Source principale :

**Bases de données annuelles des accidents corporels de la circulation routière - Années de 2005 à 2024**

Les données sont disponibles sur la plateforme officielle data.gouv.fr ainsi que sur le site de l'ONISR.

---

# Structure du projet

```text
road-accident-analysis/
│
├── data/
│   ├── processed/
│   │   ├── accidents_prepares.csv
│   │   └── usagers_prepares.csv
│   │
│   └── raw/
│       ├── caract-2024.csv
│       ├── lieux-2024.csv
│       ├── usagers-2024.csv
│       └── vehicules-2024.csv
│
├── notebooks/
│   ├── analyse_donnees.ipynb
│   ├── data_exploration.ipynb
│   └── machine_learning.ipynb
│
├── src/
│   ├── explore_data.py
│   ├── machine_learning.py
│   ├── prepare_data.py
│   └── test_prediction.py
│
├── visualizations/
│
├── .gitignore
├── README.md
└── requirements.txt
```

Le modèle Random Forest sauvegardé est conservé localement et n'est pas versionné sur GitHub en raison de sa taille.

---

# 1. Exploration des données

La première étape du projet consiste à explorer les données afin de comprendre leur structure et leurs principales caractéristiques.

Cette partie est réalisée principalement à l'aide des notebooks :

```text
notebooks/data_exploration.ipynb
notebooks/analyse_donnees.ipynb
```

Les analyses portent notamment sur :

- la répartition des accidents selon les mois ;
- les conditions de luminosité ;
- les conditions météorologiques ;
- la localisation des accidents ;
- les catégories de routes ;
- la gravité des conséquences ;
- l'âge des usagers ;
- le sexe des usagers ;
- la catégorie des usagers.

---

## Principales observations

### Répartition par mois

Les accidents sont répartis sur l'ensemble de l'année 2024, avec des variations selon les mois.

Les mois de juin et juillet présentent notamment un nombre d'accidents plus important que plusieurs autres mois.

### Luminosité

La majorité des accidents recensés ont eu lieu en plein jour.

Les autres accidents sont principalement répartis entre les différentes situations nocturnes ainsi que les périodes de crépuscule ou d'aube.

### Conditions météorologiques

La majorité des accidents ont été enregistrés dans des conditions atmosphériques normales.

La pluie légère représente ensuite l'une des conditions météorologiques les plus fréquentes.

### Agglomération

Les accidents recensés sont majoritairement survenus en agglomération :

- En agglomération : 34 010 accidents
- Hors agglomération : 20 392 accidents

### Catégorie de route

Les voies communales et les routes départementales représentent une part importante des accidents recensés.

Les principales catégories observées sont :

- voie communale ;
- route départementale ;
- autoroute ;
- route nationale ;
- routes de métropole urbaine.

### Gravité

La variable `grav` permet de distinguer quatre niveaux de gravité :

| Valeur | Gravité |
|---|---|
| 1 | Indemne |
| 2 | Tué |
| 3 | Blessé hospitalisé |
| 4 | Blessé léger |

Dans les données étudiées, les catégories « Indemne » et « Blessé léger » sont les plus représentées.

La catégorie « Tué » est beaucoup moins représentée.

---

# 2. Analyse des usagers

Une partie de l'analyse porte spécifiquement sur les usagers impliqués dans les accidents.

Les catégories d'usagers étudiées sont notamment :

| Catégorie | Nombre |
|---|---:|
| Conducteur | 92 581 |
| Passager | 23 205 |
| Piéton | 9 401 |

L'âge des usagers a également été étudié.

L'âge moyen observé est d'environ 38,9 ans et la médiane est de 36 ans.

La distribution des âges présente notamment un pic autour de 25 ans.

Une analyse croisée entre l'âge, le sexe et la gravité permet également d'observer différentes répartitions selon les catégories d'usagers.

---

# 3. Préparation des données

Les données provenant des différentes tables sont préparées avant leur utilisation dans les analyses et les modèles de Machine Learning.

Cette étape est principalement réalisée dans :

```text
src/prepare_data.py
```

Les données préparées sont enregistrées dans :

```text
data/processed/
├── accidents_prepares.csv
└── usagers_prepares.csv
```

Les principales étapes de préparation sont :

- sélection des variables utiles ;
- récupération des informations nécessaires depuis les différentes tables ;
- fusion des données à partir de `Num_Acc` ;
- création de nouvelles variables ;
- ajout de libellés pour faciliter l'interprétation ;
- gestion des valeurs manquantes ;
- préparation des données pour le Machine Learning.

### Création de l'âge

L'âge est calculé à partir de l'année de naissance :

```text
âge = 2024 - année de naissance
```

La variable `age` permet ensuite d'étudier la répartition des usagers selon différentes tranches d'âge.

---

# 4. Machine Learning

La partie Machine Learning cherche à prédire la **gravité d'un usager impliqué dans un accident**.

L'objectif n'est donc pas de prédire si un accident va avoir lieu, mais d'estimer la gravité d'un usager à partir des caractéristiques disponibles.

Le notebook correspondant est :

```text
notebooks/machine_learning.ipynb
```

Le script utilisé pour l'entraînement et l'évaluation est :

```text
src/machine_learning.py
```

---

## Variable cible

La variable prédite est :

```text
grav
```

avec les catégories suivantes :

| Valeur | Gravité |
|---|---|
| 1 | Indemne |
| 2 | Tué |
| 3 | Blessé hospitalisé |
| 4 | Blessé léger |

---

## Variables utilisées

Les variables utilisées pour effectuer les prédictions sont :

```text
age
sexe
catu
jour
mois
heure
lum
agg
atm
catr
```

La variable `hrmn`, qui contient l'heure et les minutes de l'accident, est transformée afin d'obtenir une variable `heure`.

---

## Prétraitement

Les variables catégorielles sont transformées grâce à un encodage One-Hot Encoding.

Les principales variables catégorielles concernées sont :

- `sexe`
- `catu`
- `lum`
- `agg`
- `atm`
- `catr`

Les données sont ensuite séparées en deux ensembles :

- 80 % pour l'entraînement ;
- 20 % pour le test.

La séparation est stratifiée afin de conserver une répartition similaire des différentes classes dans les deux ensembles.

---

# 5. Modèles utilisés

Deux modèles de classification ont été testés.

## Decision Tree

Le premier modèle utilisé est un `DecisionTreeClassifier`.

Il permet de construire une succession de règles afin de classer les usagers selon leur gravité.

## Random Forest

Le second modèle utilisé est un `RandomForestClassifier`.

Le modèle utilise :

- 100 arbres ;
- `class_weight="balanced"` afin de mieux prendre en compte le déséquilibre des classes ;
- plusieurs cœurs du processeur pour accélérer l'entraînement.

---

# 6. Résultats du Machine Learning

Les modèles sont évalués avec plusieurs indicateurs :

- Accuracy ;
- F1-score macro ;
- précision ;
- rappel ;
- matrice de confusion.

| Modèle | Accuracy | F1-score macro |
|---|---:|---:|
| Decision Tree | 0,416 | 0,32 |
| Random Forest | 0,443 | 0,34 |

Le Random Forest obtient donc une accuracy d'environ **44,3 %** et un F1-score macro d'environ **0,34** sur le jeu de test.

Ces résultats montrent que les caractéristiques disponibles permettent de retrouver certaines tendances liées à la gravité, mais que la capacité de prédiction reste limitée.

---

## Matrice de confusion

Pour le Random Forest, la matrice de confusion obtenue est :

```text
[[5277  189 1465 3139]
 [ 192   69  284  141]
 [1200  158 1311 1156]
 [3917  176 1652 4196]]
```

La classe correspondant aux personnes tuées est particulièrement difficile à prédire.

Cette classe représente environ 2,8 % des observations utilisées pour le Machine Learning, ce qui crée un déséquilibre important entre les différentes catégories.

---

# 7. Test de prédiction

Le projet contient également un script permettant de tester le modèle entraîné :

```text
src/test_prediction.py
```

Ce script charge le modèle Random Forest sauvegardé localement et effectue une prédiction sur un exemple provenant du jeu de test.

La préparation des données et la séparation entre les données d'entraînement et de test sont reproduites afin d'utiliser un exemple qui n'a pas été utilisé pour entraîner le modèle.

Le résultat permet de comparer :

```text
Gravité réelle
vs
Gravité prédite
```

Par exemple, lors d'un test :

```text
Gravité réelle : Blessé léger
Gravité prédite : Indemne
La prédiction est incorrecte.
```

Cet exemple illustre que le modèle peut produire des erreurs sur des observations individuelles, ce qui est cohérent avec ses performances globales.

---

# 8. Visualisations

Les visualisations réalisées pendant l'analyse permettent de mieux comprendre les distributions et les relations présentes dans les données.

Elles sont notamment utilisées pour étudier :

- la répartition des accidents par mois ;
- la luminosité ;
- les conditions météorologiques ;
- les catégories de routes ;
- la gravité ;
- l'âge des usagers ;
- la relation entre différentes caractéristiques des usagers et leur gravité.

Les notebooks contiennent les principales visualisations utilisées pour l'analyse.

Le dossier suivant est prévu pour regrouper les visualisations exportées :

```text
visualizations/
```

---

# 9. Limites du projet

Les résultats doivent être interprétés avec prudence.

## Déséquilibre des classes

La catégorie « Tué » représente une faible proportion des observations.

Même avec l'utilisation de `class_weight="balanced"`, cette classe reste difficile à prédire correctement.

## Variables disponibles

Les performances pourraient être améliorées en utilisant davantage de variables explicatives.

Certaines informations susceptibles d'avoir une influence sur la gravité ne sont pas nécessairement représentées dans les variables utilisées pour cette première approche.

## Interprétation des fréquences

Le nombre d'accidents observés dans une situation ne signifie pas nécessairement que cette situation est plus dangereuse.

Une catégorie très représentée peut simplement correspondre à une situation beaucoup plus fréquente dans les déplacements.

Par exemple, une grande quantité d'accidents sur un type de route ne signifie pas nécessairement que ce type de route présente le risque le plus élevé.

## Données enregistrées

Les données correspondent aux accidents corporels enregistrés par les forces de l'ordre.

Elles ne représentent donc pas nécessairement l'ensemble des incidents qui peuvent survenir sur les routes.

---

# 10. Technologies utilisées

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook
- Git
- GitHub

---

# 11. Installation

Cloner le dépôt :

```bash
git clone https://github.com/iyadelchess/road-accident-analysis.git
```

Se placer dans le dossier du projet :

```bash
cd road-accident-analysis
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Les bibliothèques nécessaires au projet sont regroupées dans :

```text
requirements.txt
```

---

# 12. Utilisation

## Préparer les données

```bash
python src/prepare_data.py
```

Cette étape prépare les données brutes et génère les fichiers présents dans :

```text
data/processed/
```

## Explorer les données

Les notebooks peuvent être ouverts avec Jupyter Notebook ou directement avec VS Code :

```text
notebooks/data_exploration.ipynb
notebooks/analyse_donnees.ipynb
```

## Entraîner les modèles

```bash
python src/machine_learning.py
```

Ce script permet notamment de :

- préparer les données pour le Machine Learning ;
- entraîner le Decision Tree ;
- entraîner le Random Forest ;
- calculer les métriques ;
- afficher les résultats ;
- sauvegarder localement le modèle Random Forest.

## Tester une prédiction

Une fois le modèle généré localement :

```bash
python src/test_prediction.py
```

---

# 13. Perspectives d'amélioration

Plusieurs améliorations pourraient être apportées au projet :

- tester d'autres algorithmes de classification ;
- optimiser les hyperparamètres des modèles ;
- améliorer la gestion du déséquilibre des classes ;
- ajouter davantage de variables explicatives ;
- analyser plus précisément les erreurs de classification ;
- comparer plusieurs techniques de Machine Learning ;
- développer une interface permettant de réaliser des prédictions ;
- créer un tableau de bord interactif ;
- approfondir l'analyse statistique des accidents.

---

# Conclusion

Ce projet m'a permis de mettre en pratique différentes étapes d'un projet de Data Analysis et de Machine Learning à partir de données réelles.

Le projet comprend notamment :

- la récupération de données publiques ;
- l'exploration de données ;
- le nettoyage et la préparation des données ;
- la création de variables ;
- l'analyse statistique ;
- la visualisation ;
- l'entraînement de modèles de classification ;
- l'évaluation des performances ;
- l'analyse des erreurs ;
- la réalisation de prédictions sur des données de test.

L'analyse exploratoire permet de mieux comprendre les caractéristiques des accidents routiers enregistrés en France en 2024.

La partie Machine Learning montre également qu'il est possible d'utiliser certaines caractéristiques des accidents et des usagers pour étudier leur gravité. Cependant, les performances obtenues restent limitées, notamment en raison du déséquilibre des classes et des variables utilisées.

Ce projet constitue une base pour poursuivre l'analyse et expérimenter des approches plus avancées en Data Analysis et en Machine Learning.