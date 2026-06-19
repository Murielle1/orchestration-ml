# ✈️ SkyScore — Airline Passenger Satisfaction MLOps Pipeline

> **Murielle SANOU · ESGI 5A · MLOps 2025-2026**
> Projet de classification binaire avec orchestration ML et déploiement cloud Oracle (OCI)

🌐 **Démo live :** http://141.145.217.77:8501 · [API](http://141.145.217.77:8000/docs) · [MLflow](http://141.145.217.77:5001) · [Airflow](http://141.145.217.77:8080) · [GitHub](https://github.com/Murielle1/orchestration-ml)

---

## 📌 Contexte & Problématique

Dans un secteur aérien hautement concurrentiel, la satisfaction client est un levier stratégique majeur. Les compagnies aériennes collectent de grandes quantités de données à chaque vol — profil du passager, classe de voyage, retards, qualité des services à bord — mais peinent souvent à exploiter ces informations de manière prédictive et actionnable.

**Comment, à partir des données d'un vol et du profil d'un passager, peut-on prédire automatiquement son niveau de satisfaction afin d'anticiper les insatisfactions et améliorer l'expérience client ?**

Ce projet répond à cette question en construisant un **pipeline MLOps complet** : de l'entraînement d'un modèle de classification binaire jusqu'à son déploiement sur le cloud Oracle, avec orchestration, monitoring et intégration continue.

---

## 🎯 Objectif

Entraîner, évaluer et déployer un modèle de **classification binaire** capable de prédire si un passager est :

- `satisfied` → 1 — passager satisfait
- `neutral or dissatisfied` → 0 — passager neutre ou insatisfait

Le modèle est reproductible, monitorable et exposé via une API REST en production.

---

## 📊 Données

**Source :** [Airline Passenger Satisfaction — Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

Le jeu de données contient plus de **100 000 observations** avec **22 variables prédictives** de nature mixte.

| Catégorie | Variables |
|-----------|-----------|
| Profil passager | Genre, âge, type de client (fidèle / nouveau) |
| Voyage | Type de voyage (affaires / personnel), classe (Eco / Business) |
| Vol | Distance, retard au départ, retard à l'arrivée |
| Services à bord | Wi-Fi, confort du siège, service en vol, embarquement en ligne, propreté, restauration, divertissement… |

**Variable cible :** `satisfaction` — binaire (`satisfied` / `neutral or dissatisfied`)

---

## 🏗️ Architecture du projet

```
orchestration-ml/
├── src/                        # Code source Python
│   ├── config.py               # Configuration centralisée (paths, features, MLflow)
│   ├── data.py                 # Chargement et découpage des données
│   ├── features.py             # Pipeline de prétraitement (ColumnTransformer)
│   ├── tracking.py             # Configuration MLflow centralisée
│   ├── train.py                # Entraînement baseline (LogisticRegression)
│   ├── train_models.py         # Comparaison RF / XGBoost / LightGBM (GridSearchCV)
│   ├── train_optuna.py         # Optimisation Optuna TPE
│   ├── evaluate.py             # Évaluation + porte qualité
│   └── api.py                  # API FastAPI (/predict, /predictions, /model-metrics)
├── frontend/
│   └── app.py                  # Dashboard Streamlit (SkyScore)
├── dags/
│   ├── retrain_dag.py          # DAG Airflow : ré-entraînement hebdomadaire
│   └── predictions_dag.py      # DAG Airflow : prédictions quotidiennes simulées
├── docker/
│   ├── Dockerfile.train        # Image entraînement
│   ├── Dockerfile.api          # Image API FastAPI
│   ├── Dockerfile.frontend     # Image frontend Streamlit
│   └── Dockerfile.airflow      # Image Airflow avec dépendances ML
├── .github/workflows/
│   ├── ci.yml                  # CI : lint + types + tests à chaque push
│   └── cd.yml                  # CD : build + push image Docker sur GHCR
├── data/
│   ├──                     # train.csv, test.csv (Kaggle)
│   └── processed/              # Données prétraitées
├── models/                     # model.joblib (généré)
├── docker-compose.yml          # Stack complète
├── pyproject.toml              # Dépendances et configuration (uv)
├── Makefile                    # Commandes du projet
└── README.md
```

---

## ⚙️ Stack technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.13 |
| Gestion env | uv |
| ML | Scikit-learn, XGBoost, LightGBM |
| Optimisation | Optuna (TPE) |
| Tracking | MLflow 2.x (SQLite + artifacts) |
| API | FastAPI + Pydantic v2 |
| Frontend | Streamlit |
| Orchestration | Apache Airflow 2.9 |
| Conteneurisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Registry images | GitHub Container Registry (GHCR) |
| Cloud | Oracle Cloud Infrastructure (OCI) |

---

## 🤖 Modèles ML

### Baseline
- **Logistic Regression** — entraînée via `make train` ou DAG Airflow

### Comparaison (GridSearchCV)
- **Random Forest** — `n_estimators`, `max_depth`, `min_samples_leaf`
- **XGBoost** — `n_estimators`, `max_depth`, `learning_rate`
- **LightGBM** — `n_estimators`, `num_leaves`, `learning_rate`

### Optimisation (Optuna TPE)
- Même 3 familles avec espaces de recherche élargis (`subsample`, `colsample_bytree`, `min_child_samples`…)

### Pipeline de prétraitement
```
Numériques (Age, Distance, Délais) → SimpleImputer(médiane) + StandardScaler
Catégorielles (Gender, Class…)     → OneHotEncoder
Scores 0-5 (14 colonnes)           → SimpleImputer(médiane)
```

---

## 📈 Métriques d'évaluation

| Métrique | Seuil qualité | Résultat typique |
|----------|--------------|------------------|
| ROC AUC | ≥ 0.85 | ~0.93–0.96 |
| F1-score | ≥ 0.80 | ~0.87–0.92 |
| Accuracy | — | ~0.87–0.93 |

---

## 🚀 API REST

L'API FastAPI expose 5 endpoints :

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Statut de l'API |
| `/predict` | POST | Prédiction pour un passager |
| `/predictions` | GET | Journal de toutes les prédictions (session) |
| `/model-info` | GET | Version et statut du modèle chargé |
| `/model-metrics` | GET | Métriques + matrice de confusion + importances |

**Exemple de requête :**
```json
POST /predict
{
  "Gender": "Male",
  "Customer Type": "Loyal Customer",
  "Age": 35,
  "Class": "Business",
  "Flight Distance": 1200,
  "Inflight wifi service": 4,
  ...
}
→ { "prediction": 1, "probability": 0.87, "label": "satisfied" }
```

---

## 🌬️ Orchestration Airflow

Deux DAGs sont déployés :

### `model_retraining` — hebdomadaire (lundis 3h)
```
prepare_data (features.py) → train (train.py + MLflow) → check_quality (porte F1 ≥ 0.80)
```

### `daily_predictions` — quotidien (10h)
```
send_predictions (20 lignes aléatoires → POST /predict → journal API)
```

---

## 🔄 CI/CD GitHub Actions

### CI (`.github/workflows/ci.yml`) — à chaque push
- `ruff check` + `ruff format --check` — style
- `mypy` — types
- `pytest --cov` — tests + couverture
- Téléchargement Kaggle + entraînement baseline

### CD (`.github/workflows/cd.yml`) — sur `main` et tags `vX.Y.Z`
- Build image Docker API
- Push sur GitHub Container Registry (GHCR)
- Tags : `latest`, `sha-XXXXXX`, version sémantique

---

## 🐳 Déploiement Docker

### Services

| Service | Port | Description |
|---------|------|-------------|
| `mlflow` | 5001 | Serveur de tracking + Model Registry |
| `train` | — | Entraînement one-shot (profil `train`) |
| `api` | 8000 | API FastAPI d'inférence |
| `frontend` | 8501 | Dashboard Streamlit SkyScore |
| `airflow` | 8080 | Orchestration des DAGs |

### Commandes

```bash
# Installation
make install

# Données
make data                    # télécharge train.csv + test.csv (Kaggle CLI)

# Pipeline ML local
make train                   # baseline
make train-models            # GridSearchCV RF/XGBoost/LightGBM
make train-optuna            # Optuna TPE
make evaluate                # porte qualité sur test.csv

# Stack Docker
make docker-up               # mlflow → train → api → frontend
make airflow                 # démarre Airflow
make airflow-trigger-retrain # déclenche le DAG de ré-entraînement
make airflow-trigger-predict # déclenche le DAG de prédictions

# Qualité
make check                   # lint + types + tests
```

---

## ☁️ Déploiement Oracle Cloud (OCI)

L'application est déployée sur une instance Oracle Cloud avec :

- Docker Compose pour orchestrer tous les services
- Ports ouverts dans les Security Lists OCI (8000, 8501, 5001, 8080)
- `restart: unless-stopped` pour la haute disponibilité
- Volumes Docker persistants pour MLflow et les modèles

**URLs de production :**

| Service | URL |
|---------|-----|
| Frontend SkyScore | http://141.145.217.77:8501 |
| API FastAPI (Swagger) | http://141.145.217.77:8000/docs |
| MLflow UI | http://141.145.217.77:5001 |
| Airflow UI | http://141.145.217.77:8080 |

---
