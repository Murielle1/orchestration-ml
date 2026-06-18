# ==============================================================================
# Airline Passenger Satisfaction - Makefile
# ==============================================================================
# Environnement géré par uv (Python 3.13) à partir de pyproject.toml.
# Aide : make help
# ==============================================================================

SHELL        := /bin/sh
PYTHON       := uv run python
RUN          := uv run
VENV_DIR     := .venv
PYTHONPATH   ?= .
export PYTHONPATH
API_HOST     ?= 127.0.0.1
API_PORT     ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT  := 5001
AIRFLOW_PORT := 8080
C            ?= 1.0
MAX_ITER     ?= 1000
CV           ?= 5
SCORING      ?= roc_auc
N_TRIALS     ?= 30

# Chemins données
RAW_DIR       := data/raw
PROCESSED_DIR := data/processed
TRAIN_CSV     := $(RAW_DIR)/train.csv
TEST_CSV      := $(RAW_DIR)/test.csv

# Couleurs ANSI
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv venv-create install sync deps-sync lock reset-env doctor \
        data features train train-models train-optuna evaluate mlflow api frontend \
        airflow airflow-logs airflow-trigger-retrain airflow-trigger-predict \
        docker-build docker-run docker-train docker-up docker-down docker-down-all docker-logs \
        lint format type test check clean


# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-26s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Setup
# ==============================================================================

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installé$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv:
	@test -d $(VENV_DIR) || { \
		echo "$(RED)[ERREUR] Virtualenv manquant : $(VENV_DIR)$(RESET)"; \
		echo "  Lance : make install"; \
		exit 1; \
	}

venv-create: check-uv ## Crée un virtualenv vide (.venv)
	@echo "$(YELLOW)>> Création du virtualenv...$(RESET)"
	uv venv $(VENV_DIR)
	@echo "$(GREEN)[OK] Virtualenv créé$(RESET)"

deps-sync: check-uv ## Synchronise les dépendances projet + dev (uv sync)
	@echo "$(YELLOW)>> Synchronisation des dépendances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Dépendances installées$(RESET)"

install: deps-sync ## Crée le venv et installe le projet + dev (alias)

sync: deps-sync ## Alias de deps-sync

lock: check-uv ## Génère/actualise uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Génération du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock généré$(RESET)"

reset-env: check-uv ## Réinitialise l'environnement (.venv + uv.lock)
	@echo "$(YELLOW)>> Réinitialisation de l'environnement...$(RESET)"
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recréé$(RESET)"

doctor: check-uv check-venv ## Diagnostique l'environnement de travail
	@uv --version
	@$(PYTHON) --version
	@echo "$(GREEN)[OK] Environnement prêt$(RESET)"


# ==============================================================================
# Pipeline ML
# ==============================================================================

data: ## Télécharge le dataset Kaggle dans data/raw/
	@echo "$(YELLOW)>> Téléchargement du dataset...$(RESET)"
	@mkdir -p $(RAW_DIR) $(PROCESSED_DIR)
	kaggle datasets download -d teejmahal20/airline-passenger-satisfaction \
		--path $(RAW_DIR) --unzip
	@echo "$(GREEN)[OK] Fichiers disponibles dans $(RAW_DIR)$(RESET)"

features: ## Génère les features prétraitées dans data/processed/
	@echo "$(YELLOW)>> Feature engineering...$(RESET)"
	$(PYTHON) -m src.features \
		--train-path $(TRAIN_CSV) \
		--test-path  $(TEST_CSV) \
		--output-dir $(PROCESSED_DIR)
	@echo "$(GREEN)[OK] Features disponibles dans $(PROCESSED_DIR)$(RESET)"

train: ## Entraîne la baseline → models/model.joblib (C=.. MAX_ITER=..)
	@echo "$(YELLOW)>> Entraînement baseline (C=$(C) max_iter=$(MAX_ITER))...$(RESET)"
	$(PYTHON) -m src.train --c $(C) --max-iter $(MAX_ITER)
	@echo "$(GREEN)[OK] Modèle sauvegardé dans models/$(RESET)"

train-models: ## Compare RF / XGBoost / LightGBM (GridSearchCV) (CV=.. SCORING=..)
	@echo "$(YELLOW)>> Comparaison de modèles (cv=$(CV) scoring=$(SCORING))...$(RESET)"
	$(PYTHON) -m src.train_models --cv $(CV) --scoring $(SCORING)
	@echo "$(GREEN)[OK] Comparaison terminée$(RESET)"

train-optuna: ## Optimise RF / XGBoost / LightGBM avec Optuna (N_TRIALS=.. CV=..)
	@echo "$(YELLOW)>> Optimisation Optuna (n_trials=$(N_TRIALS) cv=$(CV))...$(RESET)"
	$(PYTHON) -m src.train_optuna --n-trials $(N_TRIALS) --cv $(CV)
	@echo "$(GREEN)[OK] Optimisation terminée$(RESET)"

evaluate: ## Évalue le modèle sur test.csv avec porte qualité
	@echo "$(YELLOW)>> Évaluation du modèle...$(RESET)"
	$(PYTHON) -m src.evaluate
	@echo "$(GREEN)[OK] Évaluation terminée$(RESET)"

mlflow: ## Démarre le serveur MLflow sur http://localhost:5001
	@echo "$(YELLOW)>> Démarrage de MLflow...$(RESET)"
	docker compose up -d mlflow
	@echo "$(GREEN)[OK] MLflow disponible sur http://localhost:$(MLFLOW_PORT)$(RESET)"

api: ## Lance l'API FastAPI en rechargement auto (API_HOST/API_PORT)
	@echo "$(YELLOW)>> Démarrage de l'API FastAPI...$(RESET)"
	$(RUN) uvicorn src.api:app --reload --host $(API_HOST) --port $(API_PORT)

frontend: ## Lance le frontend Streamlit (FRONTEND_PORT, API_URL)
	@echo "$(YELLOW)>> Démarrage du frontend Streamlit...$(RESET)"
	API_URL=http://$(API_HOST):$(API_PORT) \
		$(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)


# ==============================================================================
# Airflow
# ==============================================================================

airflow: ## Démarre Airflow via docker compose sur http://localhost:8080
	@echo "$(YELLOW)>> Démarrage d'Airflow...$(RESET)"
	docker compose up -d airflow
	@echo "$(GREEN)[OK] Airflow disponible sur http://localhost:$(AIRFLOW_PORT)$(RESET)"
	@echo "  Login : admin / admin"

airflow-stop: ## Arrête le service Airflow
	@echo "$(YELLOW)>> Arrêt d'Airflow...$(RESET)"
	docker compose stop airflow
	@echo "$(GREEN)[OK] Airflow arrêté$(RESET)"

airflow-logs: ## Affiche les logs Airflow en temps réel
	docker compose logs -f airflow

airflow-trigger-retrain: ## Déclenche manuellement le DAG de ré-entraînement
	@echo "$(YELLOW)>> Déclenchement du DAG model_retraining...$(RESET)"
	docker compose exec airflow airflow dags trigger model_retraining
	@echo "$(GREEN)[OK] DAG model_retraining déclenché$(RESET)"
	@echo "  Suivre l'exécution sur http://localhost:$(AIRFLOW_PORT)"

airflow-trigger-predict: ## Déclenche manuellement le DAG de prédictions quotidiennes
	@echo "$(YELLOW)>> Déclenchement du DAG daily_predictions...$(RESET)"
	docker compose exec airflow airflow dags trigger daily_predictions
	@echo "$(GREEN)[OK] DAG daily_predictions déclenché$(RESET)"

airflow-dags: ## Liste les DAGs disponibles
	docker compose exec airflow airflow dags list

airflow-init: ## Initialise la DB Airflow et crée l'utilisateur admin
	@echo "$(YELLOW)>> Initialisation d'Airflow...$(RESET)"
	docker compose exec airflow airflow db migrate
	docker compose exec airflow airflow users create \
		--username admin --password admin \
		--firstname Admin --lastname User \
		--role Admin --email admin@example.com
	@echo "$(GREEN)[OK] Airflow initialisé (admin/admin)$(RESET)"


# ==============================================================================
# Docker
# ==============================================================================

docker-build: ## Construit les images train, api, frontend et airflow
	@echo "$(YELLOW)>> Build des images Docker...$(RESET)"
	docker build -f docker/Dockerfile.train    -t mlproject-train    .
	docker build -f docker/Dockerfile.api      -t mlproject-api      .
	docker build -f docker/Dockerfile.frontend -t mlproject-frontend .
	docker build -f docker/Dockerfile.airflow  -t mlproject-airflow  .
	@echo "$(GREEN)[OK] Images construites$(RESET)"

docker-run: ## Lance l'entraînement en conteneur (one-shot)
	@echo "$(YELLOW)>> Entraînement en conteneur...$(RESET)"
	docker run --rm -v "$(CURDIR)/models:/app/models" \
		-e MLFLOW_TRACKING_URI=sqlite:///mlruns.db \
		mlproject-train

docker-train: ## Entraîne via docker compose (profil train)
	@echo "$(YELLOW)>> Entraînement via docker compose...$(RESET)"
	docker compose --profile train run --rm train
	@echo "$(GREEN)[OK] Modèle disponible dans le volume models_data$(RESET)"

docker-up: ## Démarre la stack complète (mlflow → train → api → frontend → airflow)
	@echo "$(YELLOW)>> Démarrage de la stack complète...$(RESET)"
	docker compose up -d mlflow
	@echo "  Attente de MLflow..."
	sleep 5
	docker compose --profile train run --rm train
	docker compose up -d api frontend airflow
	@echo "$(GREEN)[OK] Stack démarrée$(RESET)"
	@echo "  MLflow   → http://localhost:$(MLFLOW_PORT)"
	@echo "  API      → http://localhost:$(API_PORT)"
	@echo "  Frontend → http://localhost:$(FRONTEND_PORT)"
	@echo "  Airflow  → http://localhost:$(AIRFLOW_PORT)"

docker-down: ## Arrête et supprime les conteneurs (conserve les volumes)
	@echo "$(YELLOW)>> Arrêt de la stack...$(RESET)"
	docker compose down
	@echo "$(GREEN)[OK] Stack arrêtée$(RESET)"

docker-down-all: ## Arrête les conteneurs ET supprime les volumes
	@echo "$(YELLOW)>> Arrêt complet avec suppression des volumes...$(RESET)"
	docker compose down -v
	@echo "$(GREEN)[OK] Stack et volumes supprimés$(RESET)"

docker-logs: ## Affiche les logs de tous les services en temps réel
	docker compose logs -f


# ==============================================================================
# Qualité
# ==============================================================================

lint: ## Vérifie le style (ruff)
	$(RUN) ruff check src

format: ## Formate le code (ruff)
	$(RUN) ruff format src

type: ## Vérifie les types (mypy)
	$(RUN) mypy src --namespace-packages --explicit-package-bases

test: ## Lance les tests (pytest)
	$(RUN) pytest

check: lint type test ## Workflow qualité complet (lint + types + tests)


# ==============================================================================
# Nettoyage
# ==============================================================================

clean: ## Supprime les fichiers temporaires Python
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov/ .coverage .ruff_cache
	@echo "$(GREEN)[OK] Nettoyage terminé$(RESET)"