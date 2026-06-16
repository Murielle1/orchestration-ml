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
MLFLOW_PORT  := 5000
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
        data features train train-models train-optuna mlflow api frontend \
        docker-build docker-run docker-up docker-down \
        lint format type test check clean


# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Setup - Installation de l'environnement Python (uv + pyproject.toml) [FOURNI]
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
	# TODO (S0) : nécessite la Kaggle CLI configurée (~/.kaggle/kaggle.json)
	@echo "$(YELLOW)>> Téléchargement du dataset...$(RESET)"
	@mkdir -p $(RAW_DIR) $(PROCESSED_DIR)
	kaggle datasets download -d teejmahal20/airline-passenger-satisfaction \
		--path $(RAW_DIR) --unzip
	@echo "$(GREEN)[OK] Fichiers disponibles dans $(RAW_DIR)$(RESET)"

features: ## Génère les features prétraitées dans data/processed/
	# TODO (S0) : prépare X_train, X_test encodés + pipeline joblib
	@echo "$(YELLOW)>> Feature engineering...$(RESET)"
	$(PYTHON) -m src.features \
		--train-path $(TRAIN_CSV) \
		--test-path  $(TEST_CSV) \
		--output-dir $(PROCESSED_DIR)
	@echo "$(GREEN)[OK] Features disponibles dans $(PROCESSED_DIR)$(RESET)"

train: ## Entraîne la baseline → models/model.joblib (C=.. MAX_ITER=..)
	# TODO (S5) : instrumenter avec MLflow Tracking
	@echo "$(YELLOW)>> Entraînement baseline (C=$(C) max_iter=$(MAX_ITER))...$(RESET)"
	$(PYTHON) -m src.train --c $(C) --max-iter $(MAX_ITER)
	@echo "$(GREEN)[OK] Modèle sauvegardé dans models/$(RESET)"

train-models: ## Compare RF / XGBoost / LightGBM (GridSearchCV) + SHAP (CV=.. SCORING=..)
	# TODO (S7) : $(PYTHON) -m src.train_models --cv $(CV) --scoring $(SCORING)

train-optuna: ## Optimise RF / XGBoost / LightGBM avec Optuna (N_TRIALS=.. CV=..)
	# TODO (S6) : $(PYTHON) -m src.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

mlflow: ## Démarre le serveur MLflow (docker compose)
	# TODO (S5) : docker compose -f docker-compose.yml up -d mlflow

api: ## Lance l'API FastAPI en rechargement auto (voir API_HOST/API_PORT)
	# TODO (S12) : $(RUN) uvicorn src.api:app --reload --host $(API_HOST) --port $(API_PORT)

frontend: ## Lance le frontend Streamlit (voir FRONTEND_PORT, API_URL)
	# TODO (S14bis) : $(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)


# ==============================================================================
# Docker
# ==============================================================================

docker-build: ## Construit l'image d'entraînement
	# TODO (S8) : docker build -f docker/Dockerfile.train -t src-train .

docker-run: ## Lance l'entraînement en conteneur
	# TODO (S8) : docker run --rm -v "$(CURDIR)/../models:/app/models" src-train

docker-up: ## Démarre la stack complète (mlflow, api, frontend)
	# TODO (S14) : docker compose -f docker-compose.yml up -d --build mlflow api frontend

docker-down: ## Arrête et supprime les conteneurs (conserve les volumes)
	# TODO (S14) : docker compose -f docker-compose.yml down


# ==============================================================================
# Qualité
# ==============================================================================

lint: ## Vérifie le style (ruff)
	$(RUN) ruff check src

format: ## Formate le code (ruff)
	$(RUN) ruff format src

type: ## Vérifie les types (mypy)
	$(RUN) mypy src

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