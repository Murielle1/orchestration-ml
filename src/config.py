"""Configuration centrale du projet Airline Passenger Satisfaction."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
DATA_DIR   = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"

# Chemin unique fusionné (utilisé par load_data si on travaille sur un seul CSV)
DATA_PATH  = DATA_DIR / "train.csv"

MODEL_DIR  = ROOT / "models"

# ---------------------------------------------------------------------------
# Variable cible
# ---------------------------------------------------------------------------
TARGET = "satisfaction"

# Valeurs textuelles dans le CSV → encodage binaire effectué dans data.py
TARGET_POSITIVE = "satisfied"           # → 1
TARGET_NEGATIVE = "neutral or dissatisfied"  # → 0

# Colonnes à supprimer dès le chargement (identifiants sans valeur prédictive)
DROP_COLS: list[str] = ["Unnamed: 0", "id"]

# ---------------------------------------------------------------------------
# Features numériques continues
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

# ---------------------------------------------------------------------------
# Features catégorielles (encodage OHE dans features.py)
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES: list[str] = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

# ---------------------------------------------------------------------------
# Scores de satisfaction (entiers 0-5) — pas de scaling nécessaire
# ---------------------------------------------------------------------------
RATING_FEATURES: list[str] = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

# Toutes les features utilisées par le modèle
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + RATING_FEATURES

# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ---------------------------------------------------------------------------
# MLflow — surcouche via variables d'environnement (principe 12-factor)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "airline-satisfaction-baseline")
MODEL_NAME          = os.getenv("MODEL_NAME", "airline-classifier")

MLFLOW_EXPERIMENT_DESCRIPTION = os.getenv(
    "MLFLOW_EXPERIMENT_DESCRIPTION",
    "Classification binaire — satisfaction passagers aériens (cours MLOps)",
)


def _parse_tags(raw: str) -> dict[str, str]:
    """Parse une chaîne `cle=valeur,cle2=valeur2` en dictionnaire de tags."""
    tags: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key, value = key.strip(), value.strip()
        if key:
            tags[key] = value
    return tags


MLFLOW_EXPERIMENT_TAGS = _parse_tags(
    os.getenv("MLFLOW_EXPERIMENT_TAGS", "course=mlops,dataset=airline-satisfaction")
)

# ---------------------------------------------------------------------------
# Seuils de la porte qualité (evaluate.py)
# ---------------------------------------------------------------------------
EVAL_ROC_AUC_MIN = float(os.getenv("EVAL_ROC_AUC_MIN", "0.85"))
EVAL_F1_MIN      = float(os.getenv("EVAL_F1_MIN", "0.80"))

# ---------------------------------------------------------------------------
# API FastAPI
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")