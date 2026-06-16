"""Configuration partagée du suivi MLflow.

Séance 5 - TP MLflow Tracking (suite)
    Centralise la configuration du tracking pour éviter de la dupliquer dans
    chaque script d'entraînement, et ajoute la traçabilité des données
    (dataset lineage).

Dataset : Airline Passenger Satisfaction
"""

from __future__ import annotations

import logging

import mlflow
from mlflow.data.pandas_dataset import from_pandas

import pandas as pd

from src.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    """Configure le tracking MLflow et les métadonnées de l'expérience.

    - Pointe vers MLFLOW_TRACKING_URI
    - Crée ou sélectionne l'expérience MLFLOW_EXPERIMENT
    - Applique la description et les tags définis dans config.py

    L'opération est idempotente (re-appelable sans erreur).
    """
    # S5-8 : configuration du tracking et de l'expérience
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)

    client = mlflow.MlflowClient()

    if MLFLOW_EXPERIMENT_DESCRIPTION:
        client.set_experiment_tag(
            experiment.experiment_id,
            "mlflow.note.content",
            MLFLOW_EXPERIMENT_DESCRIPTION,
        )

    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)

    logger.info(
        "MLflow configuré → URI : %s | Expérience : %s (id=%s)",
        MLFLOW_TRACKING_URI,
        MLFLOW_EXPERIMENT,
        experiment.experiment_id,
    )


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Logue un dataset MLflow dans le run courant (traçabilité données → modèle).

    Rattache au run la source des données, le schéma et un profil, visibles
    dans l'onglet "Datasets" de l'UI MLflow.

    Parameters
    ----------
    df : pd.DataFrame
        Données à référencer (features + cible).
    context : str
        Rôle du dataset dans le run : "training" ou "evaluation".
    name : str
        Nom logique du dataset (ex. "train", "test").
    """
    # S5-9 : traçabilité du dataset
    dataset = from_pandas(
        df,
        source=str(DATA_PATH),
        targets=TARGET,
        name=name,
    )
    mlflow.log_input(dataset, context=context)
    logger.info("Dataset '%s' loggué (context=%s, %d lignes)", name, context, len(df))
