"""DAG Airflow - pipeline de ré-entraînement du modèle.

Séance 17 - TP Airflow
    Pipeline : préparation des données → entraînement → contrôle qualité.

Dataset : Airline Passenger Satisfaction
Schedule : tous les lundis à 3h du matin
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# Seuil F1 minimal — pipeline échoue si le modèle est en dessous
QUALITY_THRESHOLD = 0.80  # relevé à 0.80 car le dataset permet >0.85


default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# ---------------------------------------------------------------------------
# Tâches
# ---------------------------------------------------------------------------


def task_prepare_data(**context) -> None:
    """S17-1 : prépare les données en utilisant data.py et features.py.

    - Charge train.csv via load_data() (src/data.py)
    - Construit le preprocessor via build_preprocessor() (src/features.py)
    - Fit + transform sur le train, transform sur le test
    - Sauvegarde le pipeline joblib dans data/processed/
    """
    from pathlib import Path

    import joblib
    import pandas as pd

    from src.data import load_data
    from src.features import build_preprocessor
    from src.config import TARGET, TEST_PATH

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chargement via load_data (nettoyage + encodage cible inclus)
    logger.info("Chargement de train.csv...")
    train_df = load_data()
    logger.info("Chargement de test.csv...")
    test_df = load_data(path=TEST_PATH)

    logger.info("Train : %d lignes | Test : %d lignes", len(train_df), len(test_df))

    # Séparation features / cible
    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET])

    # Preprocessing via build_preprocessor (fit sur train uniquement)
    logger.info("Fit du preprocessor sur le train...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Sauvegarde du pipeline
    pipeline_path = output_dir / "preprocessing_pipeline.joblib"
    joblib.dump(preprocessor, pipeline_path)
    logger.info("Pipeline sauvegardé → %s", pipeline_path)

    context["ti"].xcom_push(key="n_train", value=len(train_df))
    context["ti"].xcom_push(key="n_test", value=len(test_df))


def task_train(**context) -> None:
    """S17-2 : entraîne la baseline et pousse le F1 dans XCom."""
    from src.train import train

    logger.info("Démarrage de l'entraînement...")
    metrics = train(c=1.0, max_iter=1000)

    f1 = metrics["f1"]
    roc_auc = metrics["roc_auc"]
    logger.info("Entraînement terminé → f1=%.3f | roc_auc=%.3f", f1, roc_auc)

    # Pousser les métriques en XCom pour la tâche de contrôle qualité
    context["ti"].xcom_push(key="f1", value=f1)
    context["ti"].xcom_push(key="roc_auc", value=roc_auc)


def task_check_quality(**context) -> None:
    """S17-3 : vérifie que le F1 dépasse le seuil minimal."""
    ti = context["ti"]

    f1 = ti.xcom_pull(task_ids="train", key="f1")
    roc_auc = ti.xcom_pull(task_ids="train", key="roc_auc")

    logger.info(
        "Contrôle qualité → f1=%.3f (seuil=%.2f) | roc_auc=%.3f",
        f1,
        QUALITY_THRESHOLD,
        roc_auc,
    )

    if f1 is None:
        raise ValueError("F1 non disponible dans XCom — la tâche train a échoué.")

    if f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Porte qualité NON passée : f1={f1:.3f} < seuil={QUALITY_THRESHOLD}. "
            "Le modèle n'est PAS déployé."
        )

    logger.info(
        "✅ Porte qualité passée : f1=%.3f >= seuil=%.2f | roc_auc=%.3f. "
        "Modèle validé et disponible dans models/model.joblib.",
        f1,
        QUALITY_THRESHOLD,
        roc_auc,
    )


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="model_retraining",
    description="Prépare les données, ré-entraîne le modèle et contrôle sa qualité",
    # S17-4 : tous les lundis à 3h du matin
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training", "airline"],
) as dag:
    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=task_prepare_data,
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=task_train,
    )

    check = PythonOperator(
        task_id="check_quality",
        python_callable=task_check_quality,
    )

    # S17-5 : ordre d'exécution
    prepare >> train_task >> check
