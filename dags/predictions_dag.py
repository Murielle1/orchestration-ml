"""DAG Airflow - trafic de prévisions quotidien.

Séance 17 - TP Airflow (suite)
    Chaque jour à 10h, échantillonne 20 lignes du dataset et les envoie
    en POST /predict pour simuler un flux de prévisions en production.

Dataset : Airline Passenger Satisfaction
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

N_PREDICTIONS = 20

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_send_predictions(**context) -> None:
    """Échantillonne N_PREDICTIONS lignes et les envoie à l'API /predict."""
    import httpx

    from src.config import API_URL, TARGET
    from src.data import load_data

    features = load_data().drop(columns=[TARGET])

    # S17-6 : échantillonnage
    sample = features.sample(n=N_PREDICTIONS, random_state=None)  # random à chaque run

    results = []
    errors = 0

    # S17-7 : envoi à l'API
    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        # Vérification que l'API est disponible
        try:
            client.get("/health").raise_for_status()
            logger.info("API joignable sur %s", API_URL)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"API non joignable sur {API_URL} : {exc}") from exc

        for i, (_, row) in enumerate(sample.iterrows()):
            # json.loads(to_json()) convertit les types numpy en types Python natifs
            payload = json.loads(row.to_json())

            try:
                response = client.post("/predict", json=payload)
                response.raise_for_status()
                result = response.json()
                results.append(result)
                logger.info(
                    "Prédiction %d/%d → %s (proba=%.3f)",
                    i + 1,
                    N_PREDICTIONS,
                    result.get("label", "?"),
                    result.get("probability", 0),
                )
            except httpx.HTTPError as exc:
                errors += 1
                logger.warning("Erreur prédiction %d : %s", i + 1, exc)

    n_satisfied = sum(1 for r in results if r.get("prediction") == 1)
    n_unsatisfied = len(results) - n_satisfied

    logger.info(
        "%d/%d prévisions envoyées (%d erreurs) → satisfaits: %d | insatisfaits: %d",
        len(results),
        N_PREDICTIONS,
        errors,
        n_satisfied,
        n_unsatisfied,
    )

    # Pousser un résumé en XCom
    context["ti"].xcom_push(key="n_sent", value=len(results))
    context["ti"].xcom_push(key="n_errors", value=errors)
    context["ti"].xcom_push(key="n_satisfied", value=n_satisfied)
    context["ti"].xcom_push(key="n_unsatisfied", value=n_unsatisfied)

    if errors == N_PREDICTIONS:
        raise RuntimeError("Toutes les prédictions ont échoué.")


with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 prévisions par jour à l'API (trafic simulé)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    # S17-8 : tous les jours à 10h
    schedule="0 10 * * *",
    catchup=False,
    tags=["classification", "predictions", "airline"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
