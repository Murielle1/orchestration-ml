"""Évaluation automatisée et validation du modèle.

Séance 11 - TP Tests Données & Modèle
    `mlflow.models.evaluate` calcule en une passe un ensemble de métriques et
    d'artefacts (matrice de confusion, courbes ROC / précision-rappel) sur un
    jeu d'évaluation. `mlflow.validate_evaluation_results` applique ensuite une
    porte qualité : le modèle est rejeté (exception) si une métrique passe sous
    son seuil.

Dataset : Airline Passenger Satisfaction
Cible   : satisfaction  →  1 (satisfied) / 0 (neutral or dissatisfied)

Lancement :
    python -m src/evaluate                        # dernière version du registry
    python -m src/evaluate --model-uri models:/airline-classifier/1
    python -m src/evaluate --no-validate          # évalue sans porte qualité
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.data
import mlflow.models
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold
from mlflow.data.pandas_dataset import from_pandas
from src.config import (
    DATA_PATH,
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    MODEL_NAME,
    TARGET,
)
from src.data import load_data, split
from src.tracking import setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Résolution de l'URI du modèle
# ---------------------------------------------------------------------------


def latest_model_uri() -> str:
    """Résout l'URI de la dernière version enregistrée de MODEL_NAME.

    Returns
    -------
    str
        URI MLflow de la forme ``models:/<MODEL_NAME>/<version>``.

    Raises
    ------
    RuntimeError
        Si aucune version n'est enregistrée dans le registry.
    """
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    if not versions:
        raise RuntimeError(
            f"Aucune version enregistrée pour '{MODEL_NAME}'. "
            "Lancez d'abord un entraînement (make train ou make train-models)."
        )

    latest = max(versions, key=lambda v: int(v.version))
    uri = f"models:/{MODEL_NAME}/{latest.version}"
    logger.info("Dernière version trouvée : %s (version %s)", MODEL_NAME, latest.version)
    return uri


# ---------------------------------------------------------------------------
# Seuils de la porte qualité
# ---------------------------------------------------------------------------


def build_thresholds() -> dict[str, MetricThreshold]:
    """Construit les seuils de validation depuis la configuration.

    Les seuils sont définis dans config.py et peuvent être surchargés via
    les variables d'environnement EVAL_ROC_AUC_MIN et EVAL_F1_MIN.

    Sur ce dataset, les modèles à base d'arbres atteignent facilement
    roc_auc > 0.95 et f1 > 0.90 — les seuils par défaut (0.85 / 0.80)
    sont donc conservateurs et garantissent un minimum raisonnable.

    Returns
    -------
    dict[str, MetricThreshold]
        Seuils minimaux sur ``roc_auc`` et ``f1_score``.
    """
    # S11-1
    thresholds = {
        "roc_auc": MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True),
        "f1_score": MetricThreshold(threshold=EVAL_F1_MIN, greater_is_better=True),
    }
    logger.info(
        "Seuils qualité → roc_auc >= %.2f | f1_score >= %.2f",
        EVAL_ROC_AUC_MIN,
        EVAL_F1_MIN,
    )
    return thresholds


# ---------------------------------------------------------------------------
# Évaluation principale
# ---------------------------------------------------------------------------


def evaluate_model(
    model_uri: str | None = None,
    validate: bool = True,
) -> mlflow.models.EvaluationResult:
    """Évalue un modèle du registry et applique optionnellement la porte qualité.

    Parameters
    ----------
    model_uri : str, optional
        URI MLflow du modèle à évaluer. Par défaut, la dernière version
        enregistrée de MODEL_NAME.
    validate : bool
        Applique la porte qualité après l'évaluation. Lève une exception
        (MlflowException) si un seuil n'est pas atteint.

    Returns
    -------
    mlflow.models.EvaluationResult
        Résultat complet (métriques + artefacts).
    """
    # --- Chargement des données --------------------------------------------
    df = load_data()
    _, x_test, _, y_test = split(df)

    # mlflow.evaluate attend un DataFrame unique features + cible
    eval_df = x_test.copy()
    eval_df[TARGET] = y_test.values
    logger.info("Jeu d'évaluation : %d lignes | %d features", len(eval_df), x_test.shape[1])

    # --- Configuration MLflow ----------------------------------------------
    setup_experiment()

    model_uri = model_uri or latest_model_uri()
    logger.info("Évaluation du modèle : %s", model_uri)

    # --- Run MLflow d'évaluation -------------------------------------------
    with mlflow.start_run(run_name="evaluate"):
        # S11-2a : traçabilité du jeu d'évaluation
        dataset = from_pandas(
            eval_df,
            source=str(DATA_PATH),
            targets=TARGET,
            name="eval",
        )
        mlflow.log_input(dataset, context="evaluation")

        # S11-2b : évaluation automatisée
        # mlflow.models.evaluate calcule automatiquement :
        #   - métriques : accuracy, f1_score, roc_auc, precision, recall, log_loss
        #   - artefacts : matrice de confusion, courbe ROC, courbe précision-rappel
        result = mlflow.models.evaluate(
            model_uri,
            data=eval_df,
            targets=TARGET,
            model_type="classifier",
            evaluators=["default"],
        )

        logger.info(
            "Résultats → f1_score=%.3f | roc_auc=%.3f | accuracy=%.3f",
            result.metrics.get("f1_score", float("nan")),
            result.metrics.get("roc_auc", float("nan")),
            result.metrics.get("accuracy_score", float("nan")),
        )

        # S11-3 : porte qualité
        if validate:
            logger.info("Application de la porte qualité...")
            mlflow.validate_evaluation_results(
                validation_thresholds=build_thresholds(),
                candidate_result=result,
            )
            logger.info("✓ Porte qualité passée — modèle validé.")

        active_run = mlflow.active_run()
        if active_run:
            logger.info("Run MLflow : %s", active_run.info.run_id)

    return result


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation automatisée — Airline Passenger Satisfaction"
    )
    parser.add_argument(
        "--model-uri",
        default=None,
        help=("URI du modèle à évaluer (défaut : dernière version de MODEL_NAME dans le registry)"),
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Évalue sans appliquer la porte qualité",
    )
    args = parser.parse_args()

    try:
        result = evaluate_model(model_uri=args.model_uri, validate=args.validate)
        logger.info("Évaluation terminée. Métriques :")
        for key, value in sorted(result.metrics.items()):
            logger.info("  %-30s %.4f", key, value)
    except MlflowException as exc:
        logger.error("Validation échouée : %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
