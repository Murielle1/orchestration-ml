"""Entraînement du modèle de classification (baseline).

Séance 5 - TP MLflow Tracking
    Ce script entraîne et évalue un modèle AVEC suivi d'expérience MLflow.

Dataset : Airline Passenger Satisfaction
Cible   : satisfaction  →  1 (satisfied) / 0 (neutral or dissatisfied)
"""
from __future__ import annotations

import argparse
import joblib

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from config import (
    MODEL_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
)
from data import load_data, split
from features import build_preprocessor


# ---------------------------------------------------------------------------
# Construction du pipeline complet
# ---------------------------------------------------------------------------

def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Entraînement + évaluation
# ---------------------------------------------------------------------------

def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    # --- Chargement et découpage -------------------------------------------
    df = load_data()
    x_train, x_val, y_train, y_val = split(df)

    print(f"Train : {len(x_train):,} lignes | Val : {len(x_val):,} lignes")
    print(f"Distribution cible (val) — satisfied : {y_val.mean():.2%}")

    # --- Configuration MLflow ----------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # --- Run MLflow --------------------------------------------------------
    with mlflow.start_run(
        tags={**MLFLOW_EXPERIMENT_TAGS, "model_type": "logistic_regression"},
        description=MLFLOW_EXPERIMENT_DESCRIPTION,
    ):
        # Entraînement
        model = build_model(c=c, max_iter=max_iter)
        model.fit(x_train, y_train)

        # Évaluation
        proba = model.predict_proba(x_val)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "f1":      float(f1_score(y_val, preds)),
            "roc_auc": float(roc_auc_score(y_val, proba)),
        }

        print(f"\nf1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")
        print(
            "\n"
            + classification_report(
                y_val, preds, target_names=["dissatisfied", "satisfied"]
            )
        )

        # Logger les paramètres
        mlflow.log_params({"c": c, "max_iter": max_iter})

        # Logger les métriques
        mlflow.log_metrics(metrics)

        # Logger le modèle
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # Logger la matrice de confusion (bonus S5-7)
        cm_path = _save_confusion_matrix(y_val, preds)
        mlflow.log_artifact(str(cm_path))

        # Afficher l'URL du run
        run_id = mlflow.active_run().info.run_id
        print(f"\nRun MLflow : {MLFLOW_TRACKING_URI}/#/runs/{run_id}")

    # --- Sauvegarde locale -------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé → {model_path}")

    return metrics


def _save_confusion_matrix(y_true, y_pred):
    """Génère et sauvegarde la matrice de confusion en PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["dissatisfied", "satisfied"],
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Matrice de confusion — validation")
    fig.tight_layout()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    return cm_path


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline LogisticRegression — Airline Passenger Satisfaction"
    )
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()