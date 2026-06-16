"""Entraînement du modèle de classification (baseline).

Séance 5 - TP MLflow Tracking
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

from src.config import MODEL_DIR, MODEL_NAME
from src.data import load_data, split
from src.features import build_preprocessor
from src.tracking import log_dataset, setup_experiment


# ---------------------------------------------------------------------------
# Construction du pipeline
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

    # --- Configuration MLflow (via tracking.py) ----------------------------
    setup_experiment()

    # --- Run MLflow --------------------------------------------------------
    with mlflow.start_run(
        tags={"model_type": "logistic_regression"},
    ):
        # Traçabilité des données
        log_dataset(df, context="training", name="train")

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

        # Logger paramètres, métriques et modèle
        mlflow.log_params({"c": c, "max_iter": max_iter})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # Matrice de confusion
        cm_path = _save_confusion_matrix(y_val, preds)
        mlflow.log_artifact(str(cm_path))

        run_id = mlflow.active_run().info.run_id
        print(f"\nRun MLflow : {run_id}")

    # --- Sauvegarde locale -------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé → {model_path}")

    return metrics


def _save_confusion_matrix(y_true, y_pred):
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