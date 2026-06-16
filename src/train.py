from __future__ import annotations

import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from config import MODEL_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
from data import load_data, split
from features import build_preprocessor

# TODO (S5-1) : importer mlflow et mlflow.sklearn
# import mlflow
# import mlflow.sklearn


# ---------------------------------------------------------------------------
# Construction du pipeline complet (preprocessing + classifieur)
# ---------------------------------------------------------------------------

def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """
    Construit le pipeline sklearn end-to-end :
      preprocessor (ColumnTransformer) → LogisticRegression

    La Régression Logistique est choisie comme baseline car :
      - rapide à entraîner sur >100 000 lignes
      - probabilités calibrées nativement (utile pour roc_auc)
      - interprétable (coefficients par feature)

    Parameters
    ----------
    c : float
        Inverse de la force de régularisation L2 (plus grand = moins régularisé).
    max_iter : int
        Nombre maximum d'itérations du solveur.

    Returns
    -------
    Pipeline
        Pipeline non encore fitté.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    class_weight="balanced",   # compense le léger déséquilibre des classes
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
    """
    Entraîne la baseline et retourne les métriques d'évaluation.

    Parameters
    ----------
    c : float
        Hyperparamètre C de la LogisticRegression.
    max_iter : int
        Nombre max d'itérations du solveur.

    Returns
    -------
    dict
        Dictionnaire contenant f1 et roc_auc sur le jeu de validation.
    """
    # --- Chargement et découpage des données --------------------------------
    df = load_data()
    x_train, x_val, y_train, y_val = split(df)

    print(f"Train : {len(x_train):,} lignes | Val : {len(x_val):,} lignes")
    print(f"Distribution cible (val) — satisfied : {y_val.mean():.2%}")

    # TODO (S5-2) : configurer l'URI de tracking et l'expérience
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # TODO (S5-3) : ouvrir un run englobant l'entraînement et l'évaluation
    # with mlflow.start_run():

    # --- Entraînement -------------------------------------------------------
    model = build_model(c=c, max_iter=max_iter)
    model.fit(x_train, y_train)

    # --- Évaluation ---------------------------------------------------------
    proba = model.predict_proba(x_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "f1":      float(f1_score(y_val, preds)),
        "roc_auc": float(roc_auc_score(y_val, proba)),
    }

    print(f"\nf1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")
    print("\n" + classification_report(y_val, preds, target_names=["dissatisfied", "satisfied"]))

    # TODO (S5-4) : logger les paramètres avec mlflow.log_params
    # mlflow.log_params({"c": c, "max_iter": max_iter})

    # TODO (S5-5) : logger les métriques avec mlflow.log_metrics
    # mlflow.log_metrics(metrics)

    # TODO (S5-6) : logger le modèle avec mlflow.sklearn.log_model
    # mlflow.sklearn.log_model(
    #     sk_model=model,
    #     artifact_path="model",
    #     registered_model_name="airline-classifier",
    # )

    # TODO (S5-7 bonus) : sauvegarder la matrice de confusion et la logger en artefact
    # _log_confusion_matrix(y_val, preds)

    # --- Sauvegarde locale du modèle ----------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModèle sauvegardé → {model_path}")

    return metrics


def _log_confusion_matrix(y_true, y_pred) -> None:
    """
    Génère la matrice de confusion en PNG et la logue comme artefact MLflow.
    Décommenter l'appel dans train() et l'import mlflow pour activer.
    """
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

    cm_path = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    # mlflow.log_artifact(str(cm_path))
    print(f"Matrice de confusion sauvegardée → {cm_path}")


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline LogisticRegression — Airline Passenger Satisfaction"
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse de la régularisation L2 (défaut : 1.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Nombre max d'itérations du solveur (défaut : 1000)",
    )
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()