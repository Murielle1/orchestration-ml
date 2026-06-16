"""Entraînement et optimisation de plusieurs modèles de classification (AutoML + SHAP).

Séance 7 - TP AutoML & SHAP
Dataset : Airline Passenger Satisfaction
Cible   : satisfaction  →  1 (satisfied) / 0 (neutral or dissatisfied)

Lancement :
    python -m src/train_models
    python -m src/train_models --cv 3 --scoring roc_auc
    python -m src/train_models --no-mlflow
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import MODEL_DIR, MODEL_NAME, RANDOM_STATE
from src.data import load_data, split
from src.features import build_preprocessor
from src.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ---------------------------------------------------------------------------
# Spécification des modèles
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    estimator: ClassifierMixin
    param_grid: dict


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
                verbosity=0,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
        ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [31, 63],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("clf", estimator)])


# ---------------------------------------------------------------------------
# Résultat d'optimisation
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray


# ---------------------------------------------------------------------------
# Optimisation par GridSearchCV
# ---------------------------------------------------------------------------


def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> FitResult:
    logger.info("Optimisation de %s (cv=%d, scoring=%s)", spec.name, cv, scoring)

    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    result = FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        f1=float(f1_score(y_test, preds)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        preds=preds,
    )
    logger.info(
        "%s → cv_%s=%.3f | f1=%.3f | roc_auc=%.3f",
        spec.name,
        scoring,
        result.cv_score,
        result.f1,
        result.roc_auc,
    )
    return result


# ---------------------------------------------------------------------------
# Logging MLflow
# ---------------------------------------------------------------------------


def log_run_to_mlflow(
    result: FitResult,
    x_test,
    y_test,
    cv: int,
    scoring: str,
    register_as: str | None = None,
) -> None:
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)
        mlflow.log_params(result.best_params)
        mlflow.log_metrics(
            {
                f"cv_{scoring}": result.cv_score,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
        )

        # Matrice de confusion
        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=["dissatisfied", "satisfied"]).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Rapport de classification
        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(
            str,
            classification_report(y_test, result.preds, target_names=["dissatisfied", "satisfied"]),
        )
        mlflow.log_text(report_text, "classification_report.txt")
        logger.info("\n%s", report_text)

        # SHAP summary (optionnel)
        _log_shap_summary(result.best_estimator, x_test, result.name)

        # Enregistrement du modèle
        signature = infer_signature(x_test, result.best_estimator.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            result.best_estimator,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        # Documentation dans le Registry (bonus S7-5)
        if register_as and model_info.registered_model_version:
            describe_registered_version(
                name=register_as,
                version=int(model_info.registered_model_version),
                result=result,
                cv=cv,
                scoring=scoring,
            )


def _log_shap_summary(pipeline: Pipeline, x_test, model_name: str) -> None:
    try:
        import shap

        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["clf"]
        x_transformed = preprocessor.transform(x_test)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, x_transformed, show=False, plot_size=None)
        plt.gca().set_title(f"SHAP Summary — {model_name}")
        plt.tight_layout()
        mlflow.log_figure(fig, "shap_summary.png")
        plt.close(fig)
        logger.info("SHAP summary loggué pour %s", model_name)
    except Exception as exc:
        logger.warning("SHAP indisponible pour %s : %s", model_name, exc)


def describe_registered_version(
    name: str, version: int, result: FitResult, cv: int, scoring: str
) -> None:
    client = mlflow.MlflowClient()
    description = (
        f"Modèle : {result.name}\n"
        f"Dataset : Airline Passenger Satisfaction\n"
        f"Recherche : GridSearchCV (cv={cv}, scoring={scoring})\n"
        f"Meilleurs hyperparamètres : {result.best_params}\n"
        f"Métriques test → f1={result.f1:.3f} | roc_auc={result.roc_auc:.3f}"
    )
    client.update_model_version(name, str(version), description=description)
    for key, value in {
        "model_family": result.name,
        "search_method": "GridSearchCV",
        "cv": str(cv),
        "scoring": scoring,
        "f1": f"{result.f1:.4f}",
        "roc_auc": f"{result.roc_auc:.4f}",
        "dataset": "airline-passenger-satisfaction",
    }.items():
        client.set_model_version_tag(name, str(version), key, value)
    logger.info("Version %d du modèle '%s' documentée dans le registry", version, name)


# ---------------------------------------------------------------------------
# Entraînement de tous les modèles
# ---------------------------------------------------------------------------


def train_all(cv: int = 5, scoring: str = "roc_auc", use_mlflow: bool = True) -> list[FitResult]:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    logger.info("Train : %d lignes | Test : %d lignes", len(x_train), len(x_test))

    if use_mlflow:
        # Configuration centralisée via tracking.py
        setup_experiment()

    results = [
        optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)
    best = results[0]
    logger.info("Meilleur modèle : %s (roc_auc=%.3f | f1=%.3f)", best.name, best.roc_auc, best.f1)

    if use_mlflow:
        with mlflow.start_run(run_name="compare-models"):
            # Traçabilité des données sur le run parent
            log_dataset(df, context="training", name="train")

            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.set_tag("best_model", best.name)
            mlflow.log_metrics({"best_roc_auc": best.roc_auc, "best_f1": best.f1})

            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as=register_as)

        logger.info("Meilleur modèle enregistré dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
    logger.info("Modèle sauvegardé → %s", MODEL_DIR / "model.joblib")

    logger.info("\n%-20s  %8s  %8s  %8s", "Modèle", "CV score", "F1", "ROC AUC")
    logger.info("-" * 50)
    for r in results:
        logger.info("%-20s  %8.3f  %8.3f  %8.3f", r.name, r.cv_score, r.f1, r.roc_auc)

    return results


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RF / XGBoost / LightGBM — Airline Passenger Satisfaction"
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    parser.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
