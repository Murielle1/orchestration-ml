"""Optimisation d'hyperparamètres avec Optuna.

Séance 6 - TP Optuna
    Ce module optimise les hyperparamètres de trois familles de modèles
    (Random Forest, XGBoost, LightGBM) avec Optuna (sampler TPE), compare
    leurs performances et persiste le meilleur dans `models/model.joblib`.

Dataset : Airline Passenger Satisfaction
Cible   : satisfaction  →  1 (satisfied) / 0 (neutral or dissatisfied)

Lancement :
    python -m src/train_optuna
    python -m src/train_optuna --n-trials 50 --cv 3
    python -m src/train_optuna --no-mlflow
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import optuna.samplers
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
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import MODEL_DIR, MODEL_NAME, RANDOM_STATE
from src.data import load_data, split
from src.features import build_preprocessor
from src.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Silencer les warnings verbeux d'Optuna et LightGBM
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Spécification des familles de modèles
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Spécification d'une famille de modèles à optimiser avec Optuna."""

    name: str
    suggest_params: Callable  # (trial) -> dict
    build_estimator: Callable  # (params) -> ClassifierMixin


def build_model_specs() -> list[ModelSpec]:
    """Construit les trois familles de modèles avec leurs espaces de recherche.

    Les espaces sont calibrés pour le dataset Airline Passenger Satisfaction
    (~100k lignes, 22 features mixtes).

    Returns
    -------
    list[ModelSpec]
        Random Forest, XGBoost et LightGBM.
    """

    # --- Random Forest ------------------------------------------------------
    def rf_suggest(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    def rf_build(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            RandomForestClassifier(
                **params,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )

    # --- XGBoost ------------------------------------------------------------
    def xgb_suggest(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    def xgb_build(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            XGBClassifier(
                **params,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
        )

    # --- LightGBM -----------------------------------------------------------
    def lgbm_suggest(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        }

    def lgbm_build(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            LGBMClassifier(
                **params,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1,
            ),
        )

    return [
        ModelSpec(name="random_forest", suggest_params=rf_suggest, build_estimator=rf_build),
        ModelSpec(name="xgboost", suggest_params=xgb_suggest, build_estimator=xgb_build),
        ModelSpec(name="lightgbm", suggest_params=lgbm_suggest, build_estimator=lgbm_build),
    ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("clf", estimator)])


# ---------------------------------------------------------------------------
# Résultat d'optimisation
# ---------------------------------------------------------------------------


@dataclass
class FamilyResult:
    spec: ModelSpec
    study: Any  # optuna.Study
    best_pipeline: Pipeline
    test_roc_auc: float
    test_f1: float
    preds: np.ndarray


# ---------------------------------------------------------------------------
# Fonction objectif Optuna
# ---------------------------------------------------------------------------


def objective(trial: optuna.Trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    """Fonction objectif : ROC AUC moyen en validation croisée (à maximiser)."""
    params = spec.suggest_params(trial)
    estimator = spec.build_estimator(params)
    pipeline = build_pipeline(estimator)

    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Étude Optuna
# ---------------------------------------------------------------------------


def run_study(
    spec: ModelSpec,
    x_train,
    y_train,
    n_trials: int,
    cv: int,
) -> optuna.Study:
    """Lance l'étude Optuna pour une famille de modèles."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=spec.name,
    )
    study.optimize(
        lambda trial: objective(trial, spec, x_train, y_train, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    logger.info(
        "%s → meilleur cv_roc_auc=%.4f | params=%s",
        spec.name,
        study.best_value,
        study.best_params,
    )
    return study


# ---------------------------------------------------------------------------
# Optimisation complète d'une famille
# ---------------------------------------------------------------------------


def optimize_family(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
) -> FamilyResult:
    """Optimise une famille avec Optuna et évalue le meilleur pipeline sur le test."""
    logger.info("Optimisation de %s (n_trials=%d, cv=%d)", spec.name, n_trials, cv)

    study = run_study(spec, x_train, y_train, n_trials=n_trials, cv=cv)

    # Ré-entraîner le meilleur pipeline sur tout le train
    best_pipeline = build_pipeline(spec.build_estimator(study.best_params))
    best_pipeline.fit(x_train, y_train)

    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return FamilyResult(
        spec=spec,
        study=study,
        best_pipeline=best_pipeline,
        test_roc_auc=float(roc_auc_score(y_test, proba)),
        test_f1=float(f1_score(y_test, preds)),
        preds=preds,
    )


# ---------------------------------------------------------------------------
# Logging MLflow
# ---------------------------------------------------------------------------


def log_family_to_mlflow(
    result: FamilyResult,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
    register_as: str | None = None,
) -> None:
    """Logue une famille dans un run MLflow imbriqué."""
    with mlflow.start_run(run_name=result.spec.name, nested=True):
        mlflow.set_tag("model_family", result.spec.name)
        mlflow.set_tag("sampler", "TPE")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv", cv)

        # S6-6 : un run imbriqué par trial pour visualiser la convergence
        for trial in result.study.trials:
            with mlflow.start_run(
                run_name=f"{result.spec.name}_trial_{trial.number}",
                nested=True,
            ):
                mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("cv_roc_auc", trial.value)

        # Meilleurs hyperparamètres et métriques finales
        mlflow.log_params(result.study.best_params)
        mlflow.log_metrics(
            {
                "cv_roc_auc": result.study.best_value,
                "test_roc_auc": result.test_roc_auc,
                "test_f1": result.test_f1,
            }
        )

        # Matrice de confusion
        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=["dissatisfied", "satisfied"]).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.spec.name}")
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

        # Courbe de convergence Optuna
        active_run = mlflow.active_run()
        if active_run:
            logger.info("Run MLflow : %s", active_run.info.run_id)

        # SHAP summary (optionnel — nécessite shap installé)
        _log_shap_summary(result.best_pipeline, x_test, result.spec.name)

        # Enregistrement du modèle
        signature = infer_signature(x_test, result.best_pipeline.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            result.best_pipeline,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        # S6-7 bonus : documentation dans le Registry
        if register_as and model_info.registered_model_version:
            describe_registered_version(
                name=register_as,
                version=int(model_info.registered_model_version),
                result=result,
                n_trials=n_trials,
                cv=cv,
            )


def _log_optuna_convergence(result: FamilyResult, run_id: str) -> None:
    """Logue la courbe de convergence Optuna (meilleur score au fil des trials)."""
    values = [t.value for t in result.study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(values) + 1), values, alpha=0.4, label="Trial ROC AUC")
    ax.plot(range(1, len(best_so_far) + 1), best_so_far, label="Meilleur ROC AUC")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV ROC AUC")
    ax.set_title(f"Convergence Optuna — {result.spec.name}")
    ax.legend()
    fig.tight_layout()
    mlflow.log_figure(fig, "optuna_convergence.png")
    plt.close(fig)


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
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, x_transformed, show=False, plot_size=None)
        plt.gca().set_title(f"SHAP Summary — {model_name}")
        plt.tight_layout()
        mlflow.log_figure(fig, "shap_summary.png")
        plt.close(fig)
        logger.info("SHAP summary loggué pour %s", model_name)
    except Exception as exc:
        logger.warning("SHAP indisponible pour %s : %s", model_name, exc)


def describe_registered_version(
    name: str,
    version: int,
    result: FamilyResult,
    n_trials: int,
    cv: int,
) -> None:
    """Documente une version enregistrée dans le Model Registry."""
    client = mlflow.MlflowClient()

    description = (
        f"Modèle : {result.spec.name}\n"
        f"Dataset : Airline Passenger Satisfaction\n"
        f"Recherche : Optuna TPE (n_trials={n_trials}, cv={cv})\n"
        f"Meilleurs hyperparamètres : {result.study.best_params}\n"
        f"Métriques test → roc_auc={result.test_roc_auc:.3f} | f1={result.test_f1:.3f}"
    )
    client.update_model_version(name, str(version), description=description)

    for key, value in {
        "model_family": result.spec.name,
        "search_method": "Optuna-TPE",
        "n_trials": str(n_trials),
        "cv": str(cv),
        "cv_roc_auc": f"{result.study.best_value:.4f}",
        "test_roc_auc": f"{result.test_roc_auc:.4f}",
        "test_f1": f"{result.test_f1:.4f}",
        "dataset": "airline-passenger-satisfaction",
    }.items():
        client.set_model_version_tag(name, str(version), key, value)

    logger.info("Version %d du modèle '%s' documentée dans le registry", version, name)


# ---------------------------------------------------------------------------
# Orchestration principale
# ---------------------------------------------------------------------------


def optimize(
    n_trials: int = 30,
    cv: int = 5,
    use_mlflow: bool = True,
) -> list[FamilyResult]:
    """Optimise RF / XGBoost / LightGBM avec Optuna et sauvegarde le meilleur."""
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    logger.info("Train : %d lignes | Test : %d lignes", len(x_train), len(x_test))
    logger.info("Distribution cible (test) — satisfied : %.2f%%", 100 * y_test.mean())

    if use_mlflow:
        setup_experiment()

    results = [
        optimize_family(spec, x_train, y_train, x_test, y_test, n_trials=n_trials, cv=cv)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.test_roc_auc, reverse=True)
    best = results[0]
    logger.info(
        "Meilleure famille : %s (test_roc_auc=%.3f | f1=%.3f)",
        best.spec.name,
        best.test_roc_auc,
        best.test_f1,
    )

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare"):
            log_dataset(df, context="training", name="train")
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv", cv)
            mlflow.set_tag("best_model", best.spec.name)
            mlflow.log_metrics(
                {
                    "best_test_roc_auc": best.test_roc_auc,
                    "best_test_f1": best.test_f1,
                }
            )
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_family_to_mlflow(result, x_test, y_test, n_trials, cv, register_as=register_as)

        logger.info("Meilleur modèle enregistré dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_pipeline, MODEL_DIR / "model.joblib")
    logger.info("Modèle sauvegardé → %s", MODEL_DIR / "model.joblib")

    # Résumé comparatif
    logger.info("\n%-20s  %10s  %8s  %8s", "Modèle", "CV ROC AUC", "Test AUC", "Test F1")
    logger.info("-" * 55)
    for r in results:
        logger.info(
            "%-20s  %10.3f  %8.3f  %8.3f",
            r.spec.name,
            r.study.best_value,
            r.test_roc_auc,
            r.test_f1,
        )

    return results


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna TPE — Airline Passenger Satisfaction")
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Nombre d'essais Optuna par famille"
    )
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis de validation croisée")
    parser.add_argument(
        "--no-mlflow", dest="use_mlflow", action="store_false", help="Désactive le suivi MLflow"
    )
    args = parser.parse_args()
    optimize(n_trials=args.n_trials, cv=args.cv, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
