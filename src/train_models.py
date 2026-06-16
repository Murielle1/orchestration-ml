"""Entraînement et optimisation de plusieurs modèles de classification (AutoML + SHAP).

Séance 7 - TP AutoML & SHAP
    Ce module compare trois familles de modèles (Random Forest, XGBoost,
    LightGBM), chacune optimisée par recherche d'hyperparamètres en grille
    (GridSearchCV), et persiste la meilleure dans `models/model.joblib`.

Dataset : Airline Passenger Satisfaction
Cible   : satisfaction  →  1 (satisfied) / 0 (neutral or dissatisfied)

Lancement :
    python -m src.train_models
    python -m src.train_models --cv 3 --scoring roc_auc
    python -m src.train_models --no-mlflow   # désactive le suivi MLflow
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

from config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_NAME,
    RANDOM_STATE,
)
from data import load_data, split
from features import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Le ColumnTransformer renvoie un tableau numpy sans noms de colonnes lors du
# scoring interne de la validation croisée : on neutralise l'avertissement
# correspondant, sans incidence sur les prédictions.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
# LightGBM génère des warnings verbeux sur certaines versions — on les filtre
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ---------------------------------------------------------------------------
# Spécification des modèles
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Spécification d'un modèle à optimiser."""

    name: str
    estimator: ClassifierMixin
    param_grid: dict


def build_model_specs() -> list[ModelSpec]:
    """Construit la liste des trois modèles à comparer.

    Les grilles d'hyperparamètres sont préfixées par ``clf__`` car le
    classifieur est la dernière étape du pipeline sklearn.

    Returns
    -------
    list[ModelSpec]
        Random Forest, XGBoost et LightGBM avec leurs grilles respectives.
    """
    return [
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",   # compense le léger déséquilibre satisfied/dissatisfied
                n_jobs=-1,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth":    [None, 10, 20],
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
                "clf__n_estimators":  [100, 200],
                "clf__max_depth":     [3, 5],
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
                "clf__n_estimators":  [100, 200],
                "clf__num_leaves":    [31, 63],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Assemble le preprocessing et un classifieur dans un pipeline.

    Parameters
    ----------
    estimator : ClassifierMixin
        Classifieur placé en dernière étape (``clf``).

    Returns
    -------
    Pipeline
        Pipeline scikit-learn prêt à être optimisé.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


# ---------------------------------------------------------------------------
# Résultat d'optimisation
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Résultat d'optimisation d'un modèle."""

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
    """Optimise un modèle par GridSearchCV et l'évalue sur le test.

    Parameters
    ----------
    spec : ModelSpec
        Modèle et grille d'hyperparamètres.
    x_train, y_train : array-like
        Données d'entraînement.
    x_test, y_test : array-like
        Données de test pour l'évaluation finale.
    cv : int
        Nombre de plis de validation croisée.
    scoring : str
        Métrique optimisée par la recherche.

    Returns
    -------
    FitResult
        Meilleur estimateur et métriques associées.
    """
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
        spec.name, scoring, result.cv_score, result.f1, result.roc_auc,
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
    """Logue un résultat d'optimisation dans un run MLflow imbriqué.

    Parameters
    ----------
    result : FitResult
        Résultat à tracer (params, métriques, estimateur).
    x_test : pandas.DataFrame
        Jeu de test — utilisé pour inférer la signature et un exemple d'entrée.
    y_test : array-like
        Cibles du jeu de test.
    cv : int
        Nombre de plis (loggué comme paramètre).
    scoring : str
        Métrique optimisée (préfixe la métrique de CV loguée).
    register_as : str, optional
        Si fourni, enregistre le modèle dans le Model Registry.
    """
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)

        # Hyperparamètres + métriques
        mlflow.log_params(result.best_params)
        mlflow.log_metrics({
            f"cv_{scoring}": result.cv_score,
            "f1":            result.f1,
            "roc_auc":       result.roc_auc,
        })

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
        report_text = cast(str, classification_report(
            y_test, result.preds,
            target_names=["dissatisfied", "satisfied"],
        ))
        mlflow.log_text(report_text, "classification_report.txt")
        logger.info("\n%s", report_text)

        # SHAP summary plot
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

        # Documentation de la version dans le Registry (bonus S7-5)
        if register_as and model_info.registered_model_version:
            describe_registered_version(
                name=register_as,
                version=int(model_info.registered_model_version),
                result=result,
                cv=cv,
                scoring=scoring,
            )


def _log_shap_summary(pipeline: Pipeline, x_test, model_name: str) -> None:
    """Génère et logue le SHAP summary plot comme artefact MLflow.

    Fonctionne avec les modèles à base d'arbres (RF, XGBoost, LightGBM)
    via TreeExplainer. En cas d'erreur (ex. modèle non supporté), un
    avertissement est émis sans interrompre le run.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline entraîné (preprocessor + clf).
    x_test : pandas.DataFrame
        Données de test en format original (avant preprocessing).
    model_name : str
        Nom du modèle — utilisé dans le titre du graphe.
    """
    try:
        import shap

        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["clf"]

        x_test_transformed = preprocessor.transform(x_test)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x_test_transformed)

        # Pour la classification binaire : shap_values peut être une liste [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            x_test_transformed,
            show=False,
            plot_size=None,
        )
        ax = plt.gca()
        ax.set_title(f"SHAP Summary — {model_name}")
        plt.tight_layout()
        mlflow.log_figure(fig, "shap_summary.png")
        plt.close(fig)
        logger.info("SHAP summary loggué pour %s", model_name)

    except Exception as exc:
        logger.warning("SHAP indisponible pour %s : %s", model_name, exc)


def describe_registered_version(
    name: str,
    version: int,
    result: FitResult,
    cv: int,
    scoring: str,
) -> None:
    """Documente une version enregistrée dans le Model Registry.

    Ajoute une description (algorithme, hyperparamètres, métriques) et des
    tags (famille de modèle, méthode de recherche, scores) sur la version du
    modèle.

    Parameters
    ----------
    name : str
        Nom du modèle enregistré dans le registry.
    version : int
        Version enregistrée à documenter.
    result : FitResult
        Résultat d'optimisation associé à cette version.
    cv : int
        Nombre de plis de validation croisée.
    scoring : str
        Métrique optimisée par GridSearchCV.
    """
    client = mlflow.MlflowClient()

    description = (
        f"Modèle : {result.name}\n"
        f"Dataset : Airline Passenger Satisfaction\n"
        f"Recherche : GridSearchCV (cv={cv}, scoring={scoring})\n"
        f"Meilleurs hyperparamètres : {result.best_params}\n"
        f"Métriques test → f1={result.f1:.3f} | roc_auc={result.roc_auc:.3f}"
    )
    client.update_model_version(name, str(version), description=description)

    tags = {
        "model_family":  result.name,
        "search_method": "GridSearchCV",
        "cv":            str(cv),
        "scoring":       scoring,
        "f1":            f"{result.f1:.4f}",
        "roc_auc":       f"{result.roc_auc:.4f}",
        "dataset":       "airline-passenger-satisfaction",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, value)

    logger.info("Version %d du modèle '%s' documentée dans le registry", version, name)


# ---------------------------------------------------------------------------
# Entraînement de tous les modèles
# ---------------------------------------------------------------------------

def train_all(
    cv: int = 5,
    scoring: str = "roc_auc",
    use_mlflow: bool = True,
) -> list[FitResult]:
    """Entraîne et compare les trois modèles, sauvegarde le meilleur.

    Parameters
    ----------
    cv : int
        Nombre de plis de validation croisée.
    scoring : str
        Métrique optimisée par GridSearchCV.
    use_mlflow : bool
        Active le suivi MLflow.

    Returns
    -------
    list[FitResult]
        Résultats triés du meilleur au moins bon (ROC AUC décroissant).
    """
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    logger.info("Train : %d lignes | Test : %d lignes", len(x_train), len(x_test))
    logger.info("Distribution cible (test) — satisfied : %.2f%%", 100 * y_test.mean())

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info(
            "Suivi MLflow : %s (expérience : %s)",
            MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
        )

    results = [
        optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)

    best = results[0]
    logger.info(
        "Meilleur modèle : %s (roc_auc=%.3f | f1=%.3f)",
        best.name, best.roc_auc, best.f1,
    )

    if use_mlflow:
        with mlflow.start_run(run_name="compare-models"):
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.set_tag("best_model", best.name)
            mlflow.log_metrics({
                "best_roc_auc": best.roc_auc,
                "best_f1":      best.f1,
            })
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as=register_as)

        logger.info("Meilleur modèle enregistré dans le registry sous '%s'", MODEL_NAME)

    # Sauvegarde locale du meilleur modèle
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(best.best_estimator, model_path)
    logger.info("Modèle sauvegardé → %s", model_path)

    # Résumé comparatif
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
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis de validation croisée")
    parser.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        help="Métrique optimisée par GridSearchCV (ex: roc_auc, f1, accuracy)",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Désactive le suivi MLflow (utile sans serveur de tracking)",
    )
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()