"""Construction du pré-processing."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RATING_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Construit le pipeline de pré-processing adapté au dataset
    Airline Passenger Satisfaction.

    Trois branches sont appliquées en parallèle via ColumnTransformer :

    1. **num** — Variables numériques continues (Age, Flight Distance, délais)
       - SimpleImputer (médiane) pour les rares valeurs manquantes
         (Arrival Delay in Minutes contient ~0.3 % de NaN)
       - StandardScaler pour normaliser les amplitudes très différentes
         (ex. Flight Distance : 50–4 983 miles vs Age : 7–85 ans)

    2. **cat** — Variables catégorielles nominales (Gender, Customer Type,
       Type of Travel, Class)
       - OneHotEncoder : pas d'ordre naturel entre les modalités
       - handle_unknown="ignore" pour robustesse en production

    3. **rating** — Scores de satisfaction sur échelle 0-5 (14 colonnes)
       - SimpleImputer (médiane) uniquement
       - Pas de scaling : l'échelle est déjà homogène et bornée,
         ce qui convient aux modèles à base d'arbres (XGBoost, LightGBM)

    Returns
    -------
    ColumnTransformer
        Pipeline de pré-processing non encore fitté, prêt pour
        `.fit_transform(X_train)` / `.transform(X_test)`.
    """
    # --- Branche 1 : numériques continues -----------------------------------
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # --- Branche 2 : catégorielles ------------------------------------------
    categorical_pipeline = Pipeline(steps=[
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ])

    # --- Branche 3 : scores de satisfaction (0-5) ---------------------------
    rating_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # --- Assemblage ---------------------------------------------------------
    return ColumnTransformer(
        transformers=[
            ("num",    numeric_pipeline,     NUMERIC_FEATURES),
            ("cat",    categorical_pipeline, CATEGORICAL_FEATURES),
            ("rating", rating_pipeline,      RATING_FEATURES),
        ],
        remainder="drop",  # toute colonne non listée est ignorée
    )