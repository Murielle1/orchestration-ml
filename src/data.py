"""Chargement et découpage des données."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    DROP_COLS,
    RANDOM_STATE,
    TARGET,
    TARGET_NEGATIVE,
    TARGET_POSITIVE,
    TEST_SIZE,
)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_data(path=DATA_PATH) -> pd.DataFrame:
    """
    Charge le CSV et applique le nettoyage minimal :
      - suppression des colonnes d'index inutiles (Unnamed: 0, id)
      - encodage binaire de la variable cible (satisfied → 1 / neutral or dissatisfied → 0)

    Parameters
    ----------
    path : Path | str
        Chemin vers le fichier CSV (train.csv par défaut).

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé avec la colonne cible encodée en int8.
    """
    df = pd.read_csv(path)

    # Suppression des colonnes d'identifiant sans valeur prédictive
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Encodage binaire de la cible
    df[TARGET] = _encode_target(df[TARGET])

    return df


def _encode_target(series: pd.Series) -> pd.Series:
    """
    Encode la variable cible textuelle en binaire.

    'satisfied'                → 1
    'neutral or dissatisfied'  → 0

    Parameters
    ----------
    series : pd.Series
        Colonne cible brute issue du CSV.

    Returns
    -------
    pd.Series
        Série encodée (int8).

    Raises
    ------
    ValueError
        Si des valeurs inattendues sont présentes.
    """
    mapping = {TARGET_POSITIVE: 1, TARGET_NEGATIVE: 0}
    valid   = set(mapping.keys())
    found   = set(series.dropna().unique())

    if unexpected := found - valid:
        raise ValueError(
            f"Valeurs inattendues dans '{TARGET}' : {unexpected}. "
            f"Valeurs attendues : {valid}"
        )

    return series.map(mapping).astype("int8")


# ---------------------------------------------------------------------------
# Découpage train / validation
# ---------------------------------------------------------------------------

def split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    Sépare le DataFrame en jeux d'entraînement et de validation.

    Le découpage est stratifié sur la cible pour conserver la distribution
    des classes dans chaque split.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame complet avec la colonne cible déjà encodée.
    test_size : float
        Proportion du jeu de validation (défaut : 0.2).

    Returns
    -------
    X_train, X_val, y_train, y_val : tuple[pd.DataFrame, ...]
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )