"""API d'inférence — Airline Passenger Satisfaction.

Endpoints :
    GET  /health          vérification de santé
    POST /predict         prédiction pour un passager
    GET  /predictions     journal de toutes les prédictions
    GET  /model-info      infos sur le modèle servi
    GET  /model-metrics   métriques d'évaluation + matrice + importances
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import MODEL_DIR, TEST_PATH, TARGET
from data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}
# Journal en mémoire des prédictions
predictions_log: list[dict] = []


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Modèle introuvable : {model_path}.")
    ml["model"] = joblib.load(model_path)
    logger.info("Modèle chargé depuis %s", model_path)

    # Pré-calcul des métriques sur test.csv au démarrage
    _compute_metrics()
    yield
    ml.clear()


def _compute_metrics() -> None:
    """Calcule et met en cache les métriques sur test.csv."""
    try:
        from sklearn.metrics import (
            accuracy_score, f1_score, recall_score, roc_auc_score,
            confusion_matrix,
        )
        df   = load_data(path=TEST_PATH)
        X    = df.drop(columns=[TARGET])
        y    = df[TARGET]
        model = ml["model"]

        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)

        ml["metrics"] = {
            "roc_auc":  round(float(roc_auc_score(y, proba)), 4),
            "f1":       round(float(f1_score(y, preds)), 4),
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "recall":   round(float(recall_score(y, preds)), 4),
        }
        cm = confusion_matrix(y, preds).tolist()
        ml["confusion_matrix"] = cm

        # Feature importances (arbres) ou coefficients (linéaire)
        clf = model.named_steps["clf"]
        preprocessor = model.named_steps["preprocessor"]
        try:
            feature_names = list(preprocessor.get_feature_names_out())
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                importances = np.abs(clf.coef_[0])
            else:
                importances = None

            if importances is not None:
                ml["feature_importances"] = dict(
                    zip(feature_names, [round(float(v), 5) for v in importances])
                )
        except Exception:
            ml["feature_importances"] = None

        logger.info("Métriques pré-calculées : %s", ml["metrics"])
    except Exception as exc:
        logger.warning("Pré-calcul des métriques échoué : %s", exc)
        ml["metrics"] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="✈️ Airline Passenger Satisfaction API",
    description="Prédit la satisfaction d'un passager à partir de ses caractéristiques de vol.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------

class Features(BaseModel):
    Gender: str             = Field(..., examples=["Male"])
    Customer_Type: str      = Field(..., alias="Customer Type", examples=["Loyal Customer"])
    Age: int                = Field(..., ge=1, le=120, examples=[35])
    Type_of_Travel: str     = Field(..., alias="Type of Travel", examples=["Business travel"])
    Class: str              = Field(..., examples=["Business"])
    Flight_Distance: int    = Field(..., alias="Flight Distance", ge=0, examples=[1200])
    Departure_Delay: int    = Field(..., alias="Departure Delay in Minutes", ge=0, examples=[0])
    Arrival_Delay: float    = Field(..., alias="Arrival Delay in Minutes", ge=0, examples=[0.0])
    Inflight_wifi: int              = Field(..., alias="Inflight wifi service", ge=0, le=5, examples=[4])
    Time_convenient: int            = Field(..., alias="Departure/Arrival time convenient", ge=0, le=5, examples=[3])
    Online_booking: int             = Field(..., alias="Ease of Online booking", ge=0, le=5, examples=[3])
    Gate_location: int              = Field(..., alias="Gate location", ge=0, le=5, examples=[3])
    Food_drink: int                 = Field(..., alias="Food and drink", ge=0, le=5, examples=[4])
    Online_boarding: int            = Field(..., alias="Online boarding", ge=0, le=5, examples=[4])
    Seat_comfort: int               = Field(..., alias="Seat comfort", ge=0, le=5, examples=[4])
    Inflight_entertainment: int     = Field(..., alias="Inflight entertainment", ge=0, le=5, examples=[4])
    Onboard_service: int            = Field(..., alias="On-board service", ge=0, le=5, examples=[4])
    Leg_room: int                   = Field(..., alias="Leg room service", ge=0, le=5, examples=[3])
    Baggage_handling: int           = Field(..., alias="Baggage handling", ge=0, le=5, examples=[4])
    Checkin_service: int            = Field(..., alias="Checkin service", ge=0, le=5, examples=[4])
    Inflight_service: int           = Field(..., alias="Inflight service", ge=0, le=5, examples=[4])
    Cleanliness: int                = Field(..., alias="Cleanliness", ge=0, le=5, examples=[4])

    model_config = {"populate_by_name": True}


class PredictionOut(BaseModel):
    prediction:  int   = Field(..., examples=[1])
    probability: float = Field(..., examples=[0.87])
    label:       str   = Field(..., examples=["satisfied"])


class PredictionRecord(PredictionOut):
    id:        int
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Monitoring"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut, tags=["Inférence"])
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    row   = pd.DataFrame([features.model_dump(by_alias=True)])
    proba = float(model.predict_proba(row)[0, 1])
    pred  = int(proba >= 0.5)
    label = "satisfied" if pred == 1 else "neutral or dissatisfied"

    # Journal en mémoire
    predictions_log.append({
        "id":          len(predictions_log) + 1,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "prediction":  pred,
        "probability": round(proba, 4),
        "label":       label,
        "age":         features.Age,
        "class":       features.Class,
        "travel_type": features.Type_of_Travel,
    })

    logger.info("Prédiction → %s (proba=%.4f)", label, proba)
    return PredictionOut(prediction=pred, probability=round(proba, 4), label=label)


@app.get("/predictions", response_model=list[PredictionRecord], tags=["Journal"])
def get_predictions() -> list[dict]:
    """Retourne toutes les prédictions effectuées depuis le démarrage de l'API."""
    return predictions_log


@app.get("/model-info", tags=["Monitoring"])
def model_info() -> dict:
    return {
        "version":    os.environ.get("MODEL_VERSION", "unknown"),
        "model_path": str(MODEL_DIR / "model.joblib"),
        "loaded":     "model" in ml,
    }


@app.get("/model-metrics", tags=["Monitoring"])
def model_metrics() -> dict:
    """Retourne les métriques pré-calculées sur test.csv au démarrage."""
    if ml.get("metrics") is None:
        raise HTTPException(status_code=503, detail="Métriques non disponibles")
    return {
        "metrics":              ml.get("metrics"),
        "confusion_matrix":     ml.get("confusion_matrix"),
        "feature_importances":  ml.get("feature_importances"),
    }