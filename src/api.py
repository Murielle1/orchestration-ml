"""API d'inférence du modèle de classification (FastAPI).

Séance 12 - TP FastAPI
    /health est fourni et fonctionne.

Dataset : Airline Passenger Satisfaction
Cible   : satisfaction  →  1 (satisfied) / 0 (neutral or dissatisfied)

Lancement :
    uvicorn src.api:app --reload
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


# ---------------------------------------------------------------------------
# Lifespan : chargement du modèle au démarrage, libération à l'arrêt
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge le modèle une seule fois au démarrage de l'API."""
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"Modèle introuvable : {model_path}. "
            "Lancez d'abord `make train` ou `make train-models`."
        )
    ml["model"] = joblib.load(model_path)
    logger.info("Modèle chargé depuis %s", model_path)
    yield
    ml.clear()
    logger.info("Modèle déchargé.")


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Airline Passenger Satisfaction — API d'inférence",
    description=(
        "Prédit si un passager sera **satisfait** (1) ou "
        "**neutre / insatisfait** (0) à partir de ses caractéristiques de vol."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schéma d'entrée — S12-1
# ---------------------------------------------------------------------------


class Features(BaseModel):
    """Caractéristiques d'un passager et de son vol."""

    # --- Profil passager ----------------------------------------------------
    Gender: str = Field(
        ...,
        description="Genre du passager",
        examples=["Male", "Female"],
    )
    Customer_Type: str = Field(
        ...,
        alias="Customer Type",
        description="Type de client : 'Loyal Customer' ou 'disloyal Customer'",
        examples=["Loyal Customer"],
    )
    Age: int = Field(..., ge=1, le=120, description="Âge du passager")
    Type_of_Travel: str = Field(
        ...,
        alias="Type of Travel",
        description="Motif du voyage : 'Business travel' ou 'Personal Travel'",
        examples=["Business travel"],
    )
    Class: str = Field(
        ...,
        description="Classe de voyage : 'Business', 'Eco' ou 'Eco Plus'",
        examples=["Business"],
    )

    # --- Informations du vol ------------------------------------------------
    Flight_Distance: int = Field(
        ...,
        alias="Flight Distance",
        ge=0,
        description="Distance du vol en miles",
        examples=[1200],
    )
    Departure_Delay_in_Minutes: int = Field(
        ...,
        alias="Departure Delay in Minutes",
        ge=0,
        description="Retard au départ en minutes",
        examples=[0],
    )
    Arrival_Delay_in_Minutes: float = Field(
        ...,
        alias="Arrival Delay in Minutes",
        ge=0,
        description="Retard à l'arrivée en minutes",
        examples=[0.0],
    )

    # --- Scores de satisfaction (0 = N/A, 1-5) ------------------------------
    Inflight_wifi_service: int = Field(
        ...,
        alias="Inflight wifi service",
        ge=0,
        le=5,
        examples=[4],
    )
    Departure_Arrival_time_convenient: int = Field(
        ...,
        alias="Departure/Arrival time convenient",
        ge=0,
        le=5,
        examples=[3],
    )
    Ease_of_Online_booking: int = Field(
        ...,
        alias="Ease of Online booking",
        ge=0,
        le=5,
        examples=[3],
    )
    Gate_location: int = Field(
        ...,
        alias="Gate location",
        ge=0,
        le=5,
        examples=[3],
    )
    Food_and_drink: int = Field(
        ...,
        alias="Food and drink",
        ge=0,
        le=5,
        examples=[4],
    )
    Online_boarding: int = Field(
        ...,
        alias="Online boarding",
        ge=0,
        le=5,
        examples=[4],
    )
    Seat_comfort: int = Field(
        ...,
        alias="Seat comfort",
        ge=0,
        le=5,
        examples=[4],
    )
    Inflight_entertainment: int = Field(
        ...,
        alias="Inflight entertainment",
        ge=0,
        le=5,
        examples=[4],
    )
    On_board_service: int = Field(
        ...,
        alias="On-board service",
        ge=0,
        le=5,
        examples=[4],
    )
    Leg_room_service: int = Field(
        ...,
        alias="Leg room service",
        ge=0,
        le=5,
        examples=[3],
    )
    Baggage_handling: int = Field(
        ...,
        alias="Baggage handling",
        ge=0,
        le=5,
        examples=[4],
    )
    Checkin_service: int = Field(
        ...,
        alias="Checkin service",
        ge=0,
        le=5,
        examples=[4],
    )
    Inflight_service: int = Field(
        ...,
        alias="Inflight service",
        ge=0,
        le=5,
        examples=[4],
    )
    Cleanliness: int = Field(
        ...,
        alias="Cleanliness",
        ge=0,
        le=5,
        examples=[4],
    )

    model_config = {
        # Permet d'envoyer les noms originaux avec espaces via les alias
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "Gender": "Male",
                    "Customer Type": "Loyal Customer",
                    "Age": 35,
                    "Type of Travel": "Business travel",
                    "Class": "Business",
                    "Flight Distance": 1200,
                    "Departure Delay in Minutes": 0,
                    "Arrival Delay in Minutes": 0.0,
                    "Inflight wifi service": 4,
                    "Departure/Arrival time convenient": 3,
                    "Ease of Online booking": 3,
                    "Gate location": 3,
                    "Food and drink": 4,
                    "Online boarding": 4,
                    "Seat comfort": 4,
                    "Inflight entertainment": 4,
                    "On-board service": 4,
                    "Leg room service": 3,
                    "Baggage handling": 4,
                    "Checkin service": 4,
                    "Inflight service": 4,
                    "Cleanliness": 4,
                }
            ]
        },
    }


# ---------------------------------------------------------------------------
# Schéma de sortie — S12-2
# ---------------------------------------------------------------------------


class PredictionOut(BaseModel):
    """Résultat de la prédiction."""

    prediction: int = Field(
        ...,
        description="Classe prédite : 1 = satisfied | 0 = neutral or dissatisfied",
        examples=[1],
    )
    probability: float = Field(
        ...,
        description="Probabilité d'appartenir à la classe 1 (satisfied)",
        examples=[0.87],
    )
    label: str = Field(
        ...,
        description="Libellé lisible de la prédiction",
        examples=["satisfied"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Monitoring"])
def health() -> dict:
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut, tags=["Inférence"])
def predict(features: Features) -> PredictionOut:
    """Prédit la satisfaction d'un passager.

    Retourne la classe prédite (0 ou 1), la probabilité associée
    et un libellé lisible.
    """
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Reconstruction du DataFrame avec les noms de colonnes originaux
    # (ceux attendus par le pipeline sklearn entraîné)
    row = pd.DataFrame([features.model_dump(by_alias=True)])

    proba = float(model.predict_proba(row)[0, 1])
    prediction = int(proba >= 0.5)
    label = "satisfied" if prediction == 1 else "neutral or dissatisfied"

    logger.info("Prédiction → %s (proba=%.4f)", label, proba)

    return PredictionOut(
        prediction=prediction,
        probability=round(proba, 4),
        label=label,
    )


@app.get("/model-info", tags=["Monitoring"])
def model_info() -> dict:
    """Retourne les informations sur le modèle actuellement servi."""
    return {
        "version": os.environ.get("MODEL_VERSION", "unknown"),
        "model_path": str(MODEL_DIR / "model.joblib"),
        "loaded": "model" in ml,
    }
