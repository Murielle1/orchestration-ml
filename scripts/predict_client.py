"""Client de test pour l'API FastAPI du modèle.

Séance 15 - TP Tests de l'API
    Envoie quelques payloads de test à une instance locale de l'API
    (`make api`) et affiche les réponses de `/health`, `/predict` et
    `/model-info`.

    Les payloads sont échantillonnés dans le jeu de données réel — les
    colonnes envoyées correspondent exactement aux features attendues par
    le modèle.

Lancement :
    python scripts/predict_client.py
    python scripts/predict_client.py --url http://127.0.0.1:8000
    python scripts/predict_client.py --n 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import httpx

from src.config import API_URL, TARGET
from src.data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_SAMPLES = 3


# ---------------------------------------------------------------------------
# Construction des payloads
# ---------------------------------------------------------------------------

def build_payloads(n: int = N_SAMPLES) -> list[dict]:
    """Construit n payloads de test à partir du jeu de données.

    On retire la colonne cible et on convertit chaque ligne en dict JSON natif
    — ce qui garantit que les types sont sérialisables (int64 → int, etc.).

    Parameters
    ----------
    n : int
        Nombre de passagers à échantillonner.

    Returns
    -------
    list[dict]
        Liste de payloads prêts à être envoyés à POST /predict.
    """
    features = load_data().drop(columns=[TARGET])
    sample   = features.sample(n=n, random_state=42)
    payloads = [json.loads(row.to_json()) for _, row in sample.iterrows()]
    logger.info("%d payload(s) construits depuis le jeu de données.", len(payloads))
    return payloads


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Client de test — Airline Passenger Satisfaction API"
    )
    parser.add_argument(
        "--url",
        default=API_URL,
        help="URL de base de l'API (défaut: %(default)s)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_SAMPLES,
        help=f"Nombre de passagers à tester (défaut: {N_SAMPLES})",
    )
    args   = parser.parse_args()
    payloads = build_payloads(n=args.n)

    with httpx.Client(base_url=args.url, timeout=10.0) as client:

        # S15-1 : vérification que l'API est bien démarrée
        try:
            health = client.get("/health")
            logger.info("GET /health → %s %s", health.status_code, health.json())
        except httpx.ConnectError:
            logger.error(
                "Impossible de joindre l'API sur %s. "
                "Lancez d'abord `make api` dans un autre terminal.",
                args.url,
            )
            sys.exit(1)

        # S15-2a : envoi de chaque payload à POST /predict
        for i, payload in enumerate(payloads):
            logger.info(
                "--- Passager #%d ---\n  Payload : %s",
                i + 1,
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
            response = client.post("/predict", json=payload)

            if response.status_code == 200:
                result = response.json()
                logger.info(
                    "POST /predict (#%d) → %s | prédiction=%s (%s) | proba=%.4f",
                    i + 1,
                    response.status_code,
                    result["prediction"],
                    result["label"],
                    result["probability"],
                )
            else:
                logger.warning(
                    "POST /predict (#%d) → %s %s",
                    i + 1,
                    response.status_code,
                    response.text,
                )

        # S15-2b : informations sur le modèle servi
        info = client.get("/model-info")
        logger.info("GET /model-info → %s %s", info.status_code, info.json())


if __name__ == "__main__":
    main()