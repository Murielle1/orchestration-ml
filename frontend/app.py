"""Frontend Streamlit — Airline Passenger Satisfaction.
Thème : vacances d'été, soleil, mer, évasion.
Séance 14 bis - TP Streamlit
"""
from __future__ import annotations

import os
import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Style — palette été / vacances
# Corail chaud, turquoise, sable, blanc crème, orange soleil
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="✈️ Satisfaction Passager",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Nunito:wght@400;600;700;800&display=swap');

/* Fond dégradé mer / ciel */
[data-testid="stAppViewContainer"] {
    background: #FFFFFF;
    min-height: 100vh;
}
[data-testid="stHeader"] { background: #FFFFFF; }

/* Titre principal */
.hero-title {
    font-family: 'Pacifico', cursive;
    font-size: 2.8rem;
    color: #2D6A8F;
    text-shadow: none;
    margin-bottom: 0;
}
.hero-sub {
    font-family: 'Nunito', sans-serif;
    font-size: 1.1rem;
    color: #555;
    margin-top: 0;
    font-weight: 600;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    box-shadow: 0 4px 24px rgba(0,93,115,0.10);
    margin-bottom: 1.2rem;
    border: 1.5px solid rgba(255,255,255,0.7);
}

/* Badge résultat */
.badge-satisfied {
    background: linear-gradient(135deg, #06D6A0, #1B9AAA);
    color: white;
    font-family: 'Pacifico', cursive;
    font-size: 1.6rem;
    border-radius: 16px;
    padding: 0.8rem 2rem;
    display: inline-block;
    box-shadow: 0 4px 16px rgba(6,214,160,0.35);
}
.badge-unsatisfied {
    background: linear-gradient(135deg, #FF6B35, #F4A261);
    color: white;
    font-family: 'Pacifico', cursive;
    font-size: 1.6rem;
    border-radius: 16px;
    padding: 0.8rem 2rem;
    display: inline-block;
    box-shadow: 0 4px 16px rgba(255,107,53,0.35);
}

/* Section headers */
.section-label {
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2D6A8F;
    margin-bottom: 0.5rem;
}

/* Onglets */
button[data-baseweb="tab"] {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

/* Métriques */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.7);
    border-radius: 14px;
    padding: 0.8rem 1rem;
}

/* Sliders — accent corail */
[data-testid="stSlider"] [role="slider"] {
    background-color: #FF6B35 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
col_hero, col_api = st.columns([3, 1])
with col_hero:
    st.markdown('<p class="hero-title">🌴 Bon Voyage Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Découvrez si votre vol sera une expérience 5 étoiles ☀️</p>', unsafe_allow_html=True)
with col_api:
    st.markdown("<br>", unsafe_allow_html=True)
    api_url = st.text_input("URL de l'API", value=API_URL, label_visibility="collapsed")

st.divider()

# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------
predict_tab, metrics_tab, history_tab = st.tabs([
    "🔮 Prédiction",
    "📊 Performance du modèle",
    "📋 Historique",
])

# ===========================================================================
# ONGLET 1 — PRÉDICTION
# ===========================================================================
with predict_tab:
    with st.form("predict_form"):
        # --- Profil passager ------------------------------------------------
        st.markdown('<p class="section-label">👤 Profil passager</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender        = st.selectbox("Genre", ["Male", "Female"])
            customer_type = st.selectbox("Type de client", ["Loyal Customer", "disloyal Customer"])
        with c2:
            age         = st.number_input("Âge", min_value=1, max_value=120, value=35)
            travel_type = st.selectbox("Motif du voyage", ["Business travel", "Personal Travel"])
        with c3:
            travel_class = st.selectbox("Classe", ["Business", "Eco Plus", "Eco"])

        st.markdown('<p class="section-label">✈️ Informations du vol</p>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            flight_distance = st.number_input("Distance (miles)", min_value=0, value=1200)
        with c5:
            departure_delay = st.number_input("Retard départ (min)", min_value=0, value=0)
        with c6:
            arrival_delay = st.number_input("Retard arrivée (min)", min_value=0, value=0)

        st.markdown('<p class="section-label">⭐ Évaluations des services (0 = N/A, 1–5)</p>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            wifi          = st.slider("📶 Wifi à bord",                     0, 5, 3)
            time_conv     = st.slider("🕐 Horaires pratiques",               0, 5, 3)
            online_book   = st.slider("💻 Réservation en ligne",             0, 5, 3)
            gate_loc      = st.slider("🚪 Emplacement de la porte",          0, 5, 3)
            food          = st.slider("🍽️ Nourriture & boissons",            0, 5, 3)
            online_board  = st.slider("📱 Embarquement en ligne",            0, 5, 4)
            seat_comfort  = st.slider("💺 Confort du siège",                 0, 5, 4)
        with cb:
            entertainment = st.slider("🎬 Divertissement",                   0, 5, 3)
            onboard_svc   = st.slider("🛎️ Service à bord",                   0, 5, 4)
            leg_room      = st.slider("🦵 Espace pour les jambes",           0, 5, 3)
            baggage       = st.slider("🧳 Gestion des bagages",              0, 5, 4)
            checkin       = st.slider("🏷️ Enregistrement",                   0, 5, 4)
            inflight_svc  = st.slider("✈️ Service en vol",                   0, 5, 4)
            cleanliness   = st.slider("🧹 Propreté",                         0, 5, 4)

        submitted = st.form_submit_button(
            "🌴 Prédire mon expérience de vol",
            use_container_width=True,
        )

    if submitted:
        payload = {
            "Gender": gender,
            "Customer Type": customer_type,
            "Age": age,
            "Type of Travel": travel_type,
            "Class": travel_class,
            "Flight Distance": flight_distance,
            "Departure Delay in Minutes": departure_delay,
            "Arrival Delay in Minutes": float(arrival_delay),
            "Inflight wifi service": wifi,
            "Departure/Arrival time convenient": time_conv,
            "Ease of Online booking": online_book,
            "Gate location": gate_loc,
            "Food and drink": food,
            "Online boarding": online_board,
            "Seat comfort": seat_comfort,
            "Inflight entertainment": entertainment,
            "On-board service": onboard_svc,
            "Leg room service": leg_room,
            "Baggage handling": baggage,
            "Checkin service": checkin,
            "Inflight service": inflight_svc,
            "Cleanliness": cleanliness,
        }

        with st.spinner("🌊 Analyse en cours..."):
            try:
                response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as exc:
                st.error(f"❌ Appel à l'API impossible : {exc}")
                result = None

        if result:
            prediction  = result["prediction"]
            probability = result["probability"]
            label       = result["label"]

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(
                    '<div style="text-align:center"><span class="badge-satisfied">🌟 Voyage réussi — Satisfait !</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="text-align:center"><span class="badge-unsatisfied">🌧️ Expérience mitigée — Insatisfait</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Prédiction", label)
            with m2:
                st.metric("Score de satisfaction", f"{probability:.1%}")
            with m3:
                st.metric("Classe prédite", str(prediction), help="1 = satisfied | 0 = dissatisfied")

            # Jauge colorée
            color = "#06D6A0" if prediction == 1 else "#FF6B35"
            st.markdown(f"""
            <div style="background:#e8f4f8;border-radius:12px;height:22px;margin:1rem 0;overflow:hidden;">
              <div style="width:{probability*100:.1f}%;height:100%;background:linear-gradient(90deg,{color},{color}aa);
                          border-radius:12px;transition:width 0.6s ease;display:flex;align-items:center;
                          justify-content:flex-end;padding-right:8px;">
                <span style="color:white;font-size:0.75rem;font-weight:700;">{probability:.1%}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Détail JSON"):
                st.json(result)

# ===========================================================================
# ONGLET 2 — MÉTRIQUES DU MODÈLE
# ===========================================================================
with metrics_tab:
    st.markdown('<p class="section-label">Performance du modèle en production</p>', unsafe_allow_html=True)

    with st.spinner("Chargement des métriques..."):
        try:
            resp_info    = httpx.get(f"{api_url}/model-info", timeout=5.0)
            resp_metrics = httpx.get(f"{api_url}/model-metrics", timeout=5.0)
            info_ok    = resp_info.status_code == 200
            metrics_ok = resp_metrics.status_code == 200
        except httpx.HTTPError:
            info_ok = metrics_ok = False

    # --- Infos du modèle ---
    if info_ok:
        info = resp_info.json()
        ci, cv, cl = st.columns(3)
        with ci:
            st.metric("Version", info.get("version", "—"))
        with cv:
            st.metric("Modèle chargé", "✅ Oui" if info.get("loaded") else "❌ Non")
        with cl:
            st.metric("Chemin", info.get("model_path", "—"))
    else:
        st.warning("API non joignable — vérifiez que l'API est démarrée.")

    st.divider()

    # --- Métriques + matrice + importances ----------------------------------
    if metrics_ok:
        data = resp_metrics.json()

        # Métriques clés
        st.markdown('<p class="section-label">📈 Métriques d\'évaluation (test.csv)</p>', unsafe_allow_html=True)
        mk = st.columns(4)
        metrics_map = {
            "ROC AUC":  ("roc_auc",  "🎯"),
            "F1-score": ("f1",       "⚖️"),
            "Accuracy": ("accuracy", "✅"),
            "Recall":   ("recall",   "📡"),
        }
        for col, (label_m, icon) in zip(mk, metrics_map.items()):
            val = data.get("metrics", {}).get(metrics_map[label_m][0])
            col.metric(f"{icon} {label_m}", f"{val:.3f}" if val else "—")

        col_cm, col_fi = st.columns(2)

        # Matrice de confusion
        with col_cm:
            st.markdown('<p class="section-label">🗺️ Matrice de confusion</p>', unsafe_allow_html=True)
            cm = data.get("confusion_matrix")
            if cm:
                df_cm = pd.DataFrame(
                    cm,
                    index=["Réel : Insatisfait", "Réel : Satisfait"],
                    columns=["Prédit : Insatisfait", "Prédit : Satisfait"],
                )
                st.dataframe(
                    df_cm.style.background_gradient(cmap="YlOrRd"),
                    use_container_width=True,
                )
            else:
                st.info("Matrice non disponible.")

        # Features importances
        with col_fi:
            st.markdown('<p class="section-label">🏆 Variables les plus importantes</p>', unsafe_allow_html=True)
            fi = data.get("feature_importances")
            if fi:
                df_fi = (
                    pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
                    .sort_values("Importance", ascending=False)
                    .head(10)
                )
                st.bar_chart(df_fi.set_index("Feature"), color="#FF6B35")
            else:
                st.info("Importances non disponibles (modèle linéaire).")
    else:
        st.info(
            "L'endpoint `/model-metrics` n'est pas encore disponible. "
            "Ajoutez-le à l'API pour afficher les métriques ici."
        )

# ===========================================================================
# ONGLET 3 — HISTORIQUE
# ===========================================================================
with history_tab:
    st.markdown('<p class="section-label">📋 Journal des prévisions</p>', unsafe_allow_html=True)

    if st.button("🔄 Rafraîchir"):
        st.rerun()

    try:
        resp = httpx.get(f"{api_url}/predictions", timeout=5.0)
        if resp.status_code == 200:
            rows = resp.json()
            if rows:
                df_hist = pd.DataFrame(rows)
                # Mise en forme
                if "prediction" in df_hist.columns:
                    df_hist["label"] = df_hist["prediction"].map(
                        {1: "✅ Satisfait", 0: "⚠️ Insatisfait"}
                    )
                st.dataframe(df_hist, use_container_width=True, height=400)
                st.caption(f"{len(rows)} prévision(s) enregistrée(s)")
            else:
                st.info("🌴 Aucune prévision pour l'instant — lancez votre première prédiction !")
        else:
            st.info("Endpoint /predictions non disponible.")
    except httpx.HTTPError:
        st.info("🌊 API non joignable — vérifiez que l'API est démarrée.")