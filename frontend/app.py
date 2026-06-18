"""Frontend Streamlit — Airline Passenger Satisfaction.
Séance 14 bis - TP Streamlit
"""

from __future__ import annotations

import os
import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="✈️ SkyScore — Satisfaction Prédictive",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS global
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Nunito:wght@400;600;700;800;900&display=swap');

* { font-family: 'Nunito', sans-serif; }

[data-testid="stAppViewContainer"] { background: #FFFFFF; }
[data-testid="stHeader"]           { background: #FFFFFF; }
[data-testid="stSidebar"]          { background: #F7F9FC; border-right: 1px solid #E8ECF0; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    color: white;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "✈";
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 9rem;
    opacity: 0.06;
}
.hero-title {
    font-family: 'Pacifico', cursive;
    font-size: 3.2rem;
    color: #E94560;
    margin: 0 0 0.3rem 0;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 1.15rem;
    color: #A8B8D8;
    margin: 0;
    font-weight: 600;
    max-width: 600px;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(233,69,96,0.15);
    border: 1px solid rgba(233,69,96,0.4);
    color: #E94560;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Stat cards accueil */
.stat-card {
    background: #F7F9FC;
    border: 1px solid #E8ECF0;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    text-align: center;
}
.stat-number {
    font-family: 'Pacifico', cursive;
    font-size: 2.4rem;
    color: #E94560;
    display: block;
    line-height: 1.1;
}
.stat-label {
    font-size: 0.82rem;
    color: #7A8A9A;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* Feature cards accueil */
.feature-card {
    background: #FFFFFF;
    border: 1.5px solid #E8ECF0;
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
    transition: border-color 0.2s;
}
.feature-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.feature-title { font-weight: 800; color: #1A1A2E; font-size: 1rem; margin-bottom: 0.3rem; }
.feature-desc  { color: #7A8A9A; font-size: 0.88rem; line-height: 1.5; }

/* Section label */
.section-label {
    font-weight: 800;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #7A8A9A;
    margin-bottom: 0.8rem;
    border-left: 3px solid #E94560;
    padding-left: 0.6rem;
}

/* Badges prédiction */
.badge-satisfied {
    background: linear-gradient(135deg, #06D6A0, #0BA07A);
    color: white;
    font-weight: 900;
    font-size: 1.4rem;
    border-radius: 14px;
    padding: 0.8rem 2rem;
    display: inline-block;
    box-shadow: 0 6px 20px rgba(6,214,160,0.30);
}
.badge-unsatisfied {
    background: linear-gradient(135deg, #E94560, #C73652);
    color: white;
    font-weight: 900;
    font-size: 1.4rem;
    border-radius: 14px;
    padding: 0.8rem 2rem;
    display: inline-block;
    box-shadow: 0 6px 20px rgba(233,69,96,0.30);
}

/* Pill tags */
.pill {
    display: inline-block;
    background: #F0F4FF;
    color: #3D5AFE;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0.15rem;
}

/* Metrics override */
[data-testid="metric-container"] {
    background: #F7F9FC;
    border: 1px solid #E8ECF0;
    border-radius: 14px;
    padding: 1rem 1.2rem;
}

/* Sidebar nav */
.nav-item {
    font-weight: 700;
    color: #1A1A2E;
    font-size: 0.95rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center;padding:1rem 0 1.5rem 0;">
        <div style="font-family:'Pacifico',cursive;font-size:1.6rem;color:#E94560;">Murielle SANOU SkyScore</div>
        <div style="font-size:0.75rem;color:#7A8A9A;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;">
            MLOps · Classification
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🔮 Prédiction", "📊 Performance", "📋 Historique"],
        label_visibility="collapsed",
    )

    st.divider()
    api_url = st.text_input("🔗 URL de l'API", value=API_URL)

    # Statut API
    try:
        r = httpx.get(f"{api_url}/health", timeout=2.0)
        if r.status_code == 200:
            st.success("API connectée ✅")
        else:
            st.error("API erreur ❌")
    except httpx.HTTPError:
        st.error("API non joignable ❌")

    st.divider()
    st.markdown(
        """
    <div style="font-size:0.75rem;color:#AAB;text-align:center;line-height:1.7;">
        Projet MLOps · ESGI 5A<br>
        Dataset : Airline Passenger<br>Satisfaction (Kaggle)
    </div>
    """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# PAGE ACCUEIL
# ===========================================================================
if page == "🏠 Accueil":
    st.markdown(
        """
    <div class="hero-banner">
        <div class="hero-badge">🛫 Projet MLOps · Classification Binaire</div>
        <div class="hero-title">SkyScore</div>
        <p class="hero-sub">
            Prédisez en temps réel la satisfaction d'un passager aérien
            grâce à un pipeline ML complet, de l'entraînement au déploiement cloud.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Stats clés
    s1, s2, s3, s4 = st.columns(4)
    for col, num, label in zip(
        [s1, s2, s3, s4],
        ["103K", "22", ">95%", "3"],
        ["Passagers analysés", "Variables prédictives", "ROC AUC", "Modèles comparés"],
    ):
        col.markdown(
            f"""
        <div class="stat-card">
            <span class="stat-number">{num}</span>
            <div class="stat-label">{label}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Problématique
    st.markdown('<p class="section-label">🎯 Problématique</p>', unsafe_allow_html=True)
    st.markdown("""
    > **Comment, à partir des données d'un vol et du profil d'un passager,
    > peut-on prédire automatiquement son niveau de satisfaction ?**

    Les compagnies aériennes collectent des milliers de retours à chaque vol mais peinent
    à les exploiter de manière prédictive. Ce projet construit un **pipeline MLOps complet** :
    entraînement, tracking, évaluation, API REST et interface utilisateur — le tout
    conteneurisé et déployé en continu via GitHub Actions.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features du projet
    st.markdown('<p class="section-label">🏗️ Architecture du projet</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    features = [
        (
            "🤖",
            "Modèles ML",
            "Logistic Regression (baseline), Random Forest, XGBoost, LightGBM — optimisés par GridSearchCV et Optuna TPE.",
        ),
        (
            "📈",
            "MLflow Tracking",
            "Chaque run est tracé : paramètres, métriques, artefacts, matrice de confusion et modèle enregistré dans le Registry.",
        ),
        (
            "🚀",
            "API FastAPI",
            "Endpoint /predict exposé via FastAPI avec validation Pydantic, journal des prédictions et métriques en temps réel.",
        ),
        (
            "🐳",
            "Docker & CI/CD",
            "Stack conteneurisée (MLflow + API + Frontend). Pipeline CI/CD GitHub Actions pour la qualité et la livraison.",
        ),
        (
            "📊",
            "Dashboard Streamlit",
            "Interface interactive pour tester le modèle, visualiser les métriques et consulter l'historique des prédictions.",
        ),
        (
            "🔒",
            "Porte qualité",
            "Évaluation automatisée sur test.csv avec seuils ROC AUC ≥ 0.85 et F1 ≥ 0.80 — le modèle est rejeté s'il ne les atteint pas.",
        ),
    ]
    cols = [f1, f2, f3, f1, f2, f3]
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(
            f"""
        <div class="feature-card" style="margin-bottom:1rem;">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset
    st.markdown('<p class="section-label">📊 Dataset</p>', unsafe_allow_html=True)
    dc1, dc2 = st.columns([2, 1])
    with dc1:
        st.markdown("""
        **Airline Passenger Satisfaction** — Kaggle (teejmahal20)

        Le dataset contient les évaluations de passagers sur leurs vols,
        avec des informations sur leur profil, le type de voyage, la classe
        et 14 scores de satisfaction sur les services à bord.
        """)
        st.markdown(
            """
        <span class="pill">👤 Profil passager</span>
        <span class="pill">✈️ Infos du vol</span>
        <span class="pill">📶 Wifi & services</span>
        <span class="pill">💺 Confort</span>
        <span class="pill">🍽️ Restauration</span>
        <span class="pill">🎬 Divertissement</span>
        <span class="pill">⏱️ Retards</span>
        """,
            unsafe_allow_html=True,
        )
    with dc2:
        st.markdown("""
        | Classe | Exemples |
        |--------|---------|
        | `1` — Satisfait | 57 % |
        | `0` — Insatisfait | 43 % |
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # CTA
    st.info(
        "👉 Naviguez vers **🔮 Prédiction** dans la barre latérale pour tester le modèle en temps réel !"
    )

# ===========================================================================
# PAGE PRÉDICTION
# ===========================================================================
elif page == "🔮 Prédiction":
    st.markdown(
        '<p style="font-family:Pacifico,cursive;font-size:2rem;color:#1A1A2E;">🔮 Prédire la satisfaction</p>',
        unsafe_allow_html=True,
    )
    st.caption("Renseignez les caractéristiques du passager et obtenez une prédiction instantanée.")

    with st.form("predict_form"):
        st.markdown('<p class="section-label">👤 Profil passager</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Genre", ["Male", "Female"])
            customer_type = st.selectbox("Type de client", ["Loyal Customer", "disloyal Customer"])
        with c2:
            age = st.number_input("Âge", min_value=1, max_value=120, value=35)
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

        st.markdown(
            '<p class="section-label">⭐ Évaluations des services (0 = N/A, 1–5)</p>',
            unsafe_allow_html=True,
        )
        ca, cb = st.columns(2)
        with ca:
            wifi = st.slider("📶 Wifi à bord", 0, 5, 3)
            time_conv = st.slider("🕐 Horaires pratiques", 0, 5, 3)
            online_book = st.slider("💻 Réservation en ligne", 0, 5, 3)
            gate_loc = st.slider("🚪 Emplacement de la porte", 0, 5, 3)
            food = st.slider("🍽️ Nourriture & boissons", 0, 5, 3)
            online_board = st.slider("📱 Embarquement en ligne", 0, 5, 4)
            seat_comfort = st.slider("💺 Confort du siège", 0, 5, 4)
        with cb:
            entertainment = st.slider("🎬 Divertissement", 0, 5, 3)
            onboard_svc = st.slider("🛎️ Service à bord", 0, 5, 4)
            leg_room = st.slider("🦵 Espace pour les jambes", 0, 5, 3)
            baggage = st.slider("🧳 Gestion des bagages", 0, 5, 4)
            checkin = st.slider("🏷️ Enregistrement", 0, 5, 4)
            inflight_svc = st.slider("✈️ Service en vol", 0, 5, 4)
            cleanliness = st.slider("🧹 Propreté", 0, 5, 4)

        submitted = st.form_submit_button("🚀 Lancer la prédiction", use_container_width=True)

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
        with st.spinner("Analyse en cours..."):
            try:
                response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as exc:
                st.error(f"❌ Appel à l'API impossible : {exc}")
                result = None

        if result:
            prediction = result["prediction"]
            probability = result["probability"]
            label = result["label"]

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown(
                    '<div style="text-align:center"><span class="badge-satisfied">🌟 Passager Satisfait</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="text-align:center"><span class="badge-unsatisfied">⚠️ Passager Insatisfait</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Prédiction", label)
            with m2:
                st.metric("Score de satisfaction", f"{probability:.1%}")
            with m3:
                st.metric("Classe", str(prediction))

            color = "#06D6A0" if prediction == 1 else "#E94560"
            st.markdown(
                f"""
            <div style="background:#F0F4F8;border-radius:10px;height:20px;margin:1rem 0;overflow:hidden;">
              <div style="width:{probability * 100:.1f}%;height:100%;
                          background:linear-gradient(90deg,{color},{color}99);border-radius:10px;">
              </div>
            </div>
            <div style="text-align:right;font-size:0.8rem;color:#7A8A9A;margin-top:-0.5rem;">
                Score : {probability:.1%}
            </div>
            """,
                unsafe_allow_html=True,
            )

            with st.expander("🔍 Réponse JSON brute"):
                st.json(result)

# ===========================================================================
# PAGE PERFORMANCE
# ===========================================================================
elif page == "📊 Performance":
    st.markdown(
        '<p style="font-family:Pacifico,cursive;font-size:2rem;color:#1A1A2E;">📊 Performance du modèle</p>',
        unsafe_allow_html=True,
    )
    st.caption("Métriques évaluées sur test.csv — données jamais vues pendant l'entraînement.")

    with st.spinner("Chargement des métriques..."):
        try:
            resp_info = httpx.get(f"{api_url}/model-info", timeout=5.0)
            resp_metrics = httpx.get(f"{api_url}/model-metrics", timeout=5.0)
            info_ok = resp_info.status_code == 200
            metrics_ok = resp_metrics.status_code == 200
        except httpx.HTTPError:
            info_ok = metrics_ok = False

    if info_ok:
        info = resp_info.json()
        ci, cv, cl = st.columns(3)
        with ci:
            st.metric("Version", info.get("version", "—"))
        with cv:
            st.metric("Modèle chargé", "✅ Oui" if info.get("loaded") else "❌ Non")
        with cl:
            st.metric("Fichier", info.get("model_path", "—"))
    else:
        st.warning("API non joignable.")

    st.divider()

    if metrics_ok:
        data = resp_metrics.json()

        st.markdown(
            '<p class="section-label">📈 Métriques clés (test.csv)</p>', unsafe_allow_html=True
        )
        mk = st.columns(4)
        for col, (label_m, key, icon, threshold) in zip(
            mk,
            [
                ("ROC AUC", "roc_auc", "🎯", 0.85),
                ("F1-score", "f1", "⚖️", 0.80),
                ("Accuracy", "accuracy", "✅", None),
                ("Recall", "recall", "📡", None),
            ],
        ):
            val = data.get("metrics", {}).get(key)
            delta = f"seuil ≥ {threshold}" if threshold else None
            col.metric(
                f"{icon} {label_m}",
                f"{val:.3f}" if val is not None else "—",
                delta=delta,
                delta_color="normal" if threshold else "off",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_cm, col_fi = st.columns(2)

        with col_cm:
            st.markdown(
                '<p class="section-label">🗺️ Matrice de confusion</p>', unsafe_allow_html=True
            )
            cm = data.get("confusion_matrix")
            if cm:
                df_cm = pd.DataFrame(
                    cm,
                    index=["Réel : Insatisfait", "Réel : Satisfait"],
                    columns=["Prédit : Insatisfait", "Prédit : Satisfait"],
                )
                st.dataframe(
                    df_cm.style.background_gradient(cmap="RdYlGn"), use_container_width=True
                )
            else:
                st.info("Matrice non disponible.")

        with col_fi:
            st.markdown(
                '<p class="section-label">🏆 Top 10 variables importantes</p>',
                unsafe_allow_html=True,
            )
            fi = data.get("feature_importances")
            if fi:
                df_fi = (
                    pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
                    .sort_values("Importance", ascending=False)
                    .head(10)
                )
                st.bar_chart(df_fi.set_index("Feature"), color="#E94560")
            else:
                st.info("Importances non disponibles (modèle linéaire).")
    else:
        st.info(
            "L'endpoint `/model-metrics` n'est pas disponible. Vérifiez que l'API est démarrée et que test.csv est accessible."
        )

# ===========================================================================
# PAGE HISTORIQUE
# ===========================================================================
elif page == "📋 Historique":
    st.markdown(
        '<p style="font-family:Pacifico,cursive;font-size:2rem;color:#1A1A2E;">📋 Historique des prédictions</p>',
        unsafe_allow_html=True,
    )
    st.caption("Journal de toutes les prédictions effectuées depuis le démarrage de l'API.")

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Rafraîchir", use_container_width=True):
            st.rerun()

    try:
        resp = httpx.get(f"{api_url}/predictions", timeout=5.0)
        if resp.status_code == 200:
            rows = resp.json()
            if rows:
                df_hist = pd.DataFrame(rows)
                if "prediction" in df_hist.columns:
                    df_hist["résultat"] = df_hist["prediction"].map(
                        {1: "✅ Satisfait", 0: "⚠️ Insatisfait"}
                    )

                # KPIs rapides
                total = len(df_hist)
                n_sat = (df_hist["prediction"] == 1).sum()
                k1, k2, k3 = st.columns(3)
                k1.metric("Total prédictions", total)
                k2.metric("Satisfaits", f"{n_sat} ({n_sat / total:.0%})")
                k3.metric("Insatisfaits", f"{total - n_sat} ({(total - n_sat) / total:.0%})")

                st.dataframe(df_hist, use_container_width=True, height=400)
                st.caption(
                    f"{total} prévision(s) enregistrée(s) — remis à zéro au redémarrage de l'API."
                )
            else:
                st.info("🛫 Aucune prévision pour l'instant — lancez votre première prédiction !")
        else:
            st.info("Endpoint /predictions non disponible.")
    except httpx.HTTPError:
        st.error("API non joignable — vérifiez que l'API est démarrée.")
