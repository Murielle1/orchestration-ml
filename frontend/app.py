"""Frontend Streamlit — Airline Passenger Satisfaction.
Séance 14 bis - TP Streamlit · Murielle SANOU
"""
from __future__ import annotations

import os
import httpx
import pandas as pd
import streamlit as st

API_URL     = os.environ.get("API_URL",     "http://141.145.217.77:8000")
MLFLOW_URL  = os.environ.get("MLFLOW_URL",  "http://141.145.217.77:5001")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://141.145.217.77:8080")
GITHUB_URL  = "https://github.com/Murielle1/orchestration-ml"
api_url     = API_URL

st.set_page_config(
    page_title="✈️ SkyScore",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Nunito:wght@400;600;700;800;900&display=swap');
* { font-family: 'Nunito', sans-serif; }

/* Reset Streamlit */
[data-testid="stAppViewContainer"] { background:#FFFFFF; }
[data-testid="stHeader"]           { background:#FFFFFF; }
[data-testid="collapsedControl"]   { display:none; }
section[data-testid="stSidebar"]   { display:none; }
.block-container { padding-top:1rem !important; }

/* ---- HERO BANNER ---- */
.hero-banner {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 60%, #0F3460 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.hero-banner::before {
    content: "✈";
    position: absolute; right:2.5rem; top:50%;
    transform:translateY(-50%);
    font-size:8rem; opacity:0.05;
}
.hero-meta {
    font-size:0.72rem; font-weight:800; letter-spacing:0.2em;
    text-transform:uppercase; color:#6A7A9A; margin-bottom:0.5rem;
}
.hero-title {
    font-family:'Pacifico',cursive;
    font-size:2.8rem; color:#E94560; margin:0.2rem 0;
    background: transparent !important;
    -webkit-text-fill-color: #E94560;
    display: block;
}
.hero-sub {
    font-size:1rem; color:#A8B8D8; font-weight:600;
    max-width:600px; margin:0 auto 1.5rem auto; line-height:1.6;
    text-align:center;
}
.hero-btns { display:flex; gap:0.8rem; justify-content:center; flex-wrap:wrap; }
.hero-btn {
    background:rgba(255,255,255,0.08);
    border:1.5px solid rgba(255,255,255,0.2);
    color:white; border-radius:20px;
    padding:0.4rem 1.2rem; font-size:0.82rem; font-weight:700;
    text-decoration:none; transition:all 0.2s;
}
.hero-btn:hover { background:#E94560; border-color:#E94560; }

/* ---- NAVBAR TABS ---- */
.stRadio > div {
    display:flex !important;
    gap:0.2rem !important;
    background:#F7F9FC;
    border:1px solid #E8ECF0;
    border-radius:12px;
    padding:0.3rem !important;
    margin-bottom:1.2rem;
}
.stRadio > div > label {
    flex:1 !important;
    text-align:center;
    padding:0.5rem 0.8rem !important;
    border-radius:9px !important;
    font-weight:700 !important;
    font-size:0.88rem !important;
    cursor:pointer !important;
    color:#7A8A9A !important;
    transition:all 0.18s !important;
    border:none !important;
    background:transparent !important;
}
.stRadio > div > label:hover {
    background:rgba(233,69,96,0.1) !important;
    color:#E94560 !important;
}
.stRadio > div > label[data-baseweb="radio"]:has(input:checked),
.stRadio [aria-checked="true"] {
    background:#E94560 !important;
    color:white !important;
}
/* hide radio bullet */
.stRadio > div > label > div:first-child { display:none !important; }

/* ---- SIDEBAR PANEL ---- */
.side-panel {
    background:#F7F9FC;
    border:1.5px solid #E8ECF0;
    border-radius:16px;
    padding:1.2rem;
    margin-bottom:1rem;
}
.side-title {
    font-weight:800; font-size:0.75rem;
    letter-spacing:0.14em; text-transform:uppercase;
    color:#7A8A9A; margin-bottom:0.8rem;
    display:flex; align-items:center; gap:0.4rem;
}
.side-status {
    padding:0.7rem 1rem;
    border-radius:10px;
    font-weight:700; font-size:0.88rem;
    margin-bottom:0.5rem;
    display:flex; align-items:center; gap:0.5rem;
}
.status-ok  { background:#E8FAF3; color:#0BA07A; border:1px solid #A8E6CF; }
.status-err { background:#FEE8EC; color:#E94560; border:1px solid #F9B8C4; }

.ql-btn {
    display:flex; align-items:center; gap:0.6rem;
    width:100%; padding:0.65rem 1rem;
    border-radius:10px; margin-bottom:0.45rem;
    text-decoration:none; font-weight:700; font-size:0.88rem;
    color:white; transition:opacity 0.15s;
}
.ql-btn:hover { opacity:0.88; }
.ql-docs    { background:linear-gradient(135deg,#E94560,#C73652); }
.ql-mlflow  { background:linear-gradient(135deg,#7B5EA7,#5E3E8A); }
.ql-airflow { background:linear-gradient(135deg,#1B9AAA,#0D7A87); }
.ql-github  { background:#1A1A2E; }

/* ---- CARDS ---- */
.stat-card {
    background:#F7F9FC; border:1px solid #E8ECF0;
    border-radius:16px; padding:1.2rem 1.5rem; text-align:center;
}
.stat-number {
    font-family:'Pacifico',cursive;
    font-size:2.2rem; color:#E94560; display:block; line-height:1.1;
}
.stat-label {
    font-size:0.78rem; color:#7A8A9A; font-weight:700;
    text-transform:uppercase; letter-spacing:0.08em; margin-top:0.3rem;
}
.feature-card {
    background:#FFFFFF; border:1.5px solid #E8ECF0;
    border-radius:14px; padding:1.2rem; margin-bottom:0.8rem;
}
.feature-icon  { font-size:1.8rem; margin-bottom:0.4rem; }
.feature-title { font-weight:800; color:#1A1A2E; font-size:0.95rem; margin-bottom:0.2rem; }
.feature-desc  { color:#7A8A9A; font-size:0.83rem; line-height:1.5; }

/* ---- SECTION LABEL ---- */
.section-label {
    font-weight:800; font-size:0.75rem;
    letter-spacing:0.14em; text-transform:uppercase;
    color:#7A8A9A; margin-bottom:0.8rem;
    border-left:3px solid #E94560; padding-left:0.6rem;
}

/* ---- BADGES ---- */
.badge-ok  {
    background:linear-gradient(135deg,#06D6A0,#0BA07A);
    color:white; font-weight:900; font-size:1.3rem;
    border-radius:12px; padding:0.7rem 1.8rem;
    display:inline-block;
}
.badge-ko  {
    background:linear-gradient(135deg,#E94560,#C73652);
    color:white; font-weight:900; font-size:1.3rem;
    border-radius:12px; padding:0.7rem 1.8rem;
    display:inline-block;
}

/* ---- PILL ---- */
.pill {
    display:inline-block; background:#F0F4FF; color:#3D5AFE;
    border-radius:20px; padding:0.18rem 0.75rem;
    font-size:0.75rem; font-weight:700; margin:0.1rem;
}

/* ---- MONITORING CARDS ---- */
.monitor-card {
    background:#FFFFFF; border:1.5px solid #E8ECF0;
    border-radius:14px; padding:1.2rem 1.5rem;
    text-align:center;
}
.monitor-val  { font-size:2rem; font-weight:900; color:#1A1A2E; }
.monitor-label { font-size:0.78rem; color:#7A8A9A; font-weight:700; text-transform:uppercase; }

[data-testid="metric-container"] {
    background:#F7F9FC; border:1px solid #E8ECF0;
    border-radius:12px; padding:0.9rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper — API health check
# ---------------------------------------------------------------------------
def api_health() -> bool:
    try:
        r = httpx.get(f"{api_url}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# TOPBAR — nom + titre en pleine largeur
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="background:#1A1A2E;border-radius:12px;padding:0.6rem 2rem;
            display:flex;align-items:center;justify-content:space-between;
            margin-bottom:0.8rem;">
    <div style="display:flex;align-items:center;gap:0.8rem;">
        <span style="font-family:'Pacifico',cursive;font-size:1.3rem;color:#E94560;">✈️ SkyScore</span>
        <span style="font-size:0.7rem;color:#6A7A9A;font-weight:700;
                     letter-spacing:0.12em;text-transform:uppercase;">
            MLOps · ESGI 5A
        </span>
    </div>
    <div style="font-size:0.82rem;color:#A8B8D8;font-weight:700;">
        ✍️ <strong style="color:white;">Murielle SANOU</strong> · 2025-2026
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# NAVBAR — pleine largeur, toujours visible
# ---------------------------------------------------------------------------
page = st.radio(
    "nav",
    ["🏠 Accueil", "🔮 Prédiction", "📊 Monitoring", "📈 Performance", "📋 Historique"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LAYOUT : sidebar-like left panel + main content
# ---------------------------------------------------------------------------
left_col, main_col = st.columns([1, 4], gap="medium")

# ---- LEFT PANEL ----
with left_col:
    st.markdown('<div class="side-panel">', unsafe_allow_html=True)
    st.markdown('<div class="side-title">⚙️ Configuration</div>', unsafe_allow_html=True)

    st.markdown("**URL de l'API**")
    api_url_input = st.text_input("API URL", value="http://141.145.217.77:8000/docs", label_visibility="collapsed")
    api_url = api_url_input

    is_up = api_health()
    if is_up:
        st.markdown('<div class="side-status status-ok">✅ API en ligne</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="side-status status-err">❌ API hors ligne</div>', unsafe_allow_html=True)

    # MLflow status
    try:
        rm = httpx.get(f"{api_url}/model-info", timeout=2.0).json()
        model_name = rm.get("version", "unknown")
        st.markdown(f"MLflow : <code>{MLFLOW_URL}</code>", unsafe_allow_html=True)
        st.markdown(f"Modèle : <code>{model_name}</code>", unsafe_allow_html=True)
    except Exception:
        st.markdown(f"MLflow : <code>{MLFLOW_URL}</code>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Quick links
    st.markdown(f"""
    <div class="side-panel">
        <div class="side-title">🔗 Liens rapides</div>
        <a class="ql-btn ql-docs"    href="{api_url}/docs"  target="_blank">📖 API Docs</a>
        <a class="ql-btn ql-mlflow"  href="{MLFLOW_URL}"    target="_blank">📊 MLflow</a>
        <a class="ql-btn ql-airflow" href="{AIRFLOW_URL}"   target="_blank">🌬️ Airflow</a>
        <a class="ql-btn ql-github"  href="{GITHUB_URL}"    target="_blank">🐙 GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ---- MAIN CONTENT ----
with main_col:

    # =======================================================================
    # PAGE ACCUEIL
    # =======================================================================
    if page == "🏠 Accueil":
        st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-badge">🛫 Projet MLOps · Classification Binaire</div>
        <div class="hero-title" style="text-align:center;">SkyScore</div>
        <p class="hero-sub" style="text-align:center;margin:0 auto 1.5rem auto;">
            Pipeline MLOps complet — prédiction, tracking, gouvernance<br>
            et déploiement du modèle de satisfaction passager aérien.
        </p>
        <div class="hero-btns">
            <a class="hero-btn" href="{GITHUB_URL}" target="_blank">🐙 GitHub — Code source</a>
            <a class="hero-btn" href="{api_url}/docs" target="_blank">📖 API Docs</a>
            <a class="hero-btn" href="{MLFLOW_URL}" target="_blank">📊 MLflow</a>
            <a class="hero-btn" href="{AIRFLOW_URL}" target="_blank">🌬️ Airflow</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

        # Stats clés
        s1, s2, s3, s4 = st.columns(4)
        for col, num, label in zip(
            [s1, s2, s3, s4],
            ["103K", "22", ">95%", "3"],
            ["Passagers analysés", "Variables prédictives", "ROC AUC", "Modèles comparés"],
        ):
            col.markdown(f"""
            <div class="stat-card">
                <span class="stat-number">{num}</span>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

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
            ("🤖", "Modèles ML", "Logistic Regression (baseline), Random Forest, XGBoost, LightGBM — optimisés par GridSearchCV et Optuna TPE."),
            ("📈", "MLflow Tracking", "Chaque run est tracé : paramètres, métriques, artefacts, matrice de confusion et modèle enregistré dans le Registry."),
            ("🚀", "API FastAPI", "Endpoint /predict exposé via FastAPI avec validation Pydantic, journal des prédictions et métriques en temps réel."),
            ("🐳", "Docker & CI/CD", "Stack conteneurisée (MLflow + API + Frontend). Pipeline CI/CD GitHub Actions pour la qualité et la livraison."),
            ("📊", "Dashboard Streamlit", "Interface interactive pour tester le modèle, visualiser les métriques et consulter l'historique des prédictions."),
            ("🔒", "Porte qualité", "Évaluation automatisée sur test.csv avec seuils ROC AUC ≥ 0.85 et F1 ≥ 0.80 — le modèle est rejeté s'il ne les atteint pas."),
        ]
        cols = [f1, f2, f3, f1, f2, f3]
        for col, (icon, title, desc) in zip(cols, features):
            col.markdown(f"""
            <div class="feature-card" style="margin-bottom:1rem;">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

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
            st.markdown("""
            <span class="pill">👤 Profil passager</span>
            <span class="pill">✈️ Infos du vol</span>
            <span class="pill">📶 Wifi & services</span>
            <span class="pill">💺 Confort</span>
            <span class="pill">🍽️ Restauration</span>
            <span class="pill">🎬 Divertissement</span>
            <span class="pill">⏱️ Retards</span>
            """, unsafe_allow_html=True)
        with dc2:
            st.markdown("""
            | Classe | Part |
            |--------|------|
            | `1` — Satisfait | 57 % |
            | `0` — Insatisfait | 43 % |
            """)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👉 Naviguez vers **🔮 Prédiction** dans la barre latérale pour tester le modèle en temps réel !")

    # =======================================================================
    # PAGE PRÉDICTION
    # =======================================================================
    elif page == "🔮 Prédiction":
        st.markdown("### 🔮 Prédire la satisfaction")
        st.caption("Renseignez les caractéristiques du passager.")

        with st.form("predict_form"):
            st.markdown('<p class="section-label">👤 Profil passager</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                gender        = st.selectbox("Genre", ["Male", "Female"])
                customer_type = st.selectbox("Type de client", ["Loyal Customer", "disloyal Customer"])
            with c2:
                age         = st.number_input("Âge", 1, 120, 35)
                travel_type = st.selectbox("Motif", ["Business travel", "Personal Travel"])
            with c3:
                travel_class    = st.selectbox("Classe", ["Business", "Eco Plus", "Eco"])

            st.markdown('<p class="section-label">✈️ Vol</p>', unsafe_allow_html=True)
            c4, c5, c6 = st.columns(3)
            with c4: flight_distance = st.number_input("Distance (miles)", 0, value=1200)
            with c5: departure_delay = st.number_input("Retard départ (min)", 0, value=0)
            with c6: arrival_delay   = st.number_input("Retard arrivée (min)", 0, value=0)

            st.markdown('<p class="section-label">⭐ Services (0=N/A, 1–5)</p>', unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                wifi         = st.slider("📶 Wifi",                  0, 5, 3)
                time_conv    = st.slider("🕐 Horaires",               0, 5, 3)
                online_book  = st.slider("💻 Réservation en ligne",   0, 5, 3)
                gate_loc     = st.slider("🚪 Porte",                  0, 5, 3)
                food         = st.slider("🍽️ Nourriture",             0, 5, 3)
                online_board = st.slider("📱 Embarquement en ligne",  0, 5, 4)
                seat_comfort = st.slider("💺 Siège",                  0, 5, 4)
            with cb:
                entertainment = st.slider("🎬 Divertissement",        0, 5, 3)
                onboard_svc   = st.slider("🛎️ Service à bord",        0, 5, 4)
                leg_room      = st.slider("🦵 Jambes",                 0, 5, 3)
                baggage       = st.slider("🧳 Bagages",                0, 5, 4)
                checkin       = st.slider("🏷️ Enregistrement",         0, 5, 4)
                inflight_svc  = st.slider("✈️ Service en vol",         0, 5, 4)
                cleanliness   = st.slider("🧹 Propreté",               0, 5, 4)

            submitted = st.form_submit_button("🚀 Lancer la prédiction", use_container_width=True)

        if submitted:
            payload = {
                "Gender": gender, "Customer Type": customer_type, "Age": age,
                "Type of Travel": travel_type, "Class": travel_class,
                "Flight Distance": flight_distance,
                "Departure Delay in Minutes": departure_delay,
                "Arrival Delay in Minutes": float(arrival_delay),
                "Inflight wifi service": wifi,
                "Departure/Arrival time convenient": time_conv,
                "Ease of Online booking": online_book, "Gate location": gate_loc,
                "Food and drink": food, "Online boarding": online_board,
                "Seat comfort": seat_comfort, "Inflight entertainment": entertainment,
                "On-board service": onboard_svc, "Leg room service": leg_room,
                "Baggage handling": baggage, "Checkin service": checkin,
                "Inflight service": inflight_svc, "Cleanliness": cleanliness,
            }
            with st.spinner("Analyse en cours..."):
                try:
                    resp = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
                    resp.raise_for_status()
                    result = resp.json()
                except httpx.HTTPError as exc:
                    st.error(f"❌ Appel à l'API impossible : {exc}")
                    result = None

            if result:
                pred  = result["prediction"]
                prob  = result["probability"]
                label = result["label"]

                st.markdown("<br>", unsafe_allow_html=True)
                badge_cls = "badge-ok" if pred == 1 else "badge-ko"
                badge_txt = "🌟 Passager Satisfait" if pred == 1 else "⚠️ Passager Insatisfait"
                st.markdown(f'<div style="text-align:center"><span class="{badge_cls}">{badge_txt}</span></div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Prédiction", label)
                m2.metric("Score", f"{prob:.1%}")
                m3.metric("Classe", str(pred))

                color = "#06D6A0" if pred == 1 else "#E94560"
                st.markdown(f"""
                <div style="background:#F0F4F8;border-radius:10px;height:18px;margin:1rem 0;overflow:hidden;">
                  <div style="width:{prob*100:.1f}%;height:100%;background:linear-gradient(90deg,{color},{color}99);border-radius:10px;"></div>
                </div>
                <div style="text-align:right;font-size:0.78rem;color:#7A8A9A;">Score : {prob:.1%}</div>
                """, unsafe_allow_html=True)

                with st.expander("🔍 JSON brut"):
                    st.json(result)

    # =======================================================================
    # PAGE MONITORING
    # =======================================================================
    elif page == "📊 Monitoring":
        st.markdown("### 📊 Tableau de bord")

        if st.button("🔄 Rafraîchir"):
            st.rerun()

        # Statuts
        is_api_up = api_health()
        try:
            info = httpx.get(f"{api_url}/model-info", timeout=3.0).json()
            version = info.get("version", "unknown")
            loaded  = info.get("loaded", False)
        except Exception:
            info    = {}
            version = "—"
            loaded  = False

        try:
            preds_resp = httpx.get(f"{api_url}/predictions", timeout=3.0).json()
            n_preds    = len(preds_resp)
        except Exception:
            preds_resp = []
            n_preds    = 0

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f'<div class="monitor-card"><div class="monitor-val">{"OK" if is_api_up else "KO"}</div><div class="monitor-label">Statut API</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="monitor-card"><div class="monitor-val">{version}</div><div class="monitor-label">Version servie</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="monitor-card"><div class="monitor-val">{"✅" if loaded else "❌"}</div><div class="monitor-label">Modèle chargé</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="monitor-card"><div class="monitor-val">{n_preds}</div><div class="monitor-label">Prédictions (session)</div></div>', unsafe_allow_html=True)
        with k5:
            if preds_resp:
                n_sat = sum(1 for r in preds_resp if r.get("prediction") == 1)
                pct   = f"{n_sat/n_preds:.0%}"
            else:
                pct = "—"
            st.markdown(f'<div class="monitor-card"><div class="monitor-val">{pct}</div><div class="monitor-label">Taux satisfaction</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Modèle servi
        st.markdown("#### 💎 Modèle servi actuellement")
        if version and version != "—":
            st.success(f"Version active : **{version}**")
        else:
            st.info("Version active non trouvée dans le registry.")

        # Historique prédictions
        if preds_resp:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📈 Répartition des prédictions (session)")
            df_preds = pd.DataFrame(preds_resp)
            if "prediction" in df_preds.columns:
                counts = df_preds["prediction"].value_counts().rename({1: "Satisfait", 0: "Insatisfait"})
                st.bar_chart(counts, color="#E94560")

            st.markdown("#### 📋 Dernières prédictions")
            df_show = df_preds.copy()
            if "prediction" in df_show.columns:
                df_show["résultat"] = df_show["prediction"].map({1: "✅ Satisfait", 0: "⚠️ Insatisfait"})
            st.dataframe(df_show.tail(10), use_container_width=True)
        else:
            st.info("🛫 Aucune prédiction encore — lancez-en depuis l'onglet Prédiction !")

        # Liens externes
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"[📊 Voir les runs MLflow →]({MLFLOW_URL})  ·  [🌬️ DAGs Airflow →]({AIRFLOW_URL})")

    # =======================================================================
    # PAGE PERFORMANCE
    # =======================================================================
    elif page == "📈 Performance":
        st.markdown("### 📈 Performance du modèle")
        st.caption("Métriques calculées sur test.csv — données jamais vues pendant l'entraînement.")

        with st.spinner("Chargement..."):
            try:
                resp_m = httpx.get(f"{api_url}/model-metrics", timeout=5.0)
                metrics_ok = resp_m.status_code == 200
                data = resp_m.json() if metrics_ok else {}
            except Exception:
                metrics_ok = False
                data = {}

        if metrics_ok:
            mk = st.columns(4)
            for col, (lbl, key, icon, thr) in zip(mk, [
                ("ROC AUC",  "roc_auc",  "🎯", 0.85),
                ("F1-score", "f1",       "⚖️", 0.80),
                ("Accuracy", "accuracy", "✅", None),
                ("Recall",   "recall",   "📡", None),
            ]):
                val   = data.get("metrics", {}).get(key)
                delta = f"seuil ≥ {thr}" if thr else None
                col.metric(f"{icon} {lbl}", f"{val:.3f}" if val is not None else "—",
                           delta=delta, delta_color="normal" if thr else "off")

            st.markdown("<br>", unsafe_allow_html=True)
            col_cm, col_fi = st.columns(2)

            with col_cm:
                st.markdown('<p class="section-label">🗺️ Matrice de confusion</p>', unsafe_allow_html=True)
                cm = data.get("confusion_matrix")
                if cm:
                    df_cm = pd.DataFrame(cm,
                        index=["Réel : Insatisfait", "Réel : Satisfait"],
                        columns=["Prédit : Insatisfait", "Prédit : Satisfait"])
                    st.dataframe(df_cm.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
                else:
                    st.info("Matrice non disponible.")

            with col_fi:
                st.markdown('<p class="section-label">🏆 Top 10 variables importantes</p>', unsafe_allow_html=True)
                fi = data.get("feature_importances")
                if fi:
                    df_fi = (pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
                             .sort_values("Importance", ascending=False).head(10))
                    st.bar_chart(df_fi.set_index("Feature"), color="#E94560")
                else:
                    st.info("Importances non disponibles (modèle linéaire).")

            st.markdown(f"<br><a href='{MLFLOW_URL}' target='_blank' style='color:#E94560;font-weight:700;'>📊 Voir tous les runs dans MLflow →</a>", unsafe_allow_html=True)
        else:
            st.info("L'endpoint `/model-metrics` n'est pas disponible. Vérifiez que l'API est démarrée et que test.csv est accessible.")

    # =======================================================================
    # PAGE HISTORIQUE
    # =======================================================================
    elif page == "📋 Historique":
        st.markdown("### 📋 Historique des prédictions")
        st.caption("Journal de toutes les prédictions effectuées depuis le démarrage de l'API.")

        if st.button("🔄 Rafraîchir"):
            st.rerun()

        try:
            resp = httpx.get(f"{api_url}/predictions", timeout=5.0)
            rows = resp.json() if resp.status_code == 200 else []
        except Exception:
            rows = []

        if rows:
            df_hist = pd.DataFrame(rows)
            total = len(df_hist)
            n_sat = (df_hist["prediction"] == 1).sum() if "prediction" in df_hist.columns else 0

            k1, k2, k3 = st.columns(3)
            k1.metric("Total prédictions", total)
            k2.metric("Satisfaits", f"{n_sat} ({n_sat/total:.0%})")
            k3.metric("Insatisfaits", f"{total-n_sat} ({(total-n_sat)/total:.0%})")

            if "prediction" in df_hist.columns:
                df_hist["résultat"] = df_hist["prediction"].map({1: "✅ Satisfait", 0: "⚠️ Insatisfait"})
            st.dataframe(df_hist, use_container_width=True, height=420)
            st.caption(f"{total} prévision(s) — remis à zéro au redémarrage de l'API.")
        else:
            st.info("🛫 Aucune prévision pour l'instant — lancez votre première prédiction !")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem 0;border-top:1px solid #E8ECF0;margin-top:2rem;color:#7A8A9A;font-size:0.82rem;">
    <span style="font-family:'Pacifico',cursive;color:#E94560;">SkyScore</span>
    &nbsp;·&nbsp; <strong>Murielle SANOU</strong>
    &nbsp;·&nbsp; ESGI 5A · MLOps 2025/2026
</div>
""", unsafe_allow_html=True)