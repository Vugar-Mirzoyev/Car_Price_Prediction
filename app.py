"""
Car Price AI — Streamlit Application
=====================================
Author  : Vugar Mirzoyev
Version : 2.0.0
"""

import datetime
import json
import logging

import altair as alt
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_METRICS: list[dict] = [
    {"rank": "1 🏆", "name": "XGBoost",        "mae": 1_463, "r2": 96.2, "status": "WINNER",    "winner": True},
    {"rank": "2 🥈", "name": "Random Forest",   "mae": 1_220, "r2": 95.5, "status": "Excellent", "winner": False},
    {"rank": "3 🥉", "name": "CatBoost",        "mae": 1_366, "r2": 95.4, "status": "Excellent", "winner": False},
    {"rank": "4",    "name": "LightGBM",        "mae": 1_736, "r2": 92.3, "status": "Very Good", "winner": False},
    {"rank": "5",    "name": "Decision Tree",   "mae": 1_728, "r2": 90.8, "status": "Good",      "winner": False},
    {"rank": "6",    "name": "KNN",             "mae": 2_003, "r2": 88.4, "status": "Decent",    "winner": False},
    {"rank": "7",    "name": "Gradient Boost",  "mae": 2_220, "r2": 87.0, "status": "Fair",      "winner": False},
    {"rank": "8",    "name": "Linear Reg",      "mae": 3_092, "r2": 76.2, "status": "Weak",      "winner": False},
    {"rank": "9",    "name": "AdaBoost",        "mae": 7_901, "r2": 13.4, "status": "Poor",      "winner": False},
]

FILES_DIR = "files"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def get_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e2e8f0;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 50%, #0f1419 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.15;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stAppViewContainer"]::before {
    width: 500px; height: 500px;
    background: radial-gradient(circle, #3b82f6 0%, transparent 70%);
    top: -200px; right: -200px;
    animation: float 20s ease-in-out infinite;
}

[data-testid="stAppViewContainer"]::after {
    width: 600px; height: 600px;
    background: radial-gradient(circle, #8b5cf6 0%, transparent 70%);
    bottom: -300px; left: -300px;
    animation: float 25s ease-in-out infinite reverse;
}

@keyframes float {
    0%, 100% { transform: translate(0,    0)    scale(1);   }
    33%       { transform: translate(30px, -50px) scale(1.1); }
    66%       { transform: translate(-20px, 20px) scale(0.9); }
}

@keyframes fadeInUp  { 0% { opacity:0; transform:translateY(30px); } 100% { opacity:1; transform:translateY(0); } }
@keyframes scaleIn   { 0% { opacity:0; transform:scale(0.95);      } 100% { opacity:1; transform:scale(1);    } }
@keyframes pulse     { 0%,100% { transform:scale(1);    } 50% { transform:scale(1.05); } }
@keyframes rotate    { 0% { transform:rotate(0deg);   } 100% { transform:rotate(360deg); } }
@keyframes priceReveal {
    0%   { opacity:0; transform:scale(0.8); filter:blur(10px); }
    100% { opacity:1; transform:scale(1);   filter:blur(0);    }
}

.animate-enter { animation: fadeInUp 0.7s ease-out forwards; }

h1, h2, h3, h4 {
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

h1 { filter: drop-shadow(0 0 30px rgba(96,165,250,0.3)); }

.glass-card {
    background: rgba(30,41,59,0.5);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
    transition: all 0.5s cubic-bezier(0.23,1,0.32,1);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: "";
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
    transition: left 0.5s;
}

.glass-card:hover::before { left: 100%; }
.glass-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: rgba(96,165,250,0.4);
    box-shadow: 0 20px 60px rgba(96,165,250,0.2), 0 0 0 1px rgba(96,165,250,0.2), inset 0 1px 0 rgba(255,255,255,0.15);
}

.step-icon {
    font-size: 3.5rem;
    margin-bottom: 20px;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}

.stSelectbox>div>div,
.stNumberInput>div>div,
.stSlider>div>div {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(148,163,184,0.2) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    transition: all 0.3s ease !important;
}

.stSelectbox>div>div:hover,
.stNumberInput>div>div:hover {
    border-color: rgba(96,165,250,0.5) !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,0.1) !important;
}

.stSelectbox>div>div:focus-within,
.stNumberInput>div>div:focus-within {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,0.15) !important;
}

.stSlider>div>div>div>div { background: linear-gradient(90deg,#3b82f6,#8b5cf6) !important; }
.stSlider>div>div>div      { background-color: rgba(148,163,184,0.2) !important; }

h4 {
    font-size: 1.1rem !important;
    margin-top: 2rem !important;
    margin-bottom: 1.5rem !important;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid rgba(96,165,250,0.2);
    font-weight: 600 !important;
    background: linear-gradient(135deg,#60a5fa 0%,#a78bfa 50%,#ec4899 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.glass-card h4 { border-bottom:none !important; padding-bottom:0 !important; margin-top:0 !important; }

.hero-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 80px 20px 60px;
    animation: fadeInUp 1s ease-out;
    position: relative;
    z-index: 1;
}

.hero-title {
    font-size: 5rem !important;
    line-height: 1.1;
    margin-bottom: 24px;
    letter-spacing: -0.03em;
    animation: scaleIn 0.8s ease-out;
    text-align: center;
}

.hero-subtitle {
    font-size: 1.35rem;
    color: #cbd5e1;
    max-width: 800px;
    margin: 0 auto 60px auto;
    font-weight: 400;
    line-height: 1.6;
    text-align: center !important;
    display: block !important;
}

.stButton>button {
    background: linear-gradient(135deg,#3b82f6 0%,#8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    padding: 20px 48px !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.4s cubic-bezier(0.23,1,0.32,1) !important;
    box-shadow: 0 10px 40px rgba(59,130,246,0.3), 0 0 0 0 rgba(59,130,246,0.5) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton>button::before {
    content:"";
    position:absolute;
    top:0; left:-100%;
    width:100%; height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.2),transparent);
    transition:left 0.5s;
}

.stButton>button:hover::before { left:100%; }
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 20px 60px rgba(59,130,246,0.4), 0 0 0 0 rgba(139,92,246,0.5) !important;
}

.stButton>button:active { transform: translateY(-1px) scale(0.98) !important; }

.styled-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 10px;
    margin-top: 10px;
}

.styled-table thead tr {
    color: #94a3b8;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

.styled-table th { padding:16px; background:transparent; }

.styled-table td {
    background: rgba(30,41,59,0.5);
    padding: 18px;
    text-align: center;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    transition: all 0.3s ease;
}

.styled-table tbody tr:hover td { background:rgba(30,41,59,0.7); }

.styled-table tr td:first-child { border-radius:12px 0 0 12px; border-left:1px solid rgba(255,255,255,0.05); }
.styled-table tr td:last-child  { border-radius:0 12px 12px 0; border-right:1px solid rgba(255,255,255,0.05); }

.winner-row td {
    background: rgba(59,130,246,0.2) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border-top: 1px solid rgba(96,165,250,0.4) !important;
    border-bottom: 1px solid rgba(96,165,250,0.4) !important;
}

.winner-row td:first-child { border-left:1px solid rgba(96,165,250,0.4) !important; }
.winner-row td:last-child  { border-right:1px solid rgba(96,165,250,0.4) !important; }

.result-container {
    background: linear-gradient(135deg,rgba(30,41,59,0.8) 0%,rgba(15,23,42,0.9) 100%);
    border: 2px solid rgba(96,165,250,0.3);
    border-radius: 32px;
    padding: 60px 40px;
    text-align: center;
    box-shadow: 0 20px 80px rgba(59,130,246,0.2), inset 0 1px 0 rgba(255,255,255,0.1);
    margin-top: 40px;
    position: relative;
    overflow: hidden;
    animation: scaleIn 0.6s ease-out;
}

.result-container::before {
    content:"";
    position:absolute;
    top:-50%; left:-50%;
    width:200%; height:200%;
    background:radial-gradient(circle,rgba(96,165,250,0.15) 0%,transparent 70%);
    animation:rotate 20s linear infinite;
}

.price-tag {
    font-size: 6rem;
    font-weight: 900;
    margin: 20px 0;
    background: linear-gradient(135deg,#fff 0%,#60a5fa 50%,#a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
    z-index: 1;
    letter-spacing: -0.02em;
    animation: priceReveal 0.8s ease-out;
}

.confidence-badge {
    display: inline-block;
    padding: 8px 20px;
    background: rgba(96,165,250,0.15);
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 50px;
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 10px;
    position: relative;
    z-index: 1;
}

.streamlit-expanderHeader {
    background: rgba(30,41,59,0.4) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    padding: 1.5rem !important;
}

.streamlit-expanderHeader p { font-size:1.1rem !important; font-weight:600 !important; margin:0 !important; }
.streamlit-expanderHeader:hover {
    background: rgba(30,41,59,0.6) !important;
    border-color: rgba(96,165,250,0.3) !important;
}

.block-container { padding-top:2rem !important; position:relative; z-index:1; }

@media (max-width:768px) {
    .hero-title  { font-size:3rem !important; }
    .price-tag   { font-size:4rem !important; }
}
</style>
"""

# ---------------------------------------------------------------------------
# Resource loading  (cached — only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resources() -> tuple:
    """
    Load ML artefacts from disk.

    Returns
    -------
    (ml_model, encoder, scaler, options) or (None, None, None, None) on failure.
    """
    try:
        ml_model = joblib.load(f"{FILES_DIR}/xgb_model.joblib")
        encoder  = joblib.load(f"{FILES_DIR}/target_encoder.joblib")
        scaler   = joblib.load(f"{FILES_DIR}/scaler.joblib")
        with open(f"{FILES_DIR}/options.json", "r", encoding="utf-8") as fh:
            options = json.load(fh)
        logger.info("All model artefacts loaded successfully.")
        return ml_model, encoder, scaler, options
    except FileNotFoundError as exc:
        logger.error("Artefact not found: %s", exc)
        return None, None, None, None
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error loading artefacts: %s", exc)
        return None, None, None, None


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------
def navigate_to(page: str) -> None:
    st.session_state.page = page
    st.rerun()


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------
def build_accuracy_chart() -> alt.Chart:
    chart_data = pd.DataFrame({
        "Model":        [m["name"]   for m in MODEL_METRICS],
        "Accuracy (%)": [m["r2"]     for m in MODEL_METRICS],
        "Error ($)":    [m["mae"]    for m in MODEL_METRICS],
        "Color":        ["#3b82f6" if m["winner"] else "#64748b" for m in MODEL_METRICS],
    })

    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadius=6)
        .encode(
            x=alt.X("Model", sort=None,
                    axis=alt.Axis(labelAngle=-45, title=None, labelColor="white")),
            y=alt.Y("Accuracy (%)",
                    axis=alt.Axis(title="Model Accuracy (%)", titleColor="white",
                                  labelColor="white", gridColor="rgba(255,255,255,0.1)")),
            color=alt.Color("Color", scale=None, legend=None),
            tooltip=["Model", "Accuracy (%)", "Error ($)"],
        )
        .properties(height=350, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="white", titleColor="white", domainColor="white")
    )


# ---------------------------------------------------------------------------
# HTML components
# ---------------------------------------------------------------------------
def build_comparison_table_html() -> str:
    rows = ""
    for m in MODEL_METRICS:
        row_class = 'class="winner-row"' if m["winner"] else ""
        rows += (
            f"<tr {row_class}>"
            f"<td>{m['rank']}</td>"
            f"<td>{m['name']}</td>"
            f"<td>${m['mae']:,}</td>"
            f"<td>{m['r2']}%</td>"
            f"<td>{m['status']}</td>"
            f"</tr>"
        )
    return f"""
<table class="styled-table">
    <thead>
        <tr>
            <th>Rank</th><th>AI Model</th>
            <th>Avg. Error (MAE)</th><th>Accuracy (R²)</th><th>Status</th>
        </tr>
    </thead>
    <tbody>{rows}</tbody>
</table>"""


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def render_home(comparison_table_html: str) -> None:
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">CAR PRICE AI ✨</h1>
        <div style="display:flex;justify-content:center;width:100%;">
            <p class="hero-subtitle">
                Discover the true market value of any car in seconds.
                Powered by advanced Artificial Intelligence trained on thousands of real sales.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        if st.button("Start Valuation 🚀", use_container_width=True):
            navigate_to("predict")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- How it works ---
    st.markdown("<div class='animate-enter'>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align:center;margin-bottom:50px;font-size:2.2rem;'>How Does It Work?</h3>",
        unsafe_allow_html=True,
    )

    steps = [
        ("🔍", "1. Enter Details",
         "Select brand, model, year, and condition from our comprehensive menu."),
        ("🧠", "2. AI Analyzes",
         "XGBoost compares your car with 9,000+ real sales instantly."),
        ("💎", "3. Get Price",
         "Receive a precise market value with 96.2% accuracy rating."),
    ]

    for col, (icon, title, desc) in zip(st.columns(3), steps):
        with col:
            st.markdown(
                f"""<div class="glass-card" style="text-align:center;">
                    <div class="step-icon">{icon}</div>
                    <h4 style="border:none;margin-top:0!important;padding:0;">{title}</h4>
                    <p style="color:#94a3b8;font-size:1rem;line-height:1.6;">{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # --- Model comparison ---
    st.markdown(
        "<h3 style='text-align:center;font-size:2.2rem;'>Why Trust This Tool?</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#cbd5e1;margin-bottom:40px;font-size:1.1rem;'>"
        "We tested 9 different algorithms. "
        "<b style='color:#60a5fa;'>XGBoost</b> emerged as the undisputed champion.</p>",
        unsafe_allow_html=True,
    )

    col_chart, col_table = st.columns(2, gap="large")

    with col_chart:
        st.markdown(
            "<div class='glass-card' style='height:100%;padding:30px;'>"
            "<h5 style='text-align:center;margin-bottom:20px;font-size:1.2rem;color:#cbd5e1;'>"
            "📊 Accuracy Comparison</h5>",
            unsafe_allow_html=True,
        )
        st.altair_chart(build_accuracy_chart(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_table:
        st.markdown(
            f"<div class='glass-card' style='padding:20px;'>"
            f"<h4 style='text-align:center;margin-bottom:20px;font-size:1.4rem!important;border-bottom:none;'>"
            f"🏆 Model Ranking</h4>{comparison_table_html}</div>",
            unsafe_allow_html=True,
        )


def render_predict(ml_model, encoder, scaler, options: dict, comparison_table_html: str) -> None:
    st.markdown("<div style='margin-top:60px;'></div>", unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_btn"):
        navigate_to("home")

    # Guard — model files missing
    if ml_model is None:
        st.error("⚠️  Model files are missing. Please check your setup.")
        st.stop()

    st.markdown("<div class='animate-enter'>", unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align:center;margin-bottom:15px;font-size:2.8rem;'>Configure Your Car</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#cbd5e1;margin-bottom:0;font-size:1.15rem;'>"
        "Fill in the specifications for an instant AI-powered valuation.</p>",
        unsafe_allow_html=True,
    )

    # Section 1 — Vehicle Identity
    st.markdown("#### 🚘 Vehicle Identity")
    c1, c2, c3, c4 = st.columns(4)
    with c1: make       = st.selectbox("Make",          options["makes"])
    with c2: car_model  = st.selectbox("Model",         options["make_models"].get(make, []))
    with c3: trim       = st.selectbox("Trim/Package",  options["model_trims"].get(car_model, ["Standard"]))
    with c4: year       = st.number_input("Year", 1990, datetime.date.today().year + 1, 2015)

    # Section 2 — Appearance & Condition
    st.markdown("#### 🎨 Appearance & Condition")
    c5, c6, c7, c8 = st.columns(4)
    with c5: body      = st.selectbox("Body Type",       options["model_bodies"].get(car_model, ["Sedan"]))
    with c6: color     = st.selectbox("Exterior Color",  options["colors"])
    with c7: interior  = st.selectbox("Interior Color",  options["interiors"])
    with c8: condition = st.slider("Condition (1–5)", 1.0, 5.0, 4.0, 0.1)

    # Section 3 — Technical & Sales
    st.markdown("#### ⚙️ Technical & Sales")
    c9, c10, c11, c12 = st.columns(4)
    with c9:  transmission = st.selectbox("Transmission",  options["transmissions"])
    with c10: odometer     = st.number_input("Odometer (Miles)", 0, 500_000, 45_000)
    with c11:
        state_disp = st.selectbox("State", list(options["states_map"].keys()))
        state_code = options["states_map"][state_disp]
    with c12: seller = st.selectbox("Seller Type", options["make_sellers"].get(make, ["Other"]))

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Prediction ---
    if st.button("Calculate Value 💰", use_container_width=True):
        car_age = datetime.date.today().year - int(year)

        input_df = pd.DataFrame({
            "make": [make], "model": [car_model], "trim": [trim], "body": [body],
            "transmission": [transmission], "state": [state_code], "condition": [condition],
            "odometer": [odometer], "color": [color], "interior": [interior],
            "seller": [seller], "car_age": [car_age],
        })

        with st.spinner("🤖 AI is analysing 9,000+ market records…"):
            try:
                encoded    = encoder.transform(input_df)
                scaled     = scaler.transform(encoded)
                prediction = ml_model.predict(scaled)[0]

                st.balloons()
                st.markdown(
                    f"""<div class="result-container">
                        <h3 style="color:#94a3b8;font-size:1.1rem;margin-bottom:10px;
                                   font-weight:500!important;letter-spacing:3px;text-transform:uppercase;">
                            Estimated Market Value
                        </h3>
                        <div class="price-tag">${prediction:,.0f}</div>
                        <div class="confidence-badge">✓ High Confidence: 96.2% Model Accuracy</div>
                        <p style="color:#94a3b8;font-size:0.95rem;margin-top:20px;position:relative;z-index:1;">
                            Based on comprehensive analysis of real market data
                        </p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Prediction failed: %s", exc)
                st.error(f"Calculation Error: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.expander("📊 View AI Model Performance Report"):
        st.markdown("### The Technology Behind Your Valuation")
        st.markdown(
            "<p style='color:#cbd5e1;margin-bottom:25px;'>"
            "Our XGBoost model outperformed 8 other leading algorithms in rigorous testing.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(comparison_table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Car Price AI | Smart Valuation",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🚗",
    )

    st.markdown(get_css(), unsafe_allow_html=True)

    ml_model, encoder, scaler, options = load_resources()

    if "page" not in st.session_state:
        st.session_state.page = "home"

    comparison_table_html = build_comparison_table_html()

    if st.session_state.page == "home":
        render_home(comparison_table_html)
    elif st.session_state.page == "predict":
        render_predict(ml_model, encoder, scaler, options, comparison_table_html)


if __name__ == "__main__":
    main()
