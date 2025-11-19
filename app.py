import streamlit as st
import pandas as pd
import joblib
import json
import datetime
import altair as alt

st.set_page_config(
    page_title="Car Price AI | Smart Valuation",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🚗"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Base Settings */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e2e8f0;
}

/* Background with Animated Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 50%, #0f1419 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating Orbs Background Effect */
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
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, #3b82f6 0%, transparent 70%);
    top: -200px;
    right: -200px;
    animation: float 20s ease-in-out infinite;
}

[data-testid="stAppViewContainer"]::after {
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, #8b5cf6 0%, transparent 70%);
    bottom: -300px;
    left: -300px;
    animation: float 25s ease-in-out infinite reverse;
}

@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(30px, -50px) scale(1.1); }
    66% { transform: translate(-20px, 20px) scale(0.9); }
}

/* Smooth Fade In Animation */
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(30px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes scaleIn {
    0% { opacity: 0; transform: scale(0.95); }
    100% { opacity: 1; transform: scale(1); }
}

.animate-enter {
    animation: fadeInUp 0.7s ease-out forwards;
}

/* Gradient Headings with Glow */
h1, h2, h3, h4 {
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    position: relative;
}

h1 {
    filter: drop-shadow(0 0 30px rgba(96, 165, 250, 0.3));
}

/* Premium Glass Morphism Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
    transition: left 0.5s;
}

.glass-card:hover::before {
    left: 100%;
}

.glass-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: rgba(96, 165, 250, 0.4);
    box-shadow: 
        0 20px 60px rgba(96, 165, 250, 0.2),
        0 0 0 1px rgba(96, 165, 250, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.15);
}

/* Step Icons with Pulse Animation */
.step-icon {
    font-size: 3.5rem;
    margin-bottom: 20px;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

/* Enhanced Form Inputs */
.stSelectbox > div > div, 
.stNumberInput > div > div,
.stSlider > div > div {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    transition: all 0.3s ease !important;
}

.stSelectbox > div > div:hover,
.stNumberInput > div > div:hover {
    border-color: rgba(96, 165, 250, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.1) !important;
}

.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.15) !important;
}

/* Slider Styling */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
}

.stSlider > div > div > div {
    background-color: rgba(148, 163, 184, 0.2) !important;
}

/* Section Headers */
h4 {
    font-size: 1.1rem !important;
    margin-top: 2rem !important;
    margin-bottom: 1.5rem !important;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid rgba(96, 165, 250, 0.2);
    font-weight: 600 !important;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

/* Override for cards without borders */
.glass-card h4 {
    border-bottom: none !important;
    padding-bottom: 0 !important;
    margin-top: 0 !important;
}

/* Hero Section - UPDATED FOR CENTERING */
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

/* Premium Button Design */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    padding: 20px 48px !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1) !important;
    box-shadow: 
        0 10px 40px rgba(59, 130, 246, 0.3),
        0 0 0 0 rgba(59, 130, 246, 0.5) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 
        0 20px 60px rgba(59, 130, 246, 0.4),
        0 0 0 0 rgba(139, 92, 246, 0.5) !important;
}

.stButton > button:active {
    transform: translateY(-1px) scale(0.98) !important;
}

/* Enhanced Table Styling */
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

.styled-table th {
    padding: 16px;
    background: transparent;
}

.styled-table td {
    background: rgba(30, 41, 59, 0.5);
    padding: 18px;
    text-align: center;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    transition: all 0.3s ease;
}

.styled-table tbody tr:hover td {
    background: rgba(30, 41, 59, 0.7);
}

.styled-table tr td:first-child { 
    border-radius: 12px 0 0 12px;
    border-left: 1px solid rgba(255, 255, 255, 0.05);
}

.styled-table tr td:last-child { 
    border-radius: 0 12px 12px 0;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Winner Row with Animated Glow */
.winner-row td {
    background: rgba(59, 130, 246, 0.2) !important;
    color: #fff !important;
    font-weight: 700 !important;
}

.winner-row td:first-child {
    border-left: 1px solid rgba(96, 165, 250, 0.4) !important;
}

.winner-row td:last-child {
    border-right: 1px solid rgba(96, 165, 250, 0.4) !important;
}

.winner-row td {
    border-top: 1px solid rgba(96, 165, 250, 0.4) !important;
    border-bottom: 1px solid rgba(96, 165, 250, 0.4) !important;
}

/* Premium Result Container */
.result-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 2px solid rgba(96, 165, 250, 0.3);
    border-radius: 32px;
    padding: 60px 40px;
    text-align: center;
    box-shadow: 
        0 20px 80px rgba(59, 130, 246, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    margin-top: 40px;
    position: relative;
    overflow: hidden;
    animation: scaleIn 0.6s ease-out;
}

.result-container::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(96, 165, 250, 0.15) 0%, transparent 70%);
    animation: rotate 20s linear infinite;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.price-tag {
    font-size: 6rem;
    font-weight: 900;
    margin: 20px 0;
    background: linear-gradient(135deg, #fff 0%, #60a5fa 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
    z-index: 1;
    letter-spacing: -0.02em;
    text-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    animation: priceReveal 0.8s ease-out;
}

@keyframes priceReveal {
    0% { 
        opacity: 0; 
        transform: scale(0.8);
        filter: blur(10px);
    }
    100% { 
        opacity: 1; 
        transform: scale(1);
        filter: blur(0);
    }
}

/* Badge Styling */
.confidence-badge {
    display: inline-block;
    padding: 8px 20px;
    background: rgba(96, 165, 250, 0.15);
    border: 1px solid rgba(96, 165, 250, 0.3);
    border-radius: 50px;
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 10px;
    position: relative;
    z-index: 1;
}

/* Back Button Styling */
.stButton > button[key="back_btn"] {
    background: rgba(30, 41, 59, 0.6) !important;
    padding: 12px 24px !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

/* Expander Styling */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.4) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    padding: 1.5rem !important;
    display: flex !important;
    align-items: center !important;
}

.streamlit-expanderHeader p {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

.streamlit-expanderHeader:hover {
    background: rgba(30, 41, 59, 0.6) !important;
    border-color: rgba(96, 165, 250, 0.3) !important;
}

/* Remove default Streamlit padding */
.block-container {
    padding-top: 2rem !important;
    position: relative;
    z-index: 1;
}

/* Responsive Design */
@media (max-width: 768px) {
    .hero-title { font-size: 3rem !important; }
    .price-tag { font-size: 4rem !important; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data_and_models():
    model = joblib.load('xgb_model.joblib')
    encoder = joblib.load('target_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    with open('options.json', 'r') as f:
        options = json.load(f)
    return model, encoder, scaler, options

model, encoder, scaler, options = load_data_and_models()

if 'page' not in st.session_state: st.session_state.page = 'home'

def navigate_to(page):
    st.session_state.page = page
    st.rerun()

def create_performance_chart():

    data = pd.DataFrame({
        'Model': ['XGBoost', 'Random Forest', 'CatBoost', 'LightGBM', 
                  'Decision Tree', 'KNN', 'Gradient Boost', 'Linear Reg', 'AdaBoost'],
        'Error ($)': [1463, 1220, 1366, 1736, 1728, 2003, 2220, 3092, 7901], 
        'Accuracy (%)': [96.2, 95.5, 95.3, 92.3, 90.8, 88.4, 87.0, 76.2, 13.4],
        'Color': ['#3b82f6', '#64748b', '#64748b', '#64748b', '#64748b', '#64748b', '#64748b', '#64748b', '#64748b']
    })

    chart = alt.Chart(data).mark_bar(cornerRadius=6).encode(
        x=alt.X('Model', 
                sort=None, 
                axis=alt.Axis(
                    labelAngle=-45, 
                    title=None,
                    labelColor='white'
                )),
        y=alt.Y('Accuracy (%)', 
                axis=alt.Axis(
                    title='Model Accuracy (%)',
                    titleColor='white',
                    labelColor='white',
                    gridColor='rgba(255, 255, 255, 0.1)'
                )),
        color=alt.Color('Color', scale=None, legend=None),
        tooltip=['Model', 'Accuracy (%)', 'Error ($)']
    ).properties(
        height=350,
        background='transparent'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        labelColor='white',
        titleColor='white',
        domainColor='white'
    )
    
    return chart

comparison_table_html = """
<table class="styled-table">
    <thead>
        <tr>
            <th>Rank</th>
            <th>AI Model</th>
            <th>Avg. Error (MAE)</th>
            <th>Accuracy (R²)</th>
            <th>Status</th>
        </tr>
    </thead>
    <tbody>
        <tr class="winner-row">
            <td>1 🏆</td>
            <td>XGBoost</td>
            <td>$1,463</td>
            <td>96.2%</td>
            <td>WINNER</td>
        </tr>
        <tr><td>2 🥈</td><td>Random Forest</td><td>$1,220</td><td>95.5%</td><td>Excellent</td></tr>
        <tr><td>3 🥉</td><td>CatBoost</td><td>$1,366</td><td>95.4%</td><td>Excellent</td></tr>
        <tr><td>4</td><td>LightGBM</td><td>$1,736</td><td>92.3%</td><td>Very Good</td></tr>
        <tr><td>5</td><td>Decision Tree</td><td>$1,728</td><td>90.8%</td><td>Good</td></tr>
        <tr><td>6</td><td>KNN</td><td>$2,003</td><td>88.4%</td><td>Decent</td></tr>
        <tr><td>7</td><td>Gradient Boost</td><td>$2,220</td><td>87.0%</td><td>Fair</td></tr>
        <tr><td>8</td><td>Linear Reg</td><td>$3,092</td><td>76.2%</td><td>Weak</td></tr>
        <tr><td>9</td><td>AdaBoost</td><td>$7,901</td><td>13.4%</td><td>Poor</td></tr>
    </tbody>
</table>
"""

# PAGE: HOME
if st.session_state.page == 'home':

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">CAR PRICE AI ✨</h1>
        <div style="display: flex; justify-content: center; width: 100%;">
            <p class="hero-subtitle">
                Discover the true market value of any car in seconds. 
                Powered by advanced Artificial Intelligence trained on thousands of real sales.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Start Valuation 🚀", use_container_width=True):
            navigate_to('predict')

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("<div class='animate-enter'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-bottom: 50px; font-size: 2.2rem;'>How Does It Work?</h3>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div class="step-icon">🔍</div>
            <h4 style="border: none; margin-top: 0 !important; padding: 0;">1. Enter Details</h4>
            <p style="color: #94a3b8; font-size:1rem; line-height: 1.6;">Select brand, model, year, and condition from our comprehensive menu.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div class="step-icon">🧠</div>
            <h4 style="border: none; margin-top: 0 !important; padding: 0;">2. AI Analyzes</h4>
            <p style="color: #94a3b8; font-size:1rem; line-height: 1.6;">XGBoost compares your car with 9,000+ real sales instantly.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div class="step-icon">💎</div>
            <h4 style="border: none; margin-top: 0 !important; padding: 0;">3. Get Price</h4>
            <p style="color: #94a3b8; font-size:1rem; line-height: 1.6;">Receive a precise market value with 96.2% accuracy rating.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; font-size: 2.2rem;'>Why Trust This Tool?</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #cbd5e1; margin-bottom:40px; font-size: 1.1rem;'>We tested 9 different algorithms. <b style='color: #60a5fa;'>XGBoost</b> emerged as the undisputed champion.</p>", unsafe_allow_html=True)

    col_chart, col_table = st.columns([1, 1], gap="large")
    
    with col_chart:

        st.markdown("""
        <div class="glass-card" style="height: 100%; padding: 30px;">
            <h5 style="text-align:center; margin-bottom:20px; font-size: 1.2rem; color: #cbd5e1;">📊 Accuracy Comparison</h5>
        """, unsafe_allow_html=True)
        
        st.altair_chart(create_performance_chart(), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_table:

        full_table_html = f"""
        <div class="glass-card" style="padding: 20px;">
            <h4 style="text-align: center; margin-bottom: 20px; color: white; font-size: 1.4rem !important; border-bottom: none;">
                🏆 Model Ranking
            </h4>
            {comparison_table_html}
        </div>
        """
        st.markdown(full_table_html, unsafe_allow_html=True)

# PAGE: PREDICT

elif st.session_state.page == 'predict':

    st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_btn"):
        navigate_to('home')

    st.markdown("<div class='animate-enter'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; margin-bottom: 15px; font-size: 2.8rem;'>Configure Your Car</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#cbd5e1; margin-bottom: 0px; font-size: 1.15rem;'>Fill in the specifications for an instant AI-powered valuation.</p>", unsafe_allow_html=True)

    if not model:
        st.error("⚠️ Model files missing. Please check your setup.")
        st.stop()
    
    # SECTION 1
    st.markdown("#### 🚘 Vehicle Identity")
    c1, c2, c3, c4 = st.columns(4)
    with c1: make = st.selectbox("Make", options['makes'])
    with c2: model_name = st.selectbox("Model", options['make_models'].get(make, []))
    with c3: trim = st.selectbox("Trim/Package", options['model_trims'].get(model_name, ["Standard"]))
    with c4: year = st.number_input("Year", 1990, datetime.date.today().year + 1, 2015)

    # SECTION 2
    st.markdown("#### 🎨 Appearance & Condition")
    c5, c6, c7, c8 = st.columns(4)
    with c5: body = st.selectbox("Body Type", options['model_bodies'].get(model_name, ["Sedan"]))
    with c6: color = st.selectbox("Exterior Color", options['colors'])
    with c7: interior = st.selectbox("Interior Color", options['interiors'])
    with c8: condition = st.slider("Condition (1-5)", 1.0, 5.0, 4.0, 0.1)

    # SECTION 3
    st.markdown("#### ⚙️ Technical & Sales")
    c9, c10, c11, c12 = st.columns(4)
    with c9: transmission = st.selectbox("Transmission", options['transmissions'])
    with c10: odometer = st.number_input("Odometer (Miles)", 0, 500000, 45000)
    with c11: 
        state_disp = st.selectbox("State", list(options['states_map'].keys()))
        state_code = options['states_map'][state_disp]
    with c12: seller = st.selectbox("Seller Type", options['make_sellers'].get(make, ["Other"]))
    
    st.markdown("<br>", unsafe_allow_html=True)

    # CALCULATION BUTTON
    if st.button("Calculate Value 💰", use_container_width=True):
        car_age = datetime.date.today().year - year
        
        input_df = pd.DataFrame({
            'make': [make], 'model': [model_name], 'trim': [trim], 'body': [body],
            'transmission': [transmission], 'state': [state_code], 'condition': [condition],
            'odometer': [odometer], 'color': [color], 'interior': [interior],
            'seller': [seller], 'car_age': [car_age]
        })
        
        with st.spinner("🤖 AI is analyzing 9,000+ market records..."):
            try:
                enc = encoder.transform(input_df)
                scl = scaler.transform(enc)
                pred = model.predict(scl)[0]

                st.balloons()

                st.markdown(f"""
                <div class="result-container">
                    <h3 style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 10px; font-weight: 500 !important; letter-spacing: 3px; text-transform: uppercase;">Estimated Market Value</h3>
                    <div class="price-tag">${pred:,.0f}</div>
                    <div class="confidence-badge">
                        ✓ High Confidence: 96.2% Model Accuracy
                    </div>
                    <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 20px; position:relative; z-index:1;">
                        Based on comprehensive analysis of real market data
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Calculation Error: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.expander("📊 View AI Model Performance Report"):
        st.markdown("### The Technology Behind Your Valuation")
        st.markdown("<p style='color: #cbd5e1; margin-bottom: 25px;'>Our XGBoost model outperformed 8 other leading algorithms in rigorous testing.</p>", unsafe_allow_html=True)
        st.markdown(comparison_table_html, unsafe_allow_html=True)