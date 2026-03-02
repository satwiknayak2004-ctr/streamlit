
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

st.set_page_config(
    page_title="FraudShield — AI Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def load_model():
    model  = pickle.load(open("fraud_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl",      "rb"))
    return model, scaler

model, scaler = load_model()

# -------------------------------------------------------
# ALL ANIMATIONS AND STYLING
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #060910 0%, #0d1520 50%, #060910 100%);
    background-size: 400% 400%;
    animation: gradientShift 8s ease infinite;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-8px); }
}

@keyframes glowPulse {
    0%, 100% { text-shadow: 0 0 10px rgba(0,212,255,0.5); }
    50%       { text-shadow: 0 0 30px rgba(0,212,255,1), 0 0 60px rgba(0,212,255,0.5); }
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(0,212,255,0.3); }
    50%       { box-shadow: 0 0 40px rgba(0,212,255,0.7); }
}

@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-30px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fraudAlert {
    0%  { background: rgba(255,59,59,0.1);  border-color: #ff3b3b; }
    50% { background: rgba(255,59,59,0.25); border-color: #ff6b6b; }
    100%{ background: rgba(255,59,59,0.1);  border-color: #ff3b3b; }
}

@keyframes legitimateGlow {
    0%  { background: rgba(0,255,136,0.08); border-color: #00ff88; }
    50% { background: rgba(0,255,136,0.18); border-color: #00ffaa; }
    100%{ background: rgba(0,255,136,0.08); border-color: #00ff88; }
}

.hero {
    text-align: center;
    padding: 40px 0 20px 0;
    animation: fadeSlideDown 0.8s ease both;
}

.hero-icon {
    font-size: 64px;
    animation: float 3s ease infinite;
    display: block;
    margin-bottom: 16px;
}

.hero-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff, #7f6bff, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    animation: glowPulse 3s ease infinite;
}

.hero-sub {
    color: #5a7a99;
    font-size: 15px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
    animation: fadeSlideDown 0.8s ease 0.3s both;
}

.hero-badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: #00d4ff;
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    padding: 4px 16px;
    border-radius: 100px;
    letter-spacing: 2px;
    margin-top: 12px;
    animation: pulse 2s ease infinite;
}

.stats-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}

.stat-box {
    flex: 1;
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    animation: fadeSlideUp 0.6s ease both;
}

.stat-val {
    font-size: 20px;
    font-weight: 700;
    color: #00d4ff;
    font-family: 'Space Mono', monospace;
}

.stat-label {
    font-size: 10px;
    color: #5a7a99;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    margin-top: 4px;
}

.section-label {
    font-size: 13px;
    font-family: 'Space Mono', monospace;
    color: #00d4ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 24px 0 8px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(0,212,255,0.15);
    animation: fadeSlideDown 0.5s ease both;
}

.result-fraud {
    border: 2px solid #ff3b3b;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    animation: fraudAlert 1.5s ease infinite;
    margin: 20px 0;
}

.result-legit {
    border: 2px solid #00ff88;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    animation: legitimateGlow 2s ease infinite;
    margin: 20px 0;
}

.result-title-fraud {
    font-size: 32px;
    font-weight: 700;
    color: #ff3b3b;
    font-family: 'Space Mono', monospace;
    margin-bottom: 8px;
}

.result-title-legit {
    font-size: 32px;
    font-weight: 700;
    color: #00ff88;
    font-family: 'Space Mono', monospace;
    margin-bottom: 8px;
}

.result-prob {
    font-size: 52px;
    font-weight: 700;
    margin: 8px 0;
    animation: fadeSlideUp 0.5s ease 0.2s both;
}

.result-prob-fraud { color: #ff6b6b; }
.result-prob-legit { color: #00ffaa; }

.result-sub {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #5a7a99;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.result-rec {
    margin-top: 12px;
    font-size: 13px;
    font-family: 'Space Mono', monospace;
}

.custom-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    height: 10px;
    margin: 12px 0;
    overflow: hidden;
}

.custom-bar-fill-fraud {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ff3b3b, #ff6b35);
    transition: width 1s ease;
    box-shadow: 0 0 12px rgba(255,59,59,0.6);
}

.custom-bar-fill-legit {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #00d4ff, #00ff88);
    transition: width 1s ease;
    box-shadow: 0 0 12px rgba(0,255,136,0.4);
}

.risk-pill {
    display: inline-block;
    background: rgba(255,59,59,0.1);
    border: 1px solid rgba(255,59,59,0.3);
    color: #ff8888;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    padding: 6px 14px;
    border-radius: 100px;
    margin: 4px;
    animation: fadeSlideUp 0.4s ease both;
}

.safe-pill {
    display: inline-block;
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.25);
    color: #00ff88;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    padding: 6px 14px;
    border-radius: 100px;
    margin: 4px;
    animation: fadeSlideUp 0.4s ease both;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7f6bff) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,212,255,0.5) !important;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------
st.markdown("""
<div class="hero">
    <span class="hero-icon">🛡️</span>
    <div class="hero-title">FraudShield</div>
    <div class="hero-sub">AI-Powered Transaction Security</div>
    <div class="hero-badge">XGBoost · Real-Time Detection · 97%+ Accuracy</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-box"><div class="stat-val">97.4%</div><div class="stat-label">ROC-AUC</div></div>
    <div class="stat-box"><div class="stat-val">0.30</div><div class="stat-label">Threshold</div></div>
    <div class="stat-box"><div class="stat-val">284K</div><div class="stat-label">Trained On</div></div>
    <div class="stat-box"><div class="stat-val">XGBoost</div><div class="stat-label">Model</div></div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# SECTION 1 — TRANSACTION DETAILS
# Same questions as your original app, styled with animations
# -------------------------------------------------------
st.markdown('<div class="section-label">💳 Transaction Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.01, max_value=50000.0,
        value=100.0, step=0.01,
        help="How much was spent in this transaction?"
    )
    hour = st.selectbox(
        "Time of Transaction",
        options=list(range(24)),
        format_func=lambda h: f"{h:02d}:00 {'AM' if h < 12 else 'PM'} {'(Midnight)' if h==0 else '(Noon)' if h==12 else ''}",
        index=12,
        help="What hour of the day did this transaction happen?"
    )

with col2:
    merchant_type = st.selectbox(
        "Merchant Type",
        ["Retail / Shopping", "Restaurant / Food", "Online Purchase",
         "ATM / Cash Withdrawal", "Travel / Hotels",
         "Electronics", "Fuel / Gas Station", "Other"],
        help="What kind of store or service was this transaction at?"
    )
    location_type = st.selectbox(
        "Transaction Location",
        ["Same city as usual", "Different city", "Different country", "Online"],
        help="Where did this transaction happen compared to your usual location?"
    )

# -------------------------------------------------------
# SECTION 2 — CARD BEHAVIOR
# Exact same questions as your original, just styled
# -------------------------------------------------------
st.markdown('<div class="section-label">🔍 Card Behavior</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    is_first_time = st.radio(
        "Have you transacted at this merchant before?",
        ["Yes, I have been here before", "No, this is my first time"],
        help="Fraud often happens at merchants you have never visited before."
    )
    transaction_speed = st.radio(
        "How quickly did this transaction follow the previous one?",
        ["Normal gap (hours apart)", "Quick succession (minutes apart)",
         "Very rapid (seconds apart)"],
        help="Multiple rapid transactions is a common fraud pattern."
    )

with col4:
    amount_feels = st.radio(
        "Does this amount feel normal for this type of purchase?",
        ["Yes, completely normal", "Slightly higher than usual",
         "Much higher than usual"],
        help="Unusually large amounts relative to your history are a fraud signal."
    )
    card_present = st.radio(
        "Was the physical card used or was it an online/phone transaction?",
        ["Physical card was used", "Online or phone — card not physically present"],
        help="Card-not-present transactions carry higher fraud risk."
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------------
check = st.button("🔍  CHECK THIS TRANSACTION", use_container_width=True)

# -------------------------------------------------------
# PREDICTION LOGIC — same as your original but with
# animated result cards instead of plain st.error/st.success
# -------------------------------------------------------
if check:

    with st.spinner("Scanning transaction through AI model..."):
        time.sleep(1.5)

    # Build V features from user-friendly inputs (same logic as before)
    v14 = 0.0
    if location_type == "Different country":                            v14 -= 3.5
    elif location_type == "Different city":                             v14 -= 1.5
    elif location_type == "Online":                                     v14 -= 1.0
    if is_first_time == "No, this is my first time":                    v14 -= 1.2
    if card_present == "Online or phone — card not physically present": v14 -= 0.8

    v10 = 0.0
    if transaction_speed == "Very rapid (seconds apart)":               v10 -= 3.0
    elif transaction_speed == "Quick succession (minutes apart)":       v10 -= 1.5
    if amount_feels == "Much higher than usual":                        v10 -= 2.0
    elif amount_feels == "Slightly higher than usual":                  v10 -= 0.8

    v12 = 0.0
    if merchant_type == "ATM / Cash Withdrawal":                        v12 -= 2.5
    elif merchant_type == "Electronics":                                v12 -= 1.5
    elif merchant_type == "Online Purchase":                            v12 -= 1.0
    if amount > 5000:                                                   v12 -= 1.5
    elif amount > 1000:                                                 v12 -= 0.5

    v17 = 0.0
    if 0 <= hour <= 4:                                                  v17 -= 2.0
    elif hour >= 22:                                                    v17 -= 1.0
    if transaction_speed == "Very rapid (seconds apart)":              v17 -= 1.5

    v = [0] * 28
    v[13] = v14
    v[9]  = v10
    v[11] = v12
    v[16] = v17

    amount_log    = np.log1p(amount)
    amount_zscore = (amount - 88.35) / 250.12

    row_data = v + [amount, amount_log, hour, amount_zscore,
                    v[0]*v[1], v[2]*v[3], v[9]*v[13]]

    row  = pd.DataFrame([row_data], columns=model.feature_names_in_)
    prob = model.predict_proba(row)[0][1]
    pct  = prob * 100

    # Animated result card — red pulsing for fraud, green glowing for legit
    if prob >= 0.3:
        st.markdown(f"""
        <div class="result-fraud">
            <div class="result-title-fraud">FRAUD DETECTED</div>
            <div class="result-prob result-prob-fraud">{pct:.1f}%</div>
            <div class="result-sub">Fraud Probability Score</div>
            <div class="result-rec" style="color:#ff8888;">
                Recommendation: Block transaction · Contact your bank immediately
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-legit">
            <div class="result-title-legit">LEGITIMATE</div>
            <div class="result-prob result-prob-legit">{pct:.1f}%</div>
            <div class="result-sub">Fraud Probability Score</div>
            <div class="result-rec" style="color:#00ffaa;">
                Recommendation: Transaction approved · No action needed
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Animated glowing progress bar
    bar_color = "fraud" if prob >= 0.3 else "legit"
    st.markdown(f"""
    <div class="custom-bar-wrap">
        <div class="custom-bar-fill-{bar_color}" style="width:{int(pct)}%"></div>
    </div>
    <div style="display:flex;justify-content:space-between;
                font-family:'Space Mono',monospace;font-size:10px;color:#5a7a99;">
        <span>0% Safe</span><span>30% Threshold</span><span>100% Definite Fraud</span>
    </div>
    """, unsafe_allow_html=True)

    # Risk factor analysis — same logic as original but shown as animated pills
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:12px;color:#5a7a99;
                letter-spacing:2px;text-transform:uppercase;margin:20px 0 10px 0;">
        Risk Factor Analysis
    </div>
    """, unsafe_allow_html=True)

    risk_factors = []
    safe_factors = []

    if location_type == "Different country":
        risk_factors.append("Foreign country transaction")
    elif location_type == "Same city as usual":
        safe_factors.append("Familiar location")

    if transaction_speed == "Very rapid (seconds apart)":
        risk_factors.append("Rapid successive transactions")
    elif transaction_speed == "Normal gap (hours apart)":
        safe_factors.append("Normal transaction timing")

    if amount_feels == "Much higher than usual":
        risk_factors.append("Unusually high amount")
    elif amount_feels == "Yes, completely normal":
        safe_factors.append("Normal spending amount")

    if merchant_type == "ATM / Cash Withdrawal":
        risk_factors.append("ATM cash withdrawal")
    elif merchant_type in ["Retail / Shopping", "Restaurant / Food"]:
        safe_factors.append("Common merchant type")

    if 0 <= hour <= 4:
        risk_factors.append("Late night transaction")
    elif 8 <= hour <= 20:
        safe_factors.append("Normal business hours")

    if card_present == "Online or phone — card not physically present":
        risk_factors.append("Card not physically present")
    else:
        safe_factors.append("Physical card present")

    if is_first_time == "No, this is my first time":
        risk_factors.append("First time at this merchant")
    else:
        safe_factors.append("Known merchant history")

    pills_html = ""
    for r in risk_factors:
        pills_html += f'<span class="risk-pill">⚠ {r}</span>'
    for s in safe_factors:
        pills_html += f'<span class="safe-pill">✓ {s}</span>'

    st.markdown(f'<div style="margin-top:8px">{pills_html}</div>',
                unsafe_allow_html=True)
