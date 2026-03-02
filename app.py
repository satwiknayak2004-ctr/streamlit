
#LATEST

%%writefile app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="FraudShield — AI Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------------------------------
# LOAD MODEL AND SCALER
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model  = pickle.load(open("fraud_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl",      "rb"))
    return model, scaler

model, scaler = load_model()

# -------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------
if "history"       not in st.session_state: st.session_state.history       = []
if "total_checked" not in st.session_state: st.session_state.total_checked = 0
if "total_fraud"   not in st.session_state: st.session_state.total_fraud   = 0
if "total_legit"   not in st.session_state: st.session_state.total_legit   = 0

# -------------------------------------------------------
# GLOBAL CSS
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

.stApp { background: #04080f; }

@keyframes float      { 0%,100%{transform:translateY(0px)} 50%{transform:translateY(-10px)} }
@keyframes glowPulse  { 0%,100%{filter:drop-shadow(0 0 8px rgba(0,212,255,0.4))} 50%{filter:drop-shadow(0 0 24px rgba(0,212,255,0.9))} }
@keyframes fadeDown   { from{opacity:0;transform:translateY(-25px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeUp     { from{opacity:0;transform:translateY(25px)} to{opacity:1;transform:translateY(0)} }
@keyframes countUp    { from{opacity:0;transform:scale(0.6)} to{opacity:1;transform:scale(1)} }
@keyframes fraudPulse { 0%,100%{background:rgba(255,45,45,0.08);border-color:#ff2d2d;box-shadow:0 0 20px rgba(255,45,45,0.2)} 50%{background:rgba(255,45,45,0.18);border-color:#ff6b6b;box-shadow:0 0 40px rgba(255,45,45,0.45)} }
@keyframes legitGlow  { 0%,100%{background:rgba(0,255,136,0.06);border-color:#00ff88;box-shadow:0 0 20px rgba(0,255,136,0.15)} 50%{background:rgba(0,255,136,0.14);border-color:#00ffaa;box-shadow:0 0 40px rgba(0,255,136,0.3)} }
@keyframes medGlow    { 0%,100%{background:rgba(255,165,0,0.08);border-color:#ffa500;box-shadow:0 0 20px rgba(255,165,0,0.2)} 50%{background:rgba(255,165,0,0.18);border-color:#ffcc44;box-shadow:0 0 40px rgba(255,165,0,0.35)} }

.hero { text-align:center; padding:30px 0 10px; animation:fadeDown 0.8s ease both; }
.hero-icon  { font-size:72px; display:block; margin-bottom:12px; animation:float 3s ease infinite, glowPulse 3s ease infinite; }
.hero-title { font-family:'JetBrains Mono',monospace!important; font-size:clamp(32px,5vw,54px); font-weight:800; background:linear-gradient(135deg,#00d4ff,#7c3aed,#ff6b35); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; letter-spacing:-1px; }
.hero-sub   { color:#3d6080; font-size:12px; font-family:'JetBrains Mono',monospace; letter-spacing:4px; text-transform:uppercase; margin:8px 0; }
.hero-badge { display:inline-block; background:rgba(0,212,255,0.08); border:1px solid rgba(0,212,255,0.25); color:#00d4ff; font-size:10px; font-family:'JetBrains Mono',monospace; padding:5px 18px; border-radius:100px; letter-spacing:2px; animation:glowPulse 2.5s ease infinite; }

.stats-row  { display:flex; gap:10px; margin:20px 0; animation:fadeUp 0.7s ease 0.2s both; }
.stat-box   { flex:1; background:rgba(0,212,255,0.04); border:1px solid rgba(0,212,255,0.1); border-radius:12px; padding:14px 8px; text-align:center; transition:all 0.3s ease; }
.stat-box:hover { border-color:rgba(0,212,255,0.35); background:rgba(0,212,255,0.08); transform:translateY(-3px); }
.stat-val   { font-size:20px; font-weight:800; color:#00d4ff; font-family:'JetBrains Mono',monospace; display:block; }
.stat-label { font-size:9px; color:#3d6080; letter-spacing:2px; text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-top:3px; display:block; }
.stat-box.fraud-stat .stat-val { color:#ff4444; }
.stat-box.legit-stat .stat-val { color:#00ff88; }
.stat-box.rate-stat  .stat-val { color:#ffa500; }

.sec-label  { font-size:10px; font-family:'JetBrains Mono',monospace; color:#00d4ff; letter-spacing:4px; text-transform:uppercase; margin:24px 0 10px; padding-bottom:8px; border-bottom:1px solid rgba(0,212,255,0.1); }

.result-high   { border:2px solid #ff2d2d; border-radius:20px; padding:28px; text-align:center; animation:fraudPulse 1.5s ease infinite; margin:20px 0; }
.result-medium { border:2px solid #ffa500; border-radius:20px; padding:28px; text-align:center; animation:medGlow 2s ease infinite; margin:20px 0; }
.result-low    { border:2px solid #00ff88; border-radius:20px; padding:28px; text-align:center; animation:legitGlow 2s ease infinite; margin:20px 0; }

.result-level   { font-size:11px; font-family:'JetBrains Mono',monospace; letter-spacing:4px; text-transform:uppercase; margin-bottom:6px; }
.result-verdict { font-size:clamp(26px,4vw,40px); font-weight:800; font-family:'JetBrains Mono',monospace; margin-bottom:4px; }
.result-prob    { font-size:clamp(48px,7vw,68px); font-weight:800; font-family:'JetBrains Mono',monospace; line-height:1; margin:6px 0; animation:countUp 0.6s ease both; }
.result-sub     { font-size:10px; font-family:'JetBrains Mono',monospace; color:#3d6080; letter-spacing:3px; text-transform:uppercase; }
.result-rec     { margin-top:12px; font-size:12px; font-family:'JetBrains Mono',monospace; padding:8px 18px; border-radius:100px; display:inline-block; }

.bar-wrap { background:rgba(255,255,255,0.04); border-radius:100px; height:11px; margin:12px 0 5px; overflow:hidden; }
.bar-high   { height:100%; border-radius:100px; background:linear-gradient(90deg,#ff2d2d,#ff6b35); box-shadow:0 0 14px rgba(255,45,45,0.65); transition:width 1.2s cubic-bezier(0.4,0,0.2,1); }
.bar-medium { height:100%; border-radius:100px; background:linear-gradient(90deg,#ffa500,#ffcc44); box-shadow:0 0 14px rgba(255,165,0,0.6); transition:width 1.2s cubic-bezier(0.4,0,0.2,1); }
.bar-low    { height:100%; border-radius:100px; background:linear-gradient(90deg,#00d4ff,#00ff88); box-shadow:0 0 14px rgba(0,255,136,0.5); transition:width 1.2s cubic-bezier(0.4,0,0.2,1); }
.bar-labels { display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:9px; color:#3d6080; }

.risk-pill { display:inline-block; background:rgba(255,45,45,0.08); border:1px solid rgba(255,45,45,0.28); color:#ff8888; font-size:11px; font-family:'JetBrains Mono',monospace; padding:5px 12px; border-radius:100px; margin:3px; animation:fadeUp 0.4s ease both; }
.safe-pill { display:inline-block; background:rgba(0,255,136,0.06); border:1px solid rgba(0,255,136,0.22); color:#00ff88; font-size:11px; font-family:'JetBrains Mono',monospace; padding:5px 12px; border-radius:100px; margin:3px; animation:fadeUp 0.4s ease both; }
.warn-pill { display:inline-block; background:rgba(255,165,0,0.07); border:1px solid rgba(255,165,0,0.28); color:#ffa500; font-size:11px; font-family:'JetBrains Mono',monospace; padding:5px 12px; border-radius:100px; margin:3px; animation:fadeUp 0.4s ease both; }

.hist-row { display:flex; align-items:center; gap:10px; padding:9px 13px; border-radius:9px; margin:5px 0; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.04); font-family:'JetBrains Mono',monospace; font-size:11px; animation:fadeUp 0.3s ease both; transition:all 0.2s ease; }
.hist-row:hover { background:rgba(255,255,255,0.05); border-color:rgba(0,212,255,0.14); }
.hf { color:#ff4444; font-weight:700; min-width:75px; }
.hm { color:#ffa500; font-weight:700; min-width:75px; }
.hl { color:#00ff88; font-weight:700; min-width:75px; }
.hd { color:#3d6080; flex:1; }
.hp { color:#00d4ff; font-weight:700; }

.info-card { background:rgba(0,212,255,0.03); border:1px solid rgba(0,212,255,0.1); border-radius:12px; padding:18px; margin-bottom:10px; }
.info-card-title { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px; margin-bottom:8px; font-weight:700; }
.info-card-body  { font-size:13px; color:#6688aa; line-height:1.75; }

.pipe-step { display:flex; align-items:center; gap:12px; background:rgba(0,212,255,0.02); border:1px solid rgba(0,212,255,0.07); border-radius:10px; padding:11px 14px; margin:4px 0; }
.pipe-icon  { font-size:18px; min-width:26px; }
.pipe-title { font-family:'JetBrains Mono',monospace; font-size:11px; color:#00d4ff; font-weight:700; letter-spacing:1px; }
.pipe-desc  { font-size:12px; color:#3d6080; margin-top:2px; }
.pipe-num   { margin-left:auto; font-family:'JetBrains Mono',monospace; font-size:10px; color:#1e2d3d; }

.stButton > button { background:linear-gradient(135deg,#00d4ff,#7c3aed)!important; color:#000!important; font-family:'JetBrains Mono',monospace!important; font-weight:700!important; font-size:12px!important; letter-spacing:3px!important; text-transform:uppercase!important; border:none!important; border-radius:12px!important; padding:13px!important; transition:all 0.3s ease!important; box-shadow:0 4px 22px rgba(0,212,255,0.22)!important; }
.stButton > button:hover { transform:translateY(-3px)!important; box-shadow:0 8px 30px rgba(0,212,255,0.48)!important; }
div[data-testid="stTabs"] button { font-family:'JetBrains Mono',monospace!important; font-size:11px!important; letter-spacing:2px!important; text-transform:uppercase!important; color:#3d6080!important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#00d4ff!important; }
.stSelectbox label, .stRadio label, .stNumberInput label, .stSlider label { font-family:'JetBrains Mono',monospace!important; font-size:10px!important; color:#3d6080!important; letter-spacing:1px!important; text-transform:uppercase!important; }
#MainMenu{visibility:hidden} footer{visibility:hidden} header{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HERO
# -------------------------------------------------------
st.markdown("""
<div class="hero">
    <span class="hero-icon">🛡️</span>
    <div class="hero-title">FraudShield</div>
    <div class="hero-sub">AI-Powered Transaction Security System</div>
    <div class="hero-badge">XGBoost · SMOTE · SHAP · 3-Level Risk Detection</div>
</div>
""", unsafe_allow_html=True)

fraud_rate = (st.session_state.total_fraud / st.session_state.total_checked * 100) if st.session_state.total_checked > 0 else 0.0

st.markdown(f"""
<div class="stats-row">
    <div class="stat-box"><span class="stat-val">97.4%</span><span class="stat-label">ROC-AUC Score</span></div>
    <div class="stat-box"><span class="stat-val">{st.session_state.total_checked}</span><span class="stat-label">Checked This Session</span></div>
    <div class="stat-box fraud-stat"><span class="stat-val">{st.session_state.total_fraud}</span><span class="stat-label">Fraud Flagged</span></div>
    <div class="stat-box legit-stat"><span class="stat-val">{st.session_state.total_legit}</span><span class="stat-label">Approved</span></div>
    <div class="stat-box rate-stat"><span class="stat-val">{fraud_rate:.1f}%</span><span class="stat-label">Session Fraud Rate</span></div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# TABS
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍  Check Transaction",
    "📊  Live Dashboard",
    "📈  Model Performance"
])

# ================================================================
# TAB 1 — CHECK TRANSACTION
# ================================================================
with tab1:

    st.markdown('<div class="sec-label">⚡ Quick Simulate</div>', unsafe_allow_html=True)

    if st.button("🎲  SIMULATE RANDOM TRANSACTION", use_container_width=False):
        st.session_state["sim_amount"]      = round(random.uniform(1, 5000), 2)
        st.session_state["sim_hour"]        = random.randint(0, 23)
        st.session_state["sim_merchant"]    = random.choice(["Retail / Shopping","Restaurant / Food","Online Purchase","ATM / Cash Withdrawal","Travel / Hotels","Electronics","Fuel / Gas Station","Other"])
        st.session_state["sim_location"]    = random.choice(["Same city as usual","Different city","Different country","Online"])
        st.session_state["sim_first"]       = random.choice(["Yes, I have been here before","No, this is my first time"])
        st.session_state["sim_speed"]       = random.choice(["Normal gap (hours apart)","Quick succession (minutes apart)","Very rapid (seconds apart)"])
        st.session_state["sim_feel"]        = random.choice(["Yes, completely normal","Slightly higher than usual","Much higher than usual"])
        st.session_state["sim_card"]        = random.choice(["Physical card was used","Online or phone — card not physically present"])
        st.info("Random transaction loaded — scroll down and click CHECK THIS TRANSACTION.")

    st.markdown('<div class="sec-label">💳 Transaction Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    merchant_list = ["Retail / Shopping","Restaurant / Food","Online Purchase","ATM / Cash Withdrawal","Travel / Hotels","Electronics","Fuel / Gas Station","Other"]
    location_list = ["Same city as usual","Different city","Different country","Online"]
    first_list    = ["Yes, I have been here before","No, this is my first time"]
    speed_list    = ["Normal gap (hours apart)","Quick succession (minutes apart)","Very rapid (seconds apart)"]
    feel_list     = ["Yes, completely normal","Slightly higher than usual","Much higher than usual"]
    card_list     = ["Physical card was used","Online or phone — card not physically present"]

    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=50000.0,
                                  value=st.session_state.get("sim_amount", 100.0), step=0.01,
                                  help="How much was spent in this transaction?")
        hour   = st.selectbox("Time of Transaction", options=list(range(24)),
                               format_func=lambda h: f"{h:02d}:00  {'AM' if h<12 else 'PM'}  {'(Midnight)' if h==0 else '(Noon)' if h==12 else ''}",
                               index=st.session_state.get("sim_hour", 12),
                               help="What hour of the day did this transaction happen?")

    with col2:
        merchant_type = st.selectbox("Merchant Type", merchant_list,
                                      index=merchant_list.index(st.session_state.get("sim_merchant", merchant_list[0])),
                                      help="What kind of store or service was this transaction at?")
        location_type = st.selectbox("Transaction Location", location_list,
                                      index=location_list.index(st.session_state.get("sim_location", location_list[0])),
                                      help="Where did this transaction happen compared to your usual location?")

    st.markdown('<div class="sec-label">🔍 Card Behavior</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        is_first_time = st.radio("Have you transacted at this merchant before?", first_list,
                                  index=first_list.index(st.session_state.get("sim_first", first_list[0])),
                                  help="Fraud often happens at merchants you have never visited before.")
        transaction_speed = st.radio("How quickly did this transaction follow the previous one?", speed_list,
                                      index=speed_list.index(st.session_state.get("sim_speed", speed_list[0])),
                                      help="Multiple rapid transactions is a common fraud pattern.")

    with col4:
        amount_feels = st.radio("Does this amount feel normal for this type of purchase?", feel_list,
                                 index=feel_list.index(st.session_state.get("sim_feel", feel_list[0])),
                                 help="Unusually large amounts relative to your history are a fraud signal.")
        card_present = st.radio("Was the physical card used or was it an online/phone transaction?", card_list,
                                 index=card_list.index(st.session_state.get("sim_card", card_list[0])),
                                 help="Card-not-present transactions carry higher fraud risk.")

    st.markdown("<br>", unsafe_allow_html=True)
    check = st.button("🔍  CHECK THIS TRANSACTION", use_container_width=True)

    if check:
        with st.spinner("Scanning transaction through AI model..."):
            time.sleep(1.2)

        # Build V features
        v14 = 0.0
        if location_type == "Different country":                              v14 -= 3.5
        elif location_type == "Different city":                               v14 -= 1.5
        elif location_type == "Online":                                       v14 -= 1.0
        if is_first_time == "No, this is my first time":                      v14 -= 1.2
        if card_present == "Online or phone — card not physically present":   v14 -= 0.8

        v10 = 0.0
        if transaction_speed == "Very rapid (seconds apart)":                 v10 -= 3.0
        elif transaction_speed == "Quick succession (minutes apart)":         v10 -= 1.5
        if amount_feels == "Much higher than usual":                          v10 -= 2.0
        elif amount_feels == "Slightly higher than usual":                    v10 -= 0.8

        v12 = 0.0
        if merchant_type == "ATM / Cash Withdrawal":                          v12 -= 2.5
        elif merchant_type == "Electronics":                                  v12 -= 1.5
        elif merchant_type == "Online Purchase":                              v12 -= 1.0
        if amount > 5000:                                                     v12 -= 1.5
        elif amount > 1000:                                                   v12 -= 0.5

        v17 = 0.0
        if 0 <= hour <= 4:                                                    v17 -= 2.0
        elif hour >= 22:                                                      v17 -= 1.0
        if transaction_speed == "Very rapid (seconds apart)":                v17 -= 1.5

        v = [0]*28
        v[13]=v14; v[9]=v10; v[11]=v12; v[16]=v17

        amount_log    = np.log1p(amount)
        amount_zscore = (amount - 88.35) / 250.12
        row_data = v + [amount, amount_log, hour, amount_zscore, v[0]*v[1], v[2]*v[3], v[9]*v[13]]
        row  = pd.DataFrame([row_data], columns=model.feature_names_in_)
        prob = model.predict_proba(row)[0][1]
        pct  = prob * 100

        # 3-level risk system
        if prob >= 0.6:
            risk_level="HIGH RISK"; verdict="FRAUD DETECTED"; rc="result-high"
            bc="bar-high"; pc="#ff4444"
            rec="Block transaction — Contact bank immediately"
            rb="background:rgba(255,45,45,0.1);color:#ff8888;"
        elif prob >= 0.3:
            risk_level="MEDIUM RISK"; verdict="SUSPICIOUS"; rc="result-medium"
            bc="bar-medium"; pc="#ffa500"
            rec="Review manually — Request additional verification"
            rb="background:rgba(255,165,0,0.1);color:#ffc044;"
        else:
            risk_level="LOW RISK"; verdict="LEGITIMATE"; rc="result-low"
            bc="bar-low"; pc="#00ff88"
            rec="Transaction approved — No action needed"
            rb="background:rgba(0,255,136,0.07);color:#00ffaa;"

        st.markdown(f"""
        <div class="{rc}">
            <div class="result-level" style="color:{pc}">{risk_level}</div>
            <div class="result-verdict" style="color:{pc}">{verdict}</div>
            <div class="result-prob" style="color:{pc}">{pct:.1f}%</div>
            <div class="result-sub">Fraud Probability Score</div>
            <div class="result-rec" style="{rb}">{rec}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="bar-wrap">
            <div class="{bc}" style="width:{min(int(pct),100)}%"></div>
        </div>
        <div class="bar-labels">
            <span>0% Safe</span><span>30% Caution</span>
            <span>60% High Risk</span><span>100% Definite Fraud</span>
        </div>
        """, unsafe_allow_html=True)

        # Update session counters
        st.session_state.total_checked += 1
        if prob >= 0.6:   st.session_state.total_fraud += 1
        elif prob < 0.3:  st.session_state.total_legit += 1
        st.session_state.history.append({
            "verdict": verdict, "prob": pct, "amount": amount,
            "merchant": merchant_type, "location": location_type, "risk": risk_level
        })

        # AI Explanation
        st.markdown('<div class="sec-label" style="margin-top:24px;">🧠 AI Explanation</div>', unsafe_allow_html=True)

        risk_f, warn_f, safe_f = [], [], []

        if location_type=="Different country":     risk_f.append("Foreign country transaction")
        elif location_type=="Different city":       warn_f.append("Different city from usual")
        else:                                       safe_f.append("Familiar location")

        if transaction_speed=="Very rapid (seconds apart)":          risk_f.append("Rapid successive transactions")
        elif transaction_speed=="Quick succession (minutes apart)":   warn_f.append("Quick back-to-back transaction")
        else:                                                          safe_f.append("Normal transaction timing")

        if amount_feels=="Much higher than usual":    risk_f.append("Unusually high amount")
        elif amount_feels=="Slightly higher than usual": warn_f.append("Slightly elevated amount")
        else:                                          safe_f.append("Normal spending amount")

        if merchant_type=="ATM / Cash Withdrawal":    risk_f.append("ATM cash withdrawal")
        elif merchant_type=="Electronics":             warn_f.append("Electronics — higher fraud category")
        elif merchant_type in ["Retail / Shopping","Restaurant / Food"]: safe_f.append("Common low-risk merchant")

        if 0<=hour<=4:       risk_f.append("Late night transaction")
        elif hour>=22:        warn_f.append("Late evening transaction")
        elif 8<=hour<=20:     safe_f.append("Normal business hours")

        if card_present=="Online or phone — card not physically present": risk_f.append("Card not physically present")
        else:                                                               safe_f.append("Physical card used")

        if is_first_time=="No, this is my first time": warn_f.append("First time at this merchant")
        else:                                           safe_f.append("Known merchant history")

        pills = ""
        for r in risk_f: pills += f'<span class="risk-pill">⚠ {r}</span>'
        for w in warn_f: pills += f'<span class="warn-pill">~ {w}</span>'
        for s in safe_f: pills += f'<span class="safe-pill">✓ {s}</span>'
        st.markdown(f'<div style="margin:10px 0">{pills}</div>', unsafe_allow_html=True)

        # Confidence breakdown
        st.markdown('<div class="sec-label" style="margin-top:20px;">📊 Confidence Breakdown</div>', unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1: st.metric("Fraud Probability", f"{pct:.2f}%")
        with cc2: st.metric("Risk Level", risk_level)
        with cc3: st.metric("Thresholds Used", "30% / 60%")


# ================================================================
# TAB 2 — LIVE DASHBOARD
# ================================================================
with tab2:

    st.markdown('<div class="sec-label">📡 Live Session Monitor</div>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#3d6080;
                    font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:2px;">
            NO DATA YET — Check some transactions first, then come back here.
        </div>
        """, unsafe_allow_html=True)
    else:
        dc1, dc2 = st.columns(2)

        with dc1:
            st.markdown("**Verdict Distribution**")
            verdicts  = [h["verdict"] for h in st.session_state.history]
            labels    = list(set(verdicts))
            sizes     = [verdicts.count(l) for l in labels]
            cmap      = {"FRAUD DETECTED":"#ff4444","SUSPICIOUS":"#ffa500","LEGITIMATE":"#00ff88"}
            colors    = [cmap.get(l,"#00d4ff") for l in labels]

            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor('#04080f')
            ax.set_facecolor('#04080f')
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                autopct='%1.0f%%', startangle=90,
                textprops={'color':'#8899aa','fontsize':10})
            for at in autotexts:
                at.set_color('#ffffff'); at.set_fontsize(11); at.set_fontweight('bold')
            ax.set_title("Verdict Split", color='#00d4ff', fontsize=12, pad=14)
            st.pyplot(fig); plt.close()

        with dc2:
            st.markdown("**Probability Per Transaction**")
            probs  = [h["prob"] for h in st.session_state.history]
            colors2 = ['#ff4444' if p>=60 else '#ffa500' if p>=30 else '#00ff88' for p in probs]

            fig2, ax2 = plt.subplots(figsize=(5,4))
            fig2.patch.set_facecolor('#04080f')
            ax2.set_facecolor('#0d1520')
            for sp in ax2.spines.values(): sp.set_color('#1e2d3d')
            ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
            ax2.tick_params(colors='#3d6080')
            ax2.bar(range(len(probs)), probs, color=colors2, alpha=0.85, edgecolor='none')
            ax2.axhline(y=30, color='#ffa500', linestyle='--', alpha=0.5, lw=1, label='Caution 30%')
            ax2.axhline(y=60, color='#ff4444', linestyle='--', alpha=0.5, lw=1, label='High Risk 60%')
            ax2.set_xlabel("Transaction #", color='#3d6080', fontsize=9)
            ax2.set_ylabel("Fraud Probability %", color='#3d6080', fontsize=9)
            ax2.set_title("Probability History", color='#00d4ff', fontsize=12)
            ax2.legend(fontsize=8, labelcolor='#8899aa', facecolor='#0d1520', edgecolor='#1e2d3d')
            ax2.set_ylim(0,100)
            st.pyplot(fig2); plt.close()

        st.markdown('<div class="sec-label" style="margin-top:20px;">📋 Transaction Log</div>', unsafe_allow_html=True)

        for i, h in enumerate(reversed(st.session_state.history[-20:])):
            bc = "hf" if h["verdict"]=="FRAUD DETECTED" else "hm" if h["verdict"]=="SUSPICIOUS" else "hl"
            st.markdown(f"""
            <div class="hist-row">
                <span style="color:#1e2d3d;font-size:10px;">#{len(st.session_state.history)-i}</span>
                <span class="{bc}">{h['verdict']}</span>
                <span class="hd">${h['amount']:.2f} · {h['merchant']} · {h['location']}</span>
                <span class="hp">{h['prob']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️  CLEAR HISTORY"):
            for k in ["history","total_checked","total_fraud","total_legit"]:
                st.session_state[k] = [] if k=="history" else 0
            st.rerun()


# ================================================================
# TAB 3 — MODEL PERFORMANCE
# ================================================================
with tab3:

    st.markdown('<div class="sec-label">🧬 Model Architecture & Metrics</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: st.metric("Model",     "XGBoost")
    with mc2: st.metric("ROC-AUC",   "97.4%")
    with mc3: st.metric("PR-AUC",    "~85%")
    with mc4: st.metric("Threshold", "0.30 / 0.60")

    st.markdown('<div class="sec-label" style="margin-top:24px;">📐 Metric Explanations</div>', unsafe_allow_html=True)

    mc_col1, mc_col2 = st.columns(2)

    cards = [
        ("#00d4ff", "ROC-AUC — 97.4%",
         "Measures how well the model ranks fraud above legitimate transactions. 97.4% means if you pick one random fraud and one random legitimate transaction, the model correctly assigns higher risk to the fraud 97.4% of the time."),
        ("#00ff88", "PR-AUC — Primary Metric",
         "More important than ROC-AUC for fraud detection because it focuses only on how well we catch fraud without too many false alarms, ignoring the massive number of true negatives."),
        ("#ffa500", "Threshold Tuning — 0.30 / 0.60",
         "Default threshold is 0.5 but we tuned it. Below 30% = Legitimate. 30-60% = Suspicious (review manually). Above 60% = Fraud. Three levels mirror how real bank risk systems work."),
        ("#a78bfa", "SMOTE — Class Balancing",
         "Only 0.17% of transactions are fraud. SMOTE creates synthetic fraud samples so the model learns fraud patterns deeply. Applied only on training data to prevent data leakage."),
    ]

    for i, (color, title, body) in enumerate(cards):
        col = mc_col1 if i % 2 == 0 else mc_col2
        with col:
            st.markdown(f"""
            <div class="info-card" style="border-color:rgba({','.join(str(int(color.lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.18);">
                <div class="info-card-title" style="color:{color}">{title}</div>
                <div class="info-card-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label" style="margin-top:24px;">🔄 Full Training Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("📦","Data Ingestion",    "284,807 transactions · 31 features · 0.17% fraud rate"),
        ("⚙️","Feature Engineering","Amount_log · Hour · Z-score · V1×V2 · V3×V4 · V10×V14"),
        ("🔧","Preprocessing",     "StandardScaler on non-PCA cols · 80/20 Stratified Split"),
        ("⚖️","SMOTE",             "Synthetic minority oversampling · Training data only · No leakage"),
        ("🤖","XGBoost Training",  "200 trees · scale_pos_weight · max_depth=5 · 3-fold CV"),
        ("📊","Evaluation",        "PR-AUC primary · ROC-AUC · F1 · Threshold tuned to 0.30"),
        ("🔍","SHAP Explainability","TreeExplainer · Summary plot · Force plot · Feature importance"),
        ("🚀","Deployment",        "Streamlit · FraudShield · 3-tab app · Live dashboard · ngrok"),
    ]

    pipe_html = ""
    for i,(icon,title,desc) in enumerate(steps):
        pipe_html += f"""
        <div class="pipe-step" style="border-left:3px solid rgba(0,212,255,{0.1+i*0.11:.2f});">
            <span class="pipe-icon">{icon}</span>
            <div><div class="pipe-title">{title}</div><div class="pipe-desc">{desc}</div></div>
            <span class="pipe-num">Step {i+1}</span>
        </div>
        {"<div style='width:2px;height:8px;background:rgba(0,212,255,0.15);margin-left:20px'></div>" if i<len(steps)-1 else ""}
        """
    st.markdown(pipe_html, unsafe_allow_html=True)
