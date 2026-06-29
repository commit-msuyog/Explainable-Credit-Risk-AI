import streamlit as st
import joblib
import pandas as pd
import sys
import os
import time

# ── Path setup ───────────────────────────────────────────────────────────────
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from ai_explain import generate_explanation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path     = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
features_path  = os.path.join(BASE_DIR, "models", "model_features.pkl")

model    = joblib.load(model_path)
features = joblib.load(features_path)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CredInsight AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Sans:wght@500;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── White background ── */
.stApp {
    background: #F7F8FA !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1C1F2E !important;
    border-right: 1px solid #2E3248;
}
            
#section[data-testid="stSidebar"] .block-container {{ padding-top: 0.8 rem !important; }}
            
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown a,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] strong {
    color: #F1F5F9 !important;
}
[data-testid="stSidebar"] a {
    color: #818CF8 !important;
    text-decoration: none;
}
[data-testid="stSidebar"] a:hover { text-decoration: underline; }

.sidebar-pill {
    display: inline-block;
    background: #2E3248;
    border: 1px solid #3D4268;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.82rem;
    color: #A5B4FC !important;
    margin: 3px 2px;
    white-space: nowrap;
}
.sidebar-cap {
    background: #252840;
    border-left: 3px solid #6366F1;
    border-radius: 0 8px 8px 0;
    padding: 6px 12px;
    margin: 5px 0;
    font-size: 0.85rem;
    color: #94A3B8 !important;
}
.sidebar-link-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 6px;
}
.sidebar-link-row a {
    font-size: 0.83rem !important;
    color: #818CF8 !important;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(120deg, #1E1B4B 0%, #312E81 45%, #1E3A5F 100%);
    border-radius: 18px;
    padding: 2.8rem 2.4rem 2.2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.35) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 0.4rem;
    letter-spacing: -0.02em;
}
.hero p {
    color: #A5B4FC;
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.25);
    border: 1px solid rgba(165,180,252,0.35);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #C7D2FE;
    margin-bottom: 1rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Snapshot metrics bar ── */
.snap-bar {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.6rem;
}
.snap-card {
    flex: 1;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.snap-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.snap-card .value {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #1E293B;
    white-space: nowrap;
    overflow: visible;
}
.snap-card .sub {
    font-size: 0.75rem;
    color: #6366F1;
    font-weight: 500;
    margin-top: 2px;
}

/* ── Section cards ── */
.sec-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.5rem 1.6rem 1.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.sec-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #6366F1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 6px;
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 0.7rem;
}

/* ── Inputs ── */
input[type="number"], select, textarea {
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    background: #F8FAFC !important;
    color: #1E293B !important;
    font-size: 0.92rem !important;
}
label {
    color: #475569 !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}
/* Streamlit metric label override in light mode */
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #1E293B !important; font-size: 1.35rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── DTI info bar ── */
.dti-bar {
    background: #F0F4FF;
    border: 1px solid #C7D2FE;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-size: 0.88rem;
    color: #3730A3;
    font-weight: 500;
    margin-top: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Signal chips ── */
.chips-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.6rem 0 1rem; }
.chip {
    font-size: 0.8rem;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 999px;
    white-space: nowrap;
}
.chip-green  { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
.chip-yellow { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
.chip-red    { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }

/* ── CTA button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #4F46E5 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.75rem 2rem;
    border: none !important;
    border-radius: 10px !important;
    letter-spacing: 0.03em;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(79,70,229,0.45) !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-card.safe {
    background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
    border: 1.5px solid #6EE7B7;
}
.result-card.risky {
    background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
    border: 1.5px solid #FCA5A5;
}
.result-card h2 {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0 0 0.35rem;
}
.result-card.safe  h2 { color: #065F46; }
.result-card.risky h2 { color: #991B1B; }
.result-card p { color: #475569; font-size: 0.95rem; margin: 0; }

/* ── Risk gauge ── */
.gauge-wrap { margin: 1.2rem 0 0.5rem; }
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #94A3B8;
    margin-bottom: 5px;
}
.gauge-track {
    background: #E2E8F0;
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: 999px;
}
.gauge-pct {
    text-align: right;
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 4px;
}

/* ── AI box ── */
.ai-box {
    background: #F8FAFF;
    border: 1px solid #C7D2FE;
    border-left: 4px solid #6366F1;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    color: #1E293B;
    line-height: 1.8;
    font-size: 0.94rem;
    margin-top: 0.5rem;
}

/* ── Recommendation ── */
.rec-box {
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.9rem;
    font-weight: 500;
    margin-top: 0.8rem;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
.rec-box.safe   { background: #ECFDF5; border: 1px solid #6EE7B7; color: #065F46; }
.rec-box.warn   { background: #FFFBEB; border: 1px solid #FDE68A; color: #78350F; }
.rec-box.danger { background: #FEF2F2; border: 1px solid #FCA5A5; color: #7F1D1D; }

/* ── Divider ── */
hr { border-color: #E2E8F0 !important; margin: 1.2rem 0 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧩 About the Platform")
    #st.markdown("*Intelligent credit risk, explained.*")
    #st.markdown("---")

    st.markdown("#### ⚙️ Prediction Engine")
    st.markdown("""
    <span class="sidebar-pill">🌲 Random Forest</span>
    <span class="sidebar-pill">⚡ Groq LLM</span>
    """, unsafe_allow_html=True)
    #st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### ✨ Capabilities")
    caps = [
        ("📊", "Loan Risk Prediction"),
        ("📡", "AI Financial Insights"),
        ("📈", "Probability Analysis"),
        ("🔍", "Explainable AI"),
    ]
    for icon, cap in caps:
        st.markdown(f'<div class="sidebar-cap">{icon} {cap}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🚀 Developer")
    st.markdown("**Suyog Verma**")
    st.markdown(
        """
        <p style='
            font-size:13px;
            color:gray;
            margin-top:-8px;
        '>
        <i>Building intelligent AI systems and real-world ML applications</i>
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="sidebar-link-row">
        <a href="mailto:suyogverma0057@gmail.com">📧 Email</a>
        <a href="https://www.linkedin.com/in/suyog01/">🔗 LinkedIn</a>
        <a href="https://github.com/commit-msuyog">💻 GitHub</a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    #st.markdown("---")
    st.caption("Built with Streamlit · Scikit-Learn · Groq API")




# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">🔐 AI-Powered · Explainable · Real-Time</div>
  <h1>🧠 CredInsight AI</h1>
  <p>Credit risk assessment backed by machine learning and explainable AI — transparent, instant, and actionable.</p>
</div>
""", unsafe_allow_html=True)

# ── Snapshot bar ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="snap-bar">
  <div class="snap-card">
    <div class="label">Model</div>
    <div class="value">Random Forest</div>
    <div class="sub">● Active</div>
  </div>
  <div class="snap-card">
    <div class="label">Credit Grades</div>
    <div class="value">A → G</div>
    <div class="sub">7 tiers supported</div>
  </div>
  <div class="snap-card">
    <div class="label">Input Features</div>
    <div class="value">{len(features)} features</div>
    <div class="sub">Used for scoring</div>
  </div>
  <div class="snap-card">
    <div class="label">AI Explainer</div>
    <div class="value">Groq LLM</div>
    <div class="sub">● Online</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Applicant Info
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-card"><div class="sec-title">👤 Applicant Information</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=25)
with c2:
    person_income = st.number_input("Annual Income (₹)", min_value=0, value=50000, step=5000)
with c3:
    person_emp_length = st.number_input("Employment (Years)", min_value=0, max_value=40, value=2)
with c4:
    cb_person_cred_hist_length = st.number_input("Credit History (Years)", min_value=0, max_value=30, value=3)
st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Loan Info
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-card"><div class="sec-title">💰 Loan Details</div>', unsafe_allow_html=True)
c5, c6, c7, c8 = st.columns(4)
with c5:
    loan_amnt = st.number_input("Loan Amount (₹)", min_value=0, value=10000, step=1000)
with c6:
    loan_int_rate = st.slider("Interest Rate (% / yr)", 1.0, 30.0, 10.0, step=0.5)
with c7:
    loan_grade = st.selectbox("Credit Rating", ["A", "B", "C", "D", "E", "F", "G"])
with c8:
    loan_intent = st.selectbox("Loan Purpose", [
        "EDUCATION", "HOME IMPROVEMENT", "MEDICAL",
        "PERSONAL", "VENTURE", "DEBT CONSOLIDATION"
    ])

loan_percent_income = (loan_amnt / person_income) if person_income > 0 else 0
dti_label  = "Healthy" if loan_percent_income < 0.3 else ("Moderate" if loan_percent_income < 0.6 else "Stretched")
dti_icon   = "✅" if loan_percent_income < 0.3 else ("⚠️" if loan_percent_income < 0.6 else "🔴")
st.markdown(
    f'<div class="dti-bar">{dti_icon} <strong>Debt-to-Income Ratio:</strong> {loan_percent_income:.2%} — {dti_label}</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Additional Details
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-card"><div class="sec-title">🏠 Additional Details</div>', unsafe_allow_html=True)
c9, c10 = st.columns(2)
with c9:
    home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "OWN", "RENT", "OTHER"])
with c10:
    default_history = st.selectbox("Repayment History", ["Good", "Missed Payments Before"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Pre-assessment signal chips ───────────────────────────────────────────────
st.markdown("##### 🔎 Pre-Assessment Signals")
chips = [
    (f"Age: {person_age}", "chip-green" if person_age >= 25 else "chip-yellow"),
    (f"Grade: {loan_grade}", "chip-green" if loan_grade in ["A","B"] else ("chip-yellow" if loan_grade in ["C","D"] else "chip-red")),
    (f"Rate: {loan_int_rate}%", "chip-green" if loan_int_rate < 10 else ("chip-yellow" if loan_int_rate < 18 else "chip-red")),
    ("✓ Clean History" if default_history == "Good" else "⚠ Prior Default",
     "chip-green" if default_history == "Good" else "chip-red"),
    (f"DTI: {loan_percent_income:.0%}", "chip-green" if loan_percent_income < 0.3 else ("chip-yellow" if loan_percent_income < 0.6 else "chip-red")),
    (f"Exp: {person_emp_length} yrs", "chip-green" if person_emp_length >= 3 else "chip-yellow"),
    (f"Credit Hist: {cb_person_cred_hist_length} yrs", "chip-green" if cb_person_cred_hist_length >= 4 else "chip-yellow"),
]
chips_html = '<div class="chips-wrap">' + "".join(
    f'<span class="chip {cls}">{label}</span>' for label, cls in chips
) + '</div>'
st.markdown(chips_html, unsafe_allow_html=True)

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
btn_col, _ = st.columns([1, 2])
with btn_col:
    predict_clicked = st.button("⚡  Assess Credit Risk")

# ── Encode inputs ─────────────────────────────────────────────────────────────
grade_map = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6}
intent_map = {
    "EDUCATION": "EDUCATION",
    "HOME IMPROVEMENT": "HOMEIMPROVEMENT",
    "MEDICAL": "MEDICAL",
    "PERSONAL": "PERSONAL",
    "VENTURE": "VENTURE",
    "DEBT CONSOLIDATION": "DEBTCONSOLIDATION",
}
loan_intent_key = intent_map[loan_intent]

input_data = {
    "person_age": person_age,
    "person_income": person_income,
    "person_emp_length": person_emp_length,
    "loan_grade": grade_map[loan_grade],
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "person_home_ownership_OTHER": 1 if home_ownership == "OTHER" else 0,
    "person_home_ownership_OWN":   1 if home_ownership == "OWN"   else 0,
    "person_home_ownership_RENT":  1 if home_ownership == "RENT"  else 0,
    "loan_intent_EDUCATION":       1 if loan_intent_key == "EDUCATION"       else 0,
    "loan_intent_HOMEIMPROVEMENT": 1 if loan_intent_key == "HOMEIMPROVEMENT" else 0,
    "loan_intent_MEDICAL":         1 if loan_intent_key == "MEDICAL"         else 0,
    "loan_intent_PERSONAL":        1 if loan_intent_key == "PERSONAL"        else 0,
    "loan_intent_VENTURE":         1 if loan_intent_key == "VENTURE"         else 0,
    "cb_person_default_on_file_Y": 1 if default_history == "Missed Payments Before" else 0,
}
input_df = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)

# ═════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═════════════════════════════════════════════════════════════════════════════
if predict_clicked:
    with st.spinner("Running risk model…"):
        time.sleep(0.3)
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    is_safe     = prediction == 0
    panel_cls   = "safe" if is_safe else "risky"
    verdict     = "✅ Low Risk Applicant" if is_safe else "🚨 High Risk Applicant"
    verdict_sub = "This applicant demonstrates strong creditworthiness." if is_safe \
                  else "Significant default risk detected. Review required."

    # Result card
    st.markdown(f"""
    <div class="result-card {panel_cls}">
      <h2>{verdict}</h2>
      <p>{verdict_sub}</p>
    </div>""", unsafe_allow_html=True)

    # Risk gauge
    bar_color = "#10B981" if probability < 0.4 else ("#F59E0B" if probability < 0.7 else "#EF4444")
    st.markdown(f"""
    <div class="gauge-wrap">
      <div class="gauge-labels">
        <span>Default Probability</span>
        <span style="color:{bar_color}; font-weight:700;">{probability:.1%}</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{probability*100:.1f}%; background:{bar_color};"></div>
      </div>
      <div class="gauge-pct">0% — Low Risk &nbsp;&nbsp;&nbsp; 100% — High Risk</div>
    </div>""", unsafe_allow_html=True)

    # Metric columns
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Default Probability", f"{probability:.2%}")
    m2.metric("Safe Probability", f"{1 - probability:.2%}")
    risk_tier = "Low" if probability < 0.4 else ("Medium" if probability < 0.7 else "High")
    delta_txt = "✓ Approve" if probability < 0.4 else ("⚠ Review" if probability < 0.7 else "✗ Decline")
    m3.metric("Risk Tier", risk_tier, delta=delta_txt,
              delta_color="normal" if probability < 0.4 else "inverse")

    st.markdown("---")

    # AI Explanation
    st.markdown(
        """
        <h4 style="
            color:#22C55E;
            margin-bottom:0.5rem;
        ">
            📡 AI Financial Explanation
        </h4>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Generating AI explanation via Groq…"):
        ai_explanation = generate_explanation(
            prediction, probability, person_income, loan_amnt, loan_int_rate,
            loan_grade, default_history, person_emp_length,
            cb_person_cred_hist_length, loan_percent_income
        )
    st.markdown(f'<div class="ai-box">{ai_explanation}</div>', unsafe_allow_html=True)

    # Recommendation
    st.markdown("<br>", unsafe_allow_html=True)
    if probability > 0.7:
        st.markdown(
            '<div class="rec-box danger">⛔ <div><strong>Decline Recommended</strong><br>'
            'High probability of default. Further due diligence or denial advised.</div></div>',
            unsafe_allow_html=True
        )
    elif probability > 0.4:
        st.markdown(
            '<div class="rec-box warn">⚠️ <div><strong>Manual Review Required</strong><br>'
            'Moderate risk detected. Consider additional collateral or a reduced loan amount.</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="rec-box safe">✅ <div><strong>Approve with Standard Terms</strong><br>'
            'Applicant demonstrates financial stability. Proceed under normal lending conditions.</div></div>',
            unsafe_allow_html=True
        )