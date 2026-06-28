import streamlit as st
import joblib
import pandas as pd

import sys
import os

current_dir = os.path.dirname(__file__)

project_root = os.path.abspath(os.path.join(current_dir, ".."))

src_path = os.path.join(project_root, "src")

sys.path.append(src_path)

from ai_explain import generate_explanation



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")

features_path = os.path.join(BASE_DIR, "models", "model_features.pkl")

model = joblib.load(model_path)

features = joblib.load(features_path)


st.title("Explainable Credit Risk AI")
st.write("Predict whether a loan applicant is risky or safe.")
st.markdown("---")

with st.sidebar:

    st.markdown("# 💳 Credit Risk AI")

    st.markdown("---")

    st.markdown("### 🤖 Model Information")

    st.write("Random Forest Classifier")

    st.markdown("### 📌 Features")

    st.markdown("""
    - Loan Risk Prediction  
    - Risk Probability  
    - Explainable AI Logic  
    - Financial Risk Analysis  
    """)

    st.markdown("---")

    st.markdown("### 👨‍💻 Developed By")

    st.write("Suyog Verma")


st.subheader("👤 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
    )

    person_emp_length = st.number_input(
    "Employment Experience (Years)",
    min_value=0,
    max_value=40,
    value=2
    )
    st.markdown("---")

with col2:
    person_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=50000,
        step = 5000
    )

    cb_person_cred_hist_length = st.number_input(
    "Credit History Length (Years)",
    min_value=0,
    max_value=30,
    value=3
    )

    st.markdown("---")

st.subheader("💰 Loan Information")

col3, col4 = st.columns(2)

with col3:
    loan_amnt = st.number_input(
        "Loan Amount Requested",
        min_value=0,
        value=10000,
        step = 1000
    )

    loan_int_rate = st.slider(
        "Loan Interest Rate (Per Year %)",
        1.0,
        30.0,
        10.0
    )
    st.markdown("---")

with col4:
    loan_grade = st.selectbox(
        "Credit Rating",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    loan_percent_income = st.slider(
    "Income Used for Loan (%)",
    0,
    100,
    20
    ) / 100

    st.markdown("---")

st.subheader("🏠 Additional Details")

col5, col6 = st.columns(2)

with col5:
    home_ownership = st.selectbox(
        "Home Ownership",
        ["MORTGAGE", "OWN", "RENT", "OTHER"]
    )

with col6:
    default_history = st.selectbox(
        "Loan Repayment History",
        ["Good", "Missed Payments Before"]
    )

loan_intent = st.selectbox(
    "Loan Intent",
    [
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
        "DEBTCONSOLIDATION"
    ]
)





grade_mapping = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6
}

loan_grade_encoded = grade_mapping[loan_grade]


input_data = {
    "person_age": person_age,
    "person_income": person_income,
    "person_emp_length": person_emp_length,
    "loan_grade": loan_grade_encoded,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,

    "person_home_ownership_OTHER": 1 if home_ownership == "OTHER" else 0,
    "person_home_ownership_OWN": 1 if home_ownership == "OWN" else 0,
    "person_home_ownership_RENT": 1 if home_ownership == "RENT" else 0,

    "loan_intent_EDUCATION": 1 if loan_intent == "EDUCATION" else 0,
    "loan_intent_HOMEIMPROVEMENT": 1 if loan_intent == "HOMEIMPROVEMENT" else 0,
    "loan_intent_MEDICAL": 1 if loan_intent == "MEDICAL" else 0,
    "loan_intent_PERSONAL": 1 if loan_intent == "PERSONAL" else 0,
    "loan_intent_VENTURE": 1 if loan_intent == "VENTURE" else 0,

    "cb_person_default_on_file_Y": (
        1 if default_history == "Missed Payments Before" else 0
    )}

input_df = pd.DataFrame([input_data])

input_df = input_df.reindex(columns=features, fill_value=0)


if st.button("Predict Risk"):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("High Risk Loan Applicant")

    else:
        st.success("Low Risk Loan Applicant")

    st.progress(float(probability))

    st.write(f"Risk Probability: {probability:.2%}")

    if probability > 0.7:
        st.warning("High probability of loan default.")

    elif probability > 0.4:
        st.info("Moderate financial risk detected.")

    else:
        st.success("Applicant appears financially stable.")


    # AI Explanation
    ai_explanation = generate_explanation(
        prediction,
        probability,
        person_income,
        loan_amnt,
        loan_int_rate,
        loan_grade,
        default_history,
        person_emp_length,
        cb_person_cred_hist_length,
        loan_percent_income
    )

    st.markdown("---")

    st.subheader("🤖 AI Financial Explanation")

    st.write(ai_explanation)

