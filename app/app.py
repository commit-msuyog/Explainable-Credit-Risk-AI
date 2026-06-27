import streamlit as st
import joblib
import pandas as pd

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")

features_path = os.path.join(BASE_DIR, "models", "model_features.pkl")

model = joblib.load(model_path)

features = joblib.load(features_path)


st.title("Explainable Credit Risk AI")
st.write("Predict whether a loan applicant is risky or safe.")



person_age = st.number_input("Person Age", min_value=18, max_value=100, value=25)

person_income = st.number_input("Person Income", min_value=0, value=50000)

person_emp_length = st.number_input("Employment Length", min_value=0.0, value=2.0)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)

loan_int_rate = st.number_input("Interest Rate", min_value=0.0, value=10.0)

loan_percent_income = st.number_input(
    "Loan Percent Income",
    min_value=0.0,
    value=0.2
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length",
    min_value=0,
    value=3
)

home_ownership = st.selectbox(
    "Home Ownership",
    ["MORTGAGE", "OWN", "RENT", "OTHER"]
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

default_history = st.selectbox(
    "Previous Default",
    ["N", "Y"]
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

    "cb_person_default_on_file_Y": 1 if default_history == "Y" else 0
}

input_df = pd.DataFrame([input_data])

input_df = input_df.reindex(columns=features, fill_value=0)

if st.button("Predict Risk"):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"High Risk Loan Applicant")

    else:
        st.success(f"Low Risk Loan Applicant")

    st.write(f"Risk Probability: {probability:.2f}")