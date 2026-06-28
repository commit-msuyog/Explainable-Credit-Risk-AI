import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_explanation(
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
):

    risk_level = "High Risk" if prediction == 1 else "Low Risk"

    prompt = f"""
    You are a professional financial analyst.

    Applicant Risk Level:
    {risk_level}

    Risk Probability:
    {probability:.2%}

    Financial Details:
    - Annual Income: {person_income}
    - Loan Amount: {loan_amnt}
    - Interest Rate: {loan_int_rate}%
    - Credit Grade: {loan_grade}
    - Previous Missed Payments: {default_history}
    - Employment Length: {person_emp_length} years
    - Credit History Length: {cb_person_cred_hist_length} years
    - Loan Percent Income: {loan_percent_income:.2%}

    Explain:
    - key strengths
    - key financial risks
    - overall financial assessment

    Keep response concise and human-friendly.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
