import streamlit as st
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "loan_approval_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "loan_approval_scaler.pkl"))

st.title("Loan Approval Prediction")
st.write("Enter the details to predict loan approval:")

person_age        = st.number_input("Age", min_value=18.0, max_value=100.0, step=1.0, value=30.0)
person_gender     = st.selectbox("Gender", ["female", "male"])
person_education  = st.selectbox("Education", ["Associate", "Bachelor", "Doctorate", "High School", "Master"])
person_income     = st.number_input("Annual Income ($)", min_value=0.0, step=1000.0, value=50000.0)
person_emp_exp    = st.number_input("Years of Employment Experience", min_value=0, max_value=50, step=1, value=3)
person_home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"])
loan_amnt         = st.number_input("Loan Amount ($)", min_value=0.0, step=500.0, value=10000.0)
loan_intent       = st.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
loan_int_rate     = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1, value=10.0)
loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, max_value=1.0, step=0.01, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0.0, max_value=50.0, step=1.0, value=5.0)
credit_score      = st.number_input("Credit Score", min_value=300, max_value=850, step=1, value=650)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", ["No", "Yes"])

# Encode the categorical variables
gender_map    = {"female": 0, "male": 1}
education_map = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
ownership_map = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
intent_map    = {"DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2, "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5}
default_map   = {"No": 0, "Yes": 1}

if st.button("Loan Approval Predict"):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        "person_age":                      [person_age],
        "person_gender":                   [gender_map[person_gender]],
        "person_education":                [education_map[person_education]],
        "person_income":                   [person_income],
        "person_emp_exp":                  [person_emp_exp],
        "person_home_ownership":           [ownership_map[person_home_ownership]],
        "loan_amnt":                       [loan_amnt],
        "loan_intent":                     [intent_map[loan_intent]],
        "loan_int_rate":                   [loan_int_rate],
        "loan_percent_income":             [loan_percent_income],
        "cb_person_cred_hist_length":      [cb_person_cred_hist_length],
        "credit_score":                    [credit_score],
        "previous_loan_defaults_on_file":  [default_map[previous_loan_defaults_on_file]],
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.success("The loan is approved.")
    else:
        st.error("The loan is not approved.")