import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ----------------- Title -----------------
st.title('The Survivors')
st.info('Welcome to Survivors Team App')

# ----------------- Sidebar Menu -----------------
page = st.sidebar.radio(
    "Navigation",
    ["Taxi Model", "Credit Model"])
# ----------------- Load Models Once -----------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model1 = load_model("taxi_model.pkl")
model2 = load_model("best_random_forest_model.pkl")

# ==================================================
# 🚕 Taxi Model Page
# ==================================================
if page == "🚕 Taxi Model":
    st.header("Taxi Fare Prediction")

    # Features
    numeric_features = [
        'passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'improvement_surcharge',
        'trip_duration', 'pickup_month', 'pickup_day', 'pickup_hour', 'pickup_minute',
        'distance_km'
    ]

    onehot_features = [
        'VendorID_2', 'RatecodeID_2', 'RatecodeID_3', 'RatecodeID_4', 'RatecodeID_5'
    ]

    input_data = {}

    st.subheader("📊 Numeric Features")
    for col in numeric_features:
        if "count" in col or "month" in col or "day" in col or "hour" in col or "minute" in col:
            input_data[col] = st.number_input(col.replace("_", " "), value=1, step=1)
        else:
            input_data[col] = st.number_input(col.replace("_", " "), value=0.0, step=0.1, format="%.6f")

    st.subheader("🔹 Categorical / One-Hot Features")
    for col in onehot_features:
        input_data[col] = int(st.checkbox(col.replace("_", " "), value=False))

    input_df = pd.DataFrame([input_data])

    if st.button("Predict Fare"):
        pred = model1.predict(input_df)
        st.success(f"Predicted Fare = ${pred[0]:.2f}")

# ==================================================
# 💳 Credit Model Page
# ==================================================
elif page == "💳 Credit Model":
    st.header("Credit Prediction")

    # Features
    numeric_features = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 
        'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly', 
        'Monthly_Balance', 'Credit_History_Age_Months'
    ]

    boolean_features = [
        'has_Personal_Loan', 'has_Payday_Loan', 'has_Credit-Builder_Loan', 
        'has_Student_Loan', 'has_Mortgage_Loan', 'has_Home_Equity_Loan', 
        'has_Debt_Consolidation_Loan', 'has_Auto_Loan', 
        'Payment_of_Min_Amount_No', 'Payment_of_Min_Amount_Yes'
    ]

    occupation_features = [
        'Occupation_Architect', 'Occupation_Developer', 'Occupation_Doctor', 
        'Occupation_Engineer', 'Occupation_Entrepreneur', 'Occupation_Journalist', 
        'Occupation_Lawyer', 'Occupation_Manager', 'Occupation_Mechanic', 
        'Occupation_Media_Manager', 'Occupation_Musician', 'Occupation_Scientist', 
        'Occupation_Teacher', 'Occupation_Writer', 'Occupation________'
    ]

    spending_payment_features = [
        'Spending_Behaviour_Low_spent', 'Spending_Behaviour_Unknown_spent', 
        'Payment_Value_Medium_value_payments', 'Payment_Value_Small_value_payments', 
        'Payment_Value_Unknown_value_payments', 'Credit_Mix_Good', 'Credit_Mix_Standard', 
        'Credit_Mix__'
    ]

    input_data = {}

    st.subheader("📊 Numeric Features")
    for col in numeric_features:
        input_data[col] = st.number_input(col.replace("_", " "), value=0.0)

    st.subheader("✅ Boolean Features")
    for col in boolean_features:
        input_data[col] = int(st.checkbox(col.replace("_", " "), value=False))

    st.subheader("🧑 Occupation")
    for col in occupation_features:
        input_data[col] = int(st.checkbox(col.replace("_", " "), value=False))

    st.subheader("💰 Spending / Payment / Credit Mix")
    for col in spending_payment_features:
        input_data[col] = int(st.checkbox(col.replace("_", " "), value=False))

    input_df = pd.DataFrame([input_data])

    if st.button("Predict Credit"):
        pred = model2.predict(input_df)
        st.success(f"Prediction = {pred[0]}")
