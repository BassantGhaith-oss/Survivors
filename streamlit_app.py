import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------- Sidebar Menu -----------------
page = st.sidebar.radio(
    "Navigation",
    ["Home","Taxi Model","Credit Model"]
)
if page == "Home":
    st.title("The Survivors")
    st.info('Welcome to Survivors Team App')
    
# 🚕 Taxi Model Page

elif page == "Taxi Model":
    st.header("🚕 Pick up Trip")
    model1 = joblib.load("taxi_model.pkl")
    

# 💳 Credit Model Page

elif page == "Credit Model":
    st.header("💳 Cre.")
    model2 = joblib.load("best_random_forest_model.pkl")



