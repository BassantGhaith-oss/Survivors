import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------- Sidebar Menu -----------------
page = st.sidebar.radio(
    "Navigation",
    ["Home","Taxi Model","Credit Model"]
)
# ---------- CSS للألوان ----------
page_bg = """
<style>
/* الخلفية الرئيسية */
.stApp {
    background-color: #FFF8E7;  /* الصفحة فاتحة */
    color: #333333;             /* النص العادي */
}

/* Sidebar  */
[data-testid="stSidebar"] {
    background-color: #B0E0E6;
}

/* كل النصوص داخل sidebar */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio > div,
[data-testid="stSidebar"] .stCheckbox > label,
[data-testid="stSidebar"] .css-10trblm,  /* عنوان الـ sidebar */
[data-testid="stSidebar"] .stSelectbox > div {
    color: #FFF8E7 !important;  /* أصفر فاتح */
}

/* أزرار التطبيق */
.stButton>button {
    background-color: #89CFF0;  /* Baby Blue */
    color: white;
    border-radius: 8px;
    height: 40px;
    width: 100%;
    font-weight: bold;
}

/* العناوين */
h1, h2, h3, .css-1v0mbdj-StreamlitMarkdown {
    color: #1E3A8A;  /* العناوين أزرق داكن */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

if page == "Home":
    st.title("The Survivors ⚡")
    st.info('Welcome to Survivors Team App')
    st.header("Our Team :-")
    st.subheader("   Heba Hassan")
    st.subheader("   Bassant Mohammed")
# 🚕 Taxi Model Page

elif page == "Taxi Model":
    st.header("🚕 Pick up Trip")
    st.subheader("⬛⬜⬛⬜⬛⬜⬛⬜⬛⬜⬛⬜⬛⬜⬛⬜⬛⬜⬜⬛⬜⬛⬜⬛⬜⬛⬜⬛⬜⬛⬜⬛⬜⬛⬜⬛")
    
    model1 = joblib.load("taxi_model.pkl")
    input_data={}
    
    # Passenger Count (مثلاً نسمح من 1 لـ 6)
    input_data['passenger_count'] = st.selectbox(
    "Passenger Count",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    # trip distance
    input_data['trip_distance'] = st.selectbox(
    "Trip Total Distance",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    # pickup longitude
    input_data['pickup_longitude'] = st.selectbox(
    "Pickup Longitude",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    # pickup_latitude
    input_data['pickup_latitude'] = st.selectbox(
    "Pickup Latitude",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    
    # dropoff_longitude
    input_data['dropoff_longitude'] = st.selectbox(
    "dropoff longitude",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    
    # 'dropoff_latitude',
    input_data['dropoff_latitude'] = st.selectbox(
    "dropoff_latitude ",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    
    # improvement_surcharge
    input_data['improvement_surcharge'] = st.selectbox(
    "improvement_surcharge ",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    # pickup_month
    input_data['pickup_month'] = st.selectbox(
    "pickup_month ",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    # pickup_day
    input_data['pickup_day'] = st.selectbox(
    "pickup_day ",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    # pickup_hour
    input_data['pickup_hour'] = st.selectbox(
    "pickup_hour ",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    # pickup_minute
    input_data['pickup_minute'] = st.selectbox(
    "pickup_minute ",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    # distance_km
    input_data['distance_km'] = st.selectbox(
    "distance_km",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    # pickup_minute
    input_data['pickup_minute'] = st.selectbox(
    "pickup_minute ",
    options=[1, 2, 3, 4, 5, 6,7,8]
    )
    
# 💳 Credit Model Page

elif page == "Credit Model":
    st.header("💳 Cre.")
    model2 = joblib.load("best_random_forest_model.pkl")



