import streamlit as st
import pandas as pd
import numpy as np
import joblib
@st.cache_resource
def load_model(path):
    return joblib.load(path) 
model1 = load_model("taxi_model.pkl")

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
    input_data={}
    
    # Passenger Count (مثلاً نسمح من 1 لـ 6)
    input_data['passenger_count'] = st.selectbox(
    "Passenger Count",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    # trip distance
    input_data['trip_distance'] = st.selectbox(
    "Trip Total Distance",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    # pickup longitude
    input_data['pickup_longitude'] = st.selectbox(
    "Pickup Longitude",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    # pickup_latitude
    input_data['pickup_latitude'] = st.selectbox(
    "Pickup Latitude",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    
    # dropoff_longitude
    input_data['dropoff_longitude'] = st.selectbox(
    "dropoff longitude",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    
    # 'dropoff_latitude',
    input_data['dropoff_latitude'] = st.selectbox(
    "dropoff_latitude ",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    
    # improvement_surcharge
    input_data['improvement_surcharge'] = st.selectbox(
    "improvement_surcharge ",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    # trip_duration
    input_data['trip_duration'] = st.selectbox(
    "trip_duration ",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    # pickup_month
    input_data['pickup_month'] = st.selectbox(
    "pickup_month ",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    # pickup_day
    input_data['pickup_day'] = st.selectbox(
    "pickup_day ",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    # pickup_hour
    input_data['pickup_hour'] = st.selectbox(
    "pickup_hour ",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    # pickup_minute
    input_data['pickup_minute'] = st.selectbox(
    "pickup_minute ",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    # distance_km
    input_data['distance_km'] = st.selectbox(
    "distance_km",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0]
    )
    # Input
    vendor = st.selectbox("Vendor ID", ["1", "2"])
    ratecode = st.selectbox("Ratecode ID", ["1", "2", "3", "4", "5"])

    # Vendor ID
    input_data['VendorID_2'] = 1 if vendor == "2" else 0

    # Ratecode IDs
    input_data['RatecodeID_2'] = 1 if ratecode == "2" else 0
    input_data['RatecodeID_3'] = 1 if ratecode == "3" else 0
    input_data['RatecodeID_4'] = 1 if ratecode == "4" else 0
    input_data['RatecodeID_5'] = 1 if ratecode == "5" else 0
    
    input_df = pd.DataFrame([input_data])
    st.write(input_df)
    st.write("Input DataFrame columns:", input_df.columns.tolist())
    if st.button("Predict Fare"):
        prediction = model1.predict(input_df)
        st.success(f"Predicted Fare = ${prediction[0]:.2f}")
    
# 💳 Credit Model Page

elif page == "Credit Model":
    st.header("💳 Cre.")
    model2 = joblib.load("best_random_forest_model.pkl")



