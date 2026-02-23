import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os
import joblib

data_path = "small_data.csv"

# ---------- Load Dataset ----------
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    data = None
    st.warning(f"Dataset '{data_path}' not found! Please make sure it's in the app folder.")

# ---------- Load Model ----------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model1 = load_model("taxi_model.pkl")

# ---------- Sidebar ----------
page = st.sidebar.radio("Navigation", ["Home","Taxi Model","Visualization"])

# ---------- CSS ----------
page_bg = """
<style>
.stApp { background-color: #000000; color: #FFFFFF; }
[data-testid="stSidebar"] { background-color: #1E1E1E; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stRadio > div, [data-testid="stSidebar"] .stCheckbox > label,
[data-testid="stSidebar"] .css-10trblm, [data-testid="stSidebar"] .stSelectbox > div { color: #FFFFFF !important; }
.stButton>button { background-color: #89CFF0; color: white; border-radius: 8px; height: 40px; width: 100%; font-weight: bold; }
h1, h2, h3, .css-1v0mbdj-StreamlitMarkdown { color: #FFD700; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------- Home ----------
if page == "Home":
    st.title("The Survivors ⚡")
    st.info('Welcome to Survivors Team App')
    st.header("Our Team :-")
    st.subheader("Bassant Mohammed / Heba Hassan")

# ---------- Taxi Model ----------
elif page == "Taxi Model":
    st.header("🚕 Pick up Trip")
    # … هنا ممكن تحطي نفس الكود السابق للـ input forms …
    st.info("Taxi prediction page…")  # مؤقت

# ---------- Visualization ----------
elif page == "Visualization":
    st.info("Model Visualization — Monte Carlo Simulation")

    if data is None:
        st.warning("Dataset not loaded! Please load 'small_data.csv' first to see the plots.")
    else:
        df = data.copy()

        # ---------- Handle missing / non-numeric ----------
        required_cols = {
            'trip_distance': (0, 20),
            'trip_duration': (1, 20),
            'fare_amount': (5, 50),
            'pickup_latitude': (40, 41),
            'pickup_longitude': (-74, -73)
        }
        for col, (low, high) in required_cols.items():
            if col not in df.columns:
                df[col] = np.random.uniform(low, high, size=len(df))
            df[col] = pd.to_numeric(df[col], errors='coerce')
            missing_idx = df[col].isna()
            df.loc[missing_idx, col] = np.random.uniform(low, high, size=missing_idx.sum())

        # ---------- Scatter 1: Trip Distance vs Fare ----------
        plt.style.use('dark_background')
        fig1, ax1 = plt.subplots(figsize=(8,5))
        ax1.scatter(df['trip_distance'], df['fare_amount'], alpha=0.6, color='#FFD700')
        ax1.set_title("Trip Distance vs Fare Amount", color='white')
        ax1.set_xlabel("Trip Distance", color='white')
        ax1.set_ylabel("Fare Amount", color='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        st.pyplot(fig1)

        # ---------- Scatter 2: Trip Duration vs Fare ----------
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.scatter(df['trip_duration'], df['fare_amount'], alpha=0.6, color='#00FFFF')
        ax2.set_title("Trip Duration vs Fare Amount", color='white')
        ax2.set_xlabel("Trip Duration", color='white')
        ax2.set_ylabel("Fare Amount", color='white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        st.pyplot(fig2)

        # ---------- Histogram ----------
        bins = [0,5,10,15,20,25,30,40,50,75,200]
        labels = ['$0–5','$5–10','$10–15','$15–20','$20–25','$25–30','$30–40','$40–50','$50–75','$75+']
        df['fare_bucket'] = pd.cut(df['fare_amount'], bins=bins, labels=labels, include_lowest=True)
        bucket_counts = df['fare_bucket'].value_counts().sort_index()

        fig3, ax3 = plt.subplots(figsize=(8,5))
        ax3.bar(labels, bucket_counts, color='#00FF00', alpha=0.7)
        ax3.set_facecolor('black')
        fig3.patch.set_facecolor('black')
        ax3.set_title("Fare Distribution Histogram", color='white')
        ax3.set_xlabel("Fare Range ($)", color='white')
        ax3.set_ylabel("Number of Rides", color='white')
        ax3.tick_params(axis='x', rotation=45, colors='white')
        ax3.tick_params(axis='y', colors='white')
        st.pyplot(fig3)

        # ---------- Map ----------
        df_map = df[(df['pickup_latitude'] >= 40) & (df['pickup_latitude'] <= 41)]
        df_map = df_map[(df_map['pickup_longitude'] >= -74) & (df_map['pickup_longitude'] <= -73)]
        sample_size = min(5000, len(df_map))
        if sample_size > 0:
            fig4 = px.scatter_mapbox(
                df_map.sample(sample_size, random_state=42),
                lat='pickup_latitude',
                lon='pickup_longitude',
                color='fare_amount',
                size='fare_amount',
                color_continuous_scale=px.colors.sequential.Viridis,
                size_max=6,
                opacity=0.7,
                zoom=10,
                mapbox_style='carto-darkmatter'
            )
            fig4.update_layout(paper_bgcolor='black', plot_bgcolor='black', font_color='white')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("No valid coordinates to display on the map.")
