import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
import os
import plotly.graph_objects as go
data_path = "small_data.csv"

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    data = pd.DataFrame()  # نسخة فاضية لو الملف مش موجود
    st.warning(f"Dataset '{data_path}' not found! Please make sure it's in the app folder.")
if data is not None:
    df = data.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    plt.figure(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt.gcf())
           
@st.cache_resource
def load_model(path):
    return joblib.load(path) 
model1 = load_model("taxi_model.pkl")

# ----------------- Sidebar Menu -----------------
page = st.sidebar.radio(
    "Navigation",
    ["Home","Taxi Model","Visualization"]
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
    st.subheader("   Bassant Mohammed   /    Heba Hassan")
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
    options=[20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
    )
    # pickup longitude
    input_data['pickup_longitude'] = st.selectbox(
    "Pickup Longitude",
    options=[-76.0, -75.0, -74.0, -73.0, -72.0, -71.0 , -70.0 , -69.0 , -68.0 , -67.0 , -66.0,-65.0, -64.0, -63.0, -62.0, -61.0, -60.0 , -59.0 , -58.0 , -57.0 , -56.0,-55.0, -54.0, -53.0, -52.0, -51.0 ]
    )
    # pickup_latitude
    input_data['pickup_latitude'] = st.selectbox(
    "Pickup Latitude",
    options=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0 , 40.0 , 45.0 , 50.0 , 55.0]
    )
    
    # dropoff_longitude
    input_data['dropoff_longitude'] = st.selectbox(
    "dropoff longitude",
    options=[-76.0, -75.0, -74.0, -73.0, -72.0, -71.0 , -70.0 , -69.0 , -68.0 , -67.0 , -66.0,-65.0, -64.0, -63.0, -62.0, -61.0, -60.0 , -59.0 , -58.0 , -57.0 , -56.0,-55.0, -54.0, -53.0, -52.0, -51.0 ]
    )
    
    # 'dropoff_latitude',
    input_data['dropoff_latitude'] = st.selectbox(
    "dropoff_latitude ",
    options=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0 , 40.0 , 45.0 , 50.0 , 55.0]
    )
    
    # improvement_surcharge
    input_data['improvement_surcharge'] = st.selectbox(
    "improvement_surcharge ",
    options=[-0.3 , -0.2 , -0.1 , 0.0 , 0.1 ,0.2 , 0.3]
    )
    # trip_duration
    input_data['trip_duration'] = st.selectbox(
    "trip_duration ",
    options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0, 9.0 ,10.0 ,11.0 ,12.0 , 13.0, 14.0 ,15.0 , 16.0 ,17.0 ,18.0 ,19.0 ,20.0]
    )
    # pickup_month
    input_data['pickup_month'] = st.selectbox(
    "pickup_month ",
    options=[1 ]
    )
    # pickup_day
    input_data['pickup_day'] = st.selectbox(
    "pickup_day ",
    options=[1 ,2 ,3 ,4,5,6 ,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    )
    # pickup_hour
    input_data['pickup_hour'] = st.selectbox(
    "pickup_hour ",
    options=[0,1 ,2 ,3 ,4,5,6 ,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    )
    # pickup_minute
    input_data['pickup_minute'] = st.selectbox(
    "pickup_minute ",
    options=[1 ,2 ,3 ,4,5,6 ,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]
    )
    # distance_km
    input_data['distance_km'] = st.selectbox(
    "distance_km",
    options=[500.0,  1000.0 , 1500.0 , 2000.0 , 2500.0 , 3000.0 , 3500.0 , 4000.0 , 4500.0 , 5000.0 , 5500.0 , 6000.0 ,6500.0 ,7000.0 ,7500.0 ,8000.0 ,8500.0]
    )
    # Input
    vendor = st.selectbox("Vendor ID", [2])
    ratecode = st.selectbox("Ratecode ID", [2,3,4,5,6,99])

    # Vendor ID
    input_data['VendorID_2'] = 1 if vendor == 2 else 0

    # Ratecode IDs
    input_data['RatecodeID_2.0'] = 1 if ratecode == 2 else 0
    input_data['RatecodeID_3.0'] = 1 if ratecode == 3 else 0
    input_data['RatecodeID_4.0'] = 1 if ratecode == 4 else 0
    input_data['RatecodeID_5.0'] = 1 if ratecode == 5 else 0
    input_data['RatecodeID_6.0'] = 1 if ratecode == 6 else 0
    input_data['RatecodeID_99.0'] = 1 if ratecode == 99 else 0
    
    input_df = pd.DataFrame([input_data])
    st.write(input_df)
    st.write("Input DataFrame columns:", input_df.columns.tolist())
    expected_cols = model1.feature_names_in_
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)
    if st.button("Predict Fare"):
        prediction = model1.predict(input_df)
        st.success(f"Predicted Fare = ${prediction[0]:.2f}")
        
elif page == "Visualization":
    st.info("Model Visualization — Monte Carlo Simulation")

    if data.empty:
        st.warning("Dataset not loaded! Please load 'small_data.csv' first to see the plots.")
    else:
        df = data.copy()

        # ---------- Monte Carlo Plot (Plotly) ----------
        N_PATHS = st.slider("Number of Simulated Paths", min_value=50, max_value=500, value=200, step=50)
        max_distance = st.number_input("Max Distance (km)", min_value=1, max_value=500, value=100)
        distances = np.linspace(0, max_distance, 100)
        paths = [np.cumsum(np.random.rand(len(distances))*0.5) for _ in range(N_PATHS)]
        final_fares = [path[-1] for path in paths]
        paths_array = np.array(paths)
        mean_path = np.mean(paths_array, axis=0)
        p10_path = np.percentile(paths_array, 10, axis=0)
        p25_path = np.percentile(paths_array, 25, axis=0)
        p75_path = np.percentile(paths_array, 75, axis=0)
        p90_path = np.percentile(paths_array, 90, axis=0)

        def fare_to_color(fare):
            norm = min(fare / max(final_fares), 1.0)
            return f'rgba(0, {int(200*norm)}, 255, 0.3)'

        fig_mc = go.Figure()
        for i in range(N_PATHS):
            fig_mc.add_trace(go.Scatter(x=distances, y=paths[i], mode='lines',
                                        line=dict(width=0.5, color=fare_to_color(final_fares[i])),
                                        showlegend=False, hoverinfo='skip'))
        # Percentile bands
        fig_mc.add_trace(go.Scatter(
            x=np.concatenate([distances, distances[::-1]]),
            y=np.concatenate([p90_path, p10_path[::-1]]),
            fill='toself', fillcolor='rgba(0,200,255,0.07)',
            line=dict(color='rgba(0,0,0,0)'), name='P10–P90 Band'
        ))
        fig_mc.add_trace(go.Scatter(
            x=np.concatenate([distances, distances[::-1]]),
            y=np.concatenate([p75_path, p25_path[::-1]]),
            fill='toself', fillcolor='rgba(0,200,255,0.12)',
            line=dict(color='rgba(0,0,0,0)'), name='P25–P75 Band'
        ))
        fig_mc.add_trace(go.Scatter(
            x=distances, y=mean_path, mode='lines',
            line=dict(color='#FFE135', width=3.5),
            name=f'Mean Fare (${mean_path[-1]:.2f} at {max_distance}km)'
        ))
        st.plotly_chart(fig_mc, use_container_width=True)

        # ---------- Matplotlib Scatter Plots ----------
        plt.style.use('dark_background')

        # Scatter 1: Trip Distance vs Fare
        fig1, ax1 = plt.subplots(figsize=(8,5))
        ax1.scatter(df['trip_distance'], df['fare_amount'], alpha=0.5, color='#8A2BE2')
        ax1.set_title("Trip Distance vs Fare Amount", color='white')
        ax1.set_xlabel("Trip Distance", color='white')
        ax1.set_ylabel("Fare Amount", color='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        st.pyplot(fig1)

        # Scatter 2: Trip Duration vs Fare
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.scatter(df['trip_duration'], df['fare_amount'], alpha=0.5, color='#008080')
        ax2.set_title("Trip Duration vs Fare Amount", color='white')
        ax2.set_xlabel("Trip Duration", color='white')
        ax2.set_ylabel("Fare Amount", color='white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        st.pyplot(fig2)

        # Histogram 3: Fare Distribution
        bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 200]
        labels = ['$0–5','$5–10','$10–15','$15–20','$20–25','$25–30','$30–40','$40–50','$50–75','$75+']
        df['fare_bucket'] = pd.cut(df['fare_amount'], bins=bins, labels=labels, include_lowest=True)
        bucket_counts = df['fare_bucket'].value_counts().sort_index()
        fig3, ax3 = plt.subplots(figsize=(8,5))
        ax3.bar(labels, bucket_counts, color='#008080', alpha=0.7)
        ax3.set_title("Fare Distribution Histogram", color='white')
        ax3.set_xlabel("Fare Range ($)", color='white')
        ax3.set_ylabel("Number of Rides", color='white')
        ax3.tick_params(axis='x', rotation=45, colors='white')
        ax3.tick_params(axis='y', colors='white')
        st.pyplot(fig3)

        # Map 4: Pickup Locations with Fare (Plotly)
        fig4 = px.scatter_mapbox(
            df.sample(min(5000,len(df)), random_state=42),
            lat='pickup_latitude',
            lon='pickup_longitude',
            color='fare_amount',
            size='fare_amount',
            color_continuous_scale=px.colors.sequential.Viridis,
            size_max=8,
            opacity=0.7,
            zoom=10,
            mapbox_style='carto-darkmatter'
        )
        fig4.update_layout(paper_bgcolor='black', plot_bgcolor='black', font_color='white')
        st.plotly_chart(fig4, use_container_width=True)
