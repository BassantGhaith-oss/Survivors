# setup

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# headlines and welcom message

st.title('The Survivors')
st.info('Welcome to Survivors Team is App')

page = st.sidebar.selectbox(
    "Choose Model",
    ["Taxi Fare Model", "Delivery Time Model"]
)
