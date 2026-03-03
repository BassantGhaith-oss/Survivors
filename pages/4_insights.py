import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')

GOOD='#00E5A0'; STD='#FFD166'; POOR='#FF4D6D'; ACC='#9D8DF1'
BG='#0A0A0F'; PANEL='#10101A'; GRID='#16161F'; TEXT='#D8D8E8'

BASE = dict(
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(family='DM Mono, monospace', color='#555570', size=10),
    margin=dict(t=40,b=36,l=180,r=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=GRID),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, tickfont=dict(color='#555570')),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, tickfont=dict(color='#555570',size=9)),
)

@st.cache_resource
def load_model():
    with open('model.pkl','rb') as f: return pickle.load(f)
@st.cache_resource
def load_cols():
    with open('columns.pkl','rb') as f: return pickle.load(f)

model   = load_model()
columns = load_cols()

st.markdown("""
<div style='padding:32px 0 4px 0'>
    <div style='font-family:DM Mono,monospace;font-size:0.58rem;
                color:#252535;letter-spacing:4px'>INSIGHTS</div>
    <div style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;
                color:#EEEEF8;letter-spacing:-1px;margin:8px 0 12px 0'>
        What the Model Learned
    </div>
    <p style='font-family:DM Mono,monospace;font-size:0.78rem;
              color:#333348;line-height:1.9'>
        Which signals the model relies on most — and where it struggles.
    </p>
</div>
<div style='width:100%;height:1px;background:#16161F;margin:16px 0 28px 0'></div>
""", unsafe_allow_html=True)

# ── PIPELINE INFO ─────────────────────────────────────────────────────────────
m1,m2,m3,m4 = st.columns(4)
m1.metric("Algorithm",    "GradientBoosting")
m2.metric("Features",     str(len(columns)))
m3.metric("Training Set", "10,111 profiles")
m4.metric("Test Set",     "3,371 profiles")

st.markdown('<div style="width:100%;height:1px;background:#16161F;margin:28px 0"></div>', unsafe_allow_html=True)

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:DM Mono,monospace;font-size:0.58rem;
            color:#252535;letter-spacing:4px;margin-bottom:4px'>SIGNAL STRENGTH</div>
<h2 style='margin-bottom:6px'>Feature Importance</h2>
<p style='font-family:DM Mono,monospace;font-size:0.72rem;color:#333348;margin-bottom:20px'>
Top 20 features by importance score. Highlighted = engineered features.
</p>
""", unsafe_allow_html=True)

try:
    imps = model.feature_importances_
    feat_series = pd.Series(imps, index=columns).sort_values(ascending=True).tail(20)
    engineered = {'Debt_to_Income','Income_per_Loan','Stress_Index','EMI_Burden','Delay_Rate','Inquiry_Pressure'}
    bar_colors = [ACC if f in engineered else '#252535' for f in feat_series.index]

    fig = go.Figure(go.Bar(
        x=feat_series.values,
        y=feat_series.index,
        orientation='h',
        marker_color=bar_colors,
        opacity=0.9,
        text=[f'{v:.4f}' for v in feat_series.values],
        textposition='outside',
        textfont=dict(color='#333348', size=8, family='DM Mono')
    ))
    fig.update_layout(**BASE, height=560, showlegend=False,
                      title=dict(text='Top 20 Feature Importances  ·  purple = engineered',
                                 font=dict(color='#333348',size=10,family='DM Mono')),
                      xaxis_title='Importance Score')
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not extract feature importances: {e}")

st.markdown('<div style="width:100%;height:1px;background:#16161F;margin:28px 0"></div>', unsafe_allow_html=True)

# ── ENGINEERED FEATURES CALLOUT ───────────────────────────────────────────────
st.markdown("""
<div style='font-family:DM Mono,monospace;font-size:0.58rem;
            color:#252535;letter-spacing:4px;margin-bottom:4px'>FEATURE ENGINEERING</div>
<h2 style='margin-bottom:16px'>Why These 6 Features Were Created</h2>
""", unsafe_allow_html=True)

feats = [
    ('Debt_to_Income',   POOR,  'Outstanding debt ÷ monthly income. Captures affordability — same debt means different things at different income levels.'),
    ('Income_per_Loan',  GOOD,  'Annual income ÷ number of loans. How much income backs each obligation. Best single separator between Good and Poor.'),
    ('Stress_Index',     STD,   'Composite of debt burden, missed payments, and inquiry pressure — scaled 0→1. One number summarising financial health.'),
    ('EMI_Burden',       POOR,  'Monthly EMI ÷ take-home salary. What fraction of income is locked into repayments each month.'),
    ('Delay_Rate',       POOR,  'Missed payments ÷ years of history. 10 delays in 1 year is crisis. 10 delays in 20 years is normal.'),
    ('Inquiry_Pressure', STD,   'Credit inquiries ÷ years of history. Normalises inquiry count by how long the person has been building credit.'),
]

col_a, col_b = st.columns(2)
for i,(name, color, desc) in enumerate(feats):
    col = col_a if i%2==0 else col_b
    col.markdown(f"""
    <div style='background:#10101A;border-left:2px solid {color};
                padding:16px;border-radius:0 6px 6px 0;margin-bottom:12px'>
        <div style='font-family:DM Mono,monospace;font-size:0.6rem;
                    color:{color};letter-spacing:2px;margin-bottom:6px'>{name}</div>
        <div style='font-family:DM Mono,monospace;font-size:0.72rem;
                    color:#333348;line-height:1.7'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="width:100%;height:1px;background:#16161F;margin:28px 0"></div>', unsafe_allow_html=True)

# ── PIPELINE SUMMARY ──────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:DM Mono,monospace;font-size:0.58rem;
            color:#252535;letter-spacing:4px;margin-bottom:16px'>PIPELINE</div>
<div style='background:#10101A;border:1px solid #16161F;border-radius:8px;
            padding:24px;font-family:DM Mono,monospace;font-size:0.72rem;
            color:#333348;line-height:2.2'>
    <span style='color:#252535'>01 ·</span> Drop junk rows (Payment_Behaviour = !@9#%8)<br>
    <span style='color:#252535'>02 ·</span> Cap outliers at 99th percentile (salary, interest, utilization)<br>
    <span style='color:#252535'>03 ·</span> Fill NaNs with column median — before splitting<br>
    <span style='color:#252535'>04 ·</span> Loan type → 8 binary flags<br>
    <span style='color:#252535'>05 ·</span> Payment_Behaviour → Spend_Level + Payment_Size (ordinal)<br>
    <span style='color:#252535'>06 ·</span> Credit_Mix → ordinal 0/1/2 (Bad/Standard/Good)<br>
    <span style='color:#252535'>07 ·</span> Payment_of_Min_Amount → ordinal 0/1/2 (No/NM/Yes)<br>
    <span style='color:#252535'>08 ·</span> Credit_History_Age → integer months<br>
    <span style='color:#252535'>09 ·</span> 6 engineered ratio features<br>
    <span style='color:#252535'>10 ·</span> Stratified 75/25 train/test split<br>
    <span style='color:#252535'>11 ·</span> GradientBoostingClassifier — regularised<br>
</div>
""", unsafe_allow_html=True)
