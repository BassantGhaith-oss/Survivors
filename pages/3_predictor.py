import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

GOOD='#00E5A0'; STD='#FFD166'; POOR='#FF4D6D'
BG='#0A0A0F'; PANEL='#10101A'; GRID='#16161F'

@st.cache_resource
def load_model():
    with open('model.pkl','rb') as f: return pickle.load(f)
@st.cache_resource
def load_cols():
    with open('columns.pkl','rb') as f: return pickle.load(f)

model   = load_model()
columns = load_cols()

LABEL = {0:'Poor',1:'Standard',2:'Good'}
COLOR = {'Good':GOOD,'Standard':STD,'Poor':POOR}
EMOJI = {'Good':'◈','Standard':'◇','Poor':'◻'}
DESC  = {
    'Good':     'Strong financial profile. Low debt relative to income, consistent payment history, long credit track record.',
    'Standard': 'Average financial health. Some areas to improve — particularly payment consistency and debt management.',
    'Poor':     'Financial stress signals detected. High missed payments and debt burden are the primary risk factors.',
}

st.markdown("""
<div style='padding:32px 0 4px 0'>
    <div style='font-family:DM Mono,monospace;font-size:0.58rem;
                color:#252535;letter-spacing:4px'>PREDICTOR</div>
    <div style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;
                color:#EEEEF8;letter-spacing:-1px;margin:8px 0 12px 0'>
        Credit Tier Predictor
    </div>
    <p style='font-family:DM Mono,monospace;font-size:0.78rem;
              color:#333348;max-width:520px;line-height:1.9'>
        Enter a financial profile. The model classifies it into Good, Standard, or Poor.
    </p>
</div>
<div style='width:100%;height:1px;background:#16161F;margin:16px 0 28px 0'></div>
""", unsafe_allow_html=True)

# ── INPUTS ───────────────────────────────────────────────────────────────────
def section(label):
    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:0.58rem;color:#252535;letter-spacing:3px;margin-bottom:12px'>{label}</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    section("PERSONAL & INCOME")
    age            = st.slider("Age",                        18, 75,  30)
    annual_income  = st.slider("Annual Income ($)",       5000,180000,50000,step=500)
    monthly_salary = st.slider("Monthly Take-Home ($)",    500, 15000, 4000,step=100)
    num_bank       = st.slider("Bank Accounts",              1,  10,   3)
    num_cc         = st.slider("Credit Cards",               1,  11,   4)

with c2:
    section("CREDIT PROFILE")
    interest_rate  = st.slider("Interest Rate (%)",          1, 100,  14)
    num_loan       = st.slider("Number of Loans",            0,  11,   3)
    credit_hist    = st.slider("Credit History (months)",    0, 400, 120)
    credit_util    = st.slider("Credit Utilization (%)",     0, 100,  30)
    outstanding    = st.slider("Outstanding Debt ($)",       0,5000,  800,step=50)

with c3:
    section("PAYMENT BEHAVIOUR")
    delay_days     = st.slider("Avg Days Late",              0,  60,   5)
    num_delayed    = st.slider("Missed Payments",            0,  30,   3)
    changed_limit  = st.slider("Credit Limit Change ($)",  -20,  30,   5)
    num_inq        = st.slider("Credit Inquiries",           0,  50,  10)
    emi            = st.slider("Monthly EMI ($)",            0,2000, 200,step=10)
    invested       = st.slider("Monthly Investment ($)",     0,2000, 150,step=10)
    balance        = st.slider("Monthly Balance ($)",      -500,5000, 300,step=50)

st.markdown("<br>", unsafe_allow_html=True)

# Loan types
section("LOAN TYPES")
lc1,lc2,lc3,lc4 = st.columns(4)
has_personal = lc1.checkbox("Personal Loan")
has_payday   = lc1.checkbox("Payday Loan")
has_builder  = lc2.checkbox("Credit-Builder")
has_student  = lc2.checkbox("Student Loan")
has_mortgage = lc3.checkbox("Mortgage")
has_equity   = lc3.checkbox("Home Equity")
has_debtcon  = lc4.checkbox("Debt Consolidation")
has_auto     = lc4.checkbox("Auto Loan")

st.markdown("<br>", unsafe_allow_html=True)

# Categorical
mc1,mc2,mc3,mc4 = st.columns(4)
credit_mix   = mc1.selectbox("Credit Mix",       ['Good','Standard','Bad'])
pay_min      = mc2.selectbox("Pays Minimum?",    ['Yes','No','NM'])
spend_level  = mc3.selectbox("Spending Level",   ['Low','High'])
pay_size     = mc4.selectbox("Payment Size",     ['Small','Medium','Large'])

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("◈  CLASSIFY THIS PROFILE", use_container_width=True)

# ── PREDICT ──────────────────────────────────────────────────────────────────
if predict_btn:
    inp = {col:0 for col in columns}

    inp['Age']                       = age
    inp['Annual_Income']             = annual_income
    inp['Monthly_Inhand_Salary']     = monthly_salary
    inp['Num_Bank_Accounts']         = num_bank
    inp['Num_Credit_Card']           = num_cc
    inp['Interest_Rate']             = interest_rate
    inp['Num_of_Loan']               = num_loan
    inp['Delay_from_due_date']       = delay_days
    inp['Num_of_Delayed_Payment']    = num_delayed
    inp['Changed_Credit_Limit']      = changed_limit
    inp['Num_Credit_Inquiries']      = num_inq
    inp['Outstanding_Debt']          = outstanding
    inp['Credit_Utilization_Ratio']  = credit_util
    inp['Total_EMI_per_month']       = emi
    inp['Amount_invested_monthly']   = invested
    inp['Monthly_Balance']           = balance
    inp['Credit_History_Age_Months'] = credit_hist

    inp['has_Personal_Loan']           = int(has_personal)
    inp['has_Payday_Loan']             = int(has_payday)
    inp['has_Credit-Builder_Loan']     = int(has_builder)
    inp['has_Student_Loan']            = int(has_student)
    inp['has_Mortgage_Loan']           = int(has_mortgage)
    inp['has_Home_Equity_Loan']        = int(has_equity)
    inp['has_Debt_Consolidation_Loan'] = int(has_debtcon)
    inp['has_Auto_Loan']               = int(has_auto)

    inp['Credit_Mix']            = {'Bad':0,'Standard':1,'Good':2}.get(credit_mix, 1)
    inp['Payment_of_Min_Amount'] = {'No':0,'NM':1,'Yes':2}.get(pay_min, 1)
    inp['Spend_Level']           = {'Low':0,'High':1}.get(spend_level, 0)
    inp['Payment_Size']          = {'Small':0,'Medium':1,'Large':2}.get(pay_size, 1)

    # Engineered features
    dti = outstanding / max(annual_income/12, 1)
    inp['Debt_to_Income']   = dti
    inp['Income_per_Loan']  = annual_income / max(num_loan, 1)
    inp['Stress_Index']     = (min(dti,5)/5 + min(num_delayed,30)/30 + min(num_inq,50)/50)/3
    inp['EMI_Burden']       = emi / max(monthly_salary, 1)
    inp['Delay_Rate']       = num_delayed / max(credit_hist/12, 1)
    inp['Inquiry_Pressure'] = num_inq / max(credit_hist/12, 1)

    X = pd.DataFrame([inp])[columns]
    pred   = model.predict(X)[0]
    label  = pred if isinstance(pred, str) else LABEL.get(pred, str(pred))
    color  = COLOR.get(label, '#888888')

    # Try to get probabilities
    try:
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        prob_map = {}
        for cls, p in zip(classes, proba):
            cls_label = cls if isinstance(cls, str) else LABEL.get(cls, str(cls))
            prob_map[cls_label] = p
    except:
        prob_map = {label: 1.0}

    st.markdown('<div style="width:100%;height:1px;background:#16161F;margin:28px 0"></div>', unsafe_allow_html=True)

    # Result card
    st.markdown(f"""
    <div style='background:#10101A;border:1px solid {color};border-radius:8px;
                padding:36px;text-align:center;margin-bottom:24px'>
        <div style='font-family:DM Mono,monospace;font-size:0.58rem;
                    color:{color};letter-spacing:6px;margin-bottom:12px'>
            PREDICTED TIER
        </div>
        <div style='font-family:Syne,sans-serif;font-size:3.5rem;font-weight:800;
                    color:{color};letter-spacing:-2px;line-height:1'>
            {label.upper()}
        </div>
        <div style='font-family:DM Mono,monospace;font-size:0.75rem;color:#333348;
                    max-width:480px;margin:16px auto 0;line-height:1.9'>
            {DESC.get(label,'')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability breakdown
    if len(prob_map) > 1:
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:0.58rem;color:#252535;letter-spacing:3px;margin-bottom:14px'>CONFIDENCE</div>", unsafe_allow_html=True)
        pcols = st.columns(3)
        for i,(lbl,clr) in enumerate([('Poor',POOR),('Standard',STD),('Good',GOOD)]):
            p = prob_map.get(lbl, 0)
            with pcols[i]:
                st.markdown(f"""
                <div style='background:#10101A;border:1px solid #16161F;
                            border-radius:6px;padding:16px;text-align:center'>
                    <div style='font-family:DM Mono,monospace;font-size:0.55rem;
                                color:{clr};letter-spacing:3px'>{lbl.upper()}</div>
                    <div style='font-family:Syne,sans-serif;font-size:1.6rem;
                                font-weight:800;color:{clr}'>{p*100:.1f}%</div>
                    <div style='background:#16161F;border-radius:2px;height:4px;
                                margin-top:10px;overflow:hidden'>
                        <div style='background:{clr};width:{p*100:.0f}%;
                                    height:100%;border-radius:2px'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Risk signals
    st.markdown("<br><div style='font-family:DM Mono,monospace;font-size:0.58rem;color:#252535;letter-spacing:3px;margin-bottom:14px'>KEY SIGNALS</div>", unsafe_allow_html=True)
    sigs = [
        ("Debt-to-Income",   f"{dti:.2f}",   GOOD if dti<0.3 else POOR if dti>1.0 else STD),
        ("Missed Payments",  str(num_delayed),GOOD if num_delayed<5 else POOR if num_delayed>15 else STD),
        ("Credit Inquiries", str(num_inq),    GOOD if num_inq<10 else POOR if num_inq>25 else STD),
        ("Credit History",   f"{credit_hist}mo", GOOD if credit_hist>200 else POOR if credit_hist<60 else STD),
    ]
    sc1,sc2,sc3,sc4 = st.columns(4)
    for (lbl,val,col),sc in zip(sigs,[sc1,sc2,sc3,sc4]):
        sc.markdown(f"""
        <div style='background:#10101A;border-left:2px solid {col};
                    padding:12px;border-radius:0 6px 6px 0'>
            <div style='font-family:DM Mono,monospace;font-size:0.55rem;
                        color:#252535;letter-spacing:2px'>{lbl}</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;
                        font-weight:700;color:{col}'>{val}</div>
        </div>
        """, unsafe_allow_html=True)
