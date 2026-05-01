import streamlit as st
st.set_page_config(page_title="Loan Portfolio Health Monitor", page_icon="🏦", layout="wide")

st.markdown("## 🏦 Loan Portfolio Health Monitor")
st.markdown("##### BA Portfolio Project — Python · Streamlit · SQLite · XGBoost · SHAP")
st.divider()

st.info("👈 Use the **sidebar on the left** to navigate between pages.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Page 1 — Portfolio Dashboard
    - AUM, Gross/Net NPA%, Portfolio Yield
    - Delinquency Funnel (Current → 90+ DPD)
    - Collection Efficiency Trend (3 months)
    - NPA% by Loan Type, Ticket Size, Vintage
    - Early Warning Alerts with Risk Scores
    """)

with col2:
    st.markdown("""
    ### 🤖 Page 2 — NPA Prediction Model
    - 3 Models: Logistic Regression, Random Forest, XGBoost
    - ROC Curve + Precision-Recall Curves
    - SHAP Feature Explainability
    - Portfolio Risk Tier Segmentation
    - Live NPA Probability Calculator
    """)

st.divider()
st.caption("Synthetic data · For portfolio demonstration purposes only")
