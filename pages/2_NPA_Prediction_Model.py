"""
ML NPA Prediction Dashboard
============================
Page 2 of the Loan Portfolio Health Monitor
Shows: Model comparison | SHAP explainability | Risk tiers | Live prediction tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle, json, os

st.set_page_config(
    page_title="NPA Prediction Model",
    page_icon="🤖",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .model-card {
      border:1px solid #e2e8f0; border-radius:12px; padding:1.2rem;
      background:#f8fafc; margin-bottom:0.5rem;
  }
  .best-model { border:2px solid #6366f1; background:#eef2ff; }
  .shap-bar   { height:12px; border-radius:6px; background:#6366f1; display:inline-block; }
  .risk-very-high { color:#ef4444; font-weight:700; }
  .risk-high      { color:#f97316; font-weight:600; }
  .risk-medium    { color:#f59e0b; font-weight:600; }
  .risk-low       { color:#22c55e; }
  .risk-very-low  { color:#15803d; }
  .section-title  { font-size:1.1rem; font-weight:700; color:#1e293b; margin-bottom:0.4rem; }
  footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load saved outputs ────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_DIR   = "data"

@st.cache_resource
def load_models():
    models = {}
    for name, fname in [("Logistic Regression","logistic_regression"),
                         ("Random Forest","random_forest"),
                         ("XGBoost","xgboost")]:
        path = f"{MODELS_DIR}/{fname}.pkl"
        if os.path.exists(path):
            with open(path,"rb") as f:
                models[name] = pickle.load(f)
    with open(f"{MODELS_DIR}/scaler.pkl","rb") as f:
        scaler = pickle.load(f)
    return models, scaler

@st.cache_data
def load_meta():
    with open(f"{MODELS_DIR}/model_meta.json") as f:
        return json.load(f)

@st.cache_data
def load_scored():
    return pd.read_csv(f"{DATA_DIR}/scored_portfolio.csv")

@st.cache_data
def load_shap():
    return pd.read_csv(f"{DATA_DIR}/shap_importance.csv")

# ── Check training done ───────────────────────────────────────────────────────
if not os.path.exists(f"{MODELS_DIR}/model_meta.json"):
    st.error("⚠️ Models not trained yet. Run `python train_model.py` first.")
    st.stop()

models, scaler = load_models()
meta           = load_meta()
scored         = load_scored()
shap_imp       = load_shap()
results        = meta["results"]
best_model     = meta["best_model"]
feature_cols   = meta["feature_cols"]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 NPA Prediction Model")
st.caption(
    f"Trained on **{meta['n_train']:,}** loans · Tested on **{meta['n_test']:,}** · "
    f"Portfolio NPA Rate: **{meta['npa_rate']*100:.1f}%** · "
    f"Best Model: **{best_model}** (AUC {results[best_model]['auc']:.3f})"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📊 Model Performance Comparison</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
METRICS = [("Logistic Regression", col1), ("Random Forest", col2), ("XGBoost", col3)]

for name, col in METRICS:
    r   = results[name]
    is_best = (name == best_model)
    with col:
        badge = "🏆 Best Model" if is_best else ""
        st.markdown(f"**{name}** {badge}")
        m1, m2 = st.columns(2)
        m1.metric("AUC-ROC",   f"{r['auc']:.3f}")
        m2.metric("F1 Score",  f"{r['f1']:.3f}")
        m3, m4 = st.columns(2)
        m3.metric("Precision", f"{r['precision']:.3f}")
        m4.metric("Recall",    f"{r['recall']:.3f}")
        # Confusion matrix mini
        cm_fig = go.Figure(go.Heatmap(
            z=[[r["tn"], r["fp"]], [r["fn"], r["tp"]]],
            x=["Pred: No NPA", "Pred: NPA"],
            y=["Actual: No NPA", "Actual: NPA"],
            colorscale="Blues",
            text=[[r["tn"], r["fp"]], [r["fn"], r["tp"]]],
            texttemplate="%{text}",
            showscale=False,
        ))
        cm_fig.update_layout(height=180, margin=dict(t=10,b=10,l=10,r=10),
                              paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(cm_fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: ROC CURVES + PR CURVES
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns(2)

COLORS = {"Logistic Regression": "#6366f1", "Random Forest": "#22c55e", "XGBoost": "#f59e0b"}

with left:
    st.markdown('<p class="section-title">📈 ROC Curves (AUC-ROC)</p>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                      line=dict(dash="dash", color="#94a3b8", width=1))
    for name, r in results.items():
        fig_roc.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{name} (AUC={r['auc']:.3f})",
            line=dict(color=COLORS[name], width=2.5),
        ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=320, margin=dict(t=10,b=30,l=30,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.55, y=0.1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with right:
    st.markdown('<p class="section-title">🎯 Precision-Recall Curves</p>', unsafe_allow_html=True)
    fig_pr = go.Figure()
    baseline = meta["npa_rate"]
    fig_pr.add_hline(y=baseline, line_dash="dash", line_color="#94a3b8",
                     annotation_text=f"Baseline ({baseline*100:.1f}%)")
    for name, r in results.items():
        fig_pr.add_trace(go.Scatter(
            x=r["rec_curve"], y=r["prec_curve"], mode="lines",
            name=f"{name} (AP={r['avg_prec']:.3f})",
            line=dict(color=COLORS[name], width=2.5),
        ))
    fig_pr.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        height=320, margin=dict(t=10,b=30,l=30,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.55, y=0.9),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

st.caption(
    "**AUC-ROC** measures overall discrimination ability. "
    "**Precision-Recall** is more informative for imbalanced datasets (7.5% NPA rate). "
    "High recall = catches more real NPAs before they default."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SHAP FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🔍 SHAP Feature Importance (XGBoost — Why does the model predict NPA?)</p>',
            unsafe_allow_html=True)

shap_top = shap_imp.head(15).copy()

FEATURE_LABELS = {
    "last_month_pay_ratio":    "Last Month Payment Ratio",
    "collection_rate":         "Overall Collection Rate",
    "months_zero_pay":         "Months with Zero Payment",
    "months_partial_pay":      "Months with Partial Payment",
    "partial_payment":         "Partial Payment Flag",
    "payment_delayed_5d":      "Payment Delayed 5+ Days",
    "warning_score":           "Early Warning Score",
    "interest_rate":           "Interest Rate",
    "loan_amount":             "Loan Amount",
    "emi":                     "EMI Amount",
    "seasoning_ratio":         "Loan Seasoning (% tenure elapsed)",
    "months_active":           "Months Active",
    "emi_to_outstanding_ratio":"EMI-to-Outstanding Ratio",
    "loan_utilisation_pct":    "Loan Utilisation %",
    "high_rate_flag":          "High Interest Rate Flag",
    "large_ticket_flag":       "Large Ticket Loan Flag",
    "early_stage":             "Early Stage Loan (0–6M)",
    "mid_stage":               "Mid Stage Loan (6–24M)",
    "segment_high_default":    "High-Default Segment Flag",
    "type_Personal":           "Loan Type: Personal",
    "type_Business":           "Loan Type: Business",
    "type_Auto":               "Loan Type: Auto",
    "type_Home":               "Loan Type: Home",
}

shap_top["label"]   = shap_top["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
shap_top["pct"]     = shap_top["importance"] / shap_top["importance"].sum() * 100

l, r_ = st.columns([1.5, 1])

with l:
    fig_shap = go.Figure(go.Bar(
        y=shap_top["label"][::-1],
        x=shap_top["importance"][::-1],
        orientation="h",
        marker=dict(
            color=shap_top["importance"][::-1],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="SHAP Value", len=0.6),
        ),
        text=[f"{v:.3f}" for v in shap_top["importance"][::-1]],
        textposition="outside",
    ))
    fig_shap.update_layout(
        height=420, margin=dict(t=10,b=10,l=10,r=80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Mean |SHAP Value| (impact on NPA prediction)",
    )
    st.plotly_chart(fig_shap, use_container_width=True)

with r_:
    st.markdown("##### What SHAP Tells Us")
    explanations = {
        "Last Month Payment Ratio":      "🔴 If the borrower paid less than 100% last month, NPA risk rises sharply.",
        "Months with Zero Payment":      "🔴 Even 1 missed month significantly increases default probability.",
        "Months with Partial Payment":   "🟡 Partial payments signal cash-flow stress — a leading indicator.",
        "Interest Rate":                 "🟡 Higher rates → riskier borrower segment → higher NPA likelihood.",
        "Loan Utilisation %":            "🟢 Higher utilisation = more committed borrower (lower risk).",
        "Loan Seasoning":                "🟢 Older loans have survived stress — lower risk over time.",
        "Early Stage Loan (0–6M)":       "🔴 New loans have highest default risk before behaviour is established.",
    }
    for feature, explanation in explanations.items():
        st.markdown(f"**{feature}**")
        st.caption(explanation)
        st.markdown("")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: PORTFOLIO RISK TIERS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🎯 Portfolio Risk Tier Analysis</p>', unsafe_allow_html=True)

tier_order  = ["Very Low", "Low", "Medium", "High", "Very High"]
tier_colors = ["#15803d", "#22c55e", "#f59e0b", "#f97316", "#ef4444"]

tier_stats = (
    scored.groupby("risk_tier")
    .agg(
        Count=("loan_id","count"),
        Avg_NPA_Prob=("npa_probability","mean"),
        Total_Outstanding=("outstanding_principal","sum"),
        Actual_NPA_Rate=("dpd_bucket", lambda x: (x=="90+ DPD").mean()),
    )
    .reindex(tier_order)
    .reset_index()
    .fillna(0)
)

c1, c2 = st.columns(2)

with c1:
    fig_tier_count = go.Figure(go.Bar(
        x=tier_stats["risk_tier"],
        y=tier_stats["Count"],
        marker_color=tier_colors,
        text=tier_stats["Count"],
        textposition="outside",
    ))
    fig_tier_count.update_layout(
        title="Loans per Risk Tier", height=280,
        margin=dict(t=40,b=20,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_tier_count, use_container_width=True)

with c2:
    fig_tier_npa = go.Figure()
    fig_tier_npa.add_trace(go.Bar(
        x=tier_stats["risk_tier"],
        y=tier_stats["Avg_NPA_Prob"]*100,
        name="Predicted NPA Prob (%)",
        marker_color=tier_colors, opacity=0.85,
        text=[f"{v:.1f}%" for v in tier_stats["Avg_NPA_Prob"]*100],
        textposition="outside",
    ))
    fig_tier_npa.add_trace(go.Scatter(
        x=tier_stats["risk_tier"],
        y=tier_stats["Actual_NPA_Rate"]*100,
        name="Actual NPA Rate (%)",
        mode="lines+markers",
        line=dict(color="#1e293b", width=2.5, dash="dot"),
        marker=dict(size=9),
    ))
    fig_tier_npa.update_layout(
        title="Predicted vs Actual NPA Rate by Tier", height=280,
        margin=dict(t=40,b=20,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0, y=1.2, orientation="h"),
    )
    st.plotly_chart(fig_tier_npa, use_container_width=True)

# Tier table
tier_stats_display = tier_stats.copy()
tier_stats_display["Total_Outstanding"] = tier_stats_display["Total_Outstanding"].apply(
    lambda x: f"₹{x/1e7:.1f} Cr"
)
tier_stats_display["Avg_NPA_Prob"] = tier_stats_display["Avg_NPA_Prob"].apply(lambda x: f"{x*100:.1f}%")
tier_stats_display["Actual_NPA_Rate"] = tier_stats_display["Actual_NPA_Rate"].apply(lambda x: f"{x*100:.1f}%")
tier_stats_display.columns = ["Risk Tier", "No. of Loans", "Avg Predicted NPA%", "Outstanding AUM", "Actual NPA Rate"]
st.dataframe(tier_stats_display, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: HIGH RISK LOANS TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🚨 High & Very High Risk Loans — Action List</p>',
            unsafe_allow_html=True)

high_risk = scored[scored["risk_tier"].isin(["High","Very High"])].copy()
high_risk = high_risk.sort_values("npa_probability", ascending=False)
high_risk["npa_probability"] = (high_risk["npa_probability"]*100).round(1).astype(str) + "%"

cols_show = ["loan_id","loan_type","ticket_size","dpd_bucket",
             "outstanding_principal","npa_probability","risk_tier"]

st.dataframe(
    high_risk[cols_show].rename(columns={
        "loan_id":"Loan ID","loan_type":"Type","ticket_size":"Ticket",
        "dpd_bucket":"Current DPD","outstanding_principal":"Outstanding (₹)",
        "npa_probability":"NPA Probability","risk_tier":"Risk Tier",
    }).head(30).style.format({"Outstanding (₹)":"₹{:,.0f}"}),
    use_container_width=True, height=320,
)
st.caption(f"Showing top 30 of {len(high_risk):,} high-risk loans. These need immediate collections outreach.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: LIVE PREDICTION TOOL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">⚡ Live NPA Probability Calculator</p>',
            unsafe_allow_html=True)
st.caption("Enter a new loan's details to get instant NPA probability from the XGBoost model.")

with st.form("predict_form"):
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        loan_type   = st.selectbox("Loan Type", ["Personal","Home","Auto","Business"])
        loan_amount = st.number_input("Loan Amount (₹)", 50_000, 10_000_000, 300_000, step=10_000)
        rate        = st.slider("Interest Rate (%)", 7.0, 25.0, 14.0, 0.5)
    with pc2:
        tenure      = st.selectbox("Tenure (months)", [12,24,36,48,60,84,120,180,240])
        months_act  = st.slider("Months Active", 1, 120, 12)
        coll_rate   = st.slider("Collection Rate (last 3M)", 0.0, 1.0, 0.85, 0.01)
    with pc3:
        last_ratio  = st.slider("Last Month Pay Ratio", 0.0, 1.0, 0.90, 0.01)
        zero_months = st.number_input("Months with Zero Payment", 0, 6, 0)
        partial_m   = st.number_input("Months with Partial Payment", 0, 6, 1)

    submitted = st.form_submit_button("🔮 Calculate NPA Probability", use_container_width=True)

if submitted:
    emi = loan_amount * (rate/12/100) * (1+rate/12/100)**tenure / ((1+rate/12/100)**tenure - 1)
    outstanding = loan_amount * (1 - min(1.0, months_act/tenure)*0.85)

    feat = {
        "collection_rate":        coll_rate,
        "last_month_pay_ratio":   last_ratio,
        "months_zero_pay":        zero_months,
        "months_partial_pay":     partial_m,
        "loan_amount":            loan_amount,
        "interest_rate":          rate,
        "tenure_months":          tenure,
        "emi":                    emi,
        "outstanding_principal":  outstanding,
        "emi_to_outstanding_ratio": emi / (outstanding + 1),
        "loan_utilisation_pct":   1 - (outstanding / (loan_amount + 1)),
        "interest_rate_band":     0 if rate<=10 else 1 if rate<=14 else 2 if rate<=18 else 3,
        "high_rate_flag":         int(rate > 18),
        "large_ticket_flag":      int(loan_amount > 1_000_000),
        "long_tenure_flag":       int(tenure > 60),
        "months_active":          months_act,
        "seasoning_ratio":        months_act / tenure,
        "early_stage":            int(months_act <= 6),
        "mid_stage":              int(6 < months_act <= 24),
        "payment_delayed_5d":     int(last_ratio < 1.0 or zero_months > 0),
        "partial_payment":        int(partial_m > 0),
        "segment_high_default":   int(loan_type == "Personal" and rate > 18),
        "warning_score":          int(last_ratio < 1.0)*40 + int(partial_m>0)*35 + int(loan_type=="Personal" and rate>18)*25,
        "type_Auto":              int(loan_type == "Auto"),
        "type_Business":          int(loan_type == "Business"),
        "type_Home":              int(loan_type == "Home"),
        "type_Personal":          int(loan_type == "Personal"),
        "emi_burden_pct":         emi / (loan_amount / tenure + 1),
    }

    X_input = pd.DataFrame([feat])[feature_cols].fillna(0)
    xgb_model = models.get("XGBoost")
    prob = xgb_model.predict_proba(X_input)[0][1]

    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        color = "#ef4444" if prob>0.6 else "#f59e0b" if prob>0.3 else "#22c55e"
        st.markdown(f"""
        <div style="text-align:center; padding:1.5rem; border-radius:12px;
                    background:{color}22; border:2px solid {color}">
            <div style="font-size:2.5rem; font-weight:800; color:{color}">{prob*100:.1f}%</div>
            <div style="color:{color}; font-weight:600">NPA Probability</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        if prob >= 0.60:
            tier_label, tier_color, action = "🔴 Very High Risk", "#ef4444", "Immediate collections call required"
        elif prob >= 0.40:
            tier_label, tier_color, action = "🟠 High Risk", "#f97316", "Assign dedicated RM, weekly follow-up"
        elif prob >= 0.20:
            tier_label, tier_color, action = "🟡 Medium Risk", "#f59e0b", "Monitor monthly, proactive SMS"
        else:
            tier_label, tier_color, action = "🟢 Low Risk", "#22c55e", "Standard monitoring"
        st.markdown(f"**Risk Tier:** {tier_label}")
        st.markdown(f"**Recommended Action:**")
        st.info(action)
    with r3:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob*100,
            number={"suffix":"%","font":{"size":28}},
            delta={"reference": meta["npa_rate"]*100, "suffix":"%",
                   "decreasing":{"color":"green"},"increasing":{"color":"red"}},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1},
                "bar":{"color": "#ef4444" if prob>0.6 else "#f59e0b" if prob>0.3 else "#22c55e"},
                "steps":[
                    {"range":[0,20],"color":"#dcfce7"},
                    {"range":[20,40],"color":"#fef9c3"},
                    {"range":[40,60],"color":"#ffedd5"},
                    {"range":[60,100],"color":"#fee2e2"},
                ],
                "threshold":{"line":{"color":"black","width":3},"thickness":0.8,
                             "value": meta["npa_rate"]*100},
            },
            title={"text":"NPA Risk Score"},
        ))
        fig_gauge.update_layout(height=200, margin=dict(t=30,b=10,l=20,r=20),
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: BUSINESS IMPACT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">💼 Business Impact Quantification</p>', unsafe_allow_html=True)

r = results[best_model]
tp, fp, fn, tn = r["tp"], r["fp"], r["fn"], r["tn"]
total_npa_test  = tp + fn
caught_early    = tp
missed          = fn
avg_outstanding = scored["outstanding_principal"].mean()
recovery_uplift = 0.35   # industry: early intervention recovers 35% more
avg_recovery_loss = avg_outstanding * 0.60  # 60% loss given default

bi1, bi2, bi3, bi4 = st.columns(4)
bi1.metric("NPAs Caught Early",     f"{tp:,}",    f"{tp/(tp+fn)*100:.0f}% recall")
bi2.metric("False Alarms (FP)",     f"{fp:,}",    "Extra follow-ups (manageable)")
bi3.metric("NPAs Missed",           f"{fn:,}",    "Slip through undetected", delta_color="inverse")
potential_savings = tp * avg_outstanding * recovery_uplift
bi4.metric("Potential Recovery Uplift", f"₹{potential_savings/1e7:.1f} Cr",
           "vs no early warning")

st.markdown("")
st.info(
    f"📌 **How to read this in interviews:** The {best_model} model catches **{tp/(tp+fn)*100:.0f}%** of future NPAs "
    f"while the loan is still in the 30–60 DPD bucket — giving collections a 30–60 day intervention window. "
    f"At ₹{avg_outstanding/1e5:.1f}L average ticket, early intervention on {tp:,} accounts "
    f"could recover an additional **₹{potential_savings/1e7:.1f} Crore** vs waiting for 90 DPD."
)

st.divider()
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:0.8rem'>"
    "🤖 NPA Prediction Model · XGBoost + SMOTE + SHAP · Synthetic data for demonstration"
    "</div>", unsafe_allow_html=True
)
