"""
Loan Portfolio Health Monitor
Executive Dashboard with Early Warning System
Built for BA Portfolio – Python + Streamlit + SQLite
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import os, sys

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Portfolio Health Monitor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
  .metric-card {
      background: #0f172a; border-radius: 12px; padding: 1.2rem 1.5rem;
      border-left: 4px solid #3b82f6; margin-bottom: 0.5rem;
  }
  .alert-row-red   { background:#fef2f2; border-left:4px solid #ef4444; padding:8px 12px; border-radius:6px; margin:4px 0; }
  .alert-row-amber { background:#fffbeb; border-left:4px solid #f59e0b; padding:8px 12px; border-radius:6px; margin:4px 0; }
  .section-title { font-size:1.15rem; font-weight:700; color:#1e293b; margin-bottom:0.5rem; }
  footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────
DB_PATH = "data/loan_data.db"

@st.cache_data(ttl=300)
def load_data():
    if not os.path.exists(DB_PATH):
        # Auto-generate if missing
        import subprocess
        subprocess.run([sys.executable, "generate_data.py"], check=True)
    conn = sqlite3.connect(DB_PATH)
    loans    = pd.read_sql("SELECT * FROM loans",          conn)
    payments = pd.read_sql("SELECT * FROM monthly_payments", conn)
    conn.close()
    loans["disbursement_date"] = pd.to_datetime(loans["disbursement_date"])
    return loans, payments

loans, payments = load_data()

# ── Sidebar Filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=60)
    st.title("🏦 Portfolio Monitor")
    st.caption(f"Data as of: {datetime.today().strftime('%d %b %Y')}")
    st.divider()

    loan_types = st.multiselect(
        "Loan Type",
        options=loans["loan_type"].unique().tolist(),
        default=loans["loan_type"].unique().tolist(),
    )
    vintage_filter = st.multiselect(
        "Vintage Bucket",
        options=loans["vintage"].unique().tolist(),
        default=loans["vintage"].unique().tolist(),
    )
    ticket_filter = st.multiselect(
        "Ticket Size",
        options=["<2L", "2-10L", "10-50L", "50L+"],
        default=["<2L", "2-10L", "10-50L", "50L+"],
    )
    st.divider()
    st.caption("💡 Tip: Use filters to drill down by segment, vintage, or ticket size.")

# ── Apply Filters ─────────────────────────────────────────────────────────────
df = loans[
    loans["loan_type"].isin(loan_types) &
    loans["vintage"].isin(vintage_filter) &
    loans["ticket_size"].isin(ticket_filter)
].copy()

payments_f = payments[payments["loan_id"].isin(df["loan_id"])]

# ── KPI Calculations ──────────────────────────────────────────────────────────
total_aum          = df["outstanding_principal"].sum()
active             = df[df["dpd_bucket"] != "Closed"]
gross_npa_amt      = df[df["dpd_bucket"] == "90+ DPD"]["outstanding_principal"].sum()
gross_npa_pct      = gross_npa_amt / total_aum * 100 if total_aum else 0

provision_rate     = 0.50          # RBI standard: 50% provision on sub-standard assets
net_assets         = total_aum
net_npa_pct        = (gross_npa_amt * (1 - provision_rate)) / net_assets * 100 if net_assets else 0

portfolio_yield    = np.average(
    df["interest_rate"], weights=df["outstanding_principal"]
) if total_aum > 0 else 0

# Collection efficiency (last month)
last_month = payments_f[payments_f["month_offset"] == 1]
coll_eff   = (last_month["collected"].sum() / last_month["expected"].sum() * 100
              if last_month["expected"].sum() > 0 else 0)

early_warn = df[df["dpd_bucket"].isin(["30 DPD", "60 DPD"])].shape[0]
total_loans = df.shape[0]

# ── Section 1: Overview Metrics ───────────────────────────────────────────────
st.markdown("## 🏦 Loan Portfolio Health Monitor")
st.caption("Real-time executive dashboard · Early warning system for NPA prevention")
st.divider()

col1, col2, col3, col4, col5 = st.columns(5)

def fmt_cr(val):
    return f"₹{val/1e7:.1f} Cr"

col1.metric("Total AUM",          fmt_cr(total_aum))
col2.metric("Gross NPA",          f"{gross_npa_pct:.2f}%",
            delta=f"{'⚠️ High' if gross_npa_pct > 5 else '✅ In Range'}",
            delta_color="inverse")
col3.metric("Net NPA",            f"{net_npa_pct:.2f}%")
col4.metric("Portfolio Yield",    f"{portfolio_yield:.2f}%")
col5.metric("Collection Efficiency", f"{coll_eff:.1f}%",
            delta=f"{'⚠️ Below Target' if coll_eff < 95 else '✅ On Track'}",
            delta_color="inverse")

st.divider()

# ── Section 2: Delinquency Funnel ─────────────────────────────────────────────
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<p class="section-title">📉 Delinquency Funnel</p>', unsafe_allow_html=True)

    dpd_order  = ["Current", "30 DPD", "60 DPD", "90+ DPD"]
    dpd_counts = df[df["dpd_bucket"] != "Closed"]["dpd_bucket"].value_counts()
    funnel_vals = [dpd_counts.get(b, 0) for b in dpd_order]

    colors = ["#22c55e", "#f59e0b", "#f97316", "#ef4444"]
    fig_funnel = go.Figure(go.Funnel(
        y=dpd_order,
        x=funnel_vals,
        textinfo="value+percent initial",
        marker=dict(color=colors),
        connector=dict(line=dict(color="#e2e8f0", width=2)),
    ))
    fig_funnel.update_layout(
        margin=dict(t=20, b=10, l=10, r=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    # Flow annotation
    for i in range(len(dpd_order) - 1):
        a, b = dpd_order[i], dpd_order[i + 1]
        cnt_a, cnt_b = dpd_counts.get(a, 1), dpd_counts.get(b, 0)
        flow_rate = cnt_b / cnt_a * 100 if cnt_a else 0
        st.caption(f"  {a} → {b}: **{flow_rate:.1f}%** roll rate")

with right:
    st.markdown('<p class="section-title">🗓 DPD Bucket – Outstanding Principal</p>', unsafe_allow_html=True)

    dpd_aum = (
        df[df["dpd_bucket"] != "Closed"]
        .groupby("dpd_bucket")["outstanding_principal"]
        .sum()
        .reindex(dpd_order)
        .fillna(0)
    )
    fig_bar = go.Figure(go.Bar(
        x=[fmt_cr(v) for v in dpd_aum.values],
        y=dpd_aum.index,
        orientation="h",
        marker_color=colors,
        text=[fmt_cr(v) for v in dpd_aum.values],
        textposition="outside",
    ))
    fig_bar.update_layout(
        xaxis_title="Outstanding Principal",
        margin=dict(t=20, b=20, l=10, r=80),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Collection efficiency trend (last 3 months)
    st.markdown('<p class="section-title">📊 Collection Efficiency – Last 3 Months</p>', unsafe_allow_html=True)
    ce_trend = (
        payments_f.groupby("month_offset")
        .apply(lambda x: x["collected"].sum() / x["expected"].sum() * 100)
        .reset_index()
        .rename(columns={0: "CE%", "month_offset": "Month"})
        .sort_values("Month")
    )
    ce_trend["Month Label"] = ce_trend["Month"].map({1: "Last Month", 2: "2M Ago", 3: "3M Ago"})

    fig_ce = go.Figure(go.Scatter(
        x=ce_trend["Month Label"][::-1].values,
        y=ce_trend["CE%"][::-1].values,
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in ce_trend["CE%"][::-1].values],
        textposition="top center",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=9),
    ))
    fig_ce.add_hline(y=95, line_dash="dash", line_color="#f59e0b",
                     annotation_text="Target 95%", annotation_position="bottom right")
    fig_ce.update_layout(
        margin=dict(t=10, b=20, l=10, r=10),
        height=170,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[80, 102]),
    )
    st.plotly_chart(fig_ce, use_container_width=True)

st.divider()

# ── Section 3: Segmentation ───────────────────────────────────────────────────
st.markdown('<p class="section-title">🔍 Portfolio Segmentation</p>', unsafe_allow_html=True)
seg1, seg2, seg3 = st.columns(3)

# NPA% by Loan Type
with seg1:
    st.caption("**NPA% by Loan Type**")
    npa_by_type = (
        df.groupby("loan_type")
        .apply(lambda x: x[x["dpd_bucket"] == "90+ DPD"]["outstanding_principal"].sum()
               / x["outstanding_principal"].sum() * 100
               if x["outstanding_principal"].sum() > 0 else 0)
        .reset_index()
        .rename(columns={0: "NPA%"})
        .sort_values("NPA%", ascending=True)
    )
    fig = px.bar(npa_by_type, x="NPA%", y="loan_type", orientation="h",
                 color="NPA%", color_continuous_scale="RdYlGn_r",
                 text=npa_by_type["NPA%"].apply(lambda x: f"{x:.1f}%"))
    fig.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False, showlegend=False)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# NPA% by Ticket Size
with seg2:
    st.caption("**NPA% by Ticket Size**")
    order = ["<2L", "2-10L", "10-50L", "50L+"]
    npa_by_ticket = (
        df.groupby("ticket_size")
        .apply(lambda x: x[x["dpd_bucket"] == "90+ DPD"]["outstanding_principal"].sum()
               / x["outstanding_principal"].sum() * 100
               if x["outstanding_principal"].sum() > 0 else 0)
        .reindex(order).fillna(0).reset_index()
        .rename(columns={0: "NPA%"})
    )
    fig = px.bar(npa_by_ticket, x="ticket_size", y="NPA%",
                 color="NPA%", color_continuous_scale="RdYlGn_r",
                 text=npa_by_ticket["NPA%"].apply(lambda x: f"{x:.1f}%"))
    fig.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False, xaxis_title="Ticket Size",
                      yaxis_title="NPA%")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# NPA% by Vintage
with seg3:
    st.caption("**NPA% by Vintage**")
    vint_order = ["0-6M", "6-12M", "12-24M", "24M+"]
    npa_by_vint = (
        df.groupby("vintage")
        .apply(lambda x: x[x["dpd_bucket"] == "90+ DPD"]["outstanding_principal"].sum()
               / x["outstanding_principal"].sum() * 100
               if x["outstanding_principal"].sum() > 0 else 0)
        .reindex(vint_order).fillna(0).reset_index()
        .rename(columns={0: "NPA%"})
    )
    fig = px.line(npa_by_vint, x="vintage", y="NPA%", markers=True,
                  text=npa_by_vint["NPA%"].apply(lambda x: f"{x:.1f}%"))
    fig.update_traces(line=dict(color="#ef4444", width=2.5),
                      marker=dict(size=9, color="#ef4444"), textposition="top center")
    fig.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 4: Early Warning System ──────────────────────────────────────────
st.markdown('<p class="section-title">🚨 Early Warning System – At-Risk Accounts</p>', unsafe_allow_html=True)
st.caption("Loans flagged before they become NPAs. Proactive intervention window.")

# Apply early-warning rules
at_risk = df[
    (df["dpd_bucket"].isin(["Current", "30 DPD"])) &
    (
        df["payment_delayed_5d"] |
        df["partial_payment"] |
        df["segment_high_default"]
    )
].copy()

at_risk["Risk Score"] = (
    at_risk["payment_delayed_5d"].astype(int) * 40 +
    at_risk["partial_payment"].astype(int) * 35 +
    at_risk["segment_high_default"].astype(int) * 25
)
at_risk = at_risk.sort_values("Risk Score", ascending=False)

# Summary callouts
c1, c2, c3, c4 = st.columns(4)
c1.metric("⚡ At-Risk Accounts",       f"{len(at_risk):,}")
c2.metric("💰 At-Risk Outstanding",    fmt_cr(at_risk["outstanding_principal"].sum()))
c3.metric("📌 Delayed Payments (5d+)", f"{at_risk['payment_delayed_5d'].sum():,}")
c4.metric("💸 Partial Payers",         f"{at_risk['partial_payment'].sum():,}")

st.markdown("")

# Color-code rows
def risk_color(row):
    if row["Risk Score"] >= 70:
        return ['background-color: #fef2f2; color: #1a1a1a; font-weight: 600'] * len(row)
    elif row["Risk Score"] >= 40:
        return ['background-color: #fffbeb; color: #1a1a1a; font-weight: 500'] * len(row)
    return ['background-color: #f0fdf4; color: #1a1a1a'] * len(row)

display_cols = [
    "loan_id", "loan_type", "ticket_size", "dpd_bucket",
    "outstanding_principal", "emi", "Risk Score",
    "payment_delayed_5d", "partial_payment", "segment_high_default"
]

st.dataframe(
    at_risk[display_cols]
    .rename(columns={
        "loan_id": "Loan ID",
        "loan_type": "Type",
        "ticket_size": "Ticket",
        "dpd_bucket": "DPD Bucket",
        "outstanding_principal": "Outstanding (₹)",
        "emi": "EMI (₹)",
        "payment_delayed_5d": "Delayed 5d+",
        "partial_payment": "Partial Pay",
        "segment_high_default": "Seg. High Default",
    })
    .head(50)
    .style.apply(risk_color, axis=1)
    .format({"Outstanding (₹)": "₹{:,.0f}", "EMI (₹)": "₹{:,.0f}"}),
    use_container_width=True,
    height=350,
)

st.caption("🔴 Risk Score ≥ 70 = Critical  |  🟡 40–69 = Watchlist  |  Showing top 50 accounts")

st.divider()

# ── Section 5: Predictive Flag Summary ───────────────────────────────────────
st.markdown('<p class="section-title">🔮 Predictive NPA Flags – Rule Engine</p>', unsafe_allow_html=True)

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    rules_df = pd.DataFrame({
        "Rule": [
            "Payment delayed 5+ days (last 2 months)",
            "Borrower made only partial EMI payment",
            "Segment historically has >10% default rate",
        ],
        "Accounts Flagged": [
            int(df["payment_delayed_5d"].sum()),
            int(df["partial_payment"].sum()),
            int(df["segment_high_default"].sum()),
        ],
        "Risk Weight": ["40 pts", "35 pts", "25 pts"],
    })
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

with pred_col2:
    # Pie – accounts by flag overlap
    only_delay    = df["payment_delayed_5d"] & ~df["partial_payment"] & ~df["segment_high_default"]
    only_partial  = ~df["payment_delayed_5d"] & df["partial_payment"] & ~df["segment_high_default"]
    multi_flag    = (df["payment_delayed_5d"].astype(int) + df["partial_payment"].astype(int) + df["segment_high_default"].astype(int)) >= 2
    clean         = (~df["payment_delayed_5d"]) & (~df["partial_payment"]) & (~df["segment_high_default"])

    fig_pie = go.Figure(go.Pie(
        labels=["Clean Portfolio", "Delay Only", "Partial Pay Only", "Multi-Flag (High Risk)"],
        values=[clean.sum(), only_delay.sum(), only_partial.sum(), multi_flag.sum()],
        hole=0.45,
        marker_colors=["#22c55e", "#f59e0b", "#f97316", "#ef4444"],
    ))
    fig_pie.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="v", font=dict(size=11)),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; font-size:0.8rem; padding:10px 0">
    🏦 Loan Portfolio Health Monitor &nbsp;|&nbsp; Built with Python + Streamlit + SQLite
    &nbsp;|&nbsp; Synthetic data for demonstration purposes
    </div>
    """,
    unsafe_allow_html=True,
)
