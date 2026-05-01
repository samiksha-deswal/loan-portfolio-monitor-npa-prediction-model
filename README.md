# 🏦 Loan Portfolio Health Monitor

> **BA Portfolio Project** – Executive dashboard with early warning system for NPA prevention  
> **Stack:** Python · Streamlit · SQLite · Plotly

---

## 🚀 Quick Start

```bash
# 1. Clone / place files in a folder
cd loan_monitor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data (5,000 loans)
python generate_data.py

# 4. Run the dashboard
streamlit run app.py
```

The app auto-generates data on first launch if `data/loan_data.db` is missing.

---

## 📁 Project Structure

```
loan_monitor/
├── app.py              ← Main Streamlit dashboard
├── generate_data.py    ← Synthetic data generator
├── requirements.txt    ← Python dependencies
├── data/
│   ├── loans.csv
│   ├── monthly_payments.csv
│   └── loan_data.db    ← SQLite database (auto-created)
└── README.md
```

---

## 📊 Dashboard Sections

| Section | What It Shows |
|---|---|
| **1. Overview KPIs** | AUM, Gross NPA%, Net NPA%, Portfolio Yield, Collection Efficiency |
| **2. Delinquency Funnel** | Flow from Current → 30 DPD → 60 DPD → 90+ DPD with roll rates |
| **3. Segmentation** | NPA% by loan type, ticket size, and vintage |
| **4. Early Warning Alerts** | At-risk accounts before they become NPAs, with risk scores |
| **5. Predictive Flag Engine** | Rule-based flagging: delayed payments, partial EMIs, segment risk |

---

## 🧮 Key Metrics Explained

| Metric | Formula | RBI Benchmark |
|---|---|---|
| Gross NPA% | NPA Outstanding / Total AUM | < 5% |
| Net NPA% | (NPA − Provisions) / Net Assets | < 2% |
| Portfolio Yield | Weighted avg interest rate | Target > 12% |
| Collection Efficiency | Collected / Expected × 100 | > 95% |

---

## ⚡ Early Warning Rules

Three rule-based flags with weighted risk scores:

1. **Payment Delayed 5+ Days** (last 2 months) → 40 pts
2. **Partial EMI Payment** (not full EMI) → 35 pts
3. **Segment with High Historical Default** → 25 pts

Accounts scoring ≥ 70 = **Critical** (red)  
Accounts scoring 40–69 = **Watchlist** (amber)

---

*Synthetic data — for demonstration purposes only*
