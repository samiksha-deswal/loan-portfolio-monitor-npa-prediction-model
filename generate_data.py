import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3
import os

random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_LOANS = 5000
TODAY = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

LOAN_TYPES = ["Personal", "Home", "Auto", "Business"]
LOAN_TYPE_WEIGHTS = [0.40, 0.25, 0.20, 0.15]

RATE_RANGES = {
    "Personal":  (10.5, 24.0),
    "Home":      (7.5,  12.0),
    "Auto":      (8.0,  15.0),
    "Business":  (12.0, 22.0),
}

AMOUNT_RANGES = {
    "Personal":  (50_000,    1_000_000),
    "Home":      (1_000_000, 10_000_000),
    "Auto":      (200_000,   2_000_000),
    "Business":  (500_000,   5_000_000),
}

TENURE_MONTHS = {
    "Personal":  [12, 24, 36, 48, 60],
    "Home":      [120, 180, 240, 300],
    "Auto":      [36, 48, 60, 72, 84],
    "Business":  [24, 36, 48, 60],
}


def emi_calc(p, r_annual, n):
    r = r_annual / 12 / 100
    if r == 0:
        return p / n
    return p * r * (1 + r) ** n / ((1 + r) ** n - 1)


def generate_loans():
    loans = []
    for i in range(1, N_LOANS + 1):
        loan_type = np.random.choice(LOAN_TYPES, p=LOAN_TYPE_WEIGHTS)
        amount    = round(np.random.uniform(*AMOUNT_RANGES[loan_type]), -3)
        rate      = round(np.random.uniform(*RATE_RANGES[loan_type]), 2)
        tenure    = random.choice(TENURE_MONTHS[loan_type])

        # Disbursement between 4 years ago and 6 months ago
        days_ago     = random.randint(180, 4 * 365)
        disburse_dt  = TODAY - timedelta(days=days_ago)
        months_active = min(
            tenure,
            (TODAY.year - disburse_dt.year) * 12 + (TODAY.month - disburse_dt.month)
        )

        emi            = emi_calc(amount, rate, tenure)
        outstanding    = amount  # will adjust below

        # DPD assignment with realistic distribution
        dpd_bucket = np.random.choice(
            ["Current", "30 DPD", "60 DPD", "90+ DPD", "Closed"],
            p=[0.72, 0.10, 0.06, 0.07, 0.05],
        )

        if dpd_bucket == "Closed":
            months_active = tenure

        # Outstanding principal (rough)
        paid_months = months_active if dpd_bucket != "Closed" else tenure
        outstanding = max(0, amount - (amount / tenure) * paid_months)

        # 5-day delayed flag and partial payment flag (for early warning)
        payment_delayed_5d = (
            dpd_bucket in ["30 DPD", "60 DPD"]
            or (dpd_bucket == "Current" and random.random() < 0.12)
        )
        partial_payment = (
            dpd_bucket in ["30 DPD", "60 DPD"]
            or (dpd_bucket == "Current" and random.random() < 0.08)
        )

        # Segment high-default rate flag
        segment_high_default = loan_type == "Personal" and amount > 500_000

        # Vintage bucket
        if months_active <= 6:
            vintage = "0-6M"
        elif months_active <= 12:
            vintage = "6-12M"
        elif months_active <= 24:
            vintage = "12-24M"
        else:
            vintage = "24M+"

        # Ticket size bucket
        if amount < 200_000:
            ticket_size = "<2L"
        elif amount < 1_000_000:
            ticket_size = "2-10L"
        elif amount < 5_000_000:
            ticket_size = "10-50L"
        else:
            ticket_size = "50L+"

        loans.append({
            "loan_id":               f"LN{i:05d}",
            "borrower_id":           f"BR{random.randint(10000, 99999)}",
            "loan_type":             loan_type,
            "disbursement_date":     disburse_dt.date(),
            "loan_amount":           amount,
            "interest_rate":         rate,
            "tenure_months":         tenure,
            "emi":                   round(emi, 2),
            "outstanding_principal": round(outstanding, 2),
            "dpd_bucket":            dpd_bucket,
            "months_active":         months_active,
            "vintage":               vintage,
            "ticket_size":           ticket_size,
            "payment_delayed_5d":    payment_delayed_5d,
            "partial_payment":       partial_payment,
            "segment_high_default":  segment_high_default,
        })

    return pd.DataFrame(loans)


def generate_monthly_payments(loans_df):
    """Generate last-3-months payment records for collection efficiency."""
    records = []
    for _, row in loans_df.iterrows():
        if row["dpd_bucket"] == "Closed":
            continue
        for m in range(1, 4):  # last 3 months
            expected = row["emi"]
            if row["dpd_bucket"] == "Current":
                collected = expected * random.uniform(0.98, 1.0)
            elif row["dpd_bucket"] == "30 DPD":
                collected = expected * random.uniform(0.50, 0.90) if random.random() < 0.6 else 0
            elif row["dpd_bucket"] == "60 DPD":
                collected = expected * random.uniform(0.0, 0.50) if random.random() < 0.4 else 0
            else:  # 90+ DPD
                collected = 0 if random.random() < 0.85 else expected * random.uniform(0, 0.3)

            records.append({
                "loan_id":    row["loan_id"],
                "month_offset": m,          # 1 = last month
                "expected":   round(expected, 2),
                "collected":  round(collected, 2),
            })
    return pd.DataFrame(records)


def save_to_sqlite(loans_df, payments_df, db_path="loan_data.db"):
    conn = sqlite3.connect(db_path)
    loans_df.to_sql("loans", conn, if_exists="replace", index=False)
    payments_df.to_sql("monthly_payments", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Saved to {db_path}")


if __name__ == "__main__":
    print("Generating loan data...")
    loans_df    = generate_loans()
    payments_df = generate_monthly_payments(loans_df)

    os.makedirs("data", exist_ok=True)
    loans_df.to_csv("data/loans.csv", index=False)
    payments_df.to_csv("data/monthly_payments.csv", index=False)
    save_to_sqlite(loans_df, payments_df, "data/loan_data.db")

    print(f"  Loans:    {len(loans_df):,}")
    print(f"  Payments: {len(payments_df):,}")
    print(f"  DPD distribution:\n{loans_df['dpd_bucket'].value_counts()}")
