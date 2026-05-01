"""
Loan Portfolio Data Generator — v2 (No Label Leakage)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, sqlite3, os

random.seed(42)
np.random.seed(42)

N_LOANS = 5000
TODAY   = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

LOAN_TYPES        = ["Personal", "Home", "Auto", "Business"]
LOAN_TYPE_WEIGHTS = [0.40, 0.25, 0.20, 0.15]
RATE_RANGES   = {"Personal":(10.5,24.0),"Home":(7.5,12.0),"Auto":(8.0,15.0),"Business":(12.0,22.0)}
AMOUNT_RANGES = {"Personal":(50_000,800_000),"Home":(1_000_000,10_000_000),"Auto":(200_000,2_000_000),"Business":(500_000,5_000_000)}
TENURE_MONTHS = {"Personal":[12,24,36,48,60],"Home":[120,180,240,300],"Auto":[36,48,60,72,84],"Business":[24,36,48,60]}

def emi_calc(p, r, n):
    r = r/12/100
    return p*r*(1+r)**n/((1+r)**n-1) if r else p/n

def base_risk(loan_type, amount, rate, tenure, months_active):
    risk = 0.05
    risk += {"Personal":0.08,"Business":0.06,"Auto":0.03,"Home":0.01}[loan_type]
    risk += max(0,(rate-12)/100)
    if loan_type=="Personal" and amount<200_000: risk+=0.05
    if months_active<=6: risk+=0.04
    if loan_type=="Home" and tenure>=180: risk-=0.03
    return float(np.clip(risk,0.02,0.45))

def simulate_payments(emi, risk, months, loan_id):
    rng = np.random.default_rng(abs(hash(loan_id))%2**31)
    records=[]
    for m in range(1, min(months,6)+1):
        r=rng.random()
        if r < risk*1.5:
            collected=0.0; delayed=int(rng.integers(15,90))
        elif r < risk*2.5:
            collected=emi*rng.uniform(0.3,0.9); delayed=int(rng.integers(5,30))
        elif r < risk*3.5:
            collected=emi; delayed=int(rng.integers(1,15))
        else:
            collected=emi*rng.uniform(0.99,1.0); delayed=0
        records.append({"month_offset":m,"expected":round(emi,2),"collected":round(collected,2),"delayed_days":delayed})
    return records

def dpd_from_history(history):
    if not history: return "Current"
    last=[r for r in history if r["month_offset"]==1]
    if not last: return "Current"
    d=last[0]["delayed_days"]; ratio=last[0]["collected"]/(last[0]["expected"]+1e-6)
    if d>=90 or (d>=60 and ratio<0.2): return "90+ DPD"
    if d>=60 or (d>=30 and ratio<0.5): return "60 DPD"
    if d>=30 or (d>=5 and ratio<0.7):  return "30 DPD"
    return "Current"

def generate():
    loans,payments=[],[]
    for i in range(1,N_LOANS+1):
        lt=np.random.choice(LOAN_TYPES,p=LOAN_TYPE_WEIGHTS)
        amt=round(np.random.uniform(*AMOUNT_RANGES[lt]),-3)
        rate=round(np.random.uniform(*RATE_RANGES[lt]),2)
        tenure=random.choice(TENURE_MONTHS[lt])
        days_ago=random.randint(90,4*365)
        disburse=TODAY-timedelta(days=days_ago)
        months_active=min(tenure,(TODAY.year-disburse.year)*12+(TODAY.month-disburse.month))
        emi=emi_calc(amt,rate,tenure)
        lid=f"LN{i:05d}"
        risk=base_risk(lt,amt,rate,tenure,months_active)
        if random.random()<0.05:
            dpd="Closed"; hist=[]
        else:
            hist=simulate_payments(emi,risk,months_active,lid)
            dpd=dpd_from_history(hist)
        outstanding=round(amt*(1-min(1.0,months_active/tenure)*0.85),2)
        all_delayed=[r["delayed_days"] for r in hist]
        all_ratios=[r["collected"]/(r["expected"]+1e-6) for r in hist]
        if months_active<=6:    vintage="0-6M"
        elif months_active<=12: vintage="6-12M"
        elif months_active<=24: vintage="12-24M"
        else:                   vintage="24M+"
        if amt<200_000:       ticket="<2L"
        elif amt<1_000_000:   ticket="2-10L"
        elif amt<5_000_000:   ticket="10-50L"
        else:                 ticket="50L+"
        loans.append({"loan_id":lid,"borrower_id":f"BR{random.randint(10000,99999)}",
            "loan_type":lt,"disbursement_date":disburse.date(),"loan_amount":amt,
            "interest_rate":rate,"tenure_months":tenure,"emi":round(emi,2),
            "outstanding_principal":outstanding,"dpd_bucket":dpd,"months_active":months_active,
            "base_risk_score":round(risk,4),"vintage":vintage,"ticket_size":ticket,
            "payment_delayed_5d":any(d>=5 for d in all_delayed),
            "partial_payment":any(0.05<r<0.95 for r in all_ratios),
            "segment_high_default":lt=="Personal" and rate>18})
        for r in hist:
            payments.append({"loan_id":lid,**r})
    return pd.DataFrame(loans),pd.DataFrame(payments)

if __name__=="__main__":
    os.makedirs("data",exist_ok=True)
    print("Generating loan data...")
    l,p=generate()
    l.to_csv("data/loans.csv",index=False)
    p.to_csv("data/monthly_payments.csv",index=False)
    conn=sqlite3.connect("data/loan_data.db")
    l.to_sql("loans",conn,if_exists="replace",index=False)
    p.to_sql("monthly_payments",conn,if_exists="replace",index=False)
    conn.close()
    print(f"  Loans: {len(l):,} | Payments: {len(p):,}")
    print(l["dpd_bucket"].value_counts().to_string())
    print("✅ Done")
