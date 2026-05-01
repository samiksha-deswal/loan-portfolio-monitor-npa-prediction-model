"""
NPA Prediction Model — Training Pipeline
=========================================
Models trained: Logistic Regression, Random Forest, XGBoost
Technique:      SMOTE for class imbalance | Cross-validation | SHAP explainability
Output:         models/  directory with saved models + feature importance
"""

import pandas as pd
import numpy as np
import sqlite3, os, pickle, json, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ── ML Stack ─────────────────────────────────────────────────────────────────
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics         import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.calibration     import CalibratedClassifierCV
from imblearn.over_sampling  import SMOTE
import xgboost as xgb
import shap

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH     = "data/loan_data.db"
MODELS_DIR  = "models"
RANDOM_SEED = 42
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    conn     = sqlite3.connect(DB_PATH)
    loans    = pd.read_sql("SELECT * FROM loans",            conn)
    payments = pd.read_sql("SELECT * FROM monthly_payments", conn)
    conn.close()
    return loans, payments

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(loans, payments):
    """
    Create 20+ features a real credit risk team would use.
    Target: 1 = NPA (90+ DPD), 0 = Not NPA
    """
    df = loans.copy()
    df = df[df["dpd_bucket"] != "Closed"].reset_index(drop=True)

    # ── Target variable ────────────────────────────────────────────────────
    df["is_npa"] = (df["dpd_bucket"] == "90+ DPD").astype(int)

    # ── Payment behaviour features ─────────────────────────────────────────
    # Collection rate per loan across 3 months
    pay_agg = (
        payments.groupby("loan_id")
        .apply(lambda x: pd.Series({
            "total_collected":    x["collected"].sum(),
            "total_expected":     x["expected"].sum(),
            "months_zero_pay":    (x["collected"] == 0).sum(),
            "months_partial_pay": ((x["collected"] > 0) & (x["collected"] < x["expected"])).sum(),
            "last_month_collected": x[x["month_offset"] == 1]["collected"].values[0]
                                    if len(x[x["month_offset"] == 1]) > 0 else 0,
            "last_month_expected":  x[x["month_offset"] == 1]["expected"].values[0]
                                    if len(x[x["month_offset"] == 1]) > 0 else 0,
        }))
        .reset_index()
    )
    df = df.merge(pay_agg, on="loan_id", how="left")
    df["collection_rate"]      = np.where(df["total_expected"] > 0,
                                           df["total_collected"] / df["total_expected"], 1.0)
    df["last_month_pay_ratio"] = np.where(df["last_month_expected"] > 0,
                                           df["last_month_collected"] / df["last_month_expected"], 1.0)

    # ── Loan financial features ────────────────────────────────────────────
    df["emi_to_outstanding_ratio"] = df["emi"] / (df["outstanding_principal"] + 1)
    df["loan_utilisation_pct"]     = 1 - (df["outstanding_principal"] / (df["loan_amount"] + 1))
    df["emi_burden_pct"]           = df["emi"] / (df["loan_amount"] / df["tenure_months"] + 1)
    df["interest_rate_band"]       = pd.cut(df["interest_rate"],
                                             bins=[0, 10, 14, 18, 100],
                                             labels=[0, 1, 2, 3]).astype(int)
    df["high_rate_flag"]           = (df["interest_rate"] > 18).astype(int)
    df["large_ticket_flag"]        = (df["loan_amount"] > 1_000_000).astype(int)
    df["long_tenure_flag"]         = (df["tenure_months"] > 60).astype(int)

    # ── Vintage / seasoning ────────────────────────────────────────────────
    df["seasoning_ratio"]          = df["months_active"] / df["tenure_months"]
    df["early_stage"]              = (df["months_active"] <= 6).astype(int)   # Highest risk
    df["mid_stage"]                = ((df["months_active"] > 6) & (df["months_active"] <= 24)).astype(int)

    # ── Early warning flags ────────────────────────────────────────────────
    df["payment_delayed_5d"]       = df["payment_delayed_5d"].astype(int)
    df["partial_payment"]          = df["partial_payment"].astype(int)
    df["segment_high_default"]     = df["segment_high_default"].astype(int)
    df["warning_score"]            = (
        df["payment_delayed_5d"] * 40 +
        df["partial_payment"]    * 35 +
        df["segment_high_default"] * 25
    )

    # ── Encode categoricals ────────────────────────────────────────────────
    loan_type_dummies = pd.get_dummies(df["loan_type"], prefix="type")
    df = pd.concat([df, loan_type_dummies], axis=1)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPARE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Payment behaviour
    "collection_rate", "last_month_pay_ratio",
    "months_zero_pay", "months_partial_pay",
    # Loan financials
    "loan_amount", "interest_rate", "tenure_months", "emi",
    "outstanding_principal", "emi_to_outstanding_ratio",
    "loan_utilisation_pct", "interest_rate_band", "high_rate_flag",
    "large_ticket_flag", "long_tenure_flag",
    # Vintage / seasoning
    "months_active", "seasoning_ratio", "early_stage", "mid_stage",
    # Early warning flags
    "payment_delayed_5d", "partial_payment",
    "segment_high_default", "warning_score",
    # Loan type dummies
    "type_Auto", "type_Business", "type_Home", "type_Personal",
]

def prepare_xy(df):
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["is_npa"]
    return X, y, feature_cols

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
def train_all(X_train, y_train):
    """Train 3 models with SMOTE-balanced data."""
    print(f"  Class balance before SMOTE: {y_train.value_counts().to_dict()}")
    sm = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  Class balance after SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.5, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_res == 0).sum() / (y_res == 1).sum(),
            use_label_encoder=False, eval_metric="auc",
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
        ),
    }

    trained = {}
    for name, model in models.items():
        X_fit = X_scaled if name == "Logistic Regression" else X_res
        y_fit = y_res
        model.fit(X_fit, y_fit)
        trained[name] = model
        print(f"  ✅ {name} trained")

    return trained, scaler, X_res, y_res

# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(models, scaler, X_test, y_test, feature_cols):
    results = {}
    for name, model in models.items():
        X_eval = scaler.transform(X_test) if name == "Logistic Regression" else X_test
        proba  = model.predict_proba(X_eval)[:, 1]
        pred   = (proba >= 0.5).astype(int)

        auc    = roc_auc_score(y_test, proba)
        ap     = average_precision_score(y_test, proba)
        cm     = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()

        fpr, tpr, roc_thresh = roc_curve(y_test, proba)
        prec, rec, pr_thresh = precision_recall_curve(y_test, proba)

        report = classification_report(y_test, pred, output_dict=True)

        results[name] = {
            "auc":        round(auc, 4),
            "avg_prec":   round(ap,  4),
            "precision":  round(report["1"]["precision"], 4),
            "recall":     round(report["1"]["recall"],    4),
            "f1":         round(report["1"]["f1-score"],  4),
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "prec_curve": prec.tolist(), "rec_curve": rec.tolist(),
            "proba": proba.tolist(),
            "y_test": y_test.tolist(),
        }

        print(f"  {name:25s} | AUC={auc:.3f} | Precision={report['1']['precision']:.3f} | Recall={report['1']['recall']:.3f}")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# 6. SHAP EXPLAINABILITY (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
def compute_shap(xgb_model, X_test, feature_cols):
    print("  Computing SHAP values...")
    explainer  = shap.TreeExplainer(xgb_model)
    shap_vals  = explainer.shap_values(X_test)
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": mean_abs
    }).sort_values("importance", ascending=False)
    return importance, shap_vals, explainer.expected_value

# ─────────────────────────────────────────────────────────────────────────────
# 7. RISK SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def build_risk_segments(df_full, xgb_model, scaler, feature_cols):
    """Assign risk tiers to full portfolio."""
    X_all = df_full[feature_cols].fillna(0)
    proba = xgb_model.predict_proba(X_all)[:, 1]
    df_full = df_full.copy()
    df_full["npa_probability"] = proba
    df_full["risk_tier"] = pd.cut(
        proba,
        bins=[-0.001, 0.20, 0.40, 0.60, 0.80, 1.001],
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    return df_full

# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 NPA Prediction Model — Training Pipeline")
    print("=" * 55)

    # Load & engineer
    print("\n[1/6] Loading data & engineering features...")
    loans, payments = load_data()
    df = engineer_features(loans, payments)
    X, y, feature_cols = prepare_xy(df)
    print(f"  Dataset: {len(df):,} loans | {X.shape[1]} features | {y.mean()*100:.1f}% NPA rate")

    # Split
    print("\n[2/6] Splitting train/test (80/20 stratified)...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # Train
    print("\n[3/6] Training models with SMOTE balancing...")
    models, scaler, X_res, y_res = train_all(X_train, y_train)

    # Evaluate
    print("\n[4/6] Evaluating on held-out test set...")
    results = evaluate(models, scaler, X_test, y_test, feature_cols)

    # SHAP
    print("\n[5/6] SHAP explainability (XGBoost)...")
    shap_imp, shap_vals, expected_val = compute_shap(
        models["XGBoost"], X_test, feature_cols
    )
    print(f"  Top 5 features:\n{shap_imp.head(5).to_string(index=False)}")

    # Risk tiers on full portfolio
    print("\n[6/6] Scoring full portfolio & building risk tiers...")
    df_scored = build_risk_segments(df, models["XGBoost"], scaler, feature_cols)
    print(f"\n  Risk Tier Distribution:")
    print(df_scored["risk_tier"].value_counts().sort_index().to_string())

    # ── Save everything ───────────────────────────────────────────────────
    # Models
    for name, model in models.items():
        fname = name.lower().replace(" ", "_")
        with open(f"{MODELS_DIR}/{fname}.pkl", "wb") as f:
            pickle.dump(model, f)

    with open(f"{MODELS_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Metadata
    meta = {
        "feature_cols":    feature_cols,
        "trained_at":      datetime.now().isoformat(),
        "n_train":         int(len(X_train)),
        "n_test":          int(len(X_test)),
        "npa_rate":        float(y.mean()),
        "best_model":      max(results, key=lambda k: results[k]["auc"]),
        "results":         results,
        "shap_importance": shap_imp.to_dict(orient="records"),
        "expected_val":    float(expected_val),
    }
    with open(f"{MODELS_DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Scored portfolio
    scored_cols = ["loan_id", "loan_type", "ticket_size", "dpd_bucket",
                   "outstanding_principal", "interest_rate", "months_active",
                   "npa_probability", "risk_tier",
                   "payment_delayed_5d", "partial_payment", "warning_score"]
    df_scored[scored_cols].to_csv("data/scored_portfolio.csv", index=False)

    # SHAP values for dashboard
    shap_df = pd.DataFrame(shap_vals, columns=feature_cols)
    shap_df.to_csv("data/shap_values.csv", index=False)
    shap_imp.to_csv("data/shap_importance.csv", index=False)

    print("\n✅ All models saved to models/")
    print("✅ Scored portfolio → data/scored_portfolio.csv")
    print("✅ SHAP importance  → data/shap_importance.csv")

    best = meta["best_model"]
    r    = results[best]
    print(f"\n🏆 Best Model: {best}")
    print(f"   AUC:       {r['auc']}")
    print(f"   Precision: {r['precision']}")
    print(f"   Recall:    {r['recall']}")
    print(f"   F1:        {r['f1']}")
    print("\n🎉 Training complete!\n")
