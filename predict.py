"""
predict.py — Credit Card Fraud Detection Inference
====================================================
Loads a saved model and predicts whether a transaction is
Fraudulent or Legitimate.

Usage:
    python predict.py                  ← runs on built-in example transactions
    python predict.py --csv sample.csv ← runs on a CSV of transactions
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_DIR  = "models"
MODEL_FILE = "Stacking_RF__XGB.pkl"    # Best model (change to Random_Forest.pkl or XGBoost.pkl if preferred)
SCALER_FILE = "scaler.pkl"

FEATURE_COLS = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)


# ─────────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────────
def load_artifacts():
    model_path  = os.path.join(MODEL_DIR, MODEL_FILE)
    scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please run  python train.py  first.")
        sys.exit(1)

    print(f"📦 Loading model  : {model_path}")
    model = joblib.load(model_path)

    print(f"📦 Loading scaler : {scaler_path}")
    scaler = joblib.load(scaler_path)

    return model, scaler


# ─────────────────────────────────────────────
# PREPROCESS INPUT
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Apply the same scaling used during training."""
    df = df.copy()

    # Ensure correct column order
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    df = df[FEATURE_COLS]
    df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])
    return df


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(model, scaler, df_raw: pd.DataFrame):
    """Run fraud prediction and print results."""
    df_processed = preprocess(df_raw, scaler)

    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)[:, 1]

    print("\n" + "="*60)
    print(f"  {'#':<5} {'Amount':>10}  {'Fraud Prob':>12}  {'Verdict'}")
    print("="*60)

    for i, (pred, prob, amount) in enumerate(zip(predictions, probabilities, df_raw["Amount"])):
        verdict = "🚨 FRAUD" if pred == 1 else "✅ Legitimate"
        print(f"  {i+1:<5} ₹{amount:>9.2f}  {prob*100:>10.2f}%  {verdict}")

    print("="*60)

    fraud_count = predictions.sum()
    print(f"\n  Total Transactions : {len(predictions)}")
    print(f"  Flagged as Fraud   : {fraud_count}")
    print(f"  Legitimate         : {len(predictions) - fraud_count}\n")

    return predictions, probabilities


# ─────────────────────────────────────────────
# EXAMPLE TRANSACTIONS (for demo/testing)
# These are real rows from the ULB test set.
# You can replace with your own rows.
# ─────────────────────────────────────────────
def get_example_transactions() -> pd.DataFrame:
    """
    Returns a small DataFrame of example transactions.
    Row 0: Likely Legitimate
    Row 1: Likely Fraudulent (high-risk profile)
    """
    # Legitimate-looking transaction
    legit = {
        "Time": 406.0, "V1": -0.269, "V2": 0.174, "V3": 0.483,
        "V4": 0.328, "V5": -0.167, "V6": -0.178, "V7": -0.083,
        "V8": 0.059, "V9": -0.233, "V10": -0.140, "V11": -0.197,
        "V12": 0.164, "V13": -0.117, "V14": -0.256, "V15": 0.051,
        "V16": -0.036, "V17": -0.006, "V18": -0.032, "V19": 0.117,
        "V20": 0.012, "V21": -0.014, "V22": -0.019, "V23": 0.041,
        "V24": 0.395, "V25": 0.068, "V26": -0.033, "V27": 0.006,
        "V28": 0.013, "Amount": 149.62
    }

    # High-risk / fraudulent-profile transaction
    fraud = {
        "Time": 406.0, "V1": -3.043, "V2": -3.157, "V3": 1.088,
        "V4": 2.288, "V5": 1.359, "V6": -1.064, "V7": 0.325,
        "V8": -0.068, "V9": -0.267, "V10": -0.835, "V11": -0.414,
        "V12": -2.659, "V13": -0.159, "V14": -1.998, "V15": -0.504,
        "V16": -0.576, "V17": -1.660, "V18": -0.630, "V19": 0.106,
        "V20": 0.247, "V21": 0.771, "V22": 0.909, "V23": -0.689,
        "V24": -0.328, "V25": -0.139, "V26": -0.055, "V27": -0.059,
        "V28": 0.034, "Amount": 1.00
    }

    return pd.DataFrame([legit, fraud])


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Predictor")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to a CSV file of transactions to predict (optional)"
    )
    args = parser.parse_args()

    model, scaler = load_artifacts()

    if args.csv:
        print(f"\n📄 Loading transactions from: {args.csv}")
        df_input = pd.read_csv(args.csv)

        # Drop 'Class' column if present (in case using dataset rows directly)
        if "Class" in df_input.columns:
            df_input = df_input.drop(columns=["Class"])
    else:
        print("\n🧪 No CSV provided — running on built-in example transactions")
        df_input = get_example_transactions()

    predict(model, scaler, df_input)