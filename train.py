"""
train.py — Credit Card Fraud Detection
========================================
Dataset  : ULB Credit Card Fraud (creditcard.csv)
Models   : Random Forest  +  XGBoost  →  Stacking Ensemble
Resampling: SMOTE to handle extreme class imbalance (0.17% fraud)
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from utils import (
    load_data, print_class_distribution, evaluate_model,
    plot_confusion_matrix, plot_roc_curves, plot_feature_importance
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH    = "creditcard.csv"
MODEL_DIR    = "models"
PLOT_DIR     = "plots"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = load_data(DATA_PATH)

# Features and target
X = df.drop(columns=["Class"])
y = df["Class"]

feature_names = X.columns.tolist()


# ─────────────────────────────────────────────
# 2. SCALE Amount & Time
#    (V1-V28 are already PCA-scaled)
# ─────────────────────────────────────────────
print("\n⚙️  Scaling 'Amount' and 'Time' features...")
scaler = StandardScaler()
X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

# Save the scaler — needed later in predict.py
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("   ✅ Scaler saved → models/scaler.pkl")


# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT (stratified)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y          # preserves fraud ratio in both splits
)
print_class_distribution(y_train, y_test, label="(before SMOTE)")


# ─────────────────────────────────────────────
# 4. SMOTE — oversample the minority fraud class
#    Applied ONLY on training data, never on test
# ─────────────────────────────────────────────
print("\n🔄 Applying SMOTE to training data...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE → Legit: {(y_train_res==0).sum():,}  |  Fraud: {(y_train_res==1).sum():,}")


# ─────────────────────────────────────────────
# 5. DEFINE MODELS
# ─────────────────────────────────────────────
print("\n🌲 Configuring models...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=4,
    class_weight="balanced",   # extra safety on top of SMOTE
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train_res == 0).sum() / (y_train_res == 1).sum(),
    eval_metric="aucpr",       # area under precision-recall — best for imbalanced
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)

# Stacking: RF + XGB as base learners, Logistic Regression as meta-learner
stacking_model = StackingClassifier(
    estimators=[
        ("random_forest", rf_model),
        ("xgboost",       xgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1,
    passthrough=False
)


# ─────────────────────────────────────────────
# 6. TRAIN ALL MODELS
# ─────────────────────────────────────────────
models_to_train = {
    "Random Forest": rf_model,
    "XGBoost":       xgb_model,
    "Stacking (RF + XGB)": stacking_model
}

trained_results = {}   # stores predictions for ROC curve plotting

for name, model in models_to_train.items():
    print(f"\n🚀 Training: {name} ...")
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    evaluate_model(name, y_test, y_pred, y_prob)

    # Confusion matrix plot
    cm_path = os.path.join(PLOT_DIR, f"cm_{name.replace(' ', '_').replace('(','').replace(')','').replace('+','')}.png")
    plot_confusion_matrix(name, y_test, y_pred, save_path=cm_path)

    trained_results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob}


# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE (RF & XGB individually)
# ─────────────────────────────────────────────
for name in ["Random Forest", "XGBoost"]:
    fi_path = os.path.join(PLOT_DIR, f"feature_importance_{name.replace(' ', '_')}.png")
    plot_feature_importance(
        trained_results[name]["model"],
        feature_names,
        model_name=name,
        top_n=20,
        save_path=fi_path
    )


# ─────────────────────────────────────────────
# 8. ROC CURVES — all 3 models together
# ─────────────────────────────────────────────
roc_data = [
    {"name": name, "y_test": y_test, "y_prob": data["y_prob"]}
    for name, data in trained_results.items()
]
plot_roc_curves(roc_data, save_path=os.path.join(PLOT_DIR, "roc_curves.png"))


# ─────────────────────────────────────────────
# 9. CROSS-VALIDATION on best model (Stacking)
# ─────────────────────────────────────────────
print("\n🔁 5-Fold Cross Validation on Stacking Model (ROC-AUC)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(
    stacking_model, X_train_res, y_train_res,
    cv=cv, scoring="roc_auc", n_jobs=-1
)
print(f"   Fold Scores : {[round(s, 4) for s in cv_scores]}")
print(f"   Mean AUC    : {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")


# ─────────────────────────────────────────────
# 10. SAVE MODELS
# ─────────────────────────────────────────────
print("\n💾 Saving trained models...")
for name, data in trained_results.items():
    filename = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "") + ".pkl"
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(data["model"], path)
    print(f"   ✅ {name} → {path}")

print("\n🎉 Training complete! Models saved in /models, Plots saved in /plots")
