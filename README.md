# 💳 Credit Card Fraud Detection

A machine learning system that detects fraudulent credit card transactions using a **Random Forest + XGBoost Stacking Ensemble**, trained on real anonymized transaction data.

---

## 📌 Overview

Credit card fraud detection is a classic imbalanced classification problem — only **0.17%** of transactions are fraudulent. This project tackles that challenge using:

- **SMOTE** to synthetically balance the minority fraud class during training
- **Random Forest** for broad pattern detection across 200 decision trees
- **XGBoost** for sequential boosting on edge cases across 300 trees
- **Stacking Ensemble** that combines both models via a Logistic Regression meta-learner for the final verdict

The model operates at the **transaction level** — given behavioral features of a transaction, it outputs a fraud probability and a verdict (Legitimate / Fraud).

---

## 📊 Dataset

**ULB Credit Card Fraud Detection Dataset**
> Transactions made by European cardholders in September 2013.

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraud cases | 492 (0.173%) |
| Features | 30 (Time, Amount, V1–V28) |
| Target | Class (0 = Legit, 1 = Fraud) |

V1–V28 are PCA-transformed features (anonymized for privacy). `Time` and `Amount` are the only raw features.

### ⬇️ Download

Download `creditcard.csv` from Kaggle and place it in the project root:

🔗 [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> You will need a free Kaggle account to download the dataset.

---

## 🏗️ Project Structure

```
CardFraudDetection/
│
├── creditcard.csv          ← Dataset (download separately, not in repo)
├── my_transactions.csv     ← Sample transactions for prediction
│
├── train.py                ← Train all models & save to disk
├── predict.py              ← Load saved model & run inference
├── utils.py                ← Shared helpers (plots, evaluation, data loading)
├── requirements.txt        ← Python dependencies
│
├── models/                 ← Saved model files (generated after training)
│   ├── Random_Forest.pkl
│   ├── XGBoost.pkl
│   ├── Stacking_RF__XGB.pkl
│   └── scaler.pkl
│
└── plots/                  ← Generated evaluation plots
    ├── cm_Random_Forest.png
    ├── cm_XGBoost.png
    ├── cm_Stacking_RF__XGB.png
    ├── feature_importance_Random_Forest.png
    ├── feature_importance_XGBoost.png
    └── roc_curves.png
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CreditFraudDtectionModel.git
cd CreditFraudDtectionModel
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from the link above and place it in the project root.

---

## 🚀 Usage

### Train the model

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Apply SMOTE to balance the training data
- Train Random Forest, XGBoost, and the Stacking Ensemble
- Save all models to `models/`
- Save evaluation plots to `plots/`

> Training takes approximately 5–10 minutes depending on your hardware.

### Run predictions

**On built-in example transactions:**
```bash
python predict.py
```

**On your own CSV file:**
```bash
python predict.py --csv my_transactions.csv
```

Your CSV must have the same columns as `creditcard.csv` excluding the `Class` column:
```
Time, V1, V2, ..., V28, Amount
```

**Sample output:**
```
============================================================
  #         Amount    Fraud Prob  Verdict
============================================================
  1     ₹     0.00       97.14%  🚨 FRAUD
  2     ₹   529.00        0.01%  ✅ Legitimate
  3     ₹   239.93       99.98%  🚨 FRAUD
============================================================
```

---

## 📈 Model Performance

| Model | Fraud Precision | Fraud Recall | ROC-AUC |
|---|---|---|---|
| Random Forest | 0.84 | 0.83 | 0.9752 |
| XGBoost | 0.52 | 0.88 | 0.9801 |
| **Stacking (RF + XGB)** | **0.79** | **0.86** | **0.9794** |

> Evaluated on a held-out test set of 56,962 transactions (20% of dataset), never seen during training.

**Key metrics explained:**
- **Precision** — of all transactions flagged as fraud, how many actually were
- **Recall** — of all real fraud cases, how many were caught
- **ROC-AUC** — overall ability to distinguish fraud from legit (1.0 = perfect)

---

## 🧠 How It Works

```
Transaction occurs
        ↓
Features extracted (Time, Amount, V1–V28)
        ↓
Amount & Time scaled using saved StandardScaler
        ↓
Fed into Stacking Ensemble
    ├── Random Forest  →  P(fraud) = 0.82
    └── XGBoost        →  P(fraud) = 0.91
        ↓
Logistic Regression meta-learner weighs both predictions
        ↓
Final verdict: 🚨 FRAUD (94% confidence)
```

This is how real-world bank fraud systems operate — the model runs invisibly the moment a transaction occurs, without requiring any card credentials from the user.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `scikit-learn` | Random Forest, Stacking, preprocessing, evaluation |
| `xgboost` | Gradient boosted trees |
| `imbalanced-learn` | SMOTE for class balancing |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Evaluation plots |
| `joblib` | Model serialization |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
