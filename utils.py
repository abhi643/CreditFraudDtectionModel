import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)


def load_data(filepath: str) -> pd.DataFrame:
    """Load the ULB credit card fraud CSV dataset."""
    print(f"📂 Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} rows and {df.shape[1]} columns")
    print(f"   Fraud cases    : {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"   Legit cases    : {(df['Class'] == 0).sum():,}")
    return df


def print_class_distribution(y_train, y_test, label=""):
    """Print class distribution for train/test splits."""
    print(f"\n📊 Class Distribution {label}")
    print(f"   Train → Legit: {(y_train==0).sum():,}  |  Fraud: {(y_train==1).sum():,}")
    print(f"   Test  → Legit: {(y_test==0).sum():,}  |  Fraud: {(y_test==1).sum():,}")


def evaluate_model(name: str, y_test, y_pred, y_prob):
    """Print evaluation metrics for a model."""
    print(f"\n{'='*55}")
    print(f"  📈 Model Evaluation — {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"  ROC-AUC Score        : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Avg Precision Score  : {average_precision_score(y_test, y_prob):.4f}")


def plot_confusion_matrix(name: str, y_test, y_pred, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"]
    )
    plt.title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"   💾 Saved: {save_path}")
    plt.show()


def plot_roc_curves(models_data: list, save_path=None):
    """
    Plot ROC curves for multiple models.
    models_data: list of dicts with keys: name, y_test, y_prob
    """
    plt.figure(figsize=(8, 6))
    for m in models_data:
        fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
        auc = roc_auc_score(m["y_test"], m["y_prob"])
        plt.plot(fpr, tpr, label=f"{m['name']} (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Fraud Detection Models", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"   💾 Saved: {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names: list, model_name: str, top_n=20, save_path=None):
    """Plot top N feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    plt.figure(figsize=(9, 6))
    top.plot(kind="barh", color="steelblue")
    plt.title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"   💾 Saved: {save_path}")
    plt.show()
