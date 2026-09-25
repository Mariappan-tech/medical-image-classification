"""
evaluation.py — Comprehensive model evaluation:
  - Accuracy, AUC, F1, MCC, Cohen Kappa
  - Confusion Matrix (heatmap)
  - ROC Curve + AUC
  - Precision-Recall Curve
  - Per-class metrics report
"""

import os, sys, json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.dataset import get_data_pipelines


# ─── Load Model ───────────────────────────────────────────────────────────────
def load_model(model_path=None):
    # Reuse the robust loader from prediction.py
    from model.prediction import load_model as _load
    return _load(model_path)


# ─── Predict ──────────────────────────────────────────────────────────────────
def predict_dataset(model, dataset):
    """Run inference on a tf.data.Dataset → returns (y_true, y_prob)."""
    y_prob, y_true = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_prob.extend(preds.flatten().tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_prob)


# ─── Threshold Optimization ───────────────────────────────────────────────────
def find_best_threshold(y_true, y_prob):
    """Find threshold that maximises Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    best_thr  = thresholds[best_idx]
    print(f"[INFO] Optimal threshold (Youden J): {best_thr:.4f}")
    return best_thr


# ─── Plotting Utilities ───────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual",    fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Confusion matrix saved → {path}")


def plot_roc_curve(y_true, y_prob, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#4F46E5",
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] ROC curve saved → {path}")
    return roc_auc


def plot_pr_curve(y_true, y_prob, save_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2, color="#10B981",
            label=f"PR Curve (AP = {ap:.4f})")
    ax.set_xlabel("Recall",    fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "pr_curve.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] PR curve saved → {path}")
    return ap


# ─── Full Evaluation ──────────────────────────────────────────────────────────
def evaluate(model_path=None):
    os.makedirs(config.LOG_DIR, exist_ok=True)

    model = load_model(model_path)
    _, _, test_ds, _, raw = get_data_pipelines()
    _, _, _, _, te_img, te_lbl = raw

    print("\n[INFO] Running inference on test set...")
    y_true, y_prob = predict_dataset(model, test_ds)

    # Optimal threshold
    best_thr = find_best_threshold(y_true, y_prob)
    y_pred   = (y_prob >= best_thr).astype(int)

    # Metrics
    roc_auc = plot_roc_curve(y_true, y_prob, config.LOG_DIR)
    ap      = plot_pr_curve(y_true, y_prob, config.LOG_DIR)
    plot_confusion_matrix(y_true, y_pred, config.LOG_DIR)

    f1  = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)

    report = classification_report(y_true, y_pred,
                                   target_names=config.CLASS_NAMES)
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(report)
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"  Kappa    : {kap:.4f}")
    print(f"  Threshold: {best_thr:.4f}")
    print("="*60)

    # Save results JSON
    results = {
        "roc_auc":   round(roc_auc, 4),
        "avg_prec":  round(ap,      4),
        "f1":        round(f1,      4),
        "mcc":       round(mcc,     4),
        "kappa":     round(kap,     4),
        "threshold": round(best_thr, 4),
    }
    res_path = os.path.join(config.LOG_DIR, "evaluation_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved → {res_path}")
    return results


if __name__ == "__main__":
    evaluate()
