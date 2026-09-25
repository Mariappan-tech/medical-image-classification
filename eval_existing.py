"""
eval_existing.py — Evaluate the existing saved checkpoint weights
by rebuilding the model architecture from model_builder.py,
loading weights, then running the full evaluation.
No model files are modified.
"""

import os, sys, json, time
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, matthews_corrcoef,
    cohen_kappa_score, f1_score, accuracy_score,
)

from model.dataset       import get_data_pipelines
from model.model_builder import get_model

os.makedirs(config.LOG_DIR, exist_ok=True)

BANNER = "=" * 65
CKPT   = os.path.join(config.MODEL_DIR, "best_efficientnetb3_phase1.keras")


def load_model_from_weights():
    """Rebuild architecture + load weights from checkpoint."""
    print(f"[INFO] Rebuilding model architecture ({config.MODEL_NAME}) ...")
    model, _ = get_model()

    # Compile so that weight shapes are fully initialised
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )

    print(f"[INFO] Loading weights from: {CKPT}")
    model.load_weights(CKPT)
    print("[INFO] Weights loaded successfully.")
    return model


def run_evaluation(model):
    print(f"\n{BANNER}")
    print("  EVALUATION — Full Metrics on Test Set")
    print(BANNER)

    _, _, test_ds, _, _ = get_data_pipelines()

    print("\n[INFO] Running inference on test set...")
    t0 = time.perf_counter()
    y_prob_list, y_true_list = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_prob_list.extend(preds.flatten().tolist())
        y_true_list.extend(labels.numpy().tolist())
    elapsed = time.perf_counter() - t0

    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)

    # Optimal Threshold (Youden J)
    fpr_all, tpr_all, thresholds_all = roc_curve(y_true, y_prob)
    j_scores  = tpr_all - fpr_all
    best_idx  = np.argmax(j_scores)
    best_thr  = float(thresholds_all[best_idx])
    y_pred    = (y_prob >= best_thr).astype(int)

    # Metrics
    roc_auc     = float(auc(fpr_all, tpr_all))
    ap          = float(average_precision_score(y_true, y_prob))
    f1          = float(f1_score(y_true, y_pred))
    f1_macro    = float(f1_score(y_true, y_pred, average="macro"))
    mcc         = float(matthews_corrcoef(y_true, y_pred))
    kappa       = float(cohen_kappa_score(y_true, y_pred))
    acc         = float(accuracy_score(y_true, y_pred))
    cm          = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # ── Plots ─────────────────────────────────────────────────────
    _plot_confusion_matrix(cm)
    _plot_roc(fpr_all, tpr_all, roc_auc)
    _plot_pr(y_true, y_prob, ap)
    _plot_summary(roc_auc, ap, f1, mcc, kappa, acc, sensitivity, specificity)

    # ── Print Results ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'METRIC':<32} {'VALUE':>10}")
    print(f"{'─'*65}")
    metrics_table = [
        ("Accuracy",              acc),
        ("ROC-AUC",               roc_auc),
        ("Average Precision (AP)",ap),
        ("F1 Score (Pneumonia)",  f1),
        ("F1 Macro",              f1_macro),
        ("MCC (Matthews Corr.)",  mcc),
        ("Cohen Kappa",           kappa),
        ("Sensitivity (Recall)",  sensitivity),
        ("Specificity",           specificity),
        ("Precision (PPV)",       ppv),
        ("NPV",                   npv),
        ("Optimal Threshold",     best_thr),
        ("Inference Time (s)",    elapsed),
    ]
    for name, val in metrics_table:
        print(f"  {name:<32} {val:>10.4f}")
    print(f"{'─'*65}")

    print(f"\n  Confusion Matrix:")
    print(f"                   Pred Normal   Pred Pneumonia")
    print(f"  True Normal      {tn:>10}   {fp:>14}")
    print(f"  True Pneumonia   {fn:>10}   {tp:>14}")

    print(f"\n{BANNER}")
    print("  Per-Class Classification Report")
    print(BANNER)
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))

    # Save JSON
    results = {
        "accuracy":           round(acc, 4),
        "roc_auc":            round(roc_auc, 4),
        "avg_prec":           round(ap, 4),
        "f1":                 round(f1, 4),
        "f1_macro":           round(f1_macro, 4),
        "mcc":                round(mcc, 4),
        "kappa":              round(kappa, 4),
        "sensitivity":        round(sensitivity, 4),
        "specificity":        round(specificity, 4),
        "precision_ppv":      round(ppv, 4),
        "npv":                round(npv, 4),
        "threshold":          round(best_thr, 4),
        "confusion_matrix":   {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "inference_time_sec": round(elapsed, 2),
    }
    out = os.path.join(config.LOG_DIR, "evaluation_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Results JSON  → {out}")
    print(f"[DONE] Plots saved   → {config.LOG_DIR}/")
    return results


# ── Plot helpers ──────────────────────────────────────────────────
def _plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                linewidths=0.8, annot_kws={"size": 16})
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    p = os.path.join(config.LOG_DIR, "confusion_matrix.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[PLOT] Confusion matrix → {p}")


def _plot_roc(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2.5, color="#4F46E5", label=f"ROC Curve  AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#4F46E5")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title("ROC Curve", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(config.LOG_DIR, "roc_curve.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[PLOT] ROC curve        → {p}")


def _plot_pr(y_true, y_prob, ap):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2.5, color="#10B981", label=f"PR Curve  AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.08, color="#10B981")
    ax.set_xlabel("Recall",    fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(config.LOG_DIR, "pr_curve.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[PLOT] PR curve         → {p}")


def _plot_summary(roc_auc, ap, f1, mcc, kappa, acc, sens, spec):
    metrics = {
        "Accuracy":    acc,
        "ROC-AUC":     roc_auc,
        "Avg Prec":    ap,
        "F1 Score":    f1,
        "MCC (norm)":  (mcc + 1) / 2,
        "Kappa":       kappa,
        "Sensitivity": sens,
        "Specificity": spec,
    }
    colors = ["#4F46E5","#7C3AED","#10B981","#F59E0B",
              "#EF4444","#06B6D4","#84CC16","#F97316"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(list(metrics.keys()), list(metrics.values()),
                   color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Model Evaluation Summary", fontsize=16, fontweight="bold", pad=12)
    ax.axvline(x=0.9, color="gray", linestyle="--", lw=1, alpha=0.5, label="0.90 target")
    ax.legend(fontsize=10); ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    p = os.path.join(config.LOG_DIR, "metrics_summary.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[PLOT] Metrics summary  → {p}")


if __name__ == "__main__":
    total_start = time.time()
    model  = load_model_from_weights()
    results = run_evaluation(model)

    total = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  Total Evaluation Time: {total/60:.1f} min ({total:.0f} sec)")
    print(f"{'='*65}\n")
