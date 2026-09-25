"""
run_pipeline.py — Complete Training + Evaluation Pipeline Runner
Usage:
    python run_pipeline.py              → Train + Evaluate
    python run_pipeline.py --eval-only  → Evaluate existing saved model
    python run_pipeline.py --train-only → Train only
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ─── Suppress TF warnings ─────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
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
from model.model_builder import get_model, set_backbone_trainable
from model.training      import build_callbacks, compile_model, plot_history

os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR,   exist_ok=True)

BANNER = "=" * 65


# ══════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════
def run_training():
    print(f"\n{BANNER}")
    print("   MEDICAL IMAGE CLASSIFICATION — TRAINING PIPELINE")
    print(f"   Model : {config.MODEL_NAME.upper()}")
    print(f"   Input : {config.IMG_SIZE}  Classes: {config.CLASS_NAMES}")
    print(BANNER)

    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # ── Load Data ─────────────────────────────────────────────────
    train_ds, val_ds, test_ds, class_weights, raw = get_data_pipelines()

    # ── Build Model ───────────────────────────────────────────────
    model, backbone = get_model()

    # ── Phase 1: Frozen Backbone ──────────────────────────────────
    print(f"\n{BANNER}")
    print("  PHASE 1 — Training Head (Backbone Frozen)")
    print(BANNER)
    model  = compile_model(model, config.LEARNING_RATE)
    cbs_1, ckpt_1 = build_callbacks("phase1", config.LEARNING_RATE)

    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FROZEN,
        callbacks=cbs_1,
        class_weight=class_weights if config.MODEL_NAME != "custom_cnn" else None,
        verbose=1,
    )
    plot_history(hist1, "phase1")

    # ── Phase 2: Fine-Tune ────────────────────────────────────────
    if backbone is not None:
        print(f"\n{BANNER}")
        print("  PHASE 2 — Fine-Tuning (Unfreezing Last Layers)")
        print(BANNER)
        model = set_backbone_trainable(model, backbone, config.UNFREEZE_LAYERS)
        model = compile_model(model, config.FINETUNE_LR)
        cbs_2, ckpt_2 = build_callbacks("phase2", config.FINETUNE_LR)

        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS_FINETUNE,
            callbacks=cbs_2,
            class_weight=class_weights,
            verbose=1,
        )
        plot_history(hist2, "phase2")
        final_ckpt = ckpt_2
    else:
        final_ckpt = ckpt_1

    # ── Save Final Model ──────────────────────────────────────────
    final_path = os.path.join(config.MODEL_DIR, f"final_{config.MODEL_NAME}.keras")
    model.save(final_path)
    meta = {
        "model_name":  config.MODEL_NAME,
        "img_size":    list(config.IMG_SIZE),
        "class_names": config.CLASS_NAMES,
        "final_model": final_path,
        "best_ckpt":   final_ckpt,
    }
    with open(os.path.join(config.MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Final model saved  → {final_path}")
    print(f"[DONE] Best checkpoint    → {final_ckpt}")
    print(f"[DONE] model_meta.json    saved")
    return model, test_ds, raw


# ══════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════
def run_evaluation(model=None, test_ds=None):
    print(f"\n{BANNER}")
    print("  EVALUATION — Full Metrics on Test Set")
    print(BANNER)

    # Load model if not passed
    if model is None:
        model_path = os.path.join(config.MODEL_DIR, f"final_{config.MODEL_NAME}.keras")
        print(f"[INFO] Loading model → {model_path}")
        model = tf.keras.models.load_model(model_path, safe_mode=False)

    # Build test pipeline if not passed
    if test_ds is None:
        _, _, test_ds, _, _ = get_data_pipelines()

    # ── Inference ─────────────────────────────────────────────────
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

    # ── Optimal Threshold ─────────────────────────────────────────
    fpr_all, tpr_all, thresholds_all = roc_curve(y_true, y_prob)
    j_scores = tpr_all - fpr_all
    best_idx  = np.argmax(j_scores)
    best_thr  = float(thresholds_all[best_idx])
    y_pred    = (y_prob >= best_thr).astype(int)

    # ── Compute All Metrics ───────────────────────────────────────
    roc_auc   = float(auc(fpr_all, tpr_all))
    ap        = float(average_precision_score(y_true, y_prob))
    f1        = float(f1_score(y_true, y_pred))
    f1_macro  = float(f1_score(y_true, y_pred, average="macro"))
    mcc       = float(matthews_corrcoef(y_true, y_pred))
    kappa     = float(cohen_kappa_score(y_true, y_pred))
    acc       = float(accuracy_score(y_true, y_pred))
    cm        = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # Precision
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # ── Plots ─────────────────────────────────────────────────────
    _plot_confusion_matrix(y_true, y_pred, cm)
    _plot_roc_curve(fpr_all, tpr_all, roc_auc)
    _plot_pr_curve(y_true, y_prob, ap)
    _plot_metric_summary(roc_auc, ap, f1, mcc, kappa, acc, sensitivity, specificity)

    # ── Print Results ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'METRIC':<30} {'VALUE':>10}")
    print(f"{'─'*65}")
    print(f"  {'Accuracy':<30} {acc:>10.4f}")
    print(f"  {'ROC-AUC':<30} {roc_auc:>10.4f}")
    print(f"  {'Average Precision (AP)':<30} {ap:>10.4f}")
    print(f"  {'F1 Score (Pneumonia)':<30} {f1:>10.4f}")
    print(f"  {'F1 Macro':<30} {f1_macro:>10.4f}")
    print(f"  {'MCC (Matthews Corr.)':<30} {mcc:>10.4f}")
    print(f"  {'Cohen Kappa':<30} {kappa:>10.4f}")
    print(f"  {'Sensitivity (Recall)':<30} {sensitivity:>10.4f}")
    print(f"  {'Specificity':<30} {specificity:>10.4f}")
    print(f"  {'Precision (PPV)':<30} {ppv:>10.4f}")
    print(f"  {'NPV':<30} {npv:>10.4f}")
    print(f"  {'Optimal Threshold':<30} {best_thr:>10.4f}")
    print(f"  {'Inference Time (s)':<30} {elapsed:>10.2f}")
    print(f"{'─'*65}")
    print(f"\n  Confusion Matrix:\n"
          f"         Pred Normal  Pred Pneumonia\n"
          f"  True Normal   {tn:>6}  {fp:>13}\n"
          f"  True Pneumonia{fn:>6}  {tp:>13}")
    print(f"\n{BANNER}")
    print("  Per-Class Classification Report")
    print(BANNER)
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))

    # ── Save JSON ─────────────────────────────────────────────────
    results = {
        "accuracy":      round(acc, 4),
        "roc_auc":       round(roc_auc, 4),
        "avg_prec":      round(ap, 4),
        "f1":            round(f1, 4),
        "f1_macro":      round(f1_macro, 4),
        "mcc":           round(mcc, 4),
        "kappa":         round(kappa, 4),
        "sensitivity":   round(sensitivity, 4),
        "specificity":   round(specificity, 4),
        "precision_ppv": round(ppv, 4),
        "npv":           round(npv, 4),
        "threshold":     round(best_thr, 4),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        },
        "inference_time_sec": round(elapsed, 2),
    }
    out_path = os.path.join(config.LOG_DIR, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Results JSON  → {out_path}")
    print(f"[DONE] Plots saved   → {config.LOG_DIR}/")
    return results


# ══════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════
def _plot_confusion_matrix(y_true, y_pred, cm):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        linewidths=0.8, annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(config.LOG_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix → {path}")


def _plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2.5, color="#4F46E5",
            label=f"ROC Curve  AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#4F46E5")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title("ROC Curve", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(config.LOG_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] ROC curve        → {path}")


def _plot_pr_curve(y_true, y_prob, ap):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2.5, color="#10B981",
            label=f"PR Curve  AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.08, color="#10B981")
    ax.set_xlabel("Recall",    fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(config.LOG_DIR, "pr_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] PR curve         → {path}")


def _plot_metric_summary(roc_auc, ap, f1, mcc, kappa, acc, sens, spec):
    """Bar chart of all key metrics at a glance."""
    metrics = {
        "Accuracy":    acc,
        "ROC-AUC":     roc_auc,
        "Avg Prec":    ap,
        "F1 Score":    f1,
        "MCC":         (mcc + 1) / 2,     # normalise to [0,1] for display
        "Kappa":       kappa,
        "Sensitivity": sens,
        "Specificity": spec,
    }
    colors = ["#4F46E5", "#7C3AED", "#10B981", "#F59E0B",
              "#EF4444", "#06B6D4", "#84CC16", "#F97316"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(list(metrics.keys()), list(metrics.values()),
                   color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Model Evaluation Summary", fontsize=16, fontweight="bold", pad=12)
    ax.axvline(x=0.9, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="0.90 target")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(config.LOG_DIR, "metrics_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Metrics summary  → {path}")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical Image Classification — Full Pipeline Runner"
    )
    parser.add_argument(
        "--eval-only",  action="store_true",
        help="Skip training; evaluate an existing saved model"
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Train only; skip evaluation"
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Custom path to .keras model (eval-only mode)"
    )
    args = parser.parse_args()

    start = time.time()

    if args.eval_only:
        # Evaluate existing model
        model_path = args.model_path or os.path.join(
            config.MODEL_DIR, f"final_{config.MODEL_NAME}.keras"
        )
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            print("[INFO]  Run without --eval-only to train first.")
            sys.exit(1)
        model = tf.keras.models.load_model(model_path, safe_mode=False)
        run_evaluation(model=model)

    elif args.train_only:
        run_training()

    else:
        # Default: Train → Evaluate
        model, test_ds, _ = run_training()
        run_evaluation(model=model, test_ds=test_ds)

    total = time.time() - start
    print(f"\n{'='*65}")
    print(f"  Total Pipeline Time: {total/60:.1f} min ({total:.0f} sec)")
    print(f"{'='*65}\n")
