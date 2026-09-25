"""
training.py — Two-phase training pipeline with callbacks, LR scheduling,
class-weight handling, and TensorBoard logging.
"""

import os, sys, json, time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.dataset        import get_data_pipelines
from model.model_builder  import get_model, set_backbone_trainable

os.makedirs(config.MODEL_DIR, config.LOG_DIR, exist_ok=True) if False else [
    os.makedirs(d, exist_ok=True) for d in [config.MODEL_DIR, config.LOG_DIR]
]


# ─── Callbacks ────────────────────────────────────────────────────────────────
def build_callbacks(phase: str, lr: float):
    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(config.MODEL_DIR, f"best_{config.MODEL_NAME}_{phase}.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=config.PATIENCE,
            min_delta=config.MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.LOG_DIR, f"{ts}_{phase}"),
            histogram_freq=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.LOG_DIR, f"training_{phase}_{ts}.csv"),
            append=False,
        ),
    ]
    return callbacks, ckpt_path


# ─── Compile helper ───────────────────────────────────────────────────────────
def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ─── Plot helper ──────────────────────────────────────────────────────────────
def plot_history(history, phase: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["loss", "accuracy", "auc"]):
        ax.plot(history.history[metric],          label=f"Train {metric}", linewidth=2)
        ax.plot(history.history[f"val_{metric}"], label=f"Val   {metric}", linewidth=2, linestyle="--")
        ax.set_title(f"{metric.upper()} — Phase: {phase}", fontsize=14)
        ax.set_xlabel("Epoch"); ax.set_ylabel(metric)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(config.LOG_DIR, f"history_{phase}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[INFO] History plot saved → {plot_path}")


# ─── Main Training ────────────────────────────────────────────────────────────
def train():
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Load data
    train_ds, val_ds, test_ds, class_weights, raw = get_data_pipelines()

    # Build model
    model, backbone = get_model()

    # ── Phase 1: Frozen Backbone ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1 — Training Classification Head (Backbone Frozen)")
    print("="*60)
    model = compile_model(model, config.LEARNING_RATE)
    cbs_p1, ckpt_p1 = build_callbacks("phase1", config.LEARNING_RATE)

    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FROZEN,
        callbacks=cbs_p1,
        class_weight=class_weights if config.MODEL_NAME != "custom_cnn" else None,
        verbose=config.VERBOSE,
    )
    plot_history(hist1, "phase1")

    # ── Phase 2: Fine-Tuning ──────────────────────────────────────────────────
    if backbone is not None:
        print("\n" + "="*60)
        print("  PHASE 2 — Fine-Tuning (Unfreezing Last Layers)")
        print("="*60)
        model = set_backbone_trainable(model, backbone, config.UNFREEZE_LAYERS)
        model = compile_model(model, config.FINETUNE_LR)
        cbs_p2, ckpt_p2 = build_callbacks("phase2", config.FINETUNE_LR)

        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS_FINETUNE,
            callbacks=cbs_p2,
            class_weight=class_weights,
            verbose=config.VERBOSE,
        )
        plot_history(hist2, "phase2")
        final_ckpt = ckpt_p2
    else:
        final_ckpt = ckpt_p1

    # Save final model
    final_path = os.path.join(config.MODEL_DIR, f"final_{config.MODEL_NAME}.keras")
    model.save(final_path)
    print(f"\n[INFO] Final model saved → {final_path}")
    print(f"[INFO] Best checkpoint   → {final_ckpt}")

    # Save metadata
    meta = {
        "model_name":  config.MODEL_NAME,
        "img_size":    config.IMG_SIZE,
        "class_names": config.CLASS_NAMES,
        "final_model": final_path,
        "best_ckpt":   final_ckpt,
    }
    with open(os.path.join(config.MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[INFO] model_meta.json saved.")

    return model


if __name__ == "__main__":
    train()
