"""
prediction.py — Single image + batch prediction with confidence scores,
top-K uncertainty estimation, and result visualization.
"""

import os, sys, json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Load Model ───────────────────────────────────────────────────────────────
# ─── Load Model ───────────────────────────────────────────────────────────────
def load_model(model_path=None):
    """
    Load the trained model. Uses a two-step approach to handle Keras 3's
    restriction on Lambda layer deserialization:
      1. Try tf.keras.models.load_model with safe_mode=False (works on Keras 2)
      2. If Lambda shape-inference error occurs (Keras 3), rebuild the
         architecture from model_builder and transfer weights from the saved file.
    """
    if model_path is None:
        candidates = [
            os.path.join(config.MODEL_DIR, f"best_{config.MODEL_NAME}_phase1.keras"),
            os.path.join(config.MODEL_DIR, f"best_{config.MODEL_NAME}.keras"),
            os.path.join(config.MODEL_DIR, f"final_{config.MODEL_NAME}.keras"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])

    print(f"[INFO] Loading model: {model_path}")

    # ── Attempt 1: direct load (works if Keras version supports Lambda deserialization)
    try:
        model = tf.keras.models.load_model(model_path, safe_mode=False)
        print("[INFO] Model loaded directly.")
        return model
    except Exception as e:
        if "output_shape" not in str(e) and "Lambda" not in str(e):
            raise   # unrelated error — re-raise immediately
        print(f"[INFO] Direct load failed (Lambda/Keras3 issue): {e}")
        print("[INFO] Falling back to rebuild-architecture + load-weights approach...")

    # ── Attempt 2: rebuild architecture, load weights from saved model
    # Import here to avoid circular imports
    from model.model_builder import build_transfer_model, build_custom_cnn

    name         = config.MODEL_NAME.lower()
    input_shape  = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)

    if name == "custom_cnn":
        fresh_model = build_custom_cnn(input_shape)
    else:
        fresh_model, _ = build_transfer_model(name, input_shape, trainable=False)

    # Compile with dummy settings so weights can be loaded
    fresh_model.compile(optimizer="adam", loss="binary_crossentropy")

    # Load weights — by_name=False matches layers by position (architecture must match)
    try:
        fresh_model.load_weights(model_path)
        print("[INFO] Weights loaded via load_weights().")
        return fresh_model
    except Exception as e2:
        print(f"[WARN] load_weights failed: {e2}")

    # ── Attempt 3: extract weights layer-by-layer from the saved model
    print("[INFO] Trying layer-by-layer weight transfer...")
    saved = tf.keras.models.load_model(model_path, safe_mode=False,
                                       compile=False)
    # Map by layer name
    saved_weights = {l.name: l.get_weights() for l in saved.layers if l.get_weights()}
    transferred = 0
    for layer in fresh_model.layers:
        if layer.name in saved_weights and saved_weights[layer.name]:
            try:
                layer.set_weights(saved_weights[layer.name])
                transferred += 1
            except Exception:
                pass
    print(f"[INFO] Transferred weights for {transferred} layers.")
    return fresh_model


# ─── Preprocess single image (from file path or numpy) ────────────────────────
def preprocess_single(image_input):
    """
    Accepts:
      - str / os.PathLike  : file path (JPEG, PNG, TIFF)
      - np.ndarray         : H×W or H×W×C array (uint8 or float32)
    Returns: (1, H, W, 3) float32 tensor in [0,1].
    """
    if isinstance(image_input, (str, os.PathLike)):
        img = Image.open(image_input).convert("RGB")
        img = img.resize(config.IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        arr = image_input.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        if arr.shape[:2] != config.IMG_SIZE:
            img_pil = Image.fromarray((arr * 255).astype(np.uint8))
            img_pil = img_pil.resize(config.IMG_SIZE)
            arr = np.array(img_pil, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)  # (1, H, W, 3)


# ─── Single Prediction ────────────────────────────────────────────────────────
def predict_single(model, image_input, threshold=0.5):
    """
    Returns dict with:
      - predicted_class : "Normal" | "Pneumonia"
      - confidence      : 0.0 – 1.0
      - probability     : raw sigmoid output
    """
    tensor = preprocess_single(image_input)
    prob   = float(model.predict(tensor, verbose=0)[0][0])
    pred   = 1 if prob >= threshold else 0
    conf   = prob if pred == 1 else (1.0 - prob)
    return {
        "predicted_class": config.CLASS_NAMES[pred],
        "confidence":      round(conf, 4),
        "probability":     round(prob, 4),
        "label_index":     pred,
    }


# ─── Monte-Carlo Dropout Uncertainty Estimation ───────────────────────────────
def predict_with_uncertainty(model, image_input, n_passes=30, threshold=0.5):
    """
    Run N stochastic forward passes (MC Dropout) to estimate prediction uncertainty.
    Returns mean prediction + std (epistemic uncertainty).
    """
    tensor = preprocess_single(image_input)
    probs  = []
    for _ in range(n_passes):
        p = float(model(tensor, training=True).numpy()[0][0])
        probs.append(p)
    probs  = np.array(probs)
    mean_p = float(np.mean(probs))
    std_p  = float(np.std(probs))
    pred   = 1 if mean_p >= threshold else 0
    conf   = mean_p if pred == 1 else (1.0 - mean_p)
    return {
        "predicted_class": config.CLASS_NAMES[pred],
        "confidence":      round(conf, 4),
        "mean_prob":       round(mean_p, 4),
        "uncertainty_std": round(std_p,  4),
        "label_index":     pred,
    }


# ─── Batch Prediction ─────────────────────────────────────────────────────────
def predict_batch(model, image_list, threshold=0.5):
    """
    image_list: list of file paths or numpy arrays.
    Returns: list of result dicts.
    """
    results = []
    for img in image_list:
        result = predict_single(model, img, threshold)
        results.append(result)
    return results


# ─── Visualise Prediction ─────────────────────────────────────────────────────
def visualise_prediction(image_input, result, save_path=None):
    tensor = preprocess_single(image_input)[0]  # (H,W,3)
    label  = result["predicted_class"]
    conf   = result["confidence"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tensor, cmap="gray" if tensor[:, :, 0].std() < 0.02 else None)
    color = "#EF4444" if label == "Pneumonia" else "#22C55E"
    ax.set_title(f"Prediction: {label}\nConfidence: {conf:.2%}",
                 fontsize=14, color=color, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Prediction viz saved → {save_path}")
    plt.close()


# ─── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict a single X-ray image")
    parser.add_argument("--image",     required=True, help="Path to image file")
    parser.add_argument("--model",     default=None,  help="Path to .keras model")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mc",        action="store_true", help="Use MC Dropout")
    parser.add_argument("--n_passes",  type=int, default=30)
    args = parser.parse_args()

    mdl = load_model(args.model)

    if args.mc:
        result = predict_with_uncertainty(mdl, args.image, args.n_passes, args.threshold)
    else:
        result = predict_single(mdl, args.image, args.threshold)

    print(json.dumps(result, indent=2))
    vis_path = os.path.join(config.LOG_DIR, "prediction_vis.png")
    visualise_prediction(args.image, result, vis_path)
