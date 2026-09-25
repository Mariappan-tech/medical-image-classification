"""
gradcam.py - Grad-CAM + Grad-CAM++ visualisations
Works with EfficientNetB3 loaded via weights-rebuild approach (Keras 3 compatible).

Fixes:
  1. Layer lookup uses the layer OBJECT (.output), not model.get_layer(name),
     which fails for layers nested inside sub-models.
  2. tape.watch(conv_outputs) added explicitly - required for intermediate tensors.
  3. AttributeError fallback when conv_layer.output is unavailable (fresh model).
  4. Grad-CAM++ nested tapes all watch conv_out.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.prediction import load_model, preprocess_single

os.makedirs(config.GRADCAM_DIR, exist_ok=True)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_last_conv_layer(model):
    """
    Returns the last Conv2D layer object, searching nested sub-models too.
    Returns the layer object (not name) so callers can use .output directly.
    """
    # Direct top-level layers first
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"[Grad-CAM] Conv layer (direct): {layer.name}")
            return layer
    # Nested sub-models (e.g. EfficientNetB3 backbone)
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    print(f"[Grad-CAM] Conv layer (nested in '{layer.name}'): {sub.name}")
                    return sub
    raise ValueError("No Conv2D layer found in model or sub-models.")


def _build_grad_model(model, conv_layer):
    """
    Build a nested-model graph that exposes both the target conv tensor and the
    parent backbone feature tensor. This keeps the conv output and the final
    classifier path in the same differentiable graph, even for sub-models such as
    the EfficientNet backbone used here.
    """
    parent = None
    for layer in model.layers:
        if hasattr(layer, "layers") and conv_layer in layer.layers:
            parent = layer
            break

    if parent is not None:
        print(f"[Grad-CAM] Building conv model from parent '{parent.name}'")
        return tf.keras.Model(
            inputs=parent.inputs,
            outputs=[conv_layer.output, parent.output],
        )

    try:
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )
    except AttributeError:
        pass

    dummy = tf.zeros([1] + list(model.input_shape[1:]))
    model(dummy, training=False)
    try:
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )
    except AttributeError:
        pass

    for layer in model.layers:
        if hasattr(layer, "layers") and any(
            isinstance(s, tf.keras.layers.Conv2D) for s in layer.layers
        ):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    print(f"[Grad-CAM] Last-resort fallback layer: {sub.name}")
                    return tf.keras.Model(
                        inputs=layer.inputs,
                        outputs=[sub.output, layer.output],
                    )
    raise RuntimeError(
        "Cannot build grad_model: no accessible Conv2D output tensor found."
    )


def _resolve_conv_layer(model, conv_layer_name):
    """Resolve conv layer by name (top-level + nested) or auto-detect last conv."""
    if conv_layer_name is None:
        return _get_last_conv_layer(model)
    # Try top-level
    try:
        return model.get_layer(conv_layer_name)
    except ValueError:
        pass
    # Try nested sub-models
    for layer in model.layers:
        if hasattr(layer, "layers"):
            try:
                return layer.get_layer(conv_layer_name)
            except ValueError:
                continue
    raise ValueError(f"Layer '{conv_layer_name}' not found in model or sub-models.")


# ─── Public helpers ───────────────────────────────────────────────────────────

def find_last_conv_layer(model):
    """Returns the name string of the last Conv2D layer (for logging/CLI use)."""
    return _get_last_conv_layer(model).name


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────

def compute_gradcam(model, image_tensor, conv_layer_name=None):
    """
    Compute Grad-CAM heatmap.

    Args:
        model           : Keras model (EfficientNetB3 or any Conv backbone)
        image_tensor    : (1, H, W, 3) float32 in [0, 1]
        conv_layer_name : target layer name; auto-detects last Conv2D if None

    Returns:
        heatmap : (H', W') float32 in [0, 1]
    """
    conv_layer = _resolve_conv_layer(model, conv_layer_name)
    grad_model = _build_grad_model(model, conv_layer)

    parent = None
    for layer in model.layers:
        if hasattr(layer, "layers") and conv_layer in layer.layers:
            parent = layer
            break

    if parent is not None:
        x = parent.output
        for name in ["gap", "fc1", "bn1", "drop1", "fc2", "drop2", "output"]:
            if name in [layer.name for layer in model.layers]:
                x = model.get_layer(name)(x)
        head_model = tf.keras.Model(inputs=parent.output, outputs=x)
    else:
        head_model = model

    with tf.GradientTape() as tape:
        conv_outputs, backbone_features = grad_model(image_tensor)
        tape.watch(conv_outputs)
        predictions = head_model(backbone_features if parent is not None else image_tensor)
        pred_score = predictions[:, 0]

    grads = tape.gradient(pred_score, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "Grad-CAM gradients are None - conv layer is not on a differentiable path."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)
    conv_out     = conv_outputs[0]                          # (H', W', C)
    heatmap      = conv_out @ pooled_grads[..., tf.newaxis] # (H', W', 1)
    heatmap      = tf.squeeze(heatmap)                      # (H', W')
    heatmap      = tf.maximum(heatmap, 0)
    heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# ─── Grad-CAM++ ───────────────────────────────────────────────────────────────

def compute_gradcam_plus(model, image_tensor, conv_layer_name=None):
    """
    Compute Grad-CAM++ heatmap.
    All three nested tapes explicitly watch conv_out.

    Args:
        model           : Keras model
        image_tensor    : (1, H, W, 3) float32 in [0, 1]
        conv_layer_name : target layer name; auto-detects last Conv2D if None

    Returns:
        heatmap : (H', W') float32 in [0, 1]
    """
    conv_layer = _resolve_conv_layer(model, conv_layer_name)
    grad_model = _build_grad_model(model, conv_layer)

    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                conv_out, preds = grad_model(image_tensor)
                tape0.watch(conv_out)
                tape1.watch(conv_out)
                tape2.watch(conv_out)
                score = preds[:, 0]
            g1 = tape0.gradient(score, conv_out)
        g2 = tape1.gradient(g1, conv_out)
    g3 = tape2.gradient(g2, conv_out)

    if g1 is None or g2 is None or g3 is None:
        raise RuntimeError("Grad-CAM++: one or more gradient orders are None.")

    global_sum  = tf.reduce_sum(conv_out, axis=(0, 1, 2), keepdims=True)
    alpha       = g2 / (2.0 * g2 + global_sum * g3 + 1e-8)
    weights     = tf.reduce_sum(alpha * tf.nn.relu(g1), axis=(0, 1, 2))

    conv_np  = conv_out[0].numpy()
    heatmap  = np.zeros(conv_np.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights.numpy()):
        heatmap += w * conv_np[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


# ─── Overlay heatmap on original image ───────────────────────────────────────

def overlay_heatmap(heatmap, original_img, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Blend Grad-CAM heatmap onto original image.

    Args:
        heatmap      : (H', W') float32 [0, 1]
        original_img : (H, W, 3) float32 [0, 1]  OR  (H, W, 3) uint8
        alpha        : heatmap blend weight (0 = original only, 1 = heatmap only)

    Returns:
        overlay : (H, W, 3) uint8
    """
    H, W = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    orig = original_img
    if orig.dtype != np.uint8:
        orig = np.uint8(np.clip(orig * 255, 0, 255))

    return cv2.addWeighted(orig, 1 - alpha, heatmap_colored, alpha, 0)


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_gradcam(image_input, model=None, model_path=None,
                method="gradcam", save_prefix="gradcam",
                conv_layer_name=None):
    """
    End-to-end Grad-CAM pipeline: preprocess -> heatmap -> overlay -> save PNG.

    Args:
        image_input     : file path, PIL image, or numpy array (H×W or H×W×3)
        model           : pre-loaded Keras model (loads from model_path if None)
        model_path      : path to .keras file
        method          : "gradcam" | "gradcam++"
        save_prefix     : filename prefix for saved PNG
        conv_layer_name : target conv layer (auto-detected if None)

    Returns:
        overlay   : (H, W, 3) uint8
        heatmap   : (H', W') float32
        save_path : str - absolute path of saved PNG
    """
    if model is None:
        model = load_model(model_path)

    tensor   = preprocess_single(image_input)  # (1, H, W, 3) float32
    orig_img = tensor[0]                       # (H, W, 3) float32

    if method == "gradcam++":
        heatmap = compute_gradcam_plus(model, tensor, conv_layer_name)
    else:
        heatmap = compute_gradcam(model, tensor, conv_layer_name)

    overlay = overlay_heatmap(heatmap, orig_img)

    # 3-panel figure: original | heatmap | overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_img);                 axes[0].set_title("Original X-Ray",         fontsize=13)
    axes[1].imshow(heatmap, cmap="jet");      axes[1].set_title("Grad-CAM Heatmap",       fontsize=13)
    axes[2].imshow(overlay);                  axes[2].set_title(f"{method.upper()} Overlay", fontsize=13)
    for ax in axes:
        ax.axis("off")

    ts        = __import__("time").strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config.GRADCAM_DIR, f"{save_prefix}_{method}_{ts}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Grad-CAM] Saved -> {save_path}")

    return overlay, heatmap, save_path


# ─── Batch ────────────────────────────────────────────────────────────────────

def batch_gradcam(image_list, model=None, model_path=None, method="gradcam"):
    if model is None:
        model = load_model(model_path)
    results = []
    for i, img in enumerate(image_list):
        overlay, heatmap, path = run_gradcam(
            img, model=model, method=method, save_prefix=f"sample_{i}"
        )
        results.append({"overlay": overlay, "heatmap": heatmap, "path": path})
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Grad-CAM Visualisation")
    parser.add_argument("--image",  required=True, help="Image file path")
    parser.add_argument("--model",  default=None,  help="Path to .keras model file")
    parser.add_argument("--method", default="gradcam", choices=["gradcam", "gradcam++"])
    parser.add_argument("--layer",  default=None,  help="Conv layer name (auto-detected if omitted)")
    args = parser.parse_args()

    run_gradcam(
        args.image,
        model_path=args.model,
        method=args.method,
        conv_layer_name=args.layer,
    )
