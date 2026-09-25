"""
dataset.py — Data loading, preprocessing, and augmentation pipeline
Loads PneumoniaMNIST .npz → returns tf.data.Dataset splits
"""

import numpy as np
import tensorflow as tf
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_npz():
    """Load raw .npz and return numpy arrays."""
    print(f"[INFO] Loading dataset from: {config.NPZ_PATH}")
    data = np.load(config.NPZ_PATH)
    print("[INFO] Available keys:", list(data.keys()))
    for k in data.keys():
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")

    train_images = data["train_images"].astype(np.float32)
    train_labels = data["train_labels"].astype(np.float32).reshape(-1)
    val_images   = data["val_images"].astype(np.float32)
    val_labels   = data["val_labels"].astype(np.float32).reshape(-1)
    test_images  = data["test_images"].astype(np.float32)
    test_labels  = data["test_labels"].astype(np.float32).reshape(-1)

    return (train_images, train_labels,
            val_images,   val_labels,
            test_images,  test_labels)


def preprocess_image(image, label):
    """Normalize, ensure 3 channels, resize."""
    image = image / 255.0
    # Grayscale (H,W) or (H,W,1) → (H,W,3)
    if len(image.shape) == 2:
        image = tf.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, config.IMG_SIZE)
    return image, label


def augment_image(image, label):
    """Apply random augmentations during training using tf.image ops (tf.function compatible)."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Random rotation using tfa-free approach: random crop + pad
    # Simulate slight rotation with shear via random crop+resize
    h, w = config.IMG_SIZE
    zoom_factor = 1.0 - config.ZOOM_RANGE
    crop_h = tf.cast(h * zoom_factor, tf.int32)
    crop_w = tf.cast(w * zoom_factor, tf.int32)
    image = tf.image.random_crop(image, [crop_h, crop_w, 3])
    image = tf.image.resize(image, [h, w])

    # Random horizontal/vertical shift via padding + crop
    pad_h = tf.cast(h * config.HEIGHT_SHIFT, tf.int32)
    pad_w = tf.cast(w * config.WIDTH_SHIFT, tf.int32)
    image = tf.pad(image, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode='REFLECT')
    image = tf.image.random_crop(image, [h, w, 3])

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_dataset(images, labels, training=False, augment=False):
    """Build a tf.data.Dataset pipeline."""
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(images), seed=config.RANDOM_SEED)

    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)

    if augment and config.AUGMENT_TRAIN:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(config.BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def get_class_weights(labels):
    """Compute inverse-frequency class weights for imbalanced data."""
    n_total = len(labels)
    n_pos   = np.sum(labels)
    n_neg   = n_total - n_pos
    weight_0 = (n_total / (2.0 * n_neg))
    weight_1 = (n_total / (2.0 * n_pos))
    print(f"[INFO] Class weights → Normal: {weight_0:.4f}, Pneumonia: {weight_1:.4f}")
    return {0: weight_0, 1: weight_1}


def get_data_pipelines():
    """High-level helper: load everything and return ready datasets + metadata."""
    (tr_img, tr_lbl, val_img, val_lbl, te_img, te_lbl) = load_npz()

    train_ds = build_dataset(tr_img, tr_lbl, training=True, augment=True)
    val_ds   = build_dataset(val_img,  val_lbl, training=False, augment=False)
    test_ds  = build_dataset(te_img,  te_lbl,  training=False, augment=False)

    class_weights = get_class_weights(tr_lbl)

    print(f"\n[INFO] Train samples : {len(tr_lbl)}")
    print(f"[INFO] Val   samples : {len(val_lbl)}")
    print(f"[INFO] Test  samples : {len(te_lbl)}")

    return train_ds, val_ds, test_ds, class_weights, (tr_img, tr_lbl, val_img, val_lbl, te_img, te_lbl)
