"""
model_builder.py — Custom CNN + Transfer Learning model factory
Supports: custom_cnn | efficientnetb3 | resnet50 | densenet121
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── 1. Custom CNN ────────────────────────────────────────────────────────────
def build_custom_cnn(input_shape=(224, 224, 3)):
    """
    Deep custom CNN with BatchNorm + Residual-like skip connections.
    Good baseline before transfer learning.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(config.L2_REG))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Head
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs, outputs, name="CustomCNN")
    return model


# ─── 2. Transfer Learning Backbone Factory ────────────────────────────────────
_BACKBONE_MAP = {
    "efficientnetb3": (tf.keras.applications.EfficientNetB3,
                       tf.keras.applications.efficientnet.preprocess_input),
    "resnet50":       (tf.keras.applications.ResNet50,
                       tf.keras.applications.resnet.preprocess_input),
    "densenet121":    (tf.keras.applications.DenseNet121,
                       tf.keras.applications.densenet.preprocess_input),
}


def build_transfer_model(backbone_name="efficientnetb3",
                         input_shape=(224, 224, 3),
                         trainable=False):
    """
    Two-phase transfer learning model:
      Phase 1 – backbone frozen, train classification head
      Phase 2 – unfreeze last N layers + fine-tune
    Call this function with trainable=False for Phase 1.
    Call set_backbone_trainable() for Phase 2 transition.
    """
    if backbone_name not in _BACKBONE_MAP:
        raise ValueError(f"Unknown backbone: {backbone_name}. "
                         f"Choose from {list(_BACKBONE_MAP.keys())}")

    BackboneCls, preprocess_fn = _BACKBONE_MAP[backbone_name]

    # Input layer — data comes in as [0,1] float32
    inputs = tf.keras.Input(shape=input_shape, name="input")

    # Scale [0,1] → [0,255] using a Lambda layer (Keras 3 compatible)
    x = layers.Lambda(lambda img: img * 255.0, name="scale_255")(inputs)

    # Apply backbone-specific preprocessing inside a Lambda layer
    x = layers.Lambda(lambda img: preprocess_fn(img), name="backbone_preprocess")(x)

    # Backbone — pass tensor directly (input_tensor is deprecated in Keras 3)
    backbone = BackboneCls(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = trainable
    x = backbone(x, training=False)

    gap = layers.GlobalAveragePooling2D(name="gap")(x)

    # Classification Head
    head = layers.Dense(512, activation="relu",
                        kernel_regularizer=regularizers.l2(config.L2_REG),
                        name="fc1")(gap)
    head = layers.BatchNormalization(name="bn1")(head)
    head = layers.Dropout(config.DROPOUT_RATE, name="drop1")(head)
    head = layers.Dense(128, activation="relu",
                        kernel_regularizer=regularizers.l2(config.L2_REG),
                        name="fc2")(head)
    head = layers.Dropout(0.3, name="drop2")(head)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(head)

    model = models.Model(inputs=inputs, outputs=outputs,
                         name=f"Transfer_{backbone_name.capitalize()}")
    return model, backbone


def set_backbone_trainable(model, backbone, unfreeze_last_n=30):
    """Unfreeze last N layers of backbone for fine-tuning."""
    backbone.trainable = True
    for layer in backbone.layers[:-unfreeze_last_n]:
        layer.trainable = False
    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    print(f"[INFO] Backbone trainable layers: {trainable_count}/{len(backbone.layers)}")
    return model


def get_model():
    """
    Factory: returns (model, backbone_or_None) based on config.MODEL_NAME.
    """
    name = config.MODEL_NAME.lower()
    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)

    if name == "custom_cnn":
        model = build_custom_cnn(input_shape)
        backbone = None
    else:
        model, backbone = build_transfer_model(name, input_shape, trainable=False)

    model.summary()
    return model, backbone
