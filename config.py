"""
config.py — Central configuration for Medical Image Classification Project
PneumoniaMNIST Binary Classification (Normal vs Pneumonia)
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data set", "raw_data")
NPZ_PATH    = os.path.join(DATA_DIR, "pneumoniamnist_224.npz")
PROC_DIR    = os.path.join(BASE_DIR, "data set", "processed_data")
MODEL_DIR   = os.path.join(BASE_DIR, "saved_models")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

# ─── Image Settings ───────────────────────────────────────────────────────────
IMG_HEIGHT   = 224
IMG_WIDTH    = 224
IMG_CHANNELS = 3          # grayscale → RGB for transfer learning
IMG_SIZE     = (IMG_HEIGHT, IMG_WIDTH)

# ─── Classes ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Normal", "Pneumonia"]
NUM_CLASSES = 2           # binary; model uses sigmoid output

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 15      # Phase-1: train head only (backbone frozen)
EPOCHS_FINETUNE = 25      # Phase-2: fine-tune last N layers
LEARNING_RATE   = 1e-3    # Phase-1 LR
FINETUNE_LR     = 1e-5    # Phase-2 LR
DROPOUT_RATE    = 0.5
L2_REG          = 1e-4
UNFREEZE_LAYERS = 30      # unfreeze last N layers during fine-tune

# ─── Early Stopping ───────────────────────────────────────────────────────────
PATIENCE  = 7
MIN_DELTA = 1e-4

# ─── Model Selection ──────────────────────────────────────────────────────────
# Options: "custom_cnn" | "efficientnetb3" | "resnet50" | "densenet121"
MODEL_NAME = "efficientnetb3"

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_TRAIN    = True
ROTATION_RANGE   = 15
ZOOM_RANGE       = 0.1
HORIZONTAL_FLIP  = True
WIDTH_SHIFT      = 0.1
HEIGHT_SHIFT     = 0.1

# ─── Misc ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
VERBOSE     = 1
