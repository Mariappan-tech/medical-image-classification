# 🫁 PneumoScan AI — Medical Image Classification

> Binary classification of chest X-rays: **Normal vs Pneumonia**  
> Powered by **EfficientNetB3** transfer learning · Secured with **JWT Auth**

---

## 📌 Project Overview

PneumoScan AI is a full-stack deep learning web application that classifies chest X-ray images to detect pneumonia. It uses a fine-tuned EfficientNetB3 model trained on the PneumoniaMNIST 224×224 dataset, served through a FastAPI backend and a modern HTML/CSS/JS frontend.

---

## 🏆 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | 94.07% |
| **ROC-AUC** | 0.984 |
| **F1 Score** | 0.9521 |
| **Precision (PPV)** | 96.08% |
| **Recall (Sensitivity)** | 94.36% |
| **Specificity** | 93.59% |
| **MCC** | 0.8745 |
| **Cohen's κ** | 0.8743 |
| **Avg Precision** | 0.9903 |

**Confusion Matrix** (Test set: 624 images):

|  | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **True Normal** | 219 (TN) | 15 (FP) |
| **True Pneumonia** | 22 (FN) | 368 (TP) |

---

## 🗂️ Project Structure

```
medical image classification/
│
├── main.py                    # FastAPI app — all API endpoints
├── config.py                  # Central config (paths, hyperparameters, model name)
├── run_pipeline.py            # Full training pipeline runner
├── eval_existing.py           # Evaluate saved model on test set
├── requirements.txt           # Python dependencies
│
├── backend/
│   ├── auth.py                # JWT auth, bcrypt hashing, RBAC, login/register/logout
│   ├── database.py            # SQLite DB — users, sessions, predictions
│   └── audit.py               # Audit middleware + audit log router
│
├── model/
│   ├── model_builder.py       # Custom CNN + Transfer Learning factory (EfficientNetB3, ResNet50, DenseNet121)
│   ├── dataset.py             # Dataset loading & augmentation
│   ├── load_dataset.py        # NPZ file loader for PneumoniaMNIST
│   ├── training.py            # Two-phase training (frozen + fine-tune)
│   ├── evaluation.py          # Full evaluation (ROC, PR, confusion matrix, metrics)
│   └── prediction.py          # Inference — single predict, MC-Dropout uncertainty
│
├── frontend/
│   ├── login.html             # Login / Register page
│   ├── upload.html            # Upload X-ray + analysis settings page
│   ├── result.html            # Prediction result page
│   ├── style.css              # Global stylesheet
│   └── app.js                 # Shared JS utilities
│
├── data set/
│   └── raw_data/
│       └── pneumoniamnist_224.npz   # PneumoniaMNIST dataset (224×224)
│
├── saved_models/
│   └── best_efficientnetb3_phase1.keras   # Trained model weights
│
├── logs/
│   ├── evaluation_results.json      # Latest evaluation metrics (JSON)
│   ├── confusion_matrix.png         # Confusion matrix plot
│   ├── roc_curve.png                # ROC curve plot
│   ├── pr_curve.png                 # Precision-Recall curve plot
│   ├── metrics_summary.png          # Metrics bar chart
│   └── training_phase1_*.csv        # Per-epoch training logs
│
└── saved_models/
```

---

## 🧠 Model Architecture

### EfficientNetB3 (Transfer Learning — 2-Phase)

**Phase 1 — Frozen Backbone:**
- Backbone: EfficientNetB3 pretrained on ImageNet (frozen)
- Custom head: `Dense(512, ReLU)` → `BatchNorm` → `Dropout(0.5)` → `Dense(128, ReLU)` → `Dropout(0.3)` → `Dense(1, Sigmoid)`
- Optimizer: Adam (LR = 1e-3)
- Epochs: 15 | Batch size: 32 | Early stopping patience: 7

**Phase 2 — Fine-tuning:**
- Last 30 layers of backbone unfrozen
- LR reduced to 1e-5
- Epochs: 25 (with early stopping)

**Other supported models:** `resnet50`, `densenet121`, `custom_cnn`

### Input
- Image size: 224 × 224 × 3
- Grayscale X-rays converted to RGB for transfer learning
- Augmentation: rotation ±15°, zoom 10%, horizontal flip, width/height shift 10%

---

## 🌐 API Endpoints

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login → get JWT access token |
| POST | `/auth/logout` | Revoke current session |
| POST | `/auth/logout-all` | Revoke all sessions (all devices) |
| GET | `/auth/me` | Get current user profile |

### Prediction (requires JWT)
| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Standard prediction — Normal or Pneumonia |
| POST | `/predict/uncertainty` | MC-Dropout uncertainty estimation (n=30 passes) |
| GET | `/predictions` | Get current user's prediction history |

### System
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/evaluate` | Trigger full test-set evaluation (admin/radiologist) |
| GET | `/evaluate/results` | Get last evaluation results |

### Audit (admin only)
| Method | Endpoint | Description |
|---|---|---|
| GET | `/audit/logs` | Paginated audit trail |
| GET | `/audit/stats` | Audit statistics |
| GET | `/audit/my-activity` | Current user's own activity |

---

## 🔐 Authentication & Security

- **JWT Tokens** — access + refresh token flow (`python-jose`)
- **Bcrypt** password hashing (`passlib`)
- **Role-Based Access Control (RBAC)** — roles: `user`, `radiologist`, `admin`
- **Session management** — per-device token revocation
- **Audit Middleware** — every API request is automatically logged to the audit trail
- **SQLite** database for users, sessions, predictions, audit logs

---

## 🔍 Analysis Modes

| Mode | Description |
|---|---|
| **Standard** | Fast single-pass prediction + confidence score |
| **Uncertainty** | MC-Dropout — 30 stochastic forward passes, returns mean probability + std deviation |

---

## 🖥️ Frontend Pages

### Login Page (`login.html`)
- User login and registration
- JWT token stored in localStorage

### Upload Page (`upload.html`)
- Drag & drop or click-to-browse X-ray upload
- Image validation (JPEG/PNG, grayscale check, max 25MB, min 64×64px)
- Analysis mode selection (Standard / Uncertainty)
- Classification threshold slider (0.1 – 0.9)
- Active Model Hub showing model status and performance
- Advanced Diagnostics panel (confusion matrix, ROC curve, metrics)
- Prediction history drawer

### Result Page (`result.html`)
- Prediction result with confidence score and risk level
- Uncertainty bounds display (if Uncertainty mode selected)
- Full metrics breakdown

---

## ⚙️ Setup & Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset
Place `pneumoniamnist_224.npz` in:
```
data set/raw_data/pneumoniamnist_224.npz
```

### 3. Train the model (optional — pretrained model already included)
```bash
python run_pipeline.py
```

### 4. Evaluate existing model
```bash
python eval_existing.py
```

### 5. Start the API server
```bash
python main.py
```
Server runs at: `http://localhost:8000`  
API docs: `http://localhost:8000/docs`

### 6. Open the frontend
Open `frontend/login.html` in your browser.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow >= 2.13` | Model training & inference |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `python-jose` | JWT authentication |
| `passlib[bcrypt]` | Password hashing |
| `scikit-learn` | Metrics, ROC, PR curves |
| `Pillow` | Image loading & processing |
| `numpy` | Numerical operations |
| `matplotlib / seaborn` | Plots & charts |

---

## 📊 Training Configuration

```python
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 15       # Phase 1
EPOCHS_FINETUNE = 25       # Phase 2
LEARNING_RATE   = 1e-3     # Phase 1
FINETUNE_LR     = 1e-5     # Phase 2
DROPOUT_RATE    = 0.5
L2_REG          = 1e-4
UNFREEZE_LAYERS = 30
PATIENCE        = 7
RANDOM_SEED     = 42
```

---

## 🗃️ Dataset

**PneumoniaMNIST** (MedMNIST v2)
- 224×224 chest X-ray images
- Binary labels: `0 = Normal`, `1 = Pneumonia`
- Split: train / validation / test

---

## 👤 User Roles

| Role | Permissions |
|---|---|
| `user` | Upload images, view own predictions |
| `radiologist` | All user permissions + trigger model evaluation |
| `admin` | Full access — audit logs, evaluation, all predictions |

---

*Built with TensorFlow, FastAPI, and vanilla HTML/CSS/JS.*
