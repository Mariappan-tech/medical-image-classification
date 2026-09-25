"""
main.py — FastAPI Backend for Medical Image Classification
Endpoints:
  AUTH
    POST /auth/register       — register new user
    POST /auth/login          — login + get JWT tokens
    POST /auth/logout         — revoke current session
    POST /auth/logout-all     — revoke all sessions
    GET  /auth/me             — current user profile

  PREDICTION (requires auth)
    POST /predict             — single image prediction
    POST /predict/uncertainty — MC-Dropout uncertainty prediction
    GET  /predictions         — list user's prediction history

  SYSTEM
    GET  /health              — health check
    GET  /model/info          — model metadata
    POST /evaluate            — run full test-set evaluation (admin/radiologist)
    GET  /evaluate/results    — get last evaluation results

  AUDIT (admin only)
    GET  /audit/logs          — paginated audit trail
    GET  /audit/stats         — audit statistics
    GET  /audit/my-activity   — current user's own activity
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
import base64
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import (
    FastAPI, File, UploadFile, HTTPException, Form,
    BackgroundTasks, Depends, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Backend layer
from backend.database import init_db, save_prediction, list_predictions
from backend.auth import (
    RegisterRequest, LoginRequest, TokenResponse, UserPublic,
    register_user, login_user, logout_user, logout_all_devices,
    get_current_user, get_current_active_user, require_role,
    oauth2_scheme, seed_default_users,
)
from backend.audit import AuditMiddleware, router as audit_router

# Model layer
from model.prediction import load_model as _load_model
from model.prediction import (
    predict_single,
    predict_with_uncertainty,
    preprocess_single,
    visualise_prediction,
)
from model.evaluation import evaluate as run_evaluation


# ── App Initialisation ────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical Image Classification AI",
    description=(
        "PneumoniaMNIST binary classifier — Normal vs Pneumonia.\n"
        "Powered by EfficientNetB3.\n"
        "Secured with JWT Authentication & full Audit Logging."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuditMiddleware)   # <── Auto-audit all requests

# ── Static Mounts & Web Routes ────────────────────────────────────────────────
for _dir in [config.LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)

app.mount("/static/logs", StaticFiles(directory=config.LOG_DIR), name="logs")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.get("/", tags=["Frontend"])
def serve_index():
    login_path = os.path.join(FRONTEND_DIR, "login.html")
    if os.path.exists(login_path):
        return FileResponse(login_path)
    return {"message": "PneumoScan AI API is running", "docs": "/docs"}

@app.get("/login.html", tags=["Frontend"])
def serve_login():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))

@app.get("/upload.html", tags=["Frontend"])
def serve_upload():
    return FileResponse(os.path.join(FRONTEND_DIR, "upload.html"))

@app.get("/result.html", tags=["Frontend"])
def serve_result():
    return FileResponse(os.path.join(FRONTEND_DIR, "result.html"))

@app.get("/style.css", tags=["Frontend"])
def serve_style():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"), media_type="text/css")

@app.get("/app.js", tags=["Frontend"])
def serve_app_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "app.js"), media_type="application/javascript")

# ── Include routers ───────────────────────────────────────────────────────────
app.include_router(audit_router)


# ── Startup: init DB, seed accounts & pre-load model ──────────────────────────
@app.on_event("startup")
async def on_startup():
    init_db()
    seed_default_users()
    get_model()   # warm-up


# ── Singleton Model Loading ───────────────────────────────────────────────────
_model      = None
_model_meta: dict = {}


def get_model():
    global _model, _model_meta
    if _model is None:
        _model = _load_model()
        meta_path = os.path.join(config.MODEL_DIR, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                _model_meta = json.load(f)
        print("[INFO] Model loaded and cached.")
    return _model


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction_id:    str
    predicted_class:  str
    confidence:       float
    probability:      float
    label_index:      int
    inference_time_ms: float
    saved_to_db:      bool = True


class UncertaintyResponse(BaseModel):
    prediction_id:    str
    predicted_class:  str
    confidence:       float
    mean_prob:        float
    uncertainty_std:  float
    label_index:      int
    n_passes:         int
    inference_time_ms: float
    saved_to_db:      bool = True


class EvaluationResponse(BaseModel):
    status:    str
    roc_auc:   Optional[float] = None
    avg_prec:  Optional[float] = None
    f1:        Optional[float] = None
    mcc:       Optional[float] = None
    kappa:     Optional[float] = None
    threshold: Optional[float] = None


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_name:   str
    timestamp:    float
    version:      str = "2.0.0"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_upload(file: UploadFile) -> tuple[np.ndarray, int]:
    """Read uploaded file → (numpy uint8 array, file_size_bytes)."""
    contents = file.file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    return np.array(img, dtype=np.uint8), len(contents)


# ── System Endpoints ──────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=(_model is not None),
        model_name=config.MODEL_NAME,
        timestamp=time.time(),
    )


@app.get("/model/info", tags=["System"])
def model_info():
    return {
        "model_name":  config.MODEL_NAME,
        "img_size":    config.IMG_SIZE,
        "class_names": config.CLASS_NAMES,
        "meta":        _model_meta,
    }


# ── Auth Endpoints ────────────────────────────────────────────────────────────
@app.post("/auth/register", response_model=UserPublic, tags=["Auth"], status_code=201)
async def register(data: RegisterRequest, request: Request):
    """Register a new user account."""
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    return register_user(data, ip=ip, ua=ua)


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(data: LoginRequest, request: Request):
    """Login and receive JWT access + refresh tokens."""
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    return login_user(data, ip=ip, ua=ua)


@app.post("/auth/logout", tags=["Auth"])
async def logout(
    request: Request,
    token:   str  = Depends(oauth2_scheme),
    current_user: dict = Depends(get_current_user),
):
    """Logout current session (revoke token)."""
    ip = request.client.host if request.client else None
    return logout_user(
        token=token,
        user_id=current_user.get("user_id"),
        username=current_user.get("username"),
        ip=ip,
    )


@app.post("/auth/logout-all", tags=["Auth"])
async def logout_all(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Logout from all devices (revoke all sessions)."""
    ip = request.client.host if request.client else None
    return logout_all_devices(
        user_id=current_user.get("user_id"),
        username=current_user.get("username"),
        ip=ip,
    )


@app.get("/auth/me", response_model=UserPublic, tags=["Auth"])
async def me(current_user: dict = Depends(get_current_active_user)):
    """Get current user's profile."""
    return UserPublic(
        id=current_user.get("user_id", ""),
        username=current_user["username"],
        email=current_user["email"],
        role=current_user["role"],
        is_active=current_user["is_active"],
        created_at=current_user.get("created_at", ""),
        last_login=current_user.get("last_login"),
    )


# ── Prediction Endpoints (Auth Required) ──────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file:         UploadFile = File(...),
    threshold:    float      = Form(default=0.5),
    current_user: dict       = Depends(get_current_active_user),
):
    """
    Upload a chest X-ray image (JPEG/PNG) and get a binary prediction.
    Requires authentication. Result is saved to prediction history.
    """
    model = get_model()
    img_array, img_size = _read_upload(file)

    t0 = time.perf_counter()
    result = predict_single(model, img_array, threshold=threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pid = str(uuid.uuid4())
    save_prediction(
        prediction_id=pid,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probability=result["probability"],
        label_index=result["label_index"],
        threshold=threshold,
        inference_time_ms=round(elapsed_ms, 2),
        model_name=config.MODEL_NAME,
        image_filename=file.filename,
        image_size_bytes=img_size,
        endpoint="predict",
        user_id=current_user.get("user_id"),
    )

    return PredictionResponse(
        prediction_id=pid,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probability=result["probability"],
        label_index=result["label_index"],
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post("/predict/uncertainty", response_model=UncertaintyResponse, tags=["Prediction"])
async def predict_uncertainty(
    file:         UploadFile = File(...),
    threshold:    float      = Form(default=0.5),
    n_passes:     int        = Form(default=30),
    current_user: dict       = Depends(get_current_active_user),
):
    """MC-Dropout uncertainty estimation. Requires authentication."""
    model = get_model()
    img_array, img_size = _read_upload(file)

    t0 = time.perf_counter()
    result = predict_with_uncertainty(model, img_array, n_passes=n_passes, threshold=threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pid = str(uuid.uuid4())
    save_prediction(
        prediction_id=pid,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probability=result["mean_prob"],
        label_index=result["label_index"],
        threshold=threshold,
        inference_time_ms=round(elapsed_ms, 2),
        model_name=config.MODEL_NAME,
        image_filename=file.filename,
        image_size_bytes=img_size,
        endpoint="uncertainty",
        n_passes=n_passes,
        uncertainty_std=result["uncertainty_std"],
        user_id=current_user.get("user_id"),
    )

    return UncertaintyResponse(
        prediction_id=pid,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        mean_prob=result["mean_prob"],
        uncertainty_std=result["uncertainty_std"],
        label_index=result["label_index"],
        n_passes=n_passes,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.get("/predictions", tags=["Prediction"])
async def get_predictions(
    limit:        int  = 50,
    offset:       int  = 0,
    current_user: dict = Depends(get_current_active_user),
):
    """Return the current user's prediction history."""
    is_admin = current_user.get("role") == "admin"
    uid      = None if is_admin else current_user.get("user_id")
    rows     = list_predictions(user_id=uid, limit=limit, offset=offset)
    return {"count": len(rows), "predictions": rows}


# ── Evaluation Endpoints (Radiologist / Admin only) ───────────────────────────
@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_model(
    background_tasks: BackgroundTasks,
    _user: dict = Depends(require_role("admin", "radiologist")),
):
    """Trigger full test-set evaluation. Admin/radiologist only."""
    def _run():
        try:
            run_evaluation()
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")

    background_tasks.add_task(_run)
    return EvaluationResponse(status="started — check logs/evaluation_results.json")


@app.get("/evaluate/results", tags=["Evaluation"])
def get_evaluation_results(
    _user: dict = Depends(require_role("admin", "radiologist")),
):
    res_path = os.path.join(config.LOG_DIR, "evaluation_results.json")
    if not os.path.exists(res_path):
        raise HTTPException(status_code=404, detail="No results found. Run POST /evaluate first.")
    with open(res_path) as f:
        return json.load(f)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
