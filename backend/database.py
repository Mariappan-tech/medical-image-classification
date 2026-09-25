"""
database.py — SQLite Database Layer for Medical Image Classification Backend
Handles:
  - User management (create, fetch, update)
  - Prediction history storage
  - Audit log persistence
  - Session tracking
"""

from __future__ import annotations

import os
import sys
import sqlite3
import uuid
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─── DB Path ──────────────────────────────────────────────────────────────────
DB_DIR  = os.path.join(config.BASE_DIR, "backend", "data")
DB_PATH = os.path.join(DB_DIR, "medical_app.db")
os.makedirs(DB_DIR, exist_ok=True)


# ─── Connection Context Manager ───────────────────────────────────────────────
@contextmanager
def get_db():
    """Thread-safe SQLite connection with WAL mode and row factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Schema Init ──────────────────────────────────────────────────────────────
def init_db():
    """Create all tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            -- ── Users ────────────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT    PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                username      TEXT    NOT NULL UNIQUE,
                email         TEXT    NOT NULL UNIQUE,
                hashed_password TEXT  NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'user',
                is_active     INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                last_login    TEXT
            );

            -- ── Sessions ─────────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS sessions (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash    TEXT NOT NULL UNIQUE,
                expires_at    TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                ip_address    TEXT,
                user_agent    TEXT,
                is_revoked    INTEGER NOT NULL DEFAULT 0
            );

            -- ── Predictions ───────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS predictions (
                id                TEXT PRIMARY KEY,
                user_id           TEXT REFERENCES users(id) ON DELETE SET NULL,
                predicted_class   TEXT NOT NULL,
                confidence        REAL NOT NULL,
                probability       REAL NOT NULL,
                label_index       INTEGER NOT NULL,
                threshold         REAL NOT NULL DEFAULT 0.5,
                inference_time_ms REAL,
                model_name        TEXT,
                image_filename    TEXT,
                image_size_bytes  INTEGER,
                endpoint          TEXT,
                n_passes          INTEGER,
                uncertainty_std   REAL,
                created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );

            -- ── Audit Logs ────────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS audit_logs (
                id          TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                user_id     TEXT REFERENCES users(id) ON DELETE SET NULL,
                username    TEXT,
                action      TEXT NOT NULL,
                resource    TEXT,
                status      TEXT NOT NULL,
                detail      TEXT,
                ip_address  TEXT,
                user_agent  TEXT,
                duration_ms REAL,
                created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );

            -- ── Indexes ──────────────────────────────────────────────────────
            CREATE INDEX IF NOT EXISTS idx_sessions_token    ON sessions(token_hash);
            CREATE INDEX IF NOT EXISTS idx_sessions_user     ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_user  ON predictions(user_id);
            CREATE INDEX IF NOT EXISTS idx_audit_user        ON audit_logs(user_id);
            CREATE INDEX IF NOT EXISTS idx_audit_action      ON audit_logs(action);
            CREATE INDEX IF NOT EXISTS idx_audit_created     ON audit_logs(created_at);
        """)
    print("[DB] Database initialised ->", DB_PATH)


# ─── User CRUD ────────────────────────────────────────────────────────────────
def create_user(username: str, email: str, hashed_password: str,
                role: str = "user") -> Dict[str, Any]:
    user_id = str(uuid.uuid4()).replace("-", "")
    with get_db() as conn:
        conn.execute(
            """INSERT INTO users (id, username, email, hashed_password, role)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, username, email, hashed_password, role),
        )
    return get_user_by_id(user_id)


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    return dict(row) if row else None


def update_last_login(user_id: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?", (now, user_id)
        )


def list_users(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,username,email,role,is_active,created_at,last_login FROM users LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def deactivate_user(user_id: str):
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET is_active = 0 WHERE id = ?", (user_id,)
        )


# ─── Session CRUD ─────────────────────────────────────────────────────────────
def create_session(user_id: str, token_hash: str, expires_at: str,
                   ip_address: str = None, user_agent: str = None) -> str:
    session_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO sessions (id, user_id, token_hash, expires_at, ip_address, user_agent)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, user_id, token_hash, expires_at, ip_address, user_agent),
        )
    return session_id


def get_session_by_token(token_hash: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            """SELECT s.*, u.username, u.email, u.role, u.is_active
               FROM sessions s JOIN users u ON s.user_id = u.id
               WHERE s.token_hash = ? AND s.is_revoked = 0""",
            (token_hash,),
        ).fetchone()
    return dict(row) if row else None


def revoke_session(token_hash: str):
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET is_revoked = 1 WHERE token_hash = ?", (token_hash,)
        )


def revoke_all_sessions(user_id: str):
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET is_revoked = 1 WHERE user_id = ?", (user_id,)
        )


# ─── Prediction CRUD ──────────────────────────────────────────────────────────
def save_prediction(
    prediction_id: str,
    predicted_class: str,
    confidence: float,
    probability: float,
    label_index: int,
    threshold: float = 0.5,
    inference_time_ms: float = None,
    model_name: str = None,
    image_filename: str = None,
    image_size_bytes: int = None,
    endpoint: str = "predict",
    n_passes: int = None,
    uncertainty_std: float = None,
    user_id: str = None,
) -> Dict[str, Any]:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, user_id, predicted_class, confidence, probability, label_index,
                threshold, inference_time_ms, model_name, image_filename,
                image_size_bytes, endpoint, n_passes, uncertainty_std)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                prediction_id, user_id, predicted_class, confidence, probability,
                label_index, threshold, inference_time_ms, model_name,
                image_filename, image_size_bytes, endpoint, n_passes, uncertainty_std,
            ),
        )
    return get_prediction_by_id(prediction_id)


def get_prediction_by_id(prediction_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
    return dict(row) if row else None


def list_predictions(user_id: str = None, limit: int = 50,
                     offset: int = 0) -> List[Dict[str, Any]]:
    with get_db() as conn:
        if user_id:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
    return [dict(r) for r in rows]


# ─── Audit Log CRUD ───────────────────────────────────────────────────────────
def insert_audit_log(
    action: str,
    status: str,
    user_id: str = None,
    username: str = None,
    resource: str = None,
    detail: dict = None,
    ip_address: str = None,
    user_agent: str = None,
    duration_ms: float = None,
):
    detail_str = json.dumps(detail) if detail else None
    with get_db() as conn:
        conn.execute(
            """INSERT INTO audit_logs
               (action, status, user_id, username, resource, detail,
                ip_address, user_agent, duration_ms)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (action, status, user_id, username, resource, detail_str,
             ip_address, user_agent, duration_ms),
        )


def list_audit_logs(
    action: str = None,
    user_id: str = None,
    status: str = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    clauses, params = [], []
    if action:   clauses.append("action = ?");  params.append(action)
    if user_id:  clauses.append("user_id = ?"); params.append(user_id)
    if status:   clauses.append("status = ?");  params.append(status)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params += [limit, offset]

    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM audit_logs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_audit_stats() -> Dict[str, Any]:
    with get_db() as conn:
        total   = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
        by_act  = conn.execute(
            "SELECT action, COUNT(*) as cnt FROM audit_logs GROUP BY action ORDER BY cnt DESC"
        ).fetchall()
        by_stat = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM audit_logs GROUP BY status"
        ).fetchall()
    return {
        "total": total,
        "by_action": [dict(r) for r in by_act],
        "by_status": [dict(r) for r in by_stat],
    }


# ─── Auto-init on import ──────────────────────────────────────────────────────
init_db()
